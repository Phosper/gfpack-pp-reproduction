#!/usr/bin/env python3
"""Audit saved predictions/GT without inference, training, or changing the baseline.

For failed cases, inspect the candidate with minimum *official thresholded*
overlap percentage, breaking ties by utilization. This is not guaranteed to be
the minimum exact-overlap candidate. Saved tensors must be trusted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def geometry(polygons, ids, actions):
    from shapely.geometry import Polygon
    placed = []
    for polygon_id, action in zip(ids, actions):
        theta = math.atan2(action[3], action[2])
        c, s = math.cos(theta), math.sin(theta)
        placed.append(Polygon([(x*c-y*s+action[0], x*s+y*c+action[1])
                               for x, y in polygons[polygon_id]]))
    if any(not p.is_valid or p.area <= 0 for p in placed):
        raise ValueError("Invalid input or transformed geometry")
    intersections = []
    for i, a in enumerate(placed):
        for j in range(i + 1, len(placed)):
            b = placed[j]
            if a.bounds[2] < b.bounds[0] or b.bounds[2] < a.bounds[0] or a.bounds[3] < b.bounds[1] or b.bounds[3] < a.bounds[1]:
                continue
            intersection = a.intersection(b)
            if intersection.area > 0:
                intersections.append((float(intersection.area), i, j))
    intersections.sort(reverse=True)
    total_area = sum(p.area for p in placed)
    bounds = [min(p.bounds[0] for p in placed), min(p.bounds[1] for p in placed),
              max(p.bounds[2] for p in placed), max(p.bounds[3] for p in placed)]
    def inside(height):
        return all(p.bounds[0] >= -1e-9 and p.bounds[1] >= -1e-9 and
                   p.bounds[2] <= 8000 + 1e-9 and p.bounds[3] <= height + 1e-9 for p in placed)
    return {"max_pairwise_area": intersections[0][0] if intersections else 0,
            "sum_pairwise_area": sum(x[0] for x in intersections),
            "sum_pairwise_overlap_percent": sum(x[0] for x in intersections)/total_area*100,
            "pair_count_over_1e_9": sum(x[0] > 1e-9 for x in intersections),
            "pair_count_over_50": sum(x[0] > 50 for x in intersections),
            "total_polygon_area": total_area, "bounds_xyxy": bounds,
            "within_8000x1200": inside(1200), "within_8000x1205": inside(1205),
            "top_pairs": [{"area": a, "piece_indices": [i, j]} for a, i, j in intersections[:3]]}, placed


def selection(metrics):
    enhanced = metrics["enhanced"]
    index = enhanced["best_official_valid_index"]
    if index is not None:
        return index, "best_official_valid_utilization"
    return min(range(enhanced["candidate_count"]), key=lambda i:
               (enhanced["reported_overlap_area_percent"][i], -enhanced["utilization"][i])), "minimum_official_thresholded_overlap_then_highest_utilization"


def render(path, sample, index, variants):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    for axis, (label, info, placed) in zip(axes, variants):
        for polygon in placed:
            x, y = polygon.exterior.xy
            axis.plot(x, y, color="#526775", linewidth=0.7)
        for pair in info["top_pairs"]:
            if pair["area"] <= 1e-9:
                continue
            a, b = pair["piece_indices"]
            for i in (a, b):
                x, y = placed[i].exterior.xy
                axis.plot(x, y, color="#b5483d", linewidth=1.1)
        axis.axhline(1200, color="#777777", linestyle="--", linewidth=0.6)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title("%s | max pair area %.6g | pairwise overlap %.6g%%" %
                       (label, info["max_pairwise_area"], info["sum_pairwise_overlap_percent"]), fontsize=10)
    fig.suptitle("Sample %03d | enhanced-selected candidate %d | red outlines: largest intersecting pairs" % (sample, index))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.results / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "completed" or summary["completed_cases"] != 128:
        raise ValueError("Expected complete verified baseline")
    verification = json.loads((args.results / "verification.json").read_text())
    if verification["status"] != "verified":
        raise ValueError("Baseline has not been verified")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in summary["rows"]:
        index = row["sample_index"]
        case = args.results / ("sample_%03d" % index)
        metrics = json.loads((case / "metrics.json").read_text())
        inputs = json.loads((case / "input_sample.json").read_text())
        polygons = json.loads((case / "input_geometry.json").read_text())["official_max_contours"]
        chosen, policy = selection(metrics)
        raw = torch.load(case / "predictions_raw.pt", map_location="cpu", weights_only=True)
        enhanced = torch.load(case / "predictions_enhanced.pt", map_location="cpu", weights_only=True)
        gt = torch.load(case / "gt_input.pt", map_location="cpu", weights_only=True)
        gt_enhanced = torch.load(case / "gt_enhanced.pt", map_location="cpu", weights_only=True)
        audits = {}
        for name, actions in (("raw_matched_candidate", raw[chosen]), ("enhanced_selected", enhanced[chosen]),
                              ("gt_input", gt), ("gt_enhanced", gt_enhanced)):
            audits[name], _ = geometry(polygons, inputs["polygon_ids"], actions.tolist())
        previous = metrics["enhanced_preview_independent_geometry"]
        if metrics["enhanced"]["best_official_valid_index"] is not None:
            if abs(audits["enhanced_selected"]["max_pairwise_area"] - previous["max_pairwise_intersection_area"]) > 1e-7:
                raise ValueError("Independent saved baseline geometry disagrees")
        rows.append({"sample_index": index, "selected_candidate": chosen, "selection_policy": policy,
                     "official_valid_candidate_count": metrics["enhanced"]["official_valid_count"],
                     "official_selected_utilization": metrics["enhanced"]["utilization"][chosen],
                     "official_selected_thresholded_overlap_percent": metrics["enhanced"]["reported_overlap_area_percent"][chosen],
                     "geometry": audits})
        print("audited %d/128" % (index + 1), flush=True)
    valid = [r for r in rows if r["official_valid_candidate_count"]]
    failed = [r for r in rows if not r["official_valid_candidate_count"]]
    def stats(group):
        values = [r["geometry"]["enhanced_selected"]["max_pairwise_area"] for r in group]
        return {"cases": len(group), "max_pair_area_min": min(values), "max_pair_area_median": statistics.median(values),
                "max_pair_area_max": max(values), "max_pair_area_le_0_1_cases": sum(x <= .1 for x in values),
                "within_8000x1200_cases": sum(r["geometry"]["enhanced_selected"]["within_8000x1200"] for r in group),
                "within_8000x1205_cases": sum(r["geometry"]["enhanced_selected"]["within_8000x1205"] for r in group)}
    examples = []
    for group in (valid, failed):
        ordered = sorted(group, key=lambda r: r["geometry"]["enhanced_selected"]["max_pairwise_area"])
        examples.extend([ordered[0], ordered[len(ordered)//2], ordered[-1]])
    for row in examples:
        case = args.results / ("sample_%03d" % row["sample_index"])
        inputs = json.loads((case / "input_sample.json").read_text())
        polygons = json.loads((case / "input_geometry.json").read_text())["official_max_contours"]
        variants = []
        for filename, label in (("predictions_raw.pt", "Raw matched candidate"),
                                ("predictions_enhanced.pt", "Enhanced selected candidate"),
                                ("gt_input.pt", "Original GT")):
            actions = torch.load(case / filename, map_location="cpu", weights_only=True)
            if actions.ndim == 3:
                actions = actions[row["selected_candidate"]]
            info, placed = geometry(polygons, inputs["polygon_ids"], actions.tolist())
            variants.append((label, info, placed))
        render(args.output / ("sample_%03d.png" % row["sample_index"]), row["sample_index"], row["selected_candidate"], variants)
    result = {"baseline_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
              "scope": "128 selected enhanced candidates, matched raw candidate, original/enhanced GT; not all-candidate exact geometry",
              "failure_selection_note": "Minimum official thresholded overlap, not guaranteed minimum true intersection",
              "official_valid_group": stats(valid), "no_official_valid_group": stats(failed),
              "gt_input_strict_pass_cases": sum(r["geometry"]["gt_input"]["max_pairwise_area"] <= 1e-9 for r in rows),
              "gt_enhanced_strict_pass_cases": sum(r["geometry"]["gt_enhanced"]["max_pairwise_area"] <= 1e-9 for r in rows),
              "example_samples": [r["sample_index"] for r in examples], "rows": rows}
    (args.output / "diagnosis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
