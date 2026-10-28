#!/usr/bin/env python3
"""Compare two post solvers on the exact same saved generated candidates.

No model, checkpoint, dataset pickle, CUDA, or sampling is used. A matching
paper-constraint extension and the original hashed raw artifacts are required.
These are development replays, not evidence of full paper reproduction.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

from run_inference import sha256, utc_now, write_json


POLICIES = ("legacy_unit", "guarded_backtracking")
REQUIRED = ("input_sample.json", "input_geometry.json", "predictions_raw.pt",
            "raw_metrics.json")
STRICT_AREA_EPS = 1e-9
BOUNDARY_TOLERANCE = 1e-9
HEIGHT = 1200.


def load_verified_inputs(source, extension):
    """Check recorded bytes before loading tensor data or native code."""
    source, extension = Path(source), Path(extension)
    meta_path = source / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed" or metadata.get("changed_inputs") != []:
        raise ValueError("Source run must be completed with an empty changed_inputs list")
    manifest = metadata.get("output_sha256", {})
    frozen = {str(meta_path.resolve()): sha256(meta_path)}
    for name in REQUIRED:
        path = source / name
        expected = manifest.get(name)
        if not expected or not path.is_file() or sha256(path) != expected:
            raise ValueError("Missing or hash-mismatched source artifact: " + name)
        frozen[str(path.resolve())] = expected
    build_path = extension / "build.json"
    build = json.loads(build_path.read_text(encoding="utf-8"))
    if not build.get("paper_constraints") or not build.get("source_unchanged"):
        raise ValueError("Build lacks verified paper constraint entrypoints")
    if build.get("upstream_commit") != "ff75815e24ec50cd7fa8b2abdec1acfe0b55be95":
        raise ValueError("Expected the pinned upstream extension")
    historical_inputs = metadata.get("input_sha256", {})
    historical_builds = [digest for path, digest in historical_inputs.items()
                         if path.replace("\\", "/").rsplit("/", 1)[-1] == "build.json"]
    if len(historical_builds) != 1 or sha256(build_path) != historical_builds[0]:
        raise ValueError("Replay requires the exact build.json used by the source run")
    native_hash = build.get("extension_sha256")
    historical_native = [digest for path, digest in historical_inputs.items()
                         if path.replace("\\", "/").rsplit("/", 1)[-1].startswith("collision_probe.")]
    if not native_hash or len(historical_native) != 1 or historical_native[0] != native_hash:
        raise ValueError("Source run does not bind this native extension hash")
    frozen[str(build_path.resolve())] = sha256(build_path)
    sample = json.loads((source / "input_sample.json").read_text(encoding="utf-8"))
    geometry = json.loads((source / "input_geometry.json").read_text(encoding="utf-8"))
    raw_metrics = json.loads((source / "raw_metrics.json").read_text(encoding="utf-8"))
    return metadata, build, sample, geometry, raw_metrics, frozen


def original_bbox_min_x(raw_metrics, sample_index, candidate_count):
    """Use saved native bbox, avoiding a different quantization/gauge choice."""
    samples = raw_metrics.get("samples", [])
    if len(samples) != 1 or samples[0].get("sample_index") != sample_index:
        raise ValueError("raw_metrics must describe this single source sample")
    row = samples[0]
    candidates = row.get("candidates", [])
    if row.get("planned_candidate_count") != candidate_count or len(candidates) != candidate_count:
        raise ValueError("Raw metric candidate denominator differs from saved predictions")
    result = []
    for index, item in enumerate(candidates):
        if item.get("candidate_index") != index or item.get("native_status") != "measured":
            raise ValueError("Raw native candidate records must be measured and ordered")
        bbox = item.get("native_bounds_xyxy")
        if (not isinstance(bbox, list) or len(bbox) != 4 or
                not all(math.isfinite(float(x)) for x in bbox) or
                bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            raise ValueError("Invalid saved native bounding box")
        result.append(float(bbox[0]))
    return result


def reference_report(polygons, ids, actions):
    """Report independent float64 geometry for each saved dtype; no native eps50."""
    from run_paper_alignment import exact_all
    rows = exact_all(polygons, ids, actions.tolist())
    for index, row in enumerate(rows):
        x0, y0, x1, y1 = row["bounds_xyxy"]
        span = x1 - x0
        violation = max(0., -y0, y1 - HEIGHT)
        row.update(candidate_index=index, span_x=span, height_violation=violation,
                   strip_utilization=row["total_polygon_area"] / (HEIGHT * span),
                   feasible=(row["max_pairwise_area"] <= STRICT_AREA_EPS and
                             violation <= BOUNDARY_TOLERANCE))
    eligible = [row for row in rows if row["feasible"]]
    best = max(eligible, key=lambda r: (r["strip_utilization"], -r["candidate_index"]), default=None)
    count = len(rows)
    return {"candidate_count": count, "feasible_count": len(eligible),
            "feasible_fraction": len(eligible) / count,
            "best_feasible_candidate_index": best["candidate_index"] if best else None,
            "best_feasible_strip_utilization": best["strip_utilization"] if best else None,
            "mean_sum_pairwise_overlap_fraction_all_candidates": math.fsum(
                r["sum_pairwise_area"] / r["total_polygon_area"] for r in rows) / count,
            "mean_max_pairwise_area_all_candidates": math.fsum(
                r["max_pairwise_area"] for r in rows) / count,
            "max_pairwise_area": max(r["max_pairwise_area"] for r in rows),
            "max_height_violation": max(r["height_violation"] for r in rows),
            "semantics": {"geometry": "independent float64 GEOS; saved fp32 first round-tripped",
                          "strict_area_epsilon": STRICT_AREA_EPS,
                          "boundary_tolerance": BOUNDARY_TOLERANCE, "strip_height": HEIGHT,
                          "horizontal_right_wall": None,
                          "criterion_is_project_convention_not_paper_constant": True,
                          "sum_pairwise_area": "multiway intersections may be counted more than once"},
            "candidates": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--post-iterations", type=int, default=100)
    args = parser.parse_args()
    if not 0 <= args.post_iterations <= 100:
        parser.error("post iterations must be in [0,100]")
    source, extension, out = args.source_run.resolve(), args.extension.resolve(), args.output.resolve()
    if out.exists() or any(protected == out or protected in out.parents or out in protected.parents
                           for protected in (source, extension)):
        raise ValueError("Output must be a new directory outside source run and extension trees")
    metadata, build, sample, geometry, raw_metrics, frozen = load_verified_inputs(source, extension)
    import numpy as np
    import torch
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(extension))
    import collision_probe
    native_path = Path(collision_probe.__file__).resolve()
    if native_path.parent != extension or sha256(native_path) != build["extension_sha256"]:
        raise ValueError("Loaded extension differs from the frozen source-run extension")
    frozen[str(native_path)] = sha256(native_path)
    from paper_enhancement import enhance
    raw = torch.load(source / "predictions_raw.pt", map_location="cpu", weights_only=True)
    if not isinstance(raw, torch.Tensor) or raw.dtype not in (torch.float32, torch.float64):
        raise ValueError("Expected saved float32/float64 predictions tensor")
    ids = sample["polygon_ids"]
    polygons = geometry["official_max_contours"]
    count = metadata["planned_candidates"]
    if (raw.shape != (count, len(ids), 4) or not torch.isfinite(raw).all() or
            sample["sample_index"] != metadata["sample_index"] or not ids or
            not all(type(i) is int and 0 <= i < len(polygons) for i in ids)):
        raise ValueError("Saved input geometry, sample, predictions, or denominator disagree")
    min_x = original_bbox_min_x(raw_metrics, sample["sample_index"], count)
    post_input = raw.numpy().astype(np.float64, copy=True)
    post_input[:, :, 0] -= np.asarray(min_x)[:, None]
    for name in ("replay_paper_enhancement.py", "run_paper_alignment.py", "paper_enhancement.py",
                 "paper_guarded_enhancement.py", "paper_metrics.py", "run_inference.py"):
        path = Path(__file__).resolve().with_name(name)
        frozen[str(path)] = sha256(path)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    meta = {"status": "running", "started_utc": utc_now(), "source_run": str(source),
            "sample_index": sample["sample_index"], "planned_candidates": count,
            "source_profile": metadata.get("profile"), "post_iterations": args.post_iterations,
            "policies": list(POLICIES), "input_sha256": frozen,
            "same_generated_candidates_for_both_policies": True,
            "post_input_gauge": "subtract saved raw_metrics native bbox min_x only; no y reset",
            "model_executed": False, "paper_full_reproduction": False,
            "scope": "postprocessing development replay; not an independent benchmark",
            "source_manifest_scope": "consistency with supplied metadata, not external authenticity proof",
            "software": {"python": sys.version, "pytorch": str(torch.__version__)}}
    try:
        torch.save(torch.from_numpy(post_input.copy()), out / "post_input_float64.pt")
        before = reference_report(polygons, ids, post_input)
        write_json(out / "post_input_reference.json", before)
        summaries = {}
        for policy in POLICIES:
            destination = out / policy
            destination.mkdir()
            final, diagnostics = enhance(collision_probe, polygons, ids, post_input.copy(),
                                         args.post_iterations, HEIGHT, policy=policy)
            final64 = torch.tensor(final, dtype=torch.float64)
            final32 = final64.to(dtype=torch.float32)
            torch.save(final64, destination / "predictions_enhanced_float64.pt")
            torch.save(final32, destination / "predictions_enhanced_fp32.pt")
            write_json(destination / "post.json", diagnostics)
            reports = {"float64": reference_report(polygons, ids, final64.numpy()),
                       "fp32": reference_report(polygons, ids, final32.numpy().astype(np.float64))}
            for dtype, report in reports.items():
                write_json(destination / (dtype + "_reference.json"), report)
            summaries[policy] = {"policy": policy, "statuses": diagnostics.get("statuses"),
                                 "iterations_executed": diagnostics.get("iterations_executed"),
                                 **{dtype: {k: v for k, v in report.items() if k != "candidates"}
                                    for dtype, report in reports.items()}}
        write_json(out / "comparison.json", {"sample_index": sample["sample_index"],
                   "same_post_input": True, "before": {k: v for k, v in before.items() if k != "candidates"},
                   "policies": summaries})
        meta["status"] = "completed"
    except Exception as exc:
        meta["status"] = "failed"
        meta["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        meta["finished_utc"] = utc_now()
        meta["elapsed_seconds"] = time.perf_counter() - started
        meta["changed_inputs"] = [p for p, digest in frozen.items()
                                  if not Path(p).is_file() or sha256(p) != digest]
        if meta["changed_inputs"]:
            meta["status"] = "failed"
            meta["input_change_error"] = "Frozen inputs changed; comparison must not be accepted"
        meta["output_sha256"] = {str(p.relative_to(out)): sha256(p)
                                  for p in out.rglob("*") if p.is_file()}
        write_json(out / "metadata.json", meta)
        print(json.dumps({k: meta[k] for k in ("status", "sample_index", "planned_candidates",
                                              "elapsed_seconds", "changed_inputs")}))
    if meta["status"] != "completed":
        raise RuntimeError("Replay inputs changed during execution")


if __name__ == "__main__":
    main()
