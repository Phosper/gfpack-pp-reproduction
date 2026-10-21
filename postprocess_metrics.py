"""Independent float64 geometry for frozen post-processing ablations.

Only Python's standard library and Shapely are used. Each summary row is one
selected frozen prediction or one GT through one path, never a 512-candidate
batch. Callers must supply failed rows as well as successful rows so planned
denominators remain intact. No clipping, repair, coordinate quantization, or
8000-wide right boundary is applied here.
"""

from __future__ import annotations

from collections import defaultdict
import math
import operator
import statistics


AREA_EPS = 50.0
STRICT_AREA_EPS = 1e-9
BOUNDARY_EPS = 1e-9
UTIL_EPS = 1e-12


def _finite(value, label):
    if isinstance(value, bool):
        raise ValueError("%s must be finite numeric data" % label)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite numeric data" % label)
    return value


def _actions(actions):
    rows = []
    for i, action in enumerate(actions):
        if len(action) != 4:
            raise ValueError("action %d must have four components" % i)
        rows.append([_finite(v, "action %d" % i) for v in action])
    if not rows:
        raise ValueError("at least one action is required")
    return rows


def geometry(polygons, ids, actions, height=1200):
    """Return unthresholded pair intersections and strip/bounding-box metrics.

    Polygons are indexed by ``ids``; each polygon is a simple contour of xy
    vertices. Action columns are x, y, rotation-cos component, rotation-sin
    component. Rotation is atan2(sin, cos), matching the released evaluator;
    a zero rotation vector is reported and interpreted as zero radians.
    Pair counts use > 50 and > 1e-9. The corresponding feasible thresholds
    are <= 50 and <= 1e-9, respectively. The latter is a local diagnostic,
    not a claim of exact mathematical disjointness.
    """
    from shapely.geometry import Polygon

    h = _finite(height, "height")
    if h <= 0:
        raise ValueError("height must be positive")
    actions = _actions(actions)
    ids = list(ids)
    if len(ids) != len(actions):
        raise ValueError("ids and actions must have the same length")
    placed = []
    original_area = 0.0
    zero_rotation_vectors = 0
    for i, (polygon_id, action) in enumerate(zip(ids, actions)):
        if isinstance(polygon_id, bool):
            raise ValueError("polygon id must be an integer")
        polygon_id = operator.index(polygon_id)
        if polygon_id < 0 or polygon_id >= len(polygons):
            raise ValueError("polygon id is outside the polygon list")
        vertices = []
        for vertex in polygons[polygon_id]:
            if len(vertex) != 2:
                raise ValueError("polygon vertices must have two coordinates")
            vertices.append((_finite(vertex[0], "vertex x"),
                             _finite(vertex[1], "vertex y")))
        source = Polygon(vertices)
        if not source.is_valid or source.is_empty or source.area <= 0:
            raise ValueError("invalid source polygon at action %d" % i)
        original_area += float(source.area)
        x, y, rc, rs = action
        zero_rotation_vectors += int(rc == 0 and rs == 0)
        theta = math.atan2(rs, rc)
        c, s = math.cos(theta), math.sin(theta)
        transformed = Polygon([(vx * c - vy * s + x, vx * s + vy * c + y)
                               for vx, vy in vertices])
        if not transformed.is_valid or transformed.is_empty or transformed.area <= 0:
            raise ValueError("invalid transformed polygon at action %d" % i)
        if any(not math.isfinite(v) for v in transformed.bounds):
            raise ValueError("nonfinite transformed bounds at action %d" % i)
        placed.append(transformed)

    intersections = []
    for i, first in enumerate(placed):
        ax0, ay0, ax1, ay1 = first.bounds
        for j in range(i + 1, len(placed)):
            second = placed[j]
            bx0, by0, bx1, by1 = second.bounds
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            area = float(first.intersection(second).area)
            if not math.isfinite(area) or area < 0:
                raise ValueError("invalid pair intersection area")
            if area > 0:
                intersections.append((area, i, j))
    intersections.sort(reverse=True)
    maximum = intersections[0][0] if intersections else 0.0
    total_overlap = math.fsum(item[0] for item in intersections)
    total_area = math.fsum(float(p.area) for p in placed)
    bounds = [min(p.bounds[0] for p in placed),
              min(p.bounds[1] for p in placed),
              max(p.bounds[2] for p in placed),
              max(p.bounds[3] for p in placed)]
    span_x, span_y = bounds[2] - bounds[0], bounds[3] - bounds[1]
    if span_x <= 0 or span_y <= 0 or not math.isfinite(total_area):
        raise ValueError("nonpositive or nonfinite aggregate geometry")
    lower_violation = max(0.0, -bounds[1])
    upper_violation = max(0.0, bounds[3] - h)
    violation = max(lower_violation, upper_violation)
    within_strip = violation <= BOUNDARY_EPS
    return {
        "status": "measured", "piece_count": len(placed),
        "max_pairwise_area": maximum, "sum_pairwise_area": total_overlap,
        "pair_count_over_50": sum(a > AREA_EPS for a, _, _ in intersections),
        "pair_count_over_1e_9": sum(a > STRICT_AREA_EPS for a, _, _ in intersections),
        "positive_area_pair_count": len(intersections),
        "total_polygon_area": total_area,
        "original_polygon_area": original_area,
        "rigid_area_absolute_error": abs(total_area - original_area),
        "sum_pairwise_overlap_fraction": total_overlap / total_area,
        "bounds_xyxy": bounds, "span_x": span_x, "span_y": span_y,
        "bbox_utilization": total_area / (span_x * span_y),
        "strip_utilization": total_area / (h * span_x),
        "strip_height": h, "height_violation": violation,
        "lower_y_violation": lower_violation, "upper_y_violation": upper_violation,
        "within_strip": within_strip,
        "eps50_overlap_valid": maximum <= AREA_EPS,
        "strict_overlap_valid": maximum <= STRICT_AREA_EPS,
        "eps50_strip_valid": maximum <= AREA_EPS and within_strip,
        "strict_strip_valid": maximum <= STRICT_AREA_EPS and within_strip,
        "zero_rotation_vector_count": zero_rotation_vectors,
        "top_pairs": [{"area": a, "piece_indices": [i, j]}
                      for a, i, j in intersections[:3]],
        "horizontal_domain": "infinite_strip_no_x_wall",
        "area_eps": AREA_EPS, "strict_area_eps": STRICT_AREA_EPS,
        "boundary_eps": BOUNDARY_EPS,
    }


def compare(before, after, input_actions, output_actions):
    """Describe geometric change and absolute/relative per-piece translation.

    ``rotation_unchanged`` compares the two raw rotation components exactly;
    changing their common scale is therefore also reported as a change.
    Translation RMS is sqrt(mean(dx**2 + dy**2)) over pieces. Removing the
    mean translation gives relative_translation_rms, so a global x gauge
    normalization cannot be mistaken for rearranging the pieces.
    """
    start, finish = _actions(input_actions), _actions(output_actions)
    if len(start) != len(finish):
        raise ValueError("input and output actions must have the same length")
    if before.get("status") != "measured" or after.get("status") != "measured":
        raise ValueError("compare requires measured geometry")
    if before["strip_height"] != after["strip_height"]:
        raise ValueError("cannot compare different strip heights")
    if before["piece_count"] != len(start) or after["piece_count"] != len(finish):
        raise ValueError("geometry and action piece counts disagree")
    dx = [b[0] - a[0] for a, b in zip(start, finish)]
    dy = [b[1] - a[1] for a, b in zip(start, finish)]
    mean_x = math.fsum(dx) / len(dx)
    mean_y = math.fsum(dy) / len(dy)
    distances = [math.hypot(x, y) for x, y in zip(dx, dy)]
    rotation_unchanged = all(a[2:] == b[2:] for a, b in zip(start, finish))
    rotation_component_max_change = max(abs(b[k] - a[k])
                                        for a, b in zip(start, finish)
                                        for k in (2, 3))
    delta = {name: after[name] - before[name] for name in (
        "max_pairwise_area", "sum_pairwise_area", "height_violation",
        "bbox_utilization", "strip_utilization", "span_x", "span_y")}
    overlap_worse = (delta["max_pairwise_area"] > STRICT_AREA_EPS or
                     delta["sum_pairwise_area"] > STRICT_AREA_EPS)
    overlap_improved = (not overlap_worse and
                        (delta["max_pairwise_area"] < -STRICT_AREA_EPS or
                         delta["sum_pairwise_area"] < -STRICT_AREA_EPS))
    x_gauge_only = (rotation_unchanged and all(y == 0 for y in dy) and
                    all(abs(x - mean_x) <= BOUNDARY_EPS for x in dx))
    invariant_checks = {
        name: math.isclose(before[name], after[name], rel_tol=1e-10, abs_tol=1e-7)
        for name in ("max_pairwise_area", "sum_pairwise_area",
                     "bbox_utilization", "strip_utilization", "height_violation",
                     "span_x", "span_y", "total_polygon_area")
    }
    return {
        "delta": delta,
        "overlap_improved": overlap_improved, "overlap_worse": overlap_worse,
        "bounds_improved": delta["height_violation"] < -BOUNDARY_EPS,
        "bounds_worse": delta["height_violation"] > BOUNDARY_EPS,
        "bbox_utilization_drop": delta["bbox_utilization"] < -UTIL_EPS,
        "strip_utilization_drop": delta["strip_utilization"] < -UTIL_EPS,
        "became_eps50_strip_valid": (not before["eps50_strip_valid"] and
                                        after["eps50_strip_valid"]),
        "lost_eps50_strip_validity": (before["eps50_strip_valid"] and
                                       not after["eps50_strip_valid"]),
        "became_strict_strip_valid": (not before["strict_strip_valid"] and
                                        after["strict_strip_valid"]),
        "lost_strict_strip_validity": (before["strict_strip_valid"] and
                                       not after["strict_strip_valid"]),
        "translation_rms": math.sqrt(math.fsum(d * d for d in distances) / len(dx)),
        "translation_max": max(distances),
        "translation_mean_xy": [mean_x, mean_y],
        "relative_translation_rms": math.sqrt(math.fsum(
            (x - mean_x) ** 2 + (y - mean_y) ** 2 for x, y in zip(dx, dy)) / len(dx)),
        "rotation_unchanged": rotation_unchanged,
        "rotation_component_max_change": rotation_component_max_change,
        "x_gauge_only_change": x_gauge_only,
        "x_gauge_invariance": all(invariant_checks.values()) if x_gauge_only else None,
        "x_gauge_invariance_checks": invariant_checks if x_gauge_only else None,
    }


def _stats(values):
    return ({"count": len(values), "min": min(values), "median": statistics.median(values),
             "mean": math.fsum(values) / len(values), "max": max(values)}
            if values else {"count": 0, "min": None, "median": None,
                            "mean": None, "max": None})


def _summarize_group(rows):
    planned = len(rows)
    measured = [r for r in rows if r["status"] == "completed"]
    comparisons = [r["comparison"] for r in measured if r.get("comparison") is not None]
    result = {
        "planned_outputs": planned, "measured_outputs": len(measured),
        "failed_outputs": planned - len(measured),
        "planned_unique_inputs": len({(r["sample_index"], r["source"]) for r in rows}),
        "planned_unique_samples": len({r["sample_index"] for r in rows}),
    }
    for criterion in ("eps50_overlap_valid", "strict_overlap_valid",
                      "eps50_strip_valid", "strict_strip_valid", "within_strip"):
        count = sum(bool(r["after"][criterion]) for r in measured)
        result[criterion] = {
            "outputs": count, "fraction_all_planned": count / planned if planned else None,
            "fraction_measured_only": count / len(measured) if measured else None,
        }
    for metric in ("max_pairwise_area", "sum_pairwise_area", "height_violation",
                   "bbox_utilization", "strip_utilization"):
        result[metric] = _stats([r["after"][metric] for r in measured])
    result["comparison_count"] = len(comparisons)
    result["comparison_flags"] = {
        flag: sum(bool(c[flag]) for c in comparisons)
        for flag in ("overlap_improved", "overlap_worse", "bounds_improved",
                     "bounds_worse", "bbox_utilization_drop", "strip_utilization_drop",
                     "became_eps50_strip_valid", "lost_eps50_strip_validity",
                     "became_strict_strip_valid", "lost_strict_strip_validity",
                     "rotation_unchanged", "x_gauge_only_change")}
    for metric in ("translation_rms", "translation_max", "relative_translation_rms"):
        result[metric] = _stats([c[metric] for c in comparisons])
    for metric in ("max_pairwise_area", "sum_pairwise_area", "height_violation",
                   "bbox_utilization", "strip_utilization"):
        result["delta_" + metric] = _stats([c["delta"][metric] for c in comparisons])
    for criterion in ("eps50_strip_valid", "strict_strip_valid"):
        valid_utils = [r["after"]["strip_utilization"] for r in measured
                       if r["after"][criterion]]
        result[criterion + "_strip_utilization"] = {
            "conditional_valid_only": _stats(valid_utils),
            "mean_all_planned_failure_zero": math.fsum(valid_utils) / planned if planned else None,
        }
    return result


def summarize(rows):
    """Group explicit output rows, retaining every supplied failure denominator.

    Required row keys: sample_index, source (pred/gt), baseline_group
    (success/failure), path, status (completed/failed), after (geometry or None).
    Optional comparison is the output of compare; output_precision defaults to
    float64. Baseline groups refer to the *frozen baseline prediction sample*
    and must be copied to both its prediction and its GT rows. They are never
    recomputed from the post-processing result. A duplicate input/path/precision
    row is an error. Coverage against a manifest is the caller's responsibility.
    """
    rows = list(rows)
    seen = set()
    for row in rows:
        index = row["sample_index"]
        if isinstance(index, bool) or operator.index(index) < 0:
            raise ValueError("sample_index must be a nonnegative integer")
        if row["source"] not in ("pred", "gt"):
            raise ValueError("source must be pred or gt")
        if row["baseline_group"] not in ("success", "failure"):
            raise ValueError("baseline_group must be success or failure")
        if not isinstance(row["path"], str) or not row["path"]:
            raise ValueError("path must be a nonempty string")
        if row["status"] not in ("completed", "failed"):
            raise ValueError("status must be completed or failed")
        precision = row.get("output_precision", "float64")
        if precision not in ("float64", "float32"):
            raise ValueError("output_precision must be float64 or float32")
        key = (index, row["source"], row["path"], precision)
        if key in seen:
            raise ValueError("duplicate input/path/precision output row")
        seen.add(key)
        if row["status"] == "completed":
            if not isinstance(row.get("after"), dict) or row["after"].get("status") != "measured":
                raise ValueError("completed rows require measured after geometry")
        elif row.get("after") is not None:
            raise ValueError("failed rows must have after=None; partial metrics belong in diagnostics")
    def grouped(keys):
        groups = defaultdict(list)
        for row in rows:
            value = tuple(row.get(key, "float64") if key == "output_precision"
                          else row[key] for key in keys)
            groups[value].append(row)
        return [{**dict(zip(keys, value)), **_summarize_group(group)}
                for value, group in sorted(groups.items())]
    return {
        "unit": "one frozen selected prediction or GT through one path and output precision",
        "denominator_note": "Not a 512-candidate rate; failed supplied rows remain in all-planned fractions.",
        "coverage_note": "The runner must add failed rows for missing planned outputs and verify manifest coverage.",
        "overall": _summarize_group(rows),
        "by_path": grouped(("path",)), "by_source": grouped(("source",)),
        "by_baseline_group": grouped(("baseline_group",)),
        "by_output_precision": grouped(("output_precision",)),
        "by_path_source_group": grouped(("path", "source", "baseline_group", "output_precision")),
    }


def self_test():
    """Small independent fixtures test geometry, change reporting and failures."""
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    polygons = [square]
    actions = [[0, 0, 500, 0], [5, 0, 500, 0]]
    overlap = geometry(polygons, [0, 0], actions)
    assert overlap["max_pairwise_area"] == 50
    assert overlap["pair_count_over_50"] == 0 and overlap["eps50_strip_valid"]
    assert overlap["pair_count_over_1e_9"] == 1 and not overlap["strict_strip_valid"]
    assert math.isclose(overlap["bbox_utilization"], 200 / 150)
    assert math.isclose(overlap["strip_utilization"], 200 / (1200 * 15))
    clear_actions = [[0, 0, 500, 0], [10, 0, 500, 0]]
    clear = geometry(polygons, [0, 0], clear_actions)
    assert clear["strict_strip_valid"] and clear["sum_pairwise_area"] == 0
    change = compare(overlap, clear, actions, clear_actions)
    assert change["overlap_improved"] and not change["overlap_worse"]
    assert change["rotation_unchanged"] and change["translation_max"] == 5
    assert math.isclose(change["translation_rms"], math.sqrt(12.5))
    far_actions = [[10000, 0, 500, 0], [10010, 0, 500, 0]]
    far = geometry(polygons, [0, 0], far_actions)
    gauge = compare(clear, far, clear_actions, far_actions)
    assert far["strict_strip_valid"] and gauge["x_gauge_invariance"]
    assert gauge["x_gauge_only_change"] and gauge["relative_translation_rms"] == 0
    below = geometry(polygons, [0], [[0, -0.25, 500, 0]])
    assert below["height_violation"] == 0.25 and not below["eps50_strip_valid"]
    above = geometry(polygons, [0], [[0, 1195, 500, 0]])
    assert above["height_violation"] == 5
    rotated_actions = [[0, 10, 0, -500]]
    rotated = geometry(polygons, [0], rotated_actions)
    assert rotated["strict_strip_valid"]
    assert math.isclose(rotated["total_polygon_area"], 100)
    original = geometry(polygons, [0], [[0, 0, 500, 0]])
    assert not compare(original, rotated, [[0, 0, 500, 0]], rotated_actions)["rotation_unchanged"]
    rows = [{"sample_index": 0, "source": "pred", "baseline_group": "success",
             "path": "compression_only", "status": "completed", "after": clear,
             "comparison": compare(clear, clear, clear_actions, clear_actions)},
            {"sample_index": 1, "source": "pred", "baseline_group": "failure",
             "path": "compression_only", "status": "failed", "after": None}]
    summary = summarize(rows)
    assert summary["overall"]["planned_outputs"] == 2
    assert summary["overall"]["failed_outputs"] == 1
    assert summary["overall"]["strict_strip_valid"]["fraction_all_planned"] == 0.5
    assert summary["overall"]["strict_strip_valid"]["fraction_measured_only"] == 1
    assert summary["by_baseline_group"][0]["baseline_group"] == "failure"
    assert summary["by_baseline_group"][0]["planned_outputs"] == 1
    try:
        summarize(rows + [rows[0]])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate output rows must be rejected")
    try:
        geometry(polygons, [0], [[0, math.nan, 1, 0]])
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite inputs must be rejected")
    return {"status": "passed", "checks": 12,
            "scope": "independent CPU fixtures, not real-case reproduction evidence"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2, allow_nan=False))
