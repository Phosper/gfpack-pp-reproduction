"""Pure, denominator-explicit metrics for the Dental paper-alignment run.

No official extension, model, file, or network is accessed here. ``native`` is
the unmodified output of run_inference.native_metrics (calutil, area eps=50).
``references`` is one float64 geometry record per candidate, with the keys
produced by diagnose_results.geometry: max_pairwise_area, sum_pairwise_area,
total_polygon_area, and bounds_xyxy. A missing/failed record remains explicitly
unmeasured. All planned candidates, including execution failures, are retained.

Native validity checks overlap only. Reference strip validity additionally
checks y in [0, 1200]; horizontal translation is immaterial in the infinite
horizontal strip and there is no 8000 width cap. Ratios are fractions, not %.
"""

from __future__ import annotations

import copy
import math


STRIP_HEIGHT = 1200.0
NATIVE_AREA_EPS = 50.0
STRICT_AREA_EPS = 1e-9
BOUNDARY_TOLERANCE = 1e-9

_CRITERIA = (
    "native_eps50_overlap_only",
    "native_eps50_fixed_strip",
    "reference_eps50_overlap_only",
    "reference_eps50_fixed_strip",
    "reference_strict_overlap_only",
    "reference_strict_fixed_strip",
)
_UTILIZATIONS = (
    "native_bbox_utilization", "reference_bbox_utilization",
    "reference_strip_utilization",
)


def _finite(value, name, *, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError("%s must be a finite number" % name)
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError("%s must be finite%s" %
                         (name, " and nonnegative" if nonnegative else ""))
    return result


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _bounds(value, name):
    if len(value) != 4:
        raise ValueError("%s must contain x0, y0, x1, y1" % name)
    result = [_finite(item, name) for item in value]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("%s must have positive width and height" % name)
    return result


def _reference(record, index):
    """Validate independently computed geometry; never infer it from native."""
    if record is None:
        return {"status": "not_measured"}
    result = copy.deepcopy(dict(record))
    status = result.get("status", "measured")
    if status != "measured":
        result["status"] = status
        return result
    bounds = _bounds(result["bounds_xyxy"], "reference[%d].bounds" % index)
    area = _finite(result["total_polygon_area"], "reference total area",
                   nonnegative=True)
    if area <= 0:
        raise ValueError("reference total area must be positive")
    maximum = _finite(result["max_pairwise_area"], "reference max overlap",
                      nonnegative=True)
    overlap_sum = _finite(result["sum_pairwise_area"], "reference overlap sum",
                          nonnegative=True)
    if maximum > overlap_sum + 1e-9 * max(1.0, maximum):
        raise ValueError("reference max overlap exceeds overlap sum")
    span_x, span_y = bounds[2] - bounds[0], bounds[3] - bounds[1]
    result.update(
        status="measured", bounds_xyxy=bounds, total_polygon_area=area,
        max_pairwise_area=maximum, sum_pairwise_area=overlap_sum,
        span_x=span_x, span_y=span_y,
        bbox_utilization=area / (span_x * span_y),
        strip_utilization=area / (STRIP_HEIGHT * span_x),
        strip_height=STRIP_HEIGHT,
        strip_vertical_violation=max(0.0, -bounds[1], bounds[3] - STRIP_HEIGHT),
        within_fixed_strip=(bounds[1] >= -BOUNDARY_TOLERANCE and
                            bounds[3] <= STRIP_HEIGHT + BOUNDARY_TOLERANCE),
        sum_pairwise_overlap_fraction=overlap_sum / area,
    )
    return result


def sample_metrics(sample_index, native, references=None, *,
                   planned_candidate_count=None, status="completed", error=None):
    """Return every candidate and its independent validity/utility quantities.

    Failed samples use native=None and an explicit planned_candidate_count.
    Failed individual float64 checks can use {"status": "failed", "error": ...}.
    ``None`` means not measured, and is never counted as measured geometry.
    Input arrays must include every native candidate; malformed input raises.
    """
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("sample_index must be a nonnegative integer")
    if native is None:
        count = _count(planned_candidate_count, "planned_candidate_count")
    else:
        count = _count(native["candidate_count"], "native candidate_count")
        if planned_candidate_count is not None and planned_candidate_count != count:
            raise ValueError("native and planned candidate counts disagree")
        for key in ("official_valid", "utilization", "bounding_boxes_xyxy",
                    "reported_overlap_area_percent"):
            if len(native[key]) != count:
                raise ValueError("native %s must include every candidate" % key)
        for valid in native["official_valid"]:
            if not isinstance(valid, bool):
                raise ValueError("native official_valid values must be booleans")
        if ("official_valid_count" in native and
                sum(native["official_valid"]) != native["official_valid_count"]):
            raise ValueError("native official_valid_count disagrees with flags")
    if references is None:
        references = [None] * count
    if len(references) != count:
        raise ValueError("references must include an entry for every candidate")
    candidates = []
    for index in range(count):
        reference = _reference(references[index], index)
        measured = reference["status"] == "measured"
        native_valid = native["official_valid"][index] if native is not None else None
        native_util = (_finite(native["utilization"][index], "native utilization",
                               nonnegative=True) if native is not None else None)
        native_bbox = (_bounds(native["bounding_boxes_xyxy"][index], "native bbox")
                       if native is not None else None)
        native_overlap = (_finite(native["reported_overlap_area_percent"][index],
                                  "native reported overlap", nonnegative=True)
                          if native is not None else None)
        in_strip = reference["within_fixed_strip"] if measured else None
        ref_eps50 = reference["max_pairwise_area"] <= NATIVE_AREA_EPS if measured else None
        ref_strict = reference["max_pairwise_area"] <= STRICT_AREA_EPS if measured else None
        # False AND unknown is definitely false; true AND unknown is unmeasured.
        def conjunction(left, right):
            return False if left is False or right is False else (
                True if left is True and right is True else None)
        candidates.append({
            "candidate_index": index,
            "native_status": "measured" if native is not None else "not_measured",
            "native_official_valid": native_valid,
            "native_bbox_utilization": native_util,
            "native_bounds_xyxy": native_bbox,
            "native_reported_thresholded_overlap_percent": native_overlap,
            "reference": reference,
            "reference_bbox_utilization": reference["bbox_utilization"] if measured else None,
            "reference_strip_utilization": reference["strip_utilization"] if measured else None,
            "feasible": {
                "native_eps50_overlap_only": native_valid,
                "native_eps50_fixed_strip": conjunction(native_valid, in_strip),
                "reference_eps50_overlap_only": ref_eps50,
                "reference_eps50_fixed_strip": conjunction(ref_eps50, in_strip),
                "reference_strict_overlap_only": ref_strict,
                "reference_strict_fixed_strip": conjunction(ref_strict, in_strip),
            },
        })
    return {"sample_index": sample_index, "status": status, "error": error,
            "planned_candidate_count": count, "candidates": candidates}


def aggregate_metrics(samples):
    """Aggregate all planned samples/candidates without discarding failures.

    The candidate rate denominator is every planned candidate, even when a
    sample failed execution. Unmeasured counts and a measured-only rate are
    reported separately. ``mean_batch_feasible_fraction`` is the unweighted
    mean of per-sample candidate rates (paper Sec. 6.2); it equals the pooled
    candidate rate only when all samples have the same planned batch size.

    Each utility chooses its own best feasible candidate. Failed/no-feasible
    samples contribute 0 to the all-sample mean; conditional means and their
    denominators are also reported. No unmeasured reference utility is imputed.
    """
    rows = copy.deepcopy(list(samples))
    if not rows:
        raise ValueError("at least one planned sample is required")
    seen = set()
    for row in rows:
        index = row["sample_index"]
        if index in seen:
            raise ValueError("duplicate sample_index %s" % index)
        seen.add(index)
        count = _count(row["planned_candidate_count"], "planned candidate count")
        if len(row["candidates"]) != count:
            raise ValueError("sample must retain every planned candidate")
        if [item["candidate_index"] for item in row["candidates"]] != list(range(count)):
            raise ValueError("candidate indices must be complete and ordered")
    total = sum(row["planned_candidate_count"] for row in rows)
    criteria = {}
    for criterion in _CRITERIA:
        feasible_count = 0
        infeasible_count = 0
        batch_fractions = []
        selections = []
        utilities = {name: [] for name in _UTILIZATIONS}
        for row in rows:
            candidates = row["candidates"]
            eligible = [item for item in candidates if item["feasible"][criterion] is True]
            feasible_count += len(eligible)
            infeasible_count += sum(item["feasible"][criterion] is False for item in candidates)
            batch_fractions.append(len(eligible) / row["planned_candidate_count"])
            best = {}
            for name in _UTILIZATIONS:
                measured = [item for item in eligible if item[name] is not None]
                chosen = max(measured, key=lambda item: (item[name], -item["candidate_index"])) if measured else None
                if chosen is not None:
                    utilities[name].append(chosen[name])
                best[name] = {
                    "candidate_index": chosen["candidate_index"] if chosen else None,
                    "value": chosen[name] if chosen else None,
                    "failure_as_zero": chosen[name] if chosen else 0.0,
                    "feasible_candidates_missing_this_utilization": len(eligible) - len(measured),
                }
            selections.append({"sample_index": row["sample_index"],
                               "feasible_candidate_count": len(eligible),
                               "unmeasured_feasibility_count": sum(
                                   item["feasible"][criterion] is None for item in candidates),
                               "best": best})
        measured_count = feasible_count + infeasible_count
        utility_summary = {}
        for name, values in utilities.items():
            utility_summary[name] = {
                "samples_with_measured_feasible_best": len(values),
                "all_sample_denominator": len(rows),
                "mean_best_all_samples_failure_as_zero": math.fsum(values) / len(rows),
                "mean_best_measured_feasible_samples_only": math.fsum(values) / len(values) if values else None,
                "min_best_measured_feasible_samples_only": min(values) if values else None,
                "max_best_measured_feasible_samples_only": max(values) if values else None,
                "all_sample_mean_is_fully_measured": all(
                    item["unmeasured_feasibility_count"] == 0 and
                    item["best"][name]["feasible_candidates_missing_this_utilization"] == 0
                    for item in selections),
            }
        criteria[criterion] = {
            "candidate_denominator": total, "feasible_candidates": feasible_count,
            "infeasible_candidates": infeasible_count,
            "unmeasured_candidates": total - measured_count,
            "feasible_fraction_all_planned_candidates": feasible_count / total,
            "mean_batch_feasible_fraction": math.fsum(batch_fractions) / len(rows),
            "feasible_fraction_measured_candidates_only": feasible_count / measured_count if measured_count else None,
            "samples_with_feasible_candidate": sum(item["feasible_candidate_count"] > 0 for item in selections),
            "sample_denominator": len(rows), "utilization": utility_summary,
            "sample_selections": selections,
        }
    return {
        "schema_version": 1, "planned_samples": len(rows),
        "planned_candidates": total,
        "sample_status_counts": {status: sum(row["status"] == status for row in rows)
                                 for status in sorted({row["status"] for row in rows})},
        "semantics": {
            "native_area_eps": NATIVE_AREA_EPS,
            "strict_area_eps": STRICT_AREA_EPS,
            "boundary_tolerance": BOUNDARY_TOLERANCE,
            "fixed_strip_height": STRIP_HEIGHT,
            "horizontal_width_limit": None,
            "bbox_utilization": "sum polygon area / (spanX * spanY)",
            "strip_utilization": "sum polygon area / (1200 * spanX)",
            "overlap_sum": "sum of pairwise intersections; multiway intersections may be counted more than once",
            "native_overlap": "official quantized calutil eps=50; no boundary check",
            "reference_overlap": "independent float64 geometry; not native calutil",
            "paper_mean_failure_policy": "not specified for batches with no feasible candidate; report both means",
            "paper_dataset_identity": "unconfirmed; released validation PKL need not be the paper's newly generated test set",
            "partial_reference_warning": "unmeasured geometry/utility remains explicit; partial rates are lower bounds over the planned denominator",
        },
        "criteria": criteria, "samples": rows,
    }
