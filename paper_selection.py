"""Select from measured collision-free saved layouts (paper Sec. 5.2).

Numerical epsilons belong to this project's evaluation contract, not to the
paper. No relaxed-threshold or candidate-zero fallback is a final solution.
"""
from paper_metrics import STRICT_AREA_EPS, BOUNDARY_TOLERANCE


def select_saved_result(aggregates, storage_dtype="float64"):
    if storage_dtype not in ("float64", "fp32"):
        raise ValueError("Result dtype must be float64 or fp32")
    stage = "enhanced_" + storage_dtype
    criterion = "reference_strict_fixed_strip"
    report = aggregates[stage]
    selections = report["criteria"][criterion]["sample_selections"]
    if len(selections) != 1:
        raise ValueError("Expected a single sample for result selection")
    item = selections[0]
    best = item["best"]["reference_strip_utilization"]
    index = best["candidate_index"]
    return {"status": "feasible" if index is not None else "no_feasible_candidate",
            "sample_index": item["sample_index"], "candidate_index": index,
            "strip_utilization": best["value"], "source_stage": stage,
            "criterion": criterion, "feasible_candidates": item["feasible_candidate_count"],
            "candidate_denominator": report["planned_candidates"],
            "strict_area_epsilon": STRICT_AREA_EPS,
            "boundary_tolerance": BOUNDARY_TOLERANCE,
            "thresholds_are_project_conventions": True,
            "storage_dtype": storage_dtype,
            "selection_rule": "highest utilization among independently valid saved " + storage_dtype + " candidates",
            "fallback_used": False}
