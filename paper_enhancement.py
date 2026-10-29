"""Documented post-generation constraint/gap reconstruction (Sec. 5.2).

The paper supplies a 100-iteration ceiling but no post-gradient step size,
stopping tolerances, or exact interleaving. We declare a unit repulsion step,
then x/y gap removal each iteration. The gap primitive is checked independently
with native unthresholded overlap/bounds; unsafe gap moves are rolled back.
This remains the default legacy_unit policy. The optional guarded_backtracking
policy adds independently measured step acceptance in its own module.
paper_fixed_step exposes only a declared fixed-step numerical probe; it does
not imply that the author used the selected step size.
"""
import math
import time

import numpy as np
from paper_metrics import STRICT_AREA_EPS, BOUNDARY_TOLERANCE


def field(native, polygons, ids, actions, height=1200., *, compute_repulsion=True):
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (len(ids), 4) or not np.isfinite(a).all():
        raise ValueError("Expected finite (batch,pieces,4) actions")
    angles = np.arctan2(a[:, :, 3], a[:, :, 2]).tolist()
    query = (native.paper_measure_all if not compute_repulsion and hasattr(native, "paper_measure_all")
             else native.paper_repulsion_all)
    result = query(ids, angles, a[:, :, :2].tolist(), polygons, height, 0.)
    return result, angles


def enhance(native, polygons, ids, actions, max_iterations=100, height=1200., *,
            policy="legacy_unit", trace_candidates=False, post_step_size=1.):
    if isinstance(post_step_size, bool) or not math.isfinite(post_step_size) or post_step_size <= 0:
        raise ValueError("post_step_size must be positive and finite")
    if policy not in ("paper_fixed_step","paper_potential_descent") and post_step_size != 1.:
        raise ValueError("Non-unit steps require the explicit paper_fixed_step numerical probe")
    if policy=="paper_potential_descent":
        from paper_potential_enhancement import enhance_potential
        return enhance_potential(native,polygons,ids,actions,max_iterations,height,
            initial_step=post_step_size,trace_candidates=trace_candidates)
    # Keep the historical round-5 path replayable, without changing its output.
    if policy == "guarded_backtracking":
        from paper_guarded_enhancement import enhance_guarded
        return enhance_guarded(native, polygons, ids, actions, max_iterations, height)
    if policy not in ("legacy_unit", "paper_fixed_step"):
        raise ValueError("Unknown post policy: %s" % policy)
    if type(max_iterations) is not int or not 0 <= max_iterations <= 100:
        raise ValueError("post iterations must be in [0,100]")
    a = np.array(actions, dtype=np.float64, copy=True)
    original = a.copy()
    initial, _ = field(native, polygons, ids, a, height, compute_repulsion=False)
    log = []
    started = time.perf_counter()
    gap_rollbacks = np.zeros(len(a), dtype=np.int64)
    statuses = ["iteration_limit"] * len(a)
    active = np.ones(len(a), dtype=bool)
    reference = getattr(native, "geometry_backend", None) == "paper_sat"
    stopping_area = STRICT_AREA_EPS if reference else 0.
    for iteration in range(max_iterations):
        previous = a.copy()
        force, angles = field(native, polygons, ids, a, height)
        repulsion = np.asarray(force["repulsion"], dtype=np.float64)
        a[active, :, :2] += post_step_size * repulsion[active]
        if not np.isfinite(a).all():
            raise FloatingPointError("Nonfinite post-generation constraint update")
        before_gap, angles = field(native, polygons, ids, a, height, compute_repulsion=False)
        after_collision = before_gap
        # A common x translation preserves every collision and the strip width.
        a[active, :, 0] -= np.asarray(before_gap["min_x"])[active, None]
        # Clipper quantization has a fixed origin, so measure again after the
        # non-grid-aligned global translation before comparing gap effects.
        before_gap, angles = field(native, polygons, ids, a, height, compute_repulsion=False)
        # Translation of the whole candidate is only a horizontal gauge choice.
        # Obtain minx from the unmodified official metric primitive; no x wall.
        # Gap code requires nonnegative starting x, as in the release entrypoint.
        candidate_gap = a.copy()
        xy = native.paper_gap_xy_all(ids, angles, a[:, :, :2].tolist(), polygons)
        candidate_gap[:, :, :2] = np.asarray(xy, dtype=np.float64)
        after_gap, _ = field(native, polygons, ids, candidate_gap, height, compute_repulsion=False)
        rollback = (np.asarray(after_gap["sum_pairwise_area"]) >
                    np.asarray(before_gap["sum_pairwise_area"]) + 1e-9) | (
                    np.asarray(after_gap["height_violation"]) >
                    np.asarray(before_gap["height_violation"]) + 1e-9)
        gap_rollbacks += rollback & active
        accept = (~rollback) & active
        a[accept] = candidate_gap[accept]
        a[~active] = previous[~active]
        checked, _ = field(native, polygons, ids, a, height, compute_repulsion=False)
        moved = np.linalg.norm(a[:, :, :2]-previous[:, :, :2], axis=-1).max(axis=1)
        # These are numerical reconstruction conventions, not paper constants.
        done = active & (moved <= 1e-3) & (np.asarray(checked["max_pairwise_area"]) <= stopping_area) & (
            np.asarray(checked["height_violation"]) <= BOUNDARY_TOLERANCE)
        for i in np.flatnonzero(done):
            statuses[int(i)] = ("reference_converged" if
                               getattr(native, "geometry_backend", None) == "paper_sat" else
                               "native_converged")
        active[done] = False
        record = {"iteration_1based": iteration+1, "active_candidates": int(active.sum()),
                    "newly_converged": int(done.sum()), "gap_rollbacks": int(rollback.sum()),
                    "max_pairwise_area": float(max(checked["max_pairwise_area"])),
                    "max_height_violation": float(max(checked["height_violation"])),
                    "max_translation_update": float(moved.max())}
        if trace_candidates:
            def geometry_rows(value):
                return [{"candidate_index": i, **{key: float(value[key][i]) for key in
                         ("max_pairwise_area", "sum_pairwise_area", "height_violation", "min_x")}}
                        for i in range(len(a))]
            record["candidate_stages"] = {
                "before_collision": geometry_rows(force),
                "after_collision_before_gauge": geometry_rows(after_collision),
                "after_collision_and_x_gauge": geometry_rows(before_gap),
                "after_gap_trial": geometry_rows(after_gap),
                "after_accepted_iteration": geometry_rows(checked),
                "gap_rolled_back": rollback.tolist(),
                "active_after_iteration": active.tolist(),
            }
            # The reference backend records x/y separately. Native legacy
            # exposes only the combined call, so do not invent substage data.
            if hasattr(native, "last_gap_trace"):
                record["candidate_stages"]["gap_substages"] = native.last_gap_trace
        log.append(record)
        if not active.any():
            break
    final, _ = field(native, polygons, ids, a, height, compute_repulsion=False)
    return a, {"max_iterations": max_iterations, "iterations_executed": len(log),
               "constraint_step_size": float(post_step_size),
               "interleaving": "fixed repulsion step then -x/-y gap", "post_policy": policy,
               "post_step_size_and_order_are_reconstruction_assumptions": True,
               "gap_tolerance": 1e-3, "overlap_stop_native_area": stopping_area,
               "stopping_tolerance_matches_reference_selection": reference,
               "stopping_motion_tolerance": 1e-3,
               "height": height, "horizontal_right_wall": None,
               "statuses": statuses, "gap_rollbacks_per_candidate": gap_rollbacks.tolist(),
               "initial_native_geometry": {k:v for k,v in initial.items() if k != "repulsion"},
               "final_native_geometry": {k:v for k,v in final.items() if k != "repulsion"},
               "trace": log, "elapsed_seconds": time.perf_counter()-started,
               "rotation_components_preserved": bool(np.array_equal(a[:, :, 2:], original[:, :, 2:])),
               "per_candidate_stage_trace": bool(trace_candidates),
               "geometry_backend": getattr(native, "geometry_backend", "legacy_native")}


def enhance_chunked(native, polygons, ids, actions, max_iterations=100, height=1200., *,
                    policy="legacy_unit", trace_candidates=False, chunk_size=1, post_step_size=1.):
    """Bound the reference backend's rotating-contour cache, independently per candidate.

    Chunking changes neither the collision field nor the 100-step per-candidate
    budget. Detailed traces retain their local indices under an explicit global
    candidate mapping; the top-level trace is an iteration summary only.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 3 or not len(a):
        raise ValueError("Expected nonempty (batch,pieces,4) actions")
    if (len(a) <= chunk_size or getattr(native, "geometry_backend", None) != "paper_sat" or
            policy == "guarded_backtracking"):
        return enhance(native, polygons, ids, a, max_iterations, height,
                       policy=policy, trace_candidates=trace_candidates, post_step_size=post_step_size)
    started = time.perf_counter()
    final, chunks = [], []
    for offset in range(0, len(a), chunk_size):
        result, post = enhance(native, polygons, ids, a[offset:offset+chunk_size], max_iterations,
                               height, policy=policy, trace_candidates=trace_candidates, post_step_size=post_step_size)
        final.append(result)
        chunks.append({"candidate_indices": list(range(offset, offset+len(result))), "post": post})
    iterations = max(c["post"]["iterations_executed"] for c in chunks)
    summary = []
    for i in range(iterations):
        executed = [c["post"]["trace"][i] for c in chunks if i < len(c["post"]["trace"])]
        latest = [c["post"]["trace"][min(i, len(c["post"]["trace"])-1)] for c in chunks if c["post"]["trace"]]
        summary.append({"iteration_1based": i+1,
            "active_candidates": sum(r["active_candidates"] for r in executed),
            "newly_converged": sum(r["newly_converged"] for r in executed),
            "gap_rollbacks": sum(r["gap_rollbacks"] for r in executed),
            "max_pairwise_area": max(r["max_pairwise_area"] for r in latest),
            "max_height_violation": max(r["max_height_violation"] for r in latest),
            "max_translation_update": max(r["max_translation_update"] for r in executed)})
    return np.concatenate(final), {"schema_version": 2, "geometry_backend": "paper_sat",
        "post_policy": policy, "max_iterations": max_iterations, "iterations_executed": iterations,
        "constraint_step_size": float(post_step_size), "chunk_size": chunk_size, "chunks": chunks, "trace": summary,
        "trace_scope": "top-level iteration summaries; detailed local traces under chunks with global candidate_indices",
        "statuses": [s for c in chunks for s in c["post"]["statuses"]],
        "gap_rollbacks_per_candidate": [s for c in chunks for s in c["post"]["gap_rollbacks_per_candidate"]],
        "rotation_components_preserved": all(c["post"]["rotation_components_preserved"] for c in chunks),
        "post_step_size_and_order_are_reconstruction_assumptions": True,
        "elapsed_seconds": time.perf_counter()-started}
