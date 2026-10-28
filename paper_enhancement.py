"""Documented post-generation constraint/gap reconstruction (Sec. 5.2).

The paper supplies a 100-iteration ceiling but no post-gradient step size,
stopping tolerances, or exact interleaving. We declare a unit repulsion step,
then x/y gap removal each iteration. The gap primitive is checked independently
with native unthresholded overlap/bounds; unsafe gap moves are rolled back.
This remains the default legacy_unit policy. The optional guarded_backtracking
policy adds independently measured step acceptance in its own module.
"""
import math
import time

import numpy as np


def field(native, polygons, ids, actions, height=1200.):
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (len(ids), 4) or not np.isfinite(a).all():
        raise ValueError("Expected finite (batch,pieces,4) actions")
    angles = np.arctan2(a[:, :, 3], a[:, :, 2]).tolist()
    result = native.paper_repulsion_all(ids, angles, a[:, :, :2].tolist(), polygons, height, 0.)
    return result, angles


def enhance(native, polygons, ids, actions, max_iterations=100, height=1200., *,
            policy="legacy_unit"):
    # Keep the historical round-5 path replayable, without changing its output.
    if policy == "guarded_backtracking":
        from paper_guarded_enhancement import enhance_guarded
        return enhance_guarded(native, polygons, ids, actions, max_iterations, height)
    if policy != "legacy_unit":
        raise ValueError("Unknown post policy: %s" % policy)
    if type(max_iterations) is not int or not 0 <= max_iterations <= 100:
        raise ValueError("post iterations must be in [0,100]")
    a = np.array(actions, dtype=np.float64, copy=True)
    original = a.copy()
    initial, _ = field(native, polygons, ids, a, height)
    log = []
    started = time.perf_counter()
    gap_rollbacks = np.zeros(len(a), dtype=np.int64)
    statuses = ["iteration_limit"] * len(a)
    active = np.ones(len(a), dtype=bool)
    for iteration in range(max_iterations):
        previous = a.copy()
        force, angles = field(native, polygons, ids, a, height)
        repulsion = np.asarray(force["repulsion"], dtype=np.float64)
        a[active, :, :2] += repulsion[active]
        if not np.isfinite(a).all():
            raise FloatingPointError("Nonfinite post-generation constraint update")
        before_gap, angles = field(native, polygons, ids, a, height)
        # A common x translation preserves every collision and the strip width.
        a[active, :, 0] -= np.asarray(before_gap["min_x"])[active, None]
        # Clipper quantization has a fixed origin, so measure again after the
        # non-grid-aligned global translation before comparing gap effects.
        before_gap, angles = field(native, polygons, ids, a, height)
        # Translation of the whole candidate is only a horizontal gauge choice.
        # Obtain minx from the unmodified official metric primitive; no x wall.
        # Gap code requires nonnegative starting x, as in the release entrypoint.
        candidate_gap = a.copy()
        xy = native.paper_gap_xy_all(ids, angles, a[:, :, :2].tolist(), polygons)
        candidate_gap[:, :, :2] = np.asarray(xy, dtype=np.float64)
        after_gap, _ = field(native, polygons, ids, candidate_gap, height)
        rollback = (np.asarray(after_gap["sum_pairwise_area"]) >
                    np.asarray(before_gap["sum_pairwise_area"]) + 1e-9) | (
                    np.asarray(after_gap["height_violation"]) >
                    np.asarray(before_gap["height_violation"]) + 1e-9)
        gap_rollbacks += rollback & active
        accept = (~rollback) & active
        a[accept] = candidate_gap[accept]
        a[~active] = previous[~active]
        checked, _ = field(native, polygons, ids, a, height)
        moved = np.linalg.norm(a[:, :, :2]-previous[:, :, :2], axis=-1).max(axis=1)
        # These are numerical reconstruction conventions, not paper constants.
        done = active & (moved <= 1e-3) & (np.asarray(checked["max_pairwise_area"]) == 0) & (
            np.asarray(checked["height_violation"]) <= 1e-9)
        for i in np.flatnonzero(done):
            statuses[int(i)] = "native_converged"
        active[done] = False
        log.append({"iteration_1based": iteration+1, "active_candidates": int(active.sum()),
                    "newly_converged": int(done.sum()), "gap_rollbacks": int(rollback.sum()),
                    "max_pairwise_area": float(max(checked["max_pairwise_area"])),
                    "max_height_violation": float(max(checked["height_violation"])),
                    "max_translation_update": float(moved.max())})
        if not active.any():
            break
    final, _ = field(native, polygons, ids, a, height)
    return a, {"max_iterations": max_iterations, "iterations_executed": len(log),
               "constraint_step_size": 1., "interleaving": "unit repulsion then -x/-y gap",
               "post_step_size_and_order_are_reconstruction_assumptions": True,
               "gap_tolerance": 1e-3, "overlap_stop_native_area": 0.,
               "stopping_motion_tolerance": 1e-3,
               "height": height, "horizontal_right_wall": None,
               "statuses": statuses, "gap_rollbacks_per_candidate": gap_rollbacks.tolist(),
               "initial_native_geometry": {k:v for k,v in initial.items() if k != "repulsion"},
               "final_native_geometry": {k:v for k,v in final.items() if k != "repulsion"},
               "trace": log, "elapsed_seconds": time.perf_counter()-started,
               "rotation_components_preserved": bool(np.array_equal(a[:, :, 2:], original[:, :, 2:]))}
