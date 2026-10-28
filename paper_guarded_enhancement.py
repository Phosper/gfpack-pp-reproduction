"""Explicit numerical variant of the Sec. 5.2 reconstruction, not author code.

Keep the existing Eq. 9 translation field and native -x/-y gap primitive.
Only change step selection/acceptance: a candidate's measured violations may
not increase. Backtracking, tolerances and interleaving are our conventions;
they are not specified by the paper and do not guarantee a feasible solution.
"""
import math
import time

import numpy as np
from shapely.geometry import Polygon

from paper_enhancement import field


AREA_EPS = 1e-9
BOUNDARY_EPS = 1e-9
MOTION_EPS = 1e-3
STEP_SIZES = tuple(2. ** -i for i in range(11))
VIOLATIONS = ("sum_pairwise_area", "max_pairwise_area", "height_violation")


def reference_geometry(polygons, ids, actions, height=1200.):
    """Independent GEOS/float64 measurement, without Clipper quantization.

    This is a floating-point geometric check, not exact arithmetic. In
    particular, native overlap==0 alone is never treated as convergence.
    """
    a = np.asarray(actions, dtype=np.float64)
    if (a.ndim != 3 or a.shape[1:] != (len(ids), 4) or not len(a) or
            not len(ids) or not np.isfinite(a).all()):
        raise ValueError("Expected nonempty finite (batch,pieces,4) actions")
    if not math.isfinite(height) or height <= 0:
        raise ValueError("Expected a positive finite strip height")
    contours = []
    for idx in ids:
        if not isinstance(idx, (int, np.integer)) or not 0 <= idx < len(polygons):
            raise ValueError("Invalid polygon ID")
        points = np.asarray(polygons[idx], dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or
                not np.isfinite(points).all()):
            raise ValueError("Invalid polygon contour")
        contours.append(points)
    records = []
    for candidate in a:
        shapes, bounds = [], []
        for points, action in zip(contours, candidate):
            theta = math.atan2(action[3], action[2])
            c, s = math.cos(theta), math.sin(theta)
            xy = points @ np.array([[c, s], [-s, c]]) + action[:2]
            shape = Polygon(xy)
            if not shape.is_valid or shape.area <= 0:
                raise ValueError("Invalid transformed contour; no automatic shape repair")
            shapes.append(shape)
            bounds.append(shape.bounds)
        boxes = np.asarray(bounds)
        mask = np.triu((boxes[:, None, 0] < boxes[None, :, 2]) &
                       (boxes[:, None, 2] > boxes[None, :, 0]) &
                       (boxes[:, None, 1] < boxes[None, :, 3]) &
                       (boxes[:, None, 3] > boxes[None, :, 1]), k=1)
        overlaps = [shapes[i].intersection(shapes[j]).area
                    for i, j in zip(*np.nonzero(mask))]
        x0, y0 = boxes[:, :2].min(axis=0)
        x1, y1 = boxes[:, 2:].max(axis=0)
        maximum = max(overlaps, default=0.)
        violation = max(0., -float(y0), float(y1) - height)
        records.append({
            "max_pairwise_area": float(maximum),
            "sum_pairwise_area": math.fsum(overlaps),
            "height_violation": violation,
            "span_x": float(x1 - x0), "min_x": float(x0),
            "bounds_xyxy": [float(x0), float(y0), float(x1), float(y1)],
            "total_polygon_area": math.fsum(p.area for p in shapes),
            "feasible": bool(maximum <= AREA_EPS and violation <= BOUNDARY_EPS),
        })
    return records


def _nonworsening(before, after):
    # No positive slack: even a previously feasible layout cannot accumulate
    # a new overlap through repeated tolerance-sized accepted increases.
    return all(after[key] <= before[key] for key in VIOLATIONS)


def _improving(before, after):
    # Reject improvements indistinguishable from numerical noise. These
    # relative tolerances select steps; they do not redefine feasibility.
    return any(before[key] - after[key] > 1e-12 * max(1., abs(before[key]))
               for key in VIOLATIONS)


def _native_geometry(result):
    return {key: value for key, value in result.items() if key != "repulsion"}


def enhance_guarded(native, polygons, ids, actions, max_iterations=100, height=1200.):
    if type(max_iterations) is not int or not 0 <= max_iterations <= 100:
        raise ValueError("post iterations must be in [0,100]")
    started = time.perf_counter()
    a = np.array(actions, dtype=np.float64, copy=True)
    initial_reference = reference_geometry(polygons, ids, a, height)
    original = a.copy()
    initial_native, _ = field(native, polygons, ids, a, height)
    count = len(a)
    statuses = ["iteration_limit"] * count
    active = np.ones(count, dtype=bool)
    gap_rollbacks = np.zeros(count, dtype=np.int64)
    rejected_trials = np.zeros(count, dtype=np.int64)
    log = []
    current_reference = initial_reference
    for iteration in range(max_iterations):
        previous = a.copy()
        before = current_reference
        force, _ = field(native, polygons, ids, a, height)
        repulsion = np.asarray(force["repulsion"], dtype=np.float64)
        if repulsion.shape != a[:, :, :2].shape or not np.isfinite(repulsion).all():
            raise FloatingPointError("Invalid post-generation translation field")
        steps = np.zeros(count)
        attempts = np.zeros(count, dtype=np.int64)
        accepted_repair = np.zeros(count, dtype=bool)
        after_repair = list(before)
        full_step = [None] * count
        pending = active & np.array([not row["feasible"] for row in before])
        for step in STEP_SIZES:
            indices = np.flatnonzero(pending)
            if not len(indices):
                break
            trials = previous[indices].copy()
            trials[:, :, :2] += step * repulsion[indices]
            attempts[indices] += 1
            if not np.isfinite(trials).all():
                raise FloatingPointError("Nonfinite trial collision update")
            trial_geometry = reference_geometry(polygons, ids, trials, height)
            for local, index in enumerate(indices):
                measured = trial_geometry[local]
                if step == STEP_SIZES[0]:
                    full_step[index] = measured
                if _nonworsening(before[index], measured) and _improving(before[index], measured):
                    a[index] = trials[local]
                    after_repair[index] = measured
                    steps[index] = step
                    accepted_repair[index] = True
                    pending[index] = False
                else:
                    rejected_trials[index] += 1
        # Gap has its own transaction. Its gauge translation and compression
        # must together pass the independent check against accepted repair.
        # Rejected gap never discards a good repair or preserves a bad gap.
        indices = np.flatnonzero(active)
        gap_input = a[indices].copy()
        gap_input[:, :, 0] -= np.array([after_repair[i]["min_x"] for i in indices])[:, None]
        angles = np.arctan2(gap_input[:, :, 3], gap_input[:, :, 2]).tolist()
        xy = np.asarray(native.paper_gap_xy_all(
            ids, angles, gap_input[:, :, :2].tolist(), polygons), dtype=np.float64)
        if xy.shape != gap_input[:, :, :2].shape or not np.isfinite(xy).all():
            raise FloatingPointError("Invalid gap update")
        gap_trial = gap_input.copy()
        gap_trial[:, :, :2] = xy
        trial_geometry = reference_geometry(polygons, ids, gap_trial, height)
        accepted_gap = np.zeros(count, dtype=bool)
        after_gap = list(after_repair)
        proposed_gap = [None] * count
        for local, index in enumerate(indices):
            measured = trial_geometry[local]
            proposed_gap[index] = measured
            if (_nonworsening(after_repair[index], measured) and
                    measured["span_x"] <= after_repair[index]["span_x"]):
                a[index] = gap_trial[local]
                after_gap[index] = measured
                accepted_gap[index] = True
            else:
                gap_rollbacks[index] += 1
        moved = np.linalg.norm(a[:, :, :2] - previous[:, :, :2], axis=-1).max(axis=1)
        feasible = np.array([row["feasible"] for row in after_gap])
        done = active & feasible & (moved <= MOTION_EPS)
        # A small accepted repair can still be useful. Stop as stalled only
        # when repair has no acceptable step and gap produced no motion.
        stalled = active & ~feasible & ~accepted_repair & (moved <= 1e-12)
        for index in np.flatnonzero(done):
            statuses[index] = "reference_converged"
        for index in np.flatnonzero(stalled):
            statuses[index] = "stalled_infeasible"
        active[done | stalled] = False
        log.append({
            "iteration_1based": iteration + 1,
            "active_candidates": int(active.sum()),
            "newly_converged": int(done.sum()), "newly_stalled": int(stalled.sum()),
            "repair_step_per_candidate": steps.tolist(),
            "repair_trials_per_candidate": attempts.tolist(),
            "repair_accepted_per_candidate": accepted_repair.tolist(),
            "gap_accepted_per_candidate": accepted_gap.tolist(),
            "gap_rollbacks": int(len(indices) - accepted_gap.sum()),
            "before_repair_reference_geometry": before,
            "full_step_trial_reference_geometry": full_step,
            "after_repair_reference_geometry": after_repair,
            "gap_trial_reference_geometry": proposed_gap,
            "after_gap_reference_geometry": after_gap,
            "max_pairwise_area": max(row["max_pairwise_area"] for row in after_gap),
            "max_height_violation": max(row["height_violation"] for row in after_gap),
            "max_translation_update": float(moved.max()),
        })
        current_reference = after_gap
        if not active.any():
            break
    final_native, _ = field(native, polygons, ids, a, height)
    # Casting includes angular components: serialization can change theta too.
    with np.errstate(over="raise", invalid="raise"):
        stored = a.astype(np.float32).astype(np.float64)
    stored_reference = reference_geometry(polygons, ids, stored, height)
    return a, {
        "policy": "guarded_backtracking", "schema_version": 1,
        "max_iterations": max_iterations, "iterations_executed": len(log),
        "constraint_step_sizes_tried": list(STEP_SIZES),
        "interleaving": "guarded repulsion then guarded -x/-y gap",
        "post_step_size_and_order_are_reconstruction_assumptions": True,
        "acceptance": "sum/max pairwise area and max vertical violation nonincreasing; repair requires improvement; gap span nonincreasing",
        "acceptance_increase_tolerance": 0.,
        "repair_improvement_relative_tolerance": 1e-12,
        "reference_engine": "GEOS float64; not exact arithmetic; no Clipper quantization",
        "gap_tolerance": 1e-3, "reference_area_tolerance": AREA_EPS,
        "boundary_tolerance": BOUNDARY_EPS, "stopping_motion_tolerance": MOTION_EPS,
        "height": height, "horizontal_right_wall": None,
        "statuses": statuses,
        "gap_rollbacks_per_candidate": gap_rollbacks.tolist(),
        "repair_rejected_trials_per_candidate": rejected_trials.tolist(),
        "initial_native_geometry": _native_geometry(initial_native),
        "final_native_geometry": _native_geometry(final_native),
        "initial_reference_geometry": initial_reference,
        "final_reference_geometry": current_reference,
        "final_fp32_reference_geometry": stored_reference,
        "fp32_reference_feasible": [row["feasible"] for row in stored_reference],
        "trace": log, "elapsed_seconds": time.perf_counter() - started,
        "rotation_components_preserved": bool(np.array_equal(a[:, :, 2:], original[:, :, 2:])),
        "limitations": [
            "Numerical reconstruction variant, not a recovered author implementation",
            "Existing concave SAT proxy/direction and sampler are unchanged",
            "Monotone local steps can stall with residual overlap",
            "reference_converged refers to float64; consult fp32_reference_feasible for saved results",
        ],
    }
