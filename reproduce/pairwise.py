"""Declared post-generation operator splitting of the paper's weighted SAT corrections.

Pair order and relaxation are numerical reconstruction choices: the paper does
not publish them. This is not round8's placement search, nor a claim of an exact
copy of the author's solver. Sampling still uses the simultaneous Eq. 9 field.
"""
import math

import numpy as np
from shapely.geometry import Polygon

from .geometry import CONTACT_MARGIN, SeparationError


DISPLACEMENT_POLICY = 'validate weighted endpoints; unsafe overrelaxation falls back to minimum separation'


def validated_pair_update(geometry, oriented, positions, shapes, i, j, relaxation=1.75):
    """Propose, validate, then return a pair update without modifying its input.

    A separating vector for concave pieces may enter another forbidden lobe
    when extended. Validate the actual area-weighted two-piece endpoints, not
    just a translation of the first piece. Overrelaxation is a preference:
    if it does not separate this pair, retry the unscaled separating vector.

    Requested factors below one retain the existing damped-update semantics;
    they need not remove this pair's overlap in a single step. Global success
    is still decided only by the final strict geometry measurement.
    """
    if not 0 < relaxation < 2 or not math.isfinite(relaxation):
        raise ValueError('Relaxation must be finite and in (0,2)')
    vector = oriented.exit_vector(i, j, positions, shapes)
    total = geometry.areas[i] + geometry.areas[j]
    weights = geometry.areas[j] / total, geometry.areas[i] / total
    factors = (relaxation, 1.) if relaxation > 1. else (relaxation,)
    trials = []
    for factor in factors:
        scaled_vector = factor * vector
        first = positions[i] + weights[0] * scaled_vector
        second = positions[j] - weights[1] * scaled_vector
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            trials.append({'factor': factor, 'finite': False})
            continue
        first_shape = Polygon(oriented.vertices[i] + first)
        second_shape = Polygon(oriented.vertices[j] + second)
        remaining = float(first_shape.intersection(second_shape).area)
        trials.append({'factor': factor, 'finite': True, 'overlap_area': remaining})
        if remaining == 0. or factor < 1.:
            return first, second, first_shape, second_shape, {
                'requested_factor': relaxation, 'applied_factor': factor,
                'minimum_step_fallback': factor != relaxation,
                'remaining_pair_overlap_area': remaining,
                'rejected_relaxed_overlap_area': trials[0].get('overlap_area', 0.) if factor != relaxation else 0.}
    raise SeparationError('Area-weighted pair update failed full-contour validation', {
        'pair': [int(i), int(j)], 'requested_relaxation': relaxation,
        'positions': [positions[i].tolist(), positions[j].tolist()],
        'oriented_vertices': [oriented.vertices[i].tolist(), oriented.vertices[j].tolist()],
        'minimum_exit_vector': vector.tolist(), 'area_weights': list(weights), 'trials': trials})


def enhance_pairwise(geometry, actions, iterations=100, relaxation=1.75):
    if not 0 <= iterations <= 100 or not 0 < relaxation < 2 or not math.isfinite(relaxation):
        raise ValueError('Iterations must be in [0,100], relaxation in (0,2)')
    result = np.array(actions, dtype=np.float64, copy=True)
    oriented = geometry.oriented(result)
    shapes = oriented.shapes(result[:, :2])
    bounds = np.asarray([p.bounds for p in shapes])
    if np.any(bounds[:, 3] - bounds[:, 1] > geometry.height):
        return result, {'status': 'fixed_orientation_exceeds_height', 'iterations': 0,
                        'trace': [], 'displacement_policy': DISPLACEMENT_POLICY,
                        'minimum_step_fallbacks': 0, 'metrics': geometry.measure(result)}
    trace, best = [], None
    best_iteration = None
    first_strict = None
    previous_metrics = geometry.measure(result)
    for iteration in range(iterations):
        previous = result[:, :2].copy()
        pair_updates = 0
        fallbacks = 0
        rejected_overlap_max = 0.
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if bounds[i, 0] >= bounds[j, 2] or bounds[j, 0] >= bounds[i, 2] or bounds[i, 1] >= bounds[j, 3] or bounds[j, 1] >= bounds[i, 3]:
                    continue
                if shapes[i].intersection(shapes[j]).area == 0:
                    continue
                first, second, first_shape, second_shape, update = validated_pair_update(
                    geometry, oriented, result[:, :2], shapes, i, j, relaxation)
                # Commit only the validated endpoints. Other pairs may still
                # overlap and are handled by the unchanged iterative solver.
                result[i, :2], result[j, :2] = first, second
                shapes[i], shapes[j] = first_shape, second_shape
                for index in (i, j):
                    bounds[index] = shapes[index].bounds
                fallbacks += int(update['minimum_step_fallback'])
                rejected_overlap_max = max(rejected_overlap_max, update['rejected_relaxed_overlap_area'])
                pair_updates += 1
        for i in range(len(result)):
            delta = 0.
            if bounds[i, 1] < 0:
                delta = -bounds[i, 1] + CONTACT_MARGIN
            elif bounds[i, 3] > geometry.height:
                delta = geometry.height - bounds[i, 3] - CONTACT_MARGIN
            if delta:
                result[i, 1] += delta
                shapes[i] = Polygon(oriented.vertices[i] + result[i, :2])
                bounds[i] = shapes[i].bounds
        before_gap = geometry.measure(result)
        candidate, gap_moves = oriented.gap_sweep(result)
        measured = geometry.measure(candidate)
        gap_rejected = before_gap['strict_valid'] and not measured['strict_valid']
        if not gap_rejected:
            result = candidate
        else:
            measured = before_gap
        shapes = oriented.shapes(result[:, :2])
        bounds = np.asarray([p.bounds for p in shapes])
        movement = float(np.linalg.norm(result[:, :2] - previous, axis=1).max())
        trace.append({'iteration': iteration+1, 'pair_updates': pair_updates,
                      'relaxation': relaxation, 'max_movement': movement,
                      'minimum_step_fallbacks': fallbacks,
                      'max_rejected_relaxed_overlap_area': rejected_overlap_max,
                      'gap_moves': gap_moves, 'gap_rejected_to_preserve_feasibility': gap_rejected,
                      'span_x_before_collision': previous_metrics['span_x'],
                      'span_x_after_collision': before_gap['span_x'],
                      'span_x_after_gap': measured['span_x'],
                      'sum_pair_overlap_area_before_collision': previous_metrics['sum_pair_overlap_area'],
                      'sum_pair_overlap_area_after_collision': before_gap['sum_pair_overlap_area'],
                      **{k: v for k, v in measured.items() if k != 'overlap_pairs'}})
        previous_metrics = measured
        if measured['strict_valid']:
            if first_strict is None:
                first_strict = iteration + 1
            if best is None or measured['span_x'] < best[1]['span_x']:
                best = result.copy(), measured
                best_iteration = iteration + 1
            if movement <= 1e-6:
                break
    if best is not None:
        result = best[0]
    return result, {'status': 'strict_valid' if best is not None else 'iteration_limit',
                    'iterations': len(trace), 'first_strict_iteration': first_strict,
                    'displacement_policy': DISPLACEMENT_POLICY,
                    'minimum_step_fallbacks': sum(t['minimum_step_fallbacks'] for t in trace),
                    'best_feasible_iteration': best_iteration, 'trace': trace,
                    'metrics': geometry.measure(result)}
