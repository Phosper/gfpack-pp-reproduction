"""Numerical integration of the paper's area-weighted translational SAT field.

The paper specifies the field, not this step-control implementation. Splitting,
backtracking, quantization checks and internal substep limits are declared
reconstruction choices, not a claim to run the authors' original solver.
"""
import math

import numpy as np


def integrate_collision_field(geometry, actions, duration, max_substeps=32,
                              storage_dtype=None, max_step=1.):
    """Advance dq/dtau=F(q) without committing an energy-increasing Euler jump.

    Rotations remain fixed. The requested duration is retained as integration
    time; early stopping and any unintegrated time are explicitly reported.
    With storage_dtype, trial positions are rounded BEFORE measuring acceptance.
    No positive overlap is reclassified as zero.
    """
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('Collision-flow duration must be finite and nonnegative')
    if not isinstance(max_substeps, int) or max_substeps < 1:
        raise ValueError('Collision-flow max_substeps must be a positive integer')
    if not math.isfinite(max_step) or max_step <= 0:
        raise ValueError('Collision-flow max_step must be positive')
    result = np.array(actions, dtype=np.float64, copy=True)
    if result.shape != (len(geometry.vertices), 4) or not np.isfinite(result).all():
        raise ValueError('Invalid collision-flow actions')
    oriented = geometry.oriented(result)
    mass = geometry.areas / geometry.areas.sum()
    force, energy, pairs = oriented.field(result[:, :2])
    initial_energy = energy
    initial_pairs = pairs
    initial_force_max = float(np.linalg.norm(force, axis=1).max())
    remaining = float(duration)
    trace = []
    rejected = 0
    stationary_time = 0.
    status = 'time_complete'
    next_step = min(max_step, remaining)
    for _ in range(max_substeps):
        if remaining <= 0:
            break
        norm = float(np.sum(mass[:, None] * force**2))
        if norm == 0:
            # The remaining flow is stationary, including infeasible fixed
            # orientations. This is not a claim that the layout is valid.
            status = 'stationary'
            stationary_time = remaining
            remaining = 0.
            break
        step = min(next_step, remaining)
        accepted = False
        for trial in range(24):
            candidate = result.copy()
            candidate[:, :2] += step * force
            if storage_dtype is not None:
                candidate[:, :2] = candidate[:, :2].astype(storage_dtype).astype(np.float64)
            delta = candidate[:, :2] - result[:, :2]
            descent = float(np.sum(mass[:, None] * force * delta))
            if not np.isfinite(candidate).all() or descent <= 0.:
                step *= .5
                rejected += 1
                continue
            trial_force, trial_energy, trial_pairs = oriented.field(candidate[:, :2])
            if trial_energy <= energy - 1e-4 * descent:
                movement = float(np.linalg.norm(delta, axis=1).max())
                trace.append({'step': step, 'energy_before': energy, 'energy_after': trial_energy,
                              'collision_pairs_before': pairs, 'collision_pairs_after': trial_pairs,
                              'max_displacement': movement, 'backtracking_trials': trial})
                result, force, energy, pairs = candidate, trial_force, trial_energy, trial_pairs
                remaining = max(0., remaining-step)
                next_step = min(max_step, 2*step)
                accepted = True
                break
            step *= .5
            rejected += 1
        if not accepted:
            status = 'step_stalled'
            break
    else:
        if remaining > 0:
            status = 'substep_limit'
    if remaining > 0 and not np.any(force):
        status, stationary_time, remaining = 'stationary', remaining, 0.
    if remaining == 0 and status not in ('stationary',):
        status = 'time_complete'
    return result, {'status': status, 'requested_time': duration,
                    'unintegrated_time': remaining, 'accepted_time': duration-remaining,
                    'stationary_time': stationary_time,
                    'substeps': len(trace), 'rejected_trials': rejected,
                    'energy_before': initial_energy, 'energy_after': energy,
                    'collision_pairs_before': initial_pairs, 'collision_pairs_after': pairs,
                    'force_max_before': initial_force_max,
                    'max_substep_displacement': max((r['max_displacement'] for r in trace), default=0.),
                    'storage_dtype': str(np.dtype(storage_dtype)) if storage_dtype is not None else 'float64',
                    'trace': trace}
