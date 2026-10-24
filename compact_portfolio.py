"""Pick the narrowest strictly feasible endpoint of two deterministic starts.

The second start must be a postprocessing of the same raw candidate, never GT
for a prediction. Both attempts and any failures remain reviewable.
"""
import copy

import numpy as np

from compact_layout import optimize
from postprocess_metrics import geometry


def optimize_two_starts(polygons, ids, raw_actions, repair_seed, *,
                        raw_result=None, height=1200., seconds=45., sweeps=4):
    if raw_result is None:
        raw_result = optimize(polygons, ids, raw_actions, height, seconds, sweeps)
    raw32 = np.asarray(raw_actions, dtype=np.float32).astype(np.float64)
    seed32 = np.asarray(repair_seed, dtype=np.float32).astype(np.float64)
    if raw32.shape != seed32.shape or not np.array_equal(raw32[:, 2:], seed32[:, 2:]):
        raise ValueError('Repair seed must preserve the same pieces and stored rotations')
    reused = np.array_equal(raw32, seed32)
    seed_result = raw_result if reused else optimize(polygons, ids, repair_seed, height, seconds, sweeps)
    attempts = []
    for start, result in [('raw', raw_result), ('legacy_repair', seed_result)]:
        measured = geometry(polygons, ids, result['actions'], height)
        attempts.append({'start': start, 'after': measured,
                         'repair_status': result['repair_status'],
                         'compaction_status': result['compaction_status']})
    feasible = [i for i, row in enumerate(attempts) if row['after']['strict_strip_valid']]
    chosen = min(feasible, key=lambda i: (attempts[i]['after']['span_x'], i)) if feasible else None
    # An all-failed portfolio remains an explicit failure with the raw partial
    # endpoint retained for diagnosis; no high-utilization illegal layout wins.
    result = raw_result if chosen in (None, 0) else seed_result
    return {'status': 'strict_feasible' if chosen is not None else 'no_strict_feasible_attempt',
            'selected_start': attempts[chosen]['start'] if chosen is not None else None,
            'actions': copy.deepcopy(result['actions']), 'attempts': attempts,
            'seed_result': None if reused else seed_result,
            'seed_reused_raw_result': bool(reused),
            'selection_rule': 'minimum horizontal span among strict stored endpoints; raw wins exact ties',
            'new_solver_seconds_without_legacy_seed_generation': raw_result['elapsed_seconds'] +
                (0. if reused else seed_result['elapsed_seconds'])}
