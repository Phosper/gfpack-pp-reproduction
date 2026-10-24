#!/usr/bin/env python3
"""Bounded CPU controls for stored-float32 concave repair and compaction.

Results are measured with postprocess_metrics, independently of the optimizer's
candidate checks. Real saved inputs are read only and remain in the report even
if a control fails. This is a control suite, not a paper-reproduction result.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import shapely

from compact_layout import BudgetExpired, CompactLayout, optimize
from postprocess_metrics import geometry, self_test as metric_self_test
from translation_intervals import self_test as interval_self_test


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def fp32(actions):
    return np.asarray(actions, dtype=np.float32).astype(np.float64).tolist()


def measured(polygons, ids, actions, height=1200.):
    return geometry(polygons, ids, actions, height)


def output_checks(polygons, ids, original, result, height=1200., strict=True):
    actions = result['actions']
    require(actions == fp32(actions), 'Output contains nonrepresentable float32 actions')
    require(len(actions) == len(original), 'Output piece count changed')
    require(all(a[2:] == b[2:] for a, b in zip(fp32(original), actions)),
            'Stored rotation components changed during optimization')
    reference = measured(polygons, ids, actions, height)
    require(reference['height_violation'] <= 1e-9, 'Stored output violates strip height')
    if strict:
        require(reference['strict_strip_valid'], 'Stored output is not strictly feasible')
    return reference


def optimized_fixture(polygons, ids, actions, height=1200., seconds=4., sweeps=4):
    before = measured(polygons, ids, actions, height)
    result = optimize(polygons, ids, actions, height, seconds, sweeps)
    after = output_checks(polygons, ids, actions, result, height)
    require(result['repair_status'] == 'strict_feasible', 'Repair not explicitly successful')
    require(result['compaction_status'] == 'completed', 'Compaction not explicitly completed')
    require(after['span_x'] <= before['span_x']+1e-9,
            'Compaction expanded this initially feasible fixture')
    return {'input_actions': actions, 'result': result, 'before': before, 'after': after}


def separated_squares():
    square = [[0., 0.], [10., 0.], [10., 10.], [0., 10.]]
    actions = [[0., 0., 1., 0.], [30., 0., 1., 0.]]
    record = optimized_fixture([square], [0, 0], actions)
    require(20. <= record['after']['span_x'] <= 20.01,
            'Two squares did not close the known empty horizontal gap')
    require(record['after']['span_x'] < record['before']['span_x']-19.,
            'Two-square fixture made no material compaction')
    return record


def nested_concave():
    u = [[0., 0.], [4., 0.], [4., 4.], [3., 4.], [3., 1.],
         [1., 1.], [1., 4.], [0., 4.]]
    inner = [[1.25, 1.25], [2.75, 1.25], [2.75, 2.75], [1.25, 2.75]]
    record = optimized_fixture([u, inner], [0, 1], [[0., 0., 1., 0.]]*2)
    require(record['after']['span_x'] <= 4.+1e-9,
            'Concave cavity was replaced by a convex or bounding-box obstruction')
    x, y = record['result']['actions'][1][:2]
    require(1.25+x >= 1.-1e-9 and 2.75+x <= 3.+1e-9 and
            1.25+y >= 1.-1e-9 and 2.75+y <= 4.+1e-9,
            'Inner square was unnecessarily pushed outside the U cavity')
    return record


def full_height():
    bar = [[0., 0.], [10., 0.], [10., 1200.], [0., 1200.]]
    actions = [[0., 0., 1., 0.]]
    record = optimized_fixture([bar], [0], actions)
    require(record['result']['actions'] == actions, 'Legal full-height piece moved')
    require(record['result']['boundary_adjustments'] == 0,
            'Full-height piece received an artificial boundary margin')
    return record


def shifted_squares(offset):
    square = [[0., 0.], [10., 0.], [10., 10.], [0., 10.]]
    record = optimized_fixture([square], [0, 0],
                               [[offset, 0., 1., 0.], [offset+30., 0., 1., 0.]])
    require(20. <= record['after']['span_x'] <= 20.02,
            'A horizontal wall prevented shifted-square compaction')
    if offset < 0:
        require(record['after']['bounds_xyxy'][2] < 0., 'Negative-x domain was removed')
    else:
        require(record['after']['bounds_xyxy'][0] > 8000., 'A numeric width8000 wall was imposed')
    return record


def float32_height_boundary():
    triangle = [[0., 0.], [10., 0.], [0., 10.00001]]
    square = [[0., 0.], [10., 0.], [10., 10.], [0., 10.]]
    polygons, ids = [triangle, square], [0, 1]
    actions = [[0., 1189.99999, 1., 0.], [30., 1180., 1., 0.]]
    initial = measured(polygons, ids, actions)
    cast = measured(polygons, ids, fp32(actions))
    require(initial['strict_strip_valid'], 'Synthetic float64 boundary fixture is not valid')
    require(cast['height_violation'] > 1e-9, 'Synthetic cast did not expose its known overflow')
    record = optimized_fixture(polygons, ids, actions)
    require(record['result']['boundary_adjustments'] == 1,
            'Storage boundary correction was not explicitly counted')
    require(record['result']['actions'][0][1] < fp32(actions)[0][1],
            'Upper boundary was not corrected to a legal representable y')
    record['unrepaired_float32_actions'] = fp32(actions)
    record['unrepaired_float32_geometry'] = cast
    return record


def real_boundary_regression(frozen, previous, index):
    case_path = frozen/'cases'/('sample_%03d.json' % index)
    job_path = previous/'jobs'/('sample_%03d_pred_repair_only.json' % index)
    case = json.loads(case_path.read_text(encoding='utf-8'))
    old_job = json.loads(job_path.read_text(encoding='utf-8'))
    actions = old_job['stages']['float64']['actions']
    polygons, ids = case['polygons'], case['polygon_ids']
    before = measured(polygons, ids, actions)
    cast_before = measured(polygons, ids, fp32(actions))
    require(before['within_strip'], 'Previous float64 repair output is already outside height')
    require(cast_before['height_violation'] > 1e-9, 'Saved FP32 regression no longer reproduced')
    layout = CompactLayout(polygons, ids, actions, 1200., 12.)
    try:
        layout.repair()
    except BudgetExpired:
        layout.repair_status = 'time_limit'
    result = {'actions': layout.output(), 'repair_status': layout.repair_status,
              'compaction_status': 'not_requested', 'moves': layout.moves,
              'boundary_adjustments': layout.boundary_moves,
              'candidate_checks': layout.candidate_checks,
              'interval_calls': layout.interval_calls}
    after = output_checks(polygons, ids, actions, result)
    require(result['repair_status'] == 'strict_feasible', 'Real saved boundary repair did not converge')
    require(result['boundary_adjustments'] > 0, 'Real boundary adjustment was not explicitly counted')
    return {'sample_index': index, 'input_case_sha256': digest(case_path),
            'previous_job_sha256': digest(job_path), 'input_actions': actions,
            'before': before, 'unrepaired_float32_geometry': cast_before,
            'result': result, 'after': after,
            'meaning': 'Stored boundary correction followed by independent full-pair collision remeasurement'}


def zero_budget():
    square = [[0., 0.], [10., 0.], [10., 10.], [0., 10.]]
    actions = [[0., 0., 1., 0.], [5., 0., 1., 0.]]
    result = optimize([square], [0, 0], actions, seconds=0.)
    after = output_checks([square], [0, 0], actions, result, strict=False)
    require(result['repair_status'] == 'time_limit', 'Zero budget was reported as a successful repair')
    require(result['compaction_status'] != 'completed', 'Zero budget was reported as successful compaction')
    require(not after['strict_strip_valid'], 'Known overlap vanished without a measured repair')
    require(after['max_pairwise_area'] == 50., 'Partial zero-budget output was silently replaced')
    return {'input_actions': actions, 'result': result, 'after': after,
            'meaning': 'Explicit timeout retains the measurable invalid output and its denominator'}


def invalid_numbers():
    square = [[0., 0.], [10., 0.], [10., 10.], [0., 10.]]
    checks = []
    for value in (math.nan, math.inf, -math.inf):
        try:
            optimize([square], [0], [[value, 0., 1., 0.]])
        except ValueError:
            checks.append({'location': 'action_x', 'value': str(value), 'rejected': True})
        else:
            raise AssertionError('Nonfinite action coordinate was accepted')
        try:
            optimize([square], [0], [[0., 0., 1., 0.]], height=value)
        except ValueError:
            checks.append({'location': 'height', 'value': str(value), 'rejected': True})
        else:
            raise AssertionError('Nonfinite strip height was accepted')
    return {'rejection_checks': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--previous-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Controls output already exists; use a new versioned path')
    tests = [
        ('known_square_gap_closes', separated_squares),
        ('concave_u_keeps_nested_square', nested_concave),
        ('legal_full_height_piece_unchanged', full_height),
        ('negative_x_has_no_wall', lambda: shifted_squares(-100.)),
        ('large_x_has_no_width8000_wall', lambda: shifted_squares(9000.)),
        ('stored_float32_height_is_repaired_and_remeasured', float32_height_boundary),
        ('real_025_float32_height_regression',
         lambda: real_boundary_regression(args.frozen, args.previous_results, 25)),
        ('real_045_float32_height_regression',
         lambda: real_boundary_regression(args.frozen, args.previous_results, 45)),
        ('zero_budget_retains_explicit_invalid_result', zero_budget),
        ('nonfinite_numbers_rejected', invalid_numbers),
        ('independent_metric_self_test', metric_self_test),
        ('translation_intervals_controls_and_random_geos', interval_self_test),
    ]
    records = []
    for name, test in tests:
        try:
            details = test()
            record = {'name': name, 'passed': True, 'details': details}
        except Exception as exc:
            record = {'name': name, 'passed': False, 'error_type': type(exc).__name__,
                      'error_message': str(exc)}
        records.append(record)
        print(json.dumps({key: record[key] for key in ('name', 'passed', 'error_type', 'error_message')
                          if key in record}), flush=True)
    sources = ['check_compact_controls.py', 'compact_layout.py',
               'translation_intervals.py', 'postprocess_metrics.py']
    result = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'verified' if all(r['passed'] for r in records) else 'failed',
              'planned_controls': len(records), 'passed_controls': sum(r['passed'] for r in records),
              'scope': 'Bounded independent CPU controls, including two saved representational regressions; no model inference',
              'thresholds': {'strict_max_pairwise_area': 1e-9, 'boundary': 1e-9, 'reference_area': 50.},
              'versions': {'python': sys.version, 'numpy': np.__version__,
                           'shapely': shapely.__version__, 'geos': shapely.geos_version_string},
              'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in sources},
              'frozen_manifest_sha256': digest(args.frozen/'manifest.json'),
              'controls': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'passed_controls': result['passed_controls'],
                      'planned_controls': result['planned_controls']}))
    if result['status'] != 'verified':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
