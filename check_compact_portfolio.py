#!/usr/bin/env python3
"""CPU controls for strict selection and reuse in the two-start portfolio.

Small selector controls supply explicit solver endpoints while independently
measuring their true geometry. The real sample45 warm start runs the actual
optimizer once; this suite does not rerun the full development experiment.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True

import numpy as np
import shapely

from compact_portfolio import optimize_two_starts
from postprocess_metrics import geometry, summarize


SQUARE = [[0., 0.], [10., 0.], [10., 10.], [0., 10.]]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoint(second_x, *, status='strict_feasible'):
    actions = [[0., 0., 1., 0.], [float(second_x), 0., 1., 0.]]
    return {'actions': actions, 'repair_actions': copy.deepcopy(actions),
            'repair_status': status, 'compaction_status': 'completed',
            'elapsed_seconds': .125, 'moves': []}


def ref(actions):
    return geometry([SQUARE], [0, 0], actions)


def select_explicit(raw, seed):
    """Exercise real selector geometry with controlled optimizer endpoints."""
    with patch('compact_portfolio.optimize', return_value=copy.deepcopy(seed)) as solver:
        result = optimize_two_starts([SQUARE], [0, 0], raw['actions'], seed['actions'],
                                     raw_result=copy.deepcopy(raw))
        require(solver.call_count == 1, 'Distinct seed endpoint did not invoke exactly one second attempt')
    return result


def illegal_high_utility_rejected():
    # Deliberately label the illegal endpoint successful: selection must check
    # geometry rather than trusting this status or its high utilization.
    raw, seed = endpoint(5.), endpoint(30.)
    before = ref(raw['actions'])
    require(before['eps50_strip_valid'] and not before['strict_strip_valid'],
            'Overlap50 fixture must distinguish reference and strict criteria')
    require(before['strip_utilization'] > ref(seed['actions'])['strip_utilization'],
            'Illegal fixture is not the high-utility alternative')
    result = select_explicit(raw, seed)
    after = ref(result['actions'])
    require(result['selected_start'] == 'legacy_repair' and after['strict_strip_valid'],
            'High-utilization colliding endpoint was allowed to win')
    require(len(result['attempts']) == 2 and not result['attempts'][0]['after']['strict_strip_valid'],
            'Rejected illegal attempt disappeared from evidence')
    return {'synthetic_solver_endpoints': True, 'result': result,
            'selected_independent_geometry': after}


def narrower_feasible_wins():
    raw, seed = endpoint(30.), endpoint(10.)
    require(ref(raw['actions'])['strict_strip_valid'] and ref(seed['actions'])['strict_strip_valid'],
            'Two feasible endpoint fixture is not feasible')
    result = select_explicit(raw, seed)
    require(result['selected_start'] == 'legacy_repair' and ref(result['actions'])['span_x'] == 20.,
            'Narrower strictly feasible seed was not selected')
    reverse = select_explicit(seed, raw)
    require(reverse['selected_start'] == 'raw' and ref(reverse['actions'])['span_x'] == 20.,
            'Selection preferred a seed label over the narrower raw endpoint')
    return {'synthetic_solver_endpoints': True, 'forward': result, 'reversed': reverse}


def equal_stored_input_reused():
    raw = endpoint(10.)
    seed = copy.deepcopy(raw['actions'])
    seed[0][0] = 1e-50  # Different Python float, identical stored float32 zero.
    require(seed != raw['actions'] and np.array_equal(np.asarray(seed, dtype=np.float32),
            np.asarray(raw['actions'], dtype=np.float32)), 'Canonical-reuse fixture is not canonical-equal')
    with patch('compact_portfolio.optimize', side_effect=AssertionError('Duplicate stored input reran solver')):
        result = optimize_two_starts([SQUARE], [0, 0], raw['actions'], seed,
                                     raw_result=raw)
    require(result['seed_reused_raw_result'] and result['seed_result'] is None,
            'Identical stored inputs were not explicitly reused')
    require(result['selected_start'] == 'raw' and len(result['attempts']) == 2,
            'Canonical reuse did not retain both attempt slots and deterministic raw tie')
    require(result['new_solver_seconds_without_legacy_seed_generation'] == raw['elapsed_seconds'],
            'Canonical reuse charged two solver attempts')
    return {'synthetic_solver_endpoints': True, 'result': result}


def exact_span_tie_prefers_raw():
    raw = endpoint(10.)
    seed = endpoint(10.)
    seed['actions'][0][0] += 100.
    seed['actions'][1][0] += 100.
    result = select_explicit(raw, seed)
    require(not result['seed_reused_raw_result'] and result['selected_start'] == 'raw',
            'Distinct equal-span endpoints did not use the documented raw tie-break')
    return {'synthetic_solver_endpoints': True, 'result': result}


def altered_seed_rotation_rejected():
    raw = endpoint(10.)
    seed = copy.deepcopy(raw['actions'])
    seed[1][2:] = [0., 1.]
    with patch('compact_portfolio.optimize', side_effect=AssertionError('Invalid seed reached optimizer')):
        try:
            optimize_two_starts([SQUARE], [0, 0], raw['actions'], seed, raw_result=raw)
        except ValueError as exc:
            return {'seed_rejected': True, 'error_type': type(exc).__name__,
                    'error_message': str(exc), 'stored_seed_rotation_changed': True}
    raise AssertionError('Seed with changed stored rotation was accepted')


def all_failed_denominators_retained():
    raw, seed = endpoint(5., status='stalled'), endpoint(8., status='stalled')
    result = select_explicit(raw, seed)
    require(result['status'] == 'no_strict_feasible_attempt' and result['selected_start'] is None,
            'All-infeasible portfolio was reported as selected or successful')
    require(result['actions'] == raw['actions'] and len(result['attempts']) == 2,
            'All-infeasible portfolio removed its raw diagnostic output or an attempt')
    measured = ref(result['actions'])
    require(not measured['strict_strip_valid'], 'All-failed fixture unexpectedly became strict feasible')
    samples = (0, 17, 25, 35, 45, 46, 47, 54, 56, 62, 74, 91, 99, 109, 115, 127)
    failures = {25, 45, 46, 47, 54, 62, 99, 115}
    rows = [{'sample_index': index, 'source': source,
             'baseline_group': 'failure' if index in failures else 'success',
             'path': 'two_start_compact', 'status': 'completed', 'output_precision': 'float32',
             'after': measured, 'comparison': None}
            for index in samples for source in ('pred', 'gt')]
    aggregate = summarize(rows)
    overall = aggregate['overall']
    require(overall['planned_outputs'] == overall['measured_outputs'] == 32 and
            overall['strict_strip_valid']['outputs'] == 0 and
            overall['strict_strip_valid']['fraction_all_planned'] == 0. and
            overall['strict_strip_valid_strip_utilization']['mean_all_planned_failure_zero'] == 0.,
            'All-failed strict denominator or failure-zero utilization was lost')
    return {'synthetic_solver_endpoints': True,
            'synthetic_32_slot_aggregation': True, 'result': result, 'aggregate': aggregate,
            'meaning': 'Measured invalid endpoints retain all planned slots; completed means measured, not feasible'}


def real_045_warm_start(frozen, previous, raw_results):
    case_path = frozen/'cases'/'sample_045.json'
    seed_path = previous/'jobs'/'sample_045_pred_repair_only.json'
    raw_path = raw_results/'jobs'/'sample_045_pred.json'
    case = json.loads(case_path.read_text(encoding='utf-8'))
    old = json.loads(seed_path.read_text(encoding='utf-8'))
    raw_job = json.loads(raw_path.read_text(encoding='utf-8'))
    require(old['input_actions'] == case['raw_actions'] and old['source'] == 'pred' and
            old['sample_index'] == raw_job['sample_index'] == 45,
            'Real seed does not derive from the same frozen prediction')
    seed = old['repair_actions']
    result = optimize_two_starts(case['polygons'], case['polygon_ids'], case['raw_actions'], seed,
                                 raw_result=raw_job['pipeline'], seconds=12., sweeps=4)
    after = geometry(case['polygons'], case['polygon_ids'], result['actions'])
    require(result['status'] == 'strict_feasible' and result['selected_start'] == 'legacy_repair',
            'Real warm start did not rescue the wider single-start sample45 endpoint')
    require(after['strict_strip_valid'] and after['max_pairwise_area'] <= 1e-9 and
            after['height_violation'] <= 1e-9, 'Real selected stored endpoint is geometrically invalid')
    require(result['actions'] == np.asarray(result['actions'], dtype=np.float32).astype(np.float64).tolist(),
            'Real selected endpoint is not float32-representable')
    require(after['span_x'] < result['attempts'][0]['after']['span_x'] and
            after['strip_utilization'] > .70,
            'Real warm start made no material improvement over the raw attempt')
    require(all(a[2:] == b[2:] for a, b in zip(case['raw_actions'], result['actions'])),
            'Real warm start changed stored rotation components')
    return {'input_case_sha256': sha(case_path), 'legacy_seed_job_sha256': sha(seed_path),
            'raw_job_sha256': sha(raw_path), 'second_attempt_seconds_budget': 12.,
            'actual_seed_optimizer_run': True, 'result': result, 'selected_independent_geometry': after}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--previous-results', type=Path, required=True)
    p.add_argument('--raw-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Controls output exists; use a new versioned path')
    tests = [('illegal_high_utilization_attempt_rejected', illegal_high_utility_rejected),
             ('two_feasible_attempts_choose_minimum_span', narrower_feasible_wins),
             ('identical_stored_start_reuses_raw_result', equal_stored_input_reused),
             ('distinct_equal_span_attempts_tie_to_raw', exact_span_tie_prefers_raw),
             ('changed_stored_seed_rotation_rejected', altered_seed_rotation_rejected),
             ('all_failed_portfolio_retains_all32_slots', all_failed_denominators_retained),
             ('real_045_legacy_repair_warm_start',
              lambda: real_045_warm_start(args.frozen, args.previous_results, args.raw_results))]
    records = []
    for name, test in tests:
        try:
            record = {'name': name, 'passed': True, 'details': test()}
        except Exception as exc:
            record = {'name': name, 'passed': False, 'error_type': type(exc).__name__,
                      'error_message': str(exc)}
        records.append(record)
        print(json.dumps({k: record[k] for k in ('name', 'passed', 'error_type', 'error_message')
                          if k in record}), flush=True)
    sources = ('check_compact_portfolio.py', 'compact_portfolio.py', 'run_compact_portfolio.py',
               'compact_layout.py', 'translation_intervals.py', 'postprocess_metrics.py')
    result = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'verified' if all(r['passed'] for r in records) else 'failed',
              'planned_controls': len(records), 'passed_controls': sum(r['passed'] for r in records),
              'scope': 'Selector unit controls plus one real saved sample45 warm-start solver run; not the full32 experiment',
              'thresholds': {'strict_max_pairwise_area': 1e-9, 'boundary': 1e-9, 'reference_area': 50.},
              'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in sources},
              'frozen_manifest_sha256': sha(args.frozen/'manifest.json'),
              'versions': {'python': sys.version, 'numpy': np.__version__,
                           'shapely': shapely.__version__, 'geos': shapely.geos_version_string},
              'controls': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'planned_controls': len(records),
                      'passed_controls': result['passed_controls']}))
    if result['status'] != 'verified':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
