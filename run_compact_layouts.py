"""Freeze and evaluate the concave postprocessor on all 16 saved cases + GT."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import shapely

sys.dont_write_bytecode = True
from check_collision_saved_cases import validate_frozen, checked_case
from compact_layout import CompactLayout, optimize
from postprocess_metrics import geometry, compare, summarize


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, value):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--round7', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=45.)
    p.add_argument('--sweeps', type=int, default=4)
    a = p.parse_args()
    if not 0 < a.seconds <= 120 or not 0 <= a.sweeps <= 16:
        p.error('Bounded time/sweep budget required')
    manifest, msha = validate_frozen(a.frozen)
    oldmeta = read(a.round7/'metadata.json')
    if oldmeta['status'] != 'completed' or oldmeta['changed_inputs']:
        raise ValueError('Round7 evidence was not completed with intact inputs')
    for rel, digest in oldmeta['output_sha256'].items():
        if sha(a.round7/rel) != digest:
            raise ValueError('Round7 output changed: '+rel)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'jobs').mkdir()
    files = ('compact_layout.py', 'translation_intervals.py', 'run_compact_layouts.py',
             'postprocess_metrics.py', 'check_collision_saved_cases.py')
    inputs = {'scripts/'+n: sha(Path(__file__).with_name(n)) for n in files}
    inputs['frozen/manifest.json'] = msha
    inputs['round7/metadata.json'] = sha(a.round7/'metadata.json')
    for e in manifest['cases']:
        checked_case(a.frozen, e)
        inputs['frozen/'+e['path']] = e['sha256']
    for rel, digest in oldmeta['output_sha256'].items():
        inputs['round7/'+rel] = digest
    save(a.output/'inputs.json', inputs)
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'samples': [e['sample_index'] for e in manifest['cases']],
            'sources': {'pred': 'raw_actions', 'gt': 'gt_actions'},
            'planned_inputs': 32,
            'routes': ['new_repair', 'new_repair_compact', 'saved_repair_compact'],
            'baseline_routes': ['gap_only', 'repair_only', 'repair_then_gap'],
            'height': 1200., 'horizontal_walls': None,
            'strict_area_tolerance': 1e-9, 'boundary_tolerance': 1e-9,
            'reference_area50': 50., 'seconds_per_new_pipeline': a.seconds,
            'compaction_sweeps': a.sweeps,
            'storage': 'All action columns cast to float32 before optimization; translations checked in float32 after each move; rotation components then fixed',
            'repair': 'Single-piece translation, four unoriented ray axes, radius16/64/256/1024/4096, nearest free intervals; motion+1000*positive-span-growth',
            'compact': 'Alternating left/right horizontal sweeps inside current aggregate span; endpoints may jump obstacles; no monotonic-collision assumption',
            'saved_repair_policy': 'Same saved round7 repair endpoint, cast to float32; compact only if strictly feasible, otherwise unchanged and explicitly skipped',
            'new_repair_failure_policy': 'Return partial measured layout and explicit status; retain in all32 denominators',
            'development_caveat': 'All16 representatives previously selected and used during development; no independent generalization or 512-candidate evaluation',
            'sampling': False, 'training': False,
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                        'shapely': shapely.__version__, 'geos': shapely.geos_version_string,
                        'platform': platform.system()},
            'legacy_native_metrics': 'Saved Linux official vectors retained in round7; new outputs use independently replayed reference geometry, no Windows calutil build'}
    save(a.output/'plan.json', plan)
    rows, jobs, baselines = [], [], []
    started = time.perf_counter()
    for entry in manifest['cases']:
        _, case = checked_case(a.frozen, entry)
        index = entry['sample_index']
        for source, field in plan['sources'].items():
            original = case[field]
            before = geometry(case['polygons'], case['polygon_ids'], original)
            base = {'sample_index': index, 'source': source,
                    'baseline_group': 'failure' if case['group']=='failure_development' else 'success'}
            old = {}
            for route in plan['baseline_routes']:
                rel = 'jobs/sample_%03d_%s_%s.json' % (index, source, route)
                prev = read(a.round7/rel)
                old[route] = prev
                for precision in ('float64', 'float32'):
                    g = geometry(case['polygons'], case['polygon_ids'], prev['stages'][precision]['actions'])
                    baselines.append(base | {'path': 'legacy_'+route, 'status': 'completed',
                        'output_precision': precision, 'after': g, 'comparison': None,
                        'round7_job_sha256': sha(a.round7/rel)})
            result = optimize(case['polygons'], case['polygon_ids'], original,
                              seconds=a.seconds, sweeps=a.sweeps)
            saved = np.asarray(old['repair_only']['repair_actions'], dtype=np.float32).astype(np.float64).tolist()
            saved_before = geometry(case['polygons'], case['polygon_ids'], saved)
            extra = {'actions': saved, 'moves': [], 'status': 'skipped_infeasible'}
            if saved_before['strict_strip_valid']:
                layout = CompactLayout(case['polygons'], case['polygon_ids'], saved, seconds=a.seconds)
                layout.repair_status = 'strict_feasible'
                try:
                    layout.compact(a.sweeps)
                    status = layout.compaction_status
                except Exception as exc:
                    from compact_layout import BudgetExpired
                    if not isinstance(exc, BudgetExpired):
                        raise
                    status = 'time_limit'
                extra = {'actions': layout.output(), 'moves': layout.moves, 'status': status,
                         'elapsed_seconds': time.perf_counter()-layout.started}
            stages = {}
            for route, actions in [('new_repair', result['repair_actions']),
                                   ('new_repair_compact', result['actions']),
                                   ('saved_repair_compact', extra['actions'])]:
                g = geometry(case['polygons'], case['polygon_ids'], actions)
                comp = compare(before, g, original, actions)
                stages[route] = {'actions': actions, 'after': g, 'comparison': comp}
                rows.append(base | {'path': route, 'status': 'completed',
                                    'output_precision': 'float32', 'after': g, 'comparison': comp})
            job = base | {'case_sha256': entry['sha256'], 'candidate_index': case['candidate_index'],
                          'input_field': field, 'before': before, 'pipeline': result,
                          'saved_repair_before': saved_before, 'saved_repair_compaction': extra,
                          'stages': stages}
            rel = 'jobs/sample_%03d_%s.json' % (index, source)
            save(a.output/rel, job)
            jobs.append({'path': rel, 'sha256': sha(a.output/rel)})
            print(json.dumps(base | {'strict': stages['new_repair_compact']['after']['strict_strip_valid'],
                                    'util': stages['new_repair_compact']['after']['strip_utilization'],
                                    'seconds': result['elapsed_seconds']}), flush=True)
    summary = {'status': 'completed', 'planned_inputs': 32, 'completed_inputs': len(jobs),
               'rows': rows, 'baseline_rows': baselines, 'aggregates': summarize(rows),
               'baseline_aggregates': summarize(baselines), 'jobs': jobs,
               'elapsed_seconds': time.perf_counter()-started}
    save(a.output/'summary.json', summary)
    changed = []
    for rel, digest in inputs.items():
        scope, name = rel.split('/', 1)
        path = ({'scripts': Path(__file__).parent, 'frozen': a.frozen, 'round7': a.round7}[scope])/name
        if sha(path) != digest:
            changed.append(rel)
    save(a.output/'metadata.json', {'status': 'completed' if not changed else 'input_changed',
         'changed_inputs': changed, 'input_count': len(inputs),
         'output_sha256': {p.relative_to(a.output).as_posix(): sha(p)
                           for p in sorted(a.output.rglob('*')) if p.is_file()}})
    if changed:
        raise RuntimeError('Frozen input changed')
    print(json.dumps({'status': 'completed', 'inputs': len(jobs),
                      'elapsed_seconds': summary['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
