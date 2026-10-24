"""Evaluate two deterministic starts, reusing the frozen one-start experiment."""
import argparse
import json
from pathlib import Path

from check_collision_saved_cases import validate_frozen, checked_case
from compact_portfolio import optimize_two_starts
from postprocess_metrics import geometry, compare, summarize
from run_compact_layouts import read, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--round7', type=Path, required=True)
    p.add_argument('--raw-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest, manifest_sha = validate_frozen(a.frozen)
    metadata = read(a.raw_results/'metadata.json')
    if metadata['status'] != 'completed' or metadata['changed_inputs']:
        raise ValueError('Raw experiment not intact')
    for rel, digest in metadata['output_sha256'].items():
        if sha(a.raw_results/rel) != digest:
            raise ValueError('Raw evidence changed')
    raw_inputs = read(a.raw_results/'inputs.json')
    for name, digest in raw_inputs.items():
        scope, rel = name.split('/', 1)
        root = {'scripts': Path(__file__).parent, 'frozen': a.frozen, 'round7': a.round7}[scope]
        if sha(root/rel) != digest:
            raise ValueError('Frozen source/seed changed: '+name)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'jobs').mkdir()
    inputs = {'frozen/manifest.json': manifest_sha}
    for name in ('compact_portfolio.py', 'run_compact_portfolio.py', 'compact_layout.py',
                 'translation_intervals.py', 'postprocess_metrics.py', 'run_compact_layouts.py',
                 'check_collision_saved_cases.py'):
        inputs['scripts/'+name] = sha(Path(__file__).with_name(name))
    inputs['raw/metadata.json'] = sha(a.raw_results/'metadata.json')
    for rel, digest in metadata['output_sha256'].items():
        inputs['raw/'+rel] = digest
    for e in manifest['cases']:
        inputs['frozen/'+e['path']] = e['sha256']
    save(a.output/'plan.json', {
        'planned_inputs': 32, 'height': 1200., 'horizontal_walls': None,
        'starts': ['frozen raw candidate', 'round7 deterministic repair of that same candidate'],
        'policy': 'Uniform two-start rule on all16 predictions and GT, no case-specific exception; equal stored starts explicitly reuse the raw result',
        'selection': 'Strict feasible stored float32 endpoints only, choose minimum span; raw wins ties',
        'seconds_per_attempt': 45., 'sweeps_per_attempt': 4,
        'scope': 'Paired development analysis, not independent generalization or a 512-candidate rate',
        'timing_caveat': 'New solver costs include both distinct attempts but exclude already recorded legacy seed generation',
        'reason': 'Single-piece local repair can be trapped by neighboring pieces; preserve the cooperative legacy seed as a second deterministic starting point',
        'original_one_start_results_retained': True})
    rows, jobs = [], []
    for entry in manifest['cases']:
        _, case = checked_case(a.frozen, entry)
        for source, field in [('pred', 'raw_actions'), ('gt', 'gt_actions')]:
            stem = 'sample_%03d_%s' % (entry['sample_index'], source)
            raw_job = read(a.raw_results/'jobs'/(stem+'.json'))
            seed_path = a.round7/'jobs'/(stem+'_repair_only.json')
            seed_job = read(seed_path)
            seed = seed_job['repair_actions']
            # This explicit provenance check prevents selecting GT or a new
            # candidate as an alternative prediction starting point.
            if seed_job['input_actions'] != case[field] or seed_job['source'] != source:
                raise ValueError('Legacy seed does not derive from the same input')
            inputs['round7/jobs/'+seed_path.name] = sha(seed_path)
            result = optimize_two_starts(case['polygons'], case['polygon_ids'], case[field],
                                         seed, raw_result=raw_job['pipeline'])
            after = geometry(case['polygons'], case['polygon_ids'], result['actions'])
            before = geometry(case['polygons'], case['polygon_ids'], case[field])
            row = {'sample_index': entry['sample_index'], 'source': source,
                   'baseline_group': raw_job['baseline_group'], 'path': 'two_start_compact',
                   'status': 'completed', 'output_precision': 'float32', 'after': after,
                   'comparison': compare(before, after, case[field], result['actions'])}
            job = row | {'case_sha256': entry['sha256'], 'candidate_index': case['candidate_index'],
                         'raw_job_sha256': sha(a.raw_results/'jobs'/(stem+'.json')),
                         'legacy_seed_job_sha256': sha(seed_path), 'portfolio': result}
            rel = 'jobs/'+stem+'.json'
            save(a.output/rel, job)
            rows.append(row)
            jobs.append({'path': rel, 'sha256': sha(a.output/rel)})
            print(json.dumps({'sample': entry['sample_index'], 'source': source,
                'selected': result['selected_start'], 'strict': after['strict_strip_valid'],
                'util': after['strip_utilization'], 'reused': result['seed_reused_raw_result']}), flush=True)
    save(a.output/'inputs.json', inputs)
    save(a.output/'summary.json', {'status': 'completed', 'planned_inputs': 32,
         'completed_inputs': len(jobs), 'jobs': jobs, 'rows': rows, 'aggregates': summarize(rows)})
    changed = []
    for name, digest in inputs.items():
        scope, rel = name.split('/', 1)
        root = {'scripts': Path(__file__).parent, 'frozen': a.frozen,
                'round7': a.round7, 'raw': a.raw_results}[scope]
        if sha(root/rel) != digest:
            changed.append(name)
    save(a.output/'metadata.json', {'status': 'completed' if not changed else 'input_changed',
         'changed_inputs': changed, 'input_count': len(inputs),
         'output_sha256': {f.relative_to(a.output).as_posix(): sha(f)
                           for f in sorted(a.output.rglob('*')) if f.is_file()}})
    if changed:
        raise RuntimeError('Frozen input changed')


if __name__ == '__main__':
    main()
