"""Run the frozen round8 algorithm on 128 saved predictions and 128 GT controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True

SCRIPTS = ('freeze_engineering_transfer.py', 'run_engineering_transfer.py',
           'prepare_transfer_seeds.py', 'compact_portfolio.py', 'compact_layout.py',
           'translation_intervals.py', 'postprocess_metrics.py',
           'check_collision_saved_cases.py', 'verify_engineering_transfer.py',
           'verify_compact_layouts.py', 'verify_compact_portfolio.py')
SOURCES = {'pred': 'raw_actions', 'gt': 'gt_actions'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, exclusive=True):
    tick = time.perf_counter()
    path = Path(path)
    if exclusive:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        temp = path.with_suffix(path.suffix+'.tmp')
        with temp.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temp, path)
    return time.perf_counter()-tick


def peak_memory_bytes():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if get_memory(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize
        return None
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def worker(args):
    import numpy as np
    from freeze_engineering_transfer import checked_transfer_case
    from compact_layout import optimize
    from compact_portfolio import optimize_two_starts
    from postprocess_metrics import geometry, compare
    started = time.perf_counter()
    manifest = read(args.frozen/'manifest.json')
    entry = next(e for e in manifest['cases'] if e['sample_index'] == args.sample_index)
    _, case = checked_transfer_case(args.frozen, entry)
    stem = 'sample_%03d_%s' % (args.sample_index, args.source)
    seed_path = args.seeds/'jobs'/(stem+'.json')
    seed = read(seed_path)
    input_loading_seconds = time.perf_counter()-started
    field = SOURCES[args.source]
    original = case[field]
    for key, expected in [('sample_index', args.sample_index), ('source', args.source),
                          ('case_sha256', entry['sha256']), ('candidate_index', case['candidate_index']),
                          ('input_actions', original)]:
        if seed.get(key) != expected:
            raise ValueError('Seed provenance mismatch: '+key)
    before = geometry(case['polygons'], case['polygon_ids'], original)
    reference_field = 'enhanced_actions' if args.source == 'pred' else 'gt_enhanced_actions'
    baseline = {'raw': before,
                'enhanced': geometry(case['polygons'], case['polygon_ids'], case[reference_field])}
    raw_result = optimize(case['polygons'], case['polygon_ids'], original,
                          height=1200., seconds=45., sweeps=4)
    portfolio = optimize_two_starts(case['polygons'], case['polygon_ids'], original,
                                    seed['repair_actions'], raw_result=raw_result,
                                    height=1200., seconds=45., sweeps=4)
    after = geometry(case['polygons'], case['polygon_ids'], portfolio['actions'])
    raw_time = raw_result['elapsed_seconds']
    extra_time = (portfolio['seed_result']['elapsed_seconds']
                  if portfolio['seed_result'] is not None else 0.)
    timing = {'seed_generation': seed['elapsed_seconds'], 'raw_solver': raw_time,
              'distinct_seed_solver': extra_time, 'solver_total': raw_time+extra_time,
              'total_with_seed_generation': seed['elapsed_seconds']+raw_time+extra_time,
              'input_loading_and_validation': input_loading_seconds,
              'job_wall': time.perf_counter()-started}
    job = {'sample_index': args.sample_index, 'source': args.source,
           'candidate_index': case['candidate_index'], 'case_sha256': entry['sha256'],
           'group': entry.get('evaluation_group', 'development' if args.sample_index in manifest['development_samples'] else 'transfer'),
           'status': 'completed', 'input_field': field, 'seed': seed, 'seed_sha256': sha(seed_path),
           'raw_result': raw_result, 'portfolio': portfolio, 'before': before, 'after': after,
           'comparison': compare(before, after, original, portfolio['actions']),
           'baseline_geometry': baseline, 'timing_seconds': timing,
           'peak_worker_memory_bytes': peak_memory_bytes(),
           'stored_float32': bool(np.array_equal(np.asarray(portfolio['actions']),
                                                np.asarray(portfolio['actions'], dtype=np.float32)))}
    path = args.output/'jobs'/(stem+'.json')
    output_io = save(path, job, exclusive=False)
    job['timing_seconds']['io_seconds'] = input_loading_seconds+output_io
    job['timing_seconds']['output_first_write'] = output_io
    job['timing_seconds']['job_wall'] = time.perf_counter()-started
    save(path, job, exclusive=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--seeds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=(1, 2, 4))
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--sample-index', type=int)
    parser.add_argument('--source', choices=SOURCES)
    args = parser.parse_args()
    for key in ('frozen', 'seeds', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    if args.worker:
        worker(args)
        return
    import numpy as np
    import shapely
    from freeze_engineering_transfer import validate_transfer_frozen, checked_transfer_case
    from verify_engineering_transfer import aggregate_checked_jobs
    manifest, manifest_sha = validate_transfer_frozen(args.frozen)
    binding = read(Path(__file__).parent/'docs/compact_layout/local_binding.json')
    for name in ('compact_layout.py', 'translation_intervals.py', 'compact_portfolio.py', 'postprocess_metrics.py'):
        if sha(Path(__file__).with_name(name)) != binding['source_sha256'][name]:
            raise ValueError('Frozen round8 algorithm changed: '+name)
    seedmeta = read(args.seeds/'metadata.json')
    if seedmeta['status'] != 'completed' or seedmeta['changed_inputs']:
        raise ValueError('Seed generation incomplete or changed')
    for relative, expected in seedmeta['output_sha256'].items():
        if sha(args.seeds/relative) != expected:
            raise ValueError('Seed inventory changed: '+relative)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'jobs').mkdir()
    (args.output/'logs').mkdir()
    files = {'scripts/'+name: Path(__file__).with_name(name) for name in SCRIPTS}
    for path in args.frozen.rglob('*'):
        if path.is_file():
            files['frozen/'+path.relative_to(args.frozen).as_posix()] = path
    files['seeds/metadata.json'] = args.seeds/'metadata.json'
    for relative in seedmeta['output_sha256']:
        files['seeds/'+relative] = args.seeds/relative
    inputs = {key: sha(path) for key, path in files.items()}
    save(args.output/'inputs.json', inputs)
    features = []
    for entry in manifest['cases']:
        _, case = checked_transfer_case(args.frozen, entry)
        old = case['baseline_metrics']['diagnosis']['geometry']['raw_matched_candidate']
        features.append({'sample_index': entry['sample_index'],
                         'raw_max_pairwise_area': old['max_pairwise_area'],
                         'vertices': sum(len(case['polygons'][i]) for i in case['polygon_ids'])})
    overlap_order = sorted(features, key=lambda row: (row['raw_max_pairwise_area'], row['sample_index']))
    vertex_order = sorted(features, key=lambda row: (row['vertices'], row['sample_index']))
    profile_samples = list(dict.fromkeys([overlap_order[0]['sample_index'], overlap_order[-1]['sample_index'],
                                        vertex_order[0]['sample_index'], vertex_order[-1]['sample_index']]))
    for row in overlap_order:
        if len(profile_samples) >= 4:
            break
        if row['sample_index'] not in profile_samples:
            profile_samples.append(row['sample_index'])
    plan = {'experiment': 'engineering_transfer_128x1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'samples': list(range(128)), 'sources': SOURCES,
            'development_samples': manifest['development_samples'], 'planned_inputs': 256,
            'seed_directory': str(args.seeds), 'height': 1200., 'horizontal_walls': None,
            'strict_area_tolerance': 1e-9, 'boundary_tolerance': 1e-9, 'reference_area50': 50.,
            'seconds_per_attempt': 45., 'sweeps_per_attempt': 4, 'workers': args.workers,
            'worker_timeout_seconds': 125., 'sampling': False, 'training': False,
            'manifest_sha256': manifest_sha,
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                        'shapely': shapely.__version__, 'geos': shapely.geos_version_string,
                        'platform': platform.system()},
            'profile_samples': profile_samples,
            'profile_features': [row for row in features if row['sample_index'] in profile_samples],
            'profile_policy': 'Min/max old raw overlap and min/max contour vertex count; ties by sample index, duplicates filled by old-overlap rank; predictions and GT serial, then independent ledger replay before fixed worker-count execution',
            'timing_scope': 'total_with_seed_generation is a compute subtotal; job wall also includes geometry/input loading, io_seconds includes input validation and the first output write; final metadata rewrite is included only in stage wall',
            'native_metrics': 'Linux native legacy starts; new Windows results independently checked with reference geometry; no new calutil replay',
            'failure_policy': 'Every failed or timed-out input remains in planned denominators; no retries or mid-run tuning'}
    save(args.output/'plan.json', plan)
    jobs = []
    started = time.perf_counter()

    def launch(index, source):
        stem = 'sample_%03d_%s' % (index, source)
        relative = 'jobs/'+stem+'.json'
        tick = time.perf_counter()
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                   '--frozen', str(args.frozen), '--seeds', str(args.seeds), '--output', str(args.output),
                   '--sample-index', str(index), '--source', source]
        with (args.output/'logs'/(stem+'.log')).open('w', encoding='utf-8') as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=125.,
                    env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
                if process.returncode:
                    raise RuntimeError('Worker exit '+str(process.returncode))
                job = read(args.output/relative)
            except (subprocess.TimeoutExpired, RuntimeError, ValueError, OSError) as exc:
                entry = manifest['cases'][index]
                _, case = checked_transfer_case(args.frozen, entry)
                from postprocess_metrics import geometry
                original = case[SOURCES[source]]
                before = geometry(case['polygons'], case['polygon_ids'], original)
                enhanced = geometry(case['polygons'], case['polygon_ids'],
                                    case['enhanced_actions' if source == 'pred' else 'gt_enhanced_actions'])
                seed_path = args.seeds/'jobs'/(stem+'.json')
                seed = read(seed_path)
                job = {'sample_index': index, 'source': source, 'candidate_index': entry['candidate_index'],
                       'case_sha256': entry['sha256'], 'input_field': SOURCES[source],
                       'group': 'development' if index in manifest['development_samples'] else 'transfer',
                       'status': 'timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'execution_failed',
                       'error': str(exc), 'after': None, 'comparison': None, 'portfolio': None, 'raw_result': None,
                       'before': before, 'baseline_geometry': {'raw': before, 'enhanced': enhanced},
                       'seed': seed, 'seed_sha256': sha(seed_path),
                       'timing_seconds': {'job_wall': time.perf_counter()-tick,
                                          'seed_generation': seed['elapsed_seconds']}}
                if (args.output/relative).exists():
                    os.replace(args.output/relative, args.output/'logs'/(stem+'_interrupted.json'))
                temporary = (args.output/relative).with_suffix('.json.tmp')
                if temporary.exists():
                    os.replace(temporary, args.output/'logs'/(stem+'_partial.json'))
                save(args.output/relative, job)
        return job

    def record(job):
        jobs.append(job)
        ordered = sorted(jobs, key=lambda row: (row['sample_index'], row['source']))
        entries = [{'path': 'jobs/sample_%03d_%s.json' % (row['sample_index'], row['source']),
                    'sha256': sha(args.output/('jobs/sample_%03d_%s.json' % (row['sample_index'], row['source'])))} for row in ordered]
        summary = {'status': 'running', 'planned_inputs': 256, 'completed_inputs': len(jobs),
                   'jobs': entries, 'aggregates': aggregate_checked_jobs(ordered, manifest['development_samples']),
                   'elapsed_seconds': time.perf_counter()-started}
        save(args.output/'summary.json', summary, exclusive=False)
        print(json.dumps({'completed_inputs': len(jobs), 'sample': job['sample_index'], 'source': job['source'],
                          'status': job['status'], 'strict': (job.get('after') or {}).get('strict_strip_valid'),
                          'util': (job.get('after') or {}).get('strip_utilization')}), flush=True)

    for index in profile_samples:
        for source in SOURCES:
            record(launch(index, source))
    profile_elapsed = time.perf_counter()-started
    from verify_compact_portfolio import replay_pipeline
    from collections import Counter
    counts = Counter()
    verification_tick = time.perf_counter()
    for job in jobs:
        if job['status'] != 'completed':
            continue
        entry = manifest['cases'][job['sample_index']]
        _, case = checked_transfer_case(args.frozen, entry)
        _, _, replayed = replay_pipeline(case, case[SOURCES[job['source']]], job['raw_result'], 'profile/raw')
        counts.update(replayed)
        if job['portfolio']['seed_result'] is not None:
            _, _, replayed = replay_pipeline(case, job['seed']['repair_actions'], job['portfolio']['seed_result'], 'profile/seed')
            counts.update(replayed)
    profile_verification_seconds = time.perf_counter()-verification_tick
    save(args.output/'profile.json', {'samples': profile_samples, 'inputs': 8,
         'elapsed_seconds': profile_elapsed, 'workers': 1,
         'independent_ledger_counts': dict(counts),
         'independent_profile_verification_seconds': profile_verification_seconds,
         'maximum_peak_worker_memory_bytes': max((j.get('peak_worker_memory_bytes') or 0) for j in jobs),
         'seed_generation_seconds': sum(j.get('timing_seconds', {}).get('seed_generation', 0.) for j in jobs),
         'solver_seconds': sum(j.get('timing_seconds', {}).get('solver_total', 0.) for j in jobs),
         'contains_gt_controls': True, 'no_budget_or_algorithm_changes': True})
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(launch, index, source) for index in range(128)
                   if index not in profile_samples for source in SOURCES]
        for future in as_completed(futures):
            record(future.result())
    summary = read(args.output/'summary.json')
    summary['status'] = 'completed'
    summary['elapsed_seconds'] = time.perf_counter()-started
    seed_seconds = sum(j.get('timing_seconds', {}).get('seed_generation', 0.) for j in jobs)
    summary['cost_profile'] = {
        'native_seed_generation_seconds_sum': seed_seconds,
        'native_seed_stage_wall_seconds': seedmeta['elapsed_seconds'],
        'native_seed_io_and_overhead_seconds': max(0., seedmeta['elapsed_seconds']-seed_seconds),
        'solver_seconds_sum': sum(j.get('timing_seconds', {}).get('solver_total', 0.) for j in jobs),
        'io_seconds_sum_including_input_validation': sum(j.get('timing_seconds', {}).get('io_seconds', 0.) for j in jobs),
        'run_wall_seconds_including_io_and_worker_startup': summary['elapsed_seconds'],
        'completed_inputs_per_run_wall_second': len(jobs)/summary['elapsed_seconds'],
        'maximum_peak_worker_memory_bytes': max((j.get('peak_worker_memory_bytes') or 0) for j in jobs),
        'independent_verification': 'Measured separately in sibling completion.json',
        'timing_scope': 'Native seeds generated remotely, solver locally; component sums and measured stage wall times are reported separately'}
    save(args.output/'summary.json', summary, exclusive=False)
    changed = [key for key, path in files.items() if sha(path) != inputs[key]]
    save(args.output/'metadata.json', {'status': 'completed' if not changed else 'input_changed',
         'changed_inputs': changed, 'input_count': len(inputs),
         'output_sha256': {path.relative_to(args.output).as_posix(): sha(path)
                           for path in sorted(args.output.rglob('*')) if path.is_file()}})
    if changed:
        raise RuntimeError('Frozen source changed during experiment')
    verification_path = args.output.parent/(args.output.name+'_verification.json')
    command = [sys.executable, '-B', str(Path(__file__).with_name('verify_engineering_transfer.py')),
               '--frozen', str(args.frozen), '--results', str(args.output), '--output', str(verification_path)]
    tick = time.perf_counter()
    with (args.output.parent/(args.output.name+'_verification.log')).open('x', encoding='utf-8') as stream:
        code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
    verification_seconds = time.perf_counter()-tick
    save(args.output.parent/(args.output.name+'_completion.json'), {
         'run_status': 'completed', 'verification_exit_code': code,
         'verification_seconds': verification_seconds, 'verification_path': str(verification_path),
         'measured_stage_wall_seconds_sum': seedmeta['elapsed_seconds']+summary['elapsed_seconds']+verification_seconds,
         'total_wall_scope': 'Sum of native seed, local run (includes I/O/profile replay), final independent verification stages; excludes freezing, network transfer and human preparation'})
    if code:
        raise RuntimeError('Independent verification failed; original experiment retained')


if __name__ == '__main__':
    main()
