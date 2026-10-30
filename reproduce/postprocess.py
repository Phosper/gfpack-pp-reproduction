"""Parallel, resumable postprocessing of frozen raw candidates. No torch import."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

from .assets import ROOT, sha256, write_json
from .geometry import Geometry, enhance
from .pairwise import enhance_pairwise
from .selection import select_candidate
from .verify import measure

_GEOMETRY = None
_POLICY = None
_RELAXATION = None
_POST_STEP = None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pose_hash(actions):
    return hashlib.sha256(json.dumps(actions, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def initialize_worker(contours, height, policy, relaxation, post_step):
    global _GEOMETRY, _POLICY, _RELAXATION, _POST_STEP
    _GEOMETRY = Geometry(contours, height)
    _POLICY, _RELAXATION, _POST_STEP = policy, relaxation, post_step


def calculate(item):
    index, actions = item
    started = time.perf_counter()
    try:
        before = _GEOMETRY.measure(actions)
        if _POLICY == 'pairwise':
            enhanced, result = enhance_pairwise(_GEOMETRY, actions, relaxation=_RELAXATION)
        else:
            enhanced, result = enhance(_GEOMETRY, actions, initial_step=_POST_STEP)
        return {'candidate': index, 'raw_candidate_sha256': pose_hash(actions),
                'raw_metrics': before, 'actions': enhanced.tolist(), 'result': result,
                'worker_pid': os.getpid(), 'worker_imported_torch': 'torch' in sys.modules,
                'postprocessing_seconds': time.perf_counter()-started}
    except Exception as error:
        return {'candidate': index, 'error': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc(), 'geometry_failure': getattr(error, 'diagnostic', None)}


def verify_payload(payload, contours, height, raw):
    index = payload['candidate']
    if payload['raw_candidate_sha256'] != pose_hash(raw[index]):
        raise ValueError('Cached candidate does not correspond to this raw input')
    if not np.array_equal(np.asarray(payload['actions'])[:, 2:], np.asarray(raw[index])[:, 2:]):
        raise ValueError('Postprocessing changed a sampled rotation')
    independent = measure(contours, payload['actions'], height)
    recorded = payload['result']['metrics']
    if independent['strict_valid'] != recorded['strict_valid']:
        raise ArithmeticError(f'Independent validity disagreement for candidate {index}')
    for key in ('max_pair_overlap_area', 'sum_pair_overlap_area', 'height_violation', 'span_x', 'strip_utilization'):
        if not math.isclose(independent[key], recorded[key], rel_tol=1e-9, abs_tol=1e-7):
            raise ArithmeticError(f'Independent metric disagreement: candidate {index}, {key}')
    return independent


def postprocess(directory, policy='pairwise', workers=1, relaxation=1.75,
                post_step=1., indices=None, resume=False):
    directory = Path(directory).resolve()
    if policy not in ('pairwise', 'simultaneous') or workers < 1:
        raise ValueError('Unknown policy or invalid worker count')
    if not math.isfinite(relaxation) or not 0 < relaxation < 2:
        raise ValueError('Invalid pairwise relaxation')
    if not math.isfinite(post_step) or post_step <= 0:
        raise ValueError('Invalid simultaneous post step')
    meta_path = directory / 'generation_metadata.json'
    if not meta_path.exists():
        meta_path = directory / 'metadata.json'
    meta = read(meta_path)
    if meta['status'] not in ('sampling_completed', 'completed'):
        raise ValueError('Saved generation did not complete')
    data, raw = read(directory / 'input.json'), read(directory / 'raw.json')
    ids = sorted(set(indices)) if indices is not None else list(range(len(raw)))
    if not ids or any(i < 0 or i >= len(raw) for i in ids):
        raise ValueError('Requested candidate IDs are outside the saved raw array')
    modules = ('geometry.py', 'pairwise.py', 'collision_flow.py',
               'postprocess.py', 'selection.py', 'verify.py')
    binding = {'input_sha256': sha256(directory / 'input.json'), 'raw_sha256': sha256(directory / 'raw.json'),
               'height': meta['height'], 'candidate_indices': ids, 'post_policy': policy,
               'relaxation': relaxation, 'post_step': post_step,
               'code_sha256': {name: sha256(ROOT / 'reproduce' / name) for name in modules}}
    binding_path = directory / 'postprocess_binding.json'
    if binding_path.exists():
        if not resume:
            raise FileExistsError('Postprocessing already exists; use --resume after checking its binding')
        if read(binding_path) != binding:
            raise ValueError('Resume refused: code, input, or postprocessing configuration changed')
    else:
        if any(directory.glob('candidate_*.json')) or (directory / 'summary.json').exists():
            raise FileExistsError('Existing results have no matching binding; refusing to overwrite them')
        write_json(binding_path, binding)
        for name in modules:
            dest = directory / 'frozen_postprocess_code' / 'reproduce' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'reproduce' / name, dest)
    payloads = {}
    pending = []
    for index in ids:
        path = directory / f'candidate_{index:04d}.json'
        if path.exists():
            payload = read(path)
            if payload['candidate'] != index:
                raise ValueError('Cached candidate filename/index mismatch')
            verify_payload(payload, data['contours'], meta['height'], raw)
            payloads[index] = payload
        else:
            pending.append((index, raw[index]))
    started = time.perf_counter()
    progress = directory / 'postprocess_progress.json'
    def status(state):
        progress.write_text(json.dumps({'status': state, 'planned_candidates': len(ids),
                            'completed_candidates': len(payloads), 'workers': workers,
                            'elapsed_seconds_this_invocation': time.perf_counter()-started}, indent=2), encoding='utf-8')
    status('running')
    try:
        if workers == 1:
            initialize_worker(data['contours'], meta['height'], policy, relaxation, post_step)
            executor = None
            iterator = map(calculate, pending)
        else:
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                         initializer=initialize_worker,
                         initargs=(data['contours'], meta['height'], policy, relaxation, post_step))
            iterator = executor.map(calculate, pending, chunksize=1)
        try:
            for payload in iterator:
                index = payload['candidate']
                if 'error' in payload:
                    failure = directory / f'postprocess_error_{index:04d}_{time.time_ns()}.json'
                    write_json(failure, payload)
                    raise RuntimeError(f'Candidate {index} runtime error; inspect {failure.name}')
                write_json(directory / f'candidate_{index:04d}.json', payload)
                persisted = read(directory / f'candidate_{index:04d}.json')
                independent = verify_payload(persisted, data['contours'], meta['height'], raw)
                payloads[index] = persisted
                status('running')
                print(f'Candidate {index}: valid={independent["strict_valid"]}; overlap={independent["sum_pair_overlap_area"]:.12g}; util={independent["strip_utilization"]:.4%}; done={len(payloads)}/{len(ids)}', flush=True)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
    except Exception:
        status('failed')
        raise
    metrics = [verify_payload(payloads[i], data['contours'], meta['height'], raw) for i in ids]
    local_selected = select_candidate(metrics)
    chosen = ids[local_selected] if local_selected is not None else None
    summary = {'sample_index': data.get('sample_index'), 'candidate_count': len(ids),
               'saved_candidate_count': len(raw), 'replayed_candidate_indices': ids,
               'replayed_all_saved_candidates': len(ids) == len(raw),
               'strict_valid_candidates': sum(m['strict_valid'] for m in metrics),
               'sample_success': chosen is not None, 'selected_candidate': chosen,
               'selected_metrics': metrics[local_selected] if chosen is not None else None,
               'sample_utilization_failure_zero': metrics[local_selected]['strip_utilization'] if chosen is not None else 0.,
               'post_policy': policy, 'post_workers': workers,
               'records': [{'candidate': i, 'raw': payloads[i]['raw_metrics'], 'enhanced': m,
                            'status': payloads[i]['result']['status'], 'iterations': payloads[i]['result']['iterations']}
                           for i, m in zip(ids, metrics)]}
    if (sha256(directory / 'input.json') != binding['input_sha256'] or
        sha256(directory / 'raw.json') != binding['raw_sha256'] or
        any(sha256(ROOT / 'reproduce' / name) != digest for name, digest in binding['code_sha256'].items())):
        status('failed')
        raise ValueError('Inputs or solver source changed during postprocessing')
    final_path = directory / 'summary.json'
    if final_path.exists():
        previous = read(final_path)
        # Worker count is execution metadata, not the algorithm or score.
        if any(previous.get(k) != v for k, v in summary.items() if k != 'post_workers'):
            status('failed')
            raise ValueError('Existing summary disagrees with verified candidates')
    else:
        write_json(final_path, summary)
    status('completed')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--post-policy', choices=('pairwise', 'simultaneous'), default='pairwise')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--relaxation', type=float, default=1.75)
    parser.add_argument('--post-step', type=float, default=1.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    postprocess(args.directory, args.post_policy, args.workers, args.relaxation, args.post_step, resume=args.resume)


if __name__ == '__main__':
    main()
