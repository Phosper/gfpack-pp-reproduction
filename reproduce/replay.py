"""Paired numerical-step diagnostic on this repository's saved raw predictions; no model sampling."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .assets import sha256, write_json
from .geometry import Geometry, enhance
from .pairwise import enhance_pairwise
from .verify import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Completed reproduce.run directory')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--post-step', default=1., type=float)
    parser.add_argument('--post-policy', choices=('simultaneous', 'pairwise'), default='simultaneous')
    parser.add_argument('--relaxation', type=float, default=1.75)
    parser.add_argument('--candidate-indices', nargs='+', type=int,
                        help='Replay only these saved candidate IDs; explicitly reported as a subset')
    args = parser.parse_args()
    source_meta = json.loads((args.input / 'metadata.json').read_text(encoding='utf-8'))
    if source_meta['status'] != 'completed':
        raise ValueError('Source generation did not complete')
    data = json.loads((args.input / 'input.json').read_text(encoding='utf-8'))
    raw = json.loads((args.input / 'raw.json').read_text(encoding='utf-8'))
    indices = sorted(set(args.candidate_indices)) if args.candidate_indices is not None else list(range(len(raw)))
    if not indices or any(i < 0 or i >= len(raw) for i in indices):
        parser.error('Candidate IDs must be inside the saved raw array')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [args.input / name for name in ('metadata.json', 'raw.json', 'input.json')]
    sources += [Path(__file__), Path(__file__).with_name('geometry.py'),
                Path(__file__).with_name('pairwise.py'), Path(__file__).with_name('verify.py')]
    before = {str(p.resolve()): sha256(p) for p in sources}
    geometry = Geometry(data['contours'], source_meta['height'])
    started = time.perf_counter()
    records = []
    for index in indices:
        actions = raw[index]
        if args.post_policy == 'pairwise':
            poses, result = enhance_pairwise(geometry, actions, relaxation=args.relaxation)
        else:
            poses, result = enhance(geometry, actions, initial_step=args.post_step)
        write_json(args.output / f'candidate_{index:04d}.json', {'actions': poses.tolist(), 'result': result})
        stored = json.loads((args.output / f'candidate_{index:04d}.json').read_text(encoding='utf-8'))
        independent = measure(data['contours'], stored['actions'], source_meta['height'])
        if independent['strict_valid'] != result['metrics']['strict_valid']:
            raise ArithmeticError('Independent final geometry disagrees')
        records.append(independent)
        print(json.dumps({'candidate': index, 'post_step': args.post_step, 'status': result['status'],
                          'iterations': result['iterations'], **independent}), flush=True)
    valid = [(i, row) for i, row in zip(indices, records) if row['strict_valid']]
    chosen = max(valid, key=lambda item: (item[1]['strip_utilization'], -item[0])) if valid else None
    write_json(args.output / 'diagnostic.json', {'model_sampling_performed': False,
               'source_generation': str(args.input.resolve()), 'post_step': args.post_step,
               'post_policy': args.post_policy, 'relaxation': args.relaxation,
               'saved_candidate_count': len(raw), 'replayed_candidate_indices': indices,
               'replayed_all_saved_candidates': len(indices) == len(raw),
               'candidate_count': len(records), 'strict_valid_candidates': len(valid),
               'selected_candidate': chosen[0] if chosen else None,
               'selected_metrics': chosen[1] if chosen else None,
               'sample_utilization_failure_zero': chosen[1]['strip_utilization'] if chosen else 0.,
               'input_sha256': before, 'input_unchanged': all(sha256(p) == value for p, value in before.items()),
               'elapsed_seconds': time.perf_counter()-started, 'metrics': records})


if __name__ == '__main__':
    main()
