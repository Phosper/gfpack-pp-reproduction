"""Paired numerical-step diagnostic on this repository's saved raw predictions; no model sampling."""
import argparse
import json
from pathlib import Path
import shutil
import time

from .assets import sha256, write_json
from .postprocess import postprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Completed reproduce.run directory')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--post-step', default=1., type=float)
    parser.add_argument('--post-policy', choices=('simultaneous', 'pairwise'), default='simultaneous')
    parser.add_argument('--relaxation', type=float, default=1.75)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--candidate-indices', nargs='+', type=int,
                        help='Replay only these saved candidate IDs; explicitly reported as a subset')
    args = parser.parse_args()
    metadata_path = args.input / 'generation_metadata.json'
    if not metadata_path.exists():
        metadata_path = args.input / 'metadata.json'
    source_meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    if source_meta['status'] not in ('completed', 'sampling_completed'):
        raise ValueError('Source generation did not complete')
    data = json.loads((args.input / 'input.json').read_text(encoding='utf-8'))
    raw = json.loads((args.input / 'raw.json').read_text(encoding='utf-8'))
    indices = sorted(set(args.candidate_indices)) if args.candidate_indices is not None else list(range(len(raw)))
    if not indices or any(i < 0 or i >= len(raw) for i in indices):
        parser.error('Candidate IDs must be inside the saved raw array')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    sources = [metadata_path, args.input / 'raw.json', args.input / 'input.json']
    sources += [Path(__file__), Path(__file__).with_name('geometry.py'),
                Path(__file__).with_name('pairwise.py'), Path(__file__).with_name('verify.py'),
                Path(__file__).with_name('postprocess.py')]
    before = {str(p.resolve()): sha256(p) for p in sources}
    for name in ('input.json', 'raw.json'):
        target = args.output / name
        if target.exists():
            if sha256(target) != sha256(args.input / name):
                raise ValueError('Replay output contains different frozen inputs')
        else:
            shutil.copyfile(args.input / name, target)
    if not (args.output / 'generation_metadata.json').exists():
        write_json(args.output / 'generation_metadata.json', {
            'status': 'sampling_completed', 'height': source_meta['height'],
            'source_generation': str(args.input.resolve()), 'replay_only': True,
            'model_executed_in_this_directory': False})
    started = time.perf_counter()
    summary = postprocess(args.output, args.post_policy, args.workers, args.relaxation,
                          args.post_step, indices=indices, resume=args.resume)
    records = [r['enhanced'] for r in summary['records']]
    diagnostic = {'model_sampling_performed': False,
               'source_generation': str(args.input.resolve()), 'post_step': args.post_step,
               'post_policy': args.post_policy, 'relaxation': args.relaxation,
               'saved_candidate_count': len(raw), 'replayed_candidate_indices': indices,
               'replayed_all_saved_candidates': len(indices) == len(raw),
               'candidate_count': len(records), 'strict_valid_candidates': summary['strict_valid_candidates'],
               'selected_candidate': summary['selected_candidate'],
               'selected_metrics': summary['selected_metrics'],
               'sample_utilization_failure_zero': summary['sample_utilization_failure_zero'],
               'post_workers': args.workers,
               'input_sha256': before, 'input_unchanged': all(sha256(p) == value for p, value in before.items()),
               'elapsed_seconds': time.perf_counter()-started, 'metrics': records}
    path = args.output / 'diagnostic.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous['input_sha256'] != before or previous['metrics'] != records:
            raise ValueError('Existing diagnostic does not match verified replay')
    else:
        write_json(path, diagnostic)


if __name__ == '__main__':
    main()
