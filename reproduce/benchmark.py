"""Sequential fixed-index benchmark; retain unsuccessful samples in the mean."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .assets import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=2,
                        help='Instances to evaluate (default: 2 for diagnosis; use 128 for full Dental evaluation)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--candidates', type=int, default=512)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--model-mode', choices=('train', 'eval'), default='train',
                        help='Preserve official default state; eval must be explicitly selected')
    parser.add_argument('--sampling-profile', choices=('paper', 'public-source'), default='paper')
    parser.add_argument('--sigma-min', type=float, choices=(.1, 1.), default=None)
    parser.add_argument('--constraint-integrator', choices=('em', 'split-flow'), default=None)
    parser.add_argument('--constraint-threads', type=int, default=1)
    parser.add_argument('--constraint-max-substeps', type=int, default=32)
    parser.add_argument('--post-policy', choices=('pairwise', 'simultaneous'), default='pairwise')
    parser.add_argument('--post-workers', type=int, default=1)
    parser.add_argument('--relaxation', type=float, default=1.75)
    args = parser.parse_args()
    if args.sampling_profile == 'public-source':
        if args.sigma_min not in (None, 1.) or args.constraint_integrator is not None:
            parser.error('public-source uses its original sigma-min=1 and no geometric callback')
        args.sigma_min = 1.
    else:
        args.sigma_min = .1 if args.sigma_min is None else args.sigma_min
        args.constraint_integrator = args.constraint_integrator or 'em'
    if not 0 <= args.start < 128 or not 1 <= args.count <= 128 - args.start:
        parser.error('Dental test index range must be inside [0,128)')
    if args.candidates < 1 or (args.candidates != 512 and not args.smoke):
        parser.error('Reduced candidate counts require --smoke')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index in range(args.start, args.start + args.count):
        target = output / f'sample_{index:03d}'
        command = [sys.executable, '-m', 'reproduce.run', '--sample-index', str(index),
                   '--candidates', str(args.candidates), '--device', args.device,
                   '--model-mode', args.model_mode, '--sigma-min', str(args.sigma_min),
                   '--sampling-profile', args.sampling_profile,
                   '--constraint-threads', str(args.constraint_threads),
                   '--constraint-max-substeps', str(args.constraint_max_substeps),
                   '--post-policy', args.post_policy, '--post-workers', str(args.post_workers),
                   '--relaxation', str(args.relaxation), '--output', str(target)]
        if args.constraint_integrator is not None:
            command.extend(['--constraint-integrator', args.constraint_integrator])
        if args.smoke:
            command.append('--smoke')
        with (output / f'sample_{index:03d}.log').open('x', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(output / 'benchmark.json', {'status': 'aborted_on_runtime_error',
                       'failed_sample': index, 'completed_samples': len(records),
                       'planned_samples': args.count, 'model_mode': args.model_mode, 'records': records})
            raise RuntimeError(f'Sample {index} runtime failed; inspect its log before continuing')
        with (output / f'sample_{index:03d}_verify.log').open('x', encoding='utf-8') as stream:
            verified = subprocess.run([sys.executable, '-m', 'reproduce.verify', str(target)], cwd=ROOT,
                                      stdout=stream, stderr=subprocess.STDOUT)
        if verified.returncode:
            raise RuntimeError(f'Sample {index} independent verification failed')
        summary = json.loads((target / 'summary.json').read_text(encoding='utf-8'))
        records.append({'sample_index': index, 'sample_success': summary['sample_success'],
                        'selected_metrics': summary['selected_metrics'],
                        'utilization_failure_zero': summary['sample_utilization_failure_zero'],
                        'strict_valid_candidates': summary['strict_valid_candidates']})
        print(f'Completed sample {index}; success={summary["sample_success"]}', flush=True)
    successes = [r for r in records if r['sample_success']]
    write_json(output / 'benchmark.json', {'status': 'completed', 'sample_count': len(records),
               'candidates_per_sample': args.candidates, 'smoke_only': args.smoke,
               'model_mode': args.model_mode,
               'sampling_profile': args.sampling_profile,
               'sigma_min': args.sigma_min, 'constraint_integrator': args.constraint_integrator,
               'constraint_threads': args.constraint_threads, 'constraint_max_substeps': args.constraint_max_substeps,
               'post_policy': args.post_policy, 'post_workers': args.post_workers, 'relaxation': args.relaxation,
               'successful_samples': len(successes), 'sample_success_rate': len(successes) / len(records),
               'mean_utilization_failure_zero': sum(r['utilization_failure_zero'] for r in records) / len(records),
               'mean_utilization_on_successes': (sum(r['utilization_failure_zero'] for r in successes) / len(successes)
                                                if successes else None), 'records': records})


if __name__ == '__main__':
    main()
