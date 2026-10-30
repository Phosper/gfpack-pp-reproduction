"""Run the unchanged official Dental network with explicit paper-mechanism reconstruction."""
import argparse
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from src.score_model import PolygonPackingTransformer
from src.sde import ve_marginal_prob
from .assets import CHECKPOINT_SHA256, ROOT, sha256, write_json
from .data import load_resources, make_features
from .geometry import Geometry, enhance
from .pairwise import DISPLACEMENT_POLICY, enhance_pairwise
from .sampling import sample


def select_candidate(metrics):
    valid = [i for i, row in enumerate(metrics) if row['strict_valid']]
    return max(valid, key=lambda i: (metrics[i]['strip_utilization'], -i)) if valid else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--candidates', type=int, default=512)
    parser.add_argument('--smoke', action='store_true', help='Required for a reduced candidate count')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', required=True, type=Path, help='New, nonexistent output directory')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--post-step', type=float, default=1.)
    parser.add_argument('--post-policy', choices=('simultaneous', 'pairwise'), default='pairwise')
    parser.add_argument('--relaxation', type=float, default=1.75,
                        help='Declared numerical choice for pairwise post updates, not an author parameter')
    parser.add_argument('--model-mode', choices=('eval', 'train'), default='train',
                        help='Default preserves the public entrypoint state; no gradients or training. eval is a diagnostic override.')
    parser.add_argument('--sigma-min', type=float, choices=(.1, 1.), default=.1,
                        help='0.1 = paper; 1 = explicitly labelled public-weight scale diagnostic')
    args = parser.parse_args()
    if args.candidates < 1 or (args.candidates != 512 and not args.smoke):
        parser.error('Use 512 candidates for paper evaluation, or explicitly mark --smoke')
    if args.threads < 1 or not np.isfinite(args.post_step) or args.post_step <= 0:
        parser.error('Threads and post-step must be positive')
    if not 0 < args.relaxation < 2 or not np.isfinite(args.relaxation):
        parser.error('Relaxation must be in (0,2)')
    checkpoint = ROOT / 'checkpoints' / 'dental.pth'
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Official checkpoint identity mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3407)
    contours, ids, gt = load_resources(args.sample_index)
    sources = list((ROOT / 'reproduce').glob('*.py')) + list((ROOT / 'src').glob('*.py'))
    sources += [ROOT / 'data/official_dental/dental_test.pkl', checkpoint]
    sources += sorted((ROOT / 'data/official_dental/polygons').glob('*.txt'))
    frozen = {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in sources}
    for source in sources:
        if source.suffix == '.py':
            target = args.output / 'frozen_code' / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    metadata = {'status': 'running', 'upstream_commit': 'ff75815e24ec50cd7fa8b2abdec1acfe0b55be95',
                'paper': 'https://arxiv.org/html/2406.07579v2', 'sample_index': args.sample_index,
                'candidates': args.candidates, 'smoke_only': args.smoke,
                'device': args.device, 'model_mode': args.model_mode, 'sigma_min': args.sigma_min,
                'not_paper_numerical_profile': args.sigma_min != .1,
                'steps': 128, 'constraints_steps': [97, 128], 'height': 1200.,
                'post_max_iterations': 100,
                'post_initial_step': args.post_step if args.post_policy == 'simultaneous' else None,
                'post_policy': args.post_policy, 'relaxation': args.relaxation,
                'pair_displacement_policy': DISPLACEMENT_POLICY if args.post_policy == 'pairwise' else None,
                'area_acceptance_tolerance': 0., 'storage': 'JSON float64 coordinates',
                'checkpoint_sha256': CHECKPOINT_SHA256, 'input_sha256': frozen,
                'model_executed': False, 'model_weights_changed': False,
                'python': platform.python_version(),
                'versions': {p: importlib.metadata.version(p) for p in ('torch', 'torch-geometric', 'numpy', 'shapely')},
                'implementation_choices': ['128 equal time intervals from 1 to 0.01, left endpoint scores',
                    'nonconvex minimum-translation extension by convex Minkowski union',
                    'translation-only geometric constraint; learned score still updates rotations',
                    ('cyclic pairwise splitting with relaxation '+str(args.relaxation)
                     if args.post_policy == 'pairwise' else 'simultaneous field and Armijo halving'),
                    ('validate actual weighted endpoints, retry minimum separation on unsafe overrelaxation'
                     if args.post_policy == 'pairwise' else 'accept by declared Armijo energy test'),
                    'post pair order and relaxation are not specified by the paper',
                    'collision updates followed by sequential -x then -y binary search',
                    'gap moves may start from overlap; only legal full-contour endpoints are accepted',
                    'contact separation margin 1e-7, exact zero-area final acceptance',
                    'standalone Python/GEOS geometry; no claim to run author native C++ callback'],
                'paper_full_reproduction_claim': False}
    write_json(args.output / 'initial_metadata.json', metadata)
    write_json(args.output / 'input.json', {'sample_index': args.sample_index,
               'polygon_ids': ids.tolist(), 'contours': [contours[i] for i in ids]})
    started = time.perf_counter()
    try:
        device = torch.device(args.device)
        model = PolygonPackingTransformer(partial(ve_marginal_prob, sigma_min=args.sigma_min,
                                                  sigma_max=1000.), device).to(device)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        def mark_model_executed(module, inputs):
            metadata['model_executed'] = True
        model.register_forward_pre_hook(mark_model_executed)
        getattr(model, args.model_mode)()
        metadata['model_training_flag'] = model.training
        metadata['model_parameters'] = sum(p.numel() for p in model.parameters())
        features = make_features(contours).to(device)
        geometry = Geometry([contours[i] for i in ids])
        sampling_start = time.perf_counter()
        raw, trajectory, sampling_trace = sample(model, torch.tensor(ids, device=device),
                                                 features, geometry, args.candidates,
                                                 sigma_min=args.sigma_min)
        metadata['sampling_seconds'] = time.perf_counter() - sampling_start
        torch.save(trajectory, args.output / 'trajectory.pt')
        write_json(args.output / 'sampling_trace.json', sampling_trace)
        write_json(args.output / 'raw.json', raw.tolist())
        records = []
        metrics = []
        for index, pose in enumerate(raw):
            before = geometry.measure(pose)
            if args.post_policy == 'pairwise':
                enhanced, result = enhance_pairwise(geometry, pose, relaxation=args.relaxation)
            else:
                enhanced, result = enhance(geometry, pose, initial_step=args.post_step)
            payload = {'candidate': index, 'raw_metrics': before, 'actions': enhanced.tolist(),
                       'result': result}
            write_json(args.output / f'candidate_{index:04d}.json', payload)
            # Re-evaluate the actual persisted values before selection.
            persisted = json.loads((args.output / f'candidate_{index:04d}.json').read_text(encoding='utf-8'))
            measured = geometry.measure(persisted['actions'])
            if measured != result['metrics']:
                raise ArithmeticError('Stored coordinate metrics changed')
            metrics.append(measured)
            records.append({'candidate': index, 'raw': before, 'enhanced': measured,
                            'status': result['status'], 'iterations': result['iterations']})
            print(f'Candidate {index}: valid={measured["strict_valid"]}; overlap={measured["sum_pair_overlap_area"]:.12g}; util={measured["strip_utilization"]:.4%}', flush=True)
        selected = select_candidate(metrics)
        summary = {'sample_index': args.sample_index, 'candidate_count': len(metrics),
                   'strict_valid_candidates': sum(m['strict_valid'] for m in metrics),
                   'sample_success': selected is not None, 'selected_candidate': selected,
                   'selected_metrics': metrics[selected] if selected is not None else None,
                   'sample_utilization_failure_zero': metrics[selected]['strip_utilization'] if selected is not None else 0.,
                   'records': records}
        write_json(args.output / 'summary.json', summary)
        metadata['output_sha256'] = {p.name: sha256(p) for p in args.output.iterdir()
                                     if p.is_file() and p.name != 'initial_metadata.json'}
        metadata.update(status='completed', elapsed_seconds=time.perf_counter() - started,
                        input_unchanged=all(sha256(ROOT / name) == value for name, value in frozen.items()),
                        model_weights_changed=sha256(checkpoint) != CHECKPOINT_SHA256)
    except Exception as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}',
                        elapsed_seconds=time.perf_counter() - started)
        if hasattr(error, 'diagnostic'):
            metadata['geometry_failure'] = error.diagnostic
        raise
    finally:
        write_json(args.output / 'metadata.json', metadata)


if __name__ == '__main__':
    main()
