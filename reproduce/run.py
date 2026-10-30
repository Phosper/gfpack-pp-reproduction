"""Run the unchanged official Dental network with explicit paper-mechanism reconstruction."""
import argparse
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from src.score_model import PolygonPackingTransformer
from src.sde import ve_marginal_prob
from .official_sampling import sample_public
from .assets import CHECKPOINT_SHA256, ROOT, sha256, write_json
from .data import load_resources, make_features
from .geometry import Geometry
from .pairwise import DISPLACEMENT_POLICY
from .sampling import sample
from .selection import select_candidate


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
    parser.add_argument('--post-workers', type=int, default=1,
                        help='Independent candidate worker processes; no torch import in post workers')
    parser.add_argument('--sampling-profile', choices=('paper', 'public-source'), default='paper',
                        help='paper reconstructs stated generation; public-source calls the unchanged author sampler')
    parser.add_argument('--constraint-integrator', choices=('em', 'split-flow'), default=None,
                        help='split-flow is a declared stable numerical reconstruction; em retains the old update')
    parser.add_argument('--constraint-threads', type=int, default=1)
    parser.add_argument('--constraint-max-substeps', type=int, default=32)
    parser.add_argument('--relaxation', type=float, default=1.75,
                        help='Declared numerical choice for pairwise post updates, not an author parameter')
    parser.add_argument('--model-mode', choices=('eval', 'train'), default='train',
                        help='Default preserves the public entrypoint state; no gradients or training. eval is a diagnostic override.')
    parser.add_argument('--sigma-min', type=float, choices=(.1, 1.), default=None,
                        help='0.1 = paper; 1 = explicitly labelled public-weight scale diagnostic')
    args = parser.parse_args()
    if args.sampling_profile == 'public-source':
        if args.sigma_min not in (None, 1.):
            parser.error('The public-source sampler uses sigma-min=1; do not mix its profile with 0.1')
        if args.constraint_integrator is not None:
            parser.error('The unchanged public sampler has no late geometry callback; constraint-integrator belongs to paper profile')
        args.sigma_min, args.constraint_integrator = 1., 'none-public-source'
    else:
        args.sigma_min = .1 if args.sigma_min is None else args.sigma_min
        args.constraint_integrator = args.constraint_integrator or 'em'
    if args.candidates < 1 or (args.candidates != 512 and not args.smoke):
        parser.error('Use 512 candidates for paper evaluation, or explicitly mark --smoke')
    if args.threads < 1 or not np.isfinite(args.post_step) or args.post_step <= 0:
        parser.error('Threads and post-step must be positive')
    if min(args.post_workers, args.constraint_threads, args.constraint_max_substeps) < 1:
        parser.error('Geometry threads, post workers and substep limit must be positive')
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
                'seed': 3407,
                'sampling_profile': args.sampling_profile,
                'not_paper_numerical_profile': (args.sigma_min != .1 or args.sampling_profile != 'paper'
                                                or args.constraint_integrator != 'em'),
                'steps': 128, 'constraints_steps': [97, 128] if args.sampling_profile == 'paper' else [], 'height': 1200.,
                'post_max_iterations': 100,
                'post_initial_step': args.post_step if args.post_policy == 'simultaneous' else None,
                'post_policy': args.post_policy, 'relaxation': args.relaxation,
                'post_workers': args.post_workers,
                'constraint_integrator': args.constraint_integrator,
                'constraint_threads': args.constraint_threads,
                'constraint_max_substeps': args.constraint_max_substeps,
                'pair_displacement_policy': DISPLACEMENT_POLICY if args.post_policy == 'pairwise' else None,
                'area_acceptance_tolerance': 0., 'storage': 'JSON float64 coordinates',
                'checkpoint_sha256': CHECKPOINT_SHA256, 'input_sha256': frozen,
                'model_executed': False, 'model_weights_changed': False,
                'python': platform.python_version(),
                'versions': {p: importlib.metadata.version(p) for p in ('torch', 'torch-geometric', 'numpy', 'shapely')},
                'implementation_choices': [
                    ('128 public linspace time points; uniform initialization; drift multiplied by 14; no geometric callback'
                     if args.sampling_profile == 'public-source' else
                     'Gaussian initialization; 128 equal time intervals from 1 to 0.01; left endpoint scores'),
                    'nonconvex minimum-translation extension by convex Minkowski union',
                    'translation-only geometric constraint; learned score still updates rotations',
                    ('unmodified author pc_sampler_state; only observational forward hooks added'
                     if args.sampling_profile == 'public-source' else
                     ('neural EM proposal followed by bounded-step collision-flow integration'
                      if args.constraint_integrator == 'split-flow' else 'single EM constrained drift at the old state')),
                    'split-flow, if explicitly requested, is a numerical extension rather than an author-confirmed implementation',
                    ('cyclic pairwise splitting with relaxation '+str(args.relaxation)
                     if args.post_policy == 'pairwise' else
                     'simultaneous field and Armijo halving'),
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
        if args.sampling_profile == 'public-source':
            raw, trajectory, sampling_trace = sample_public(model, torch.tensor(ids, device=device),
                                                             features, args.candidates)
        else:
            raw, trajectory, sampling_trace = sample(model, torch.tensor(ids, device=device),
                                                     features, geometry, args.candidates,
                                                     sigma_min=args.sigma_min,
                                                     constraint_integrator=args.constraint_integrator,
                                                     constraint_threads=args.constraint_threads,
                                                     constraint_max_substeps=args.constraint_max_substeps)
        metadata['sampling_seconds'] = time.perf_counter() - sampling_start
        torch.save(trajectory, args.output / 'trajectory.pt')
        write_json(args.output / 'sampling_trace.json', sampling_trace)
        write_json(args.output / 'raw.json', raw.tolist())
        metadata['model_runtime_parameters_unchanged'] = all(
            torch.equal(value.detach().cpu(), state[name]) for name, value in model.state_dict().items())
        if not metadata['model_runtime_parameters_unchanged']:
            raise ValueError('Neural parameters or buffers changed during generation')
        write_json(args.output / 'generation_metadata.json', {**metadata, 'status': 'sampling_completed'})
        del model, state, features, geometry, trajectory, raw
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        post_started = time.perf_counter()
        # A torch-free subprocess owns its process pool. Windows spawn workers
        # therefore do not each import the neural model/CUDA runtime.
        subprocess.run([sys.executable, '-m', 'reproduce.postprocess', '--directory', str(args.output.resolve()),
                        '--post-policy', args.post_policy, '--workers', str(args.post_workers),
                        '--relaxation', str(args.relaxation), '--post-step', str(args.post_step)],
                       cwd=ROOT, check=True)
        metadata['postprocessing_seconds'] = time.perf_counter()-post_started
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
