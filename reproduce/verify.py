"""Independent measurement of persisted contours/poses, without importing the optimizer."""
import argparse
import json
import math
from pathlib import Path

from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from .assets import ROOT, sha256, write_json


def measure(contours, actions, height=1200.):
    if len(contours) != len(actions) or not actions:
        raise ValueError('Action/contour count mismatch')
    shapes = []
    for vertices, (x, y, cosine, sine) in zip(contours, actions):
        if not all(math.isfinite(v) for v in (x, y, cosine, sine)):
            raise ValueError('Nonfinite pose')
        theta = math.atan2(sine, cosine)
        c, s = math.cos(theta), math.sin(theta)
        shapes.append(affine_transform(Polygon(vertices), [c, -s, s, c, x, y]))
    intersections = [shapes[i].intersection(shapes[j]).area
                     for i in range(len(shapes)) for j in range(i + 1, len(shapes))]
    overlap = sum(intersections)
    violation = max(0., -min(p.bounds[1] for p in shapes), max(p.bounds[3] for p in shapes) - height)
    span = max(p.bounds[2] for p in shapes) - min(p.bounds[0] for p in shapes)
    return {'max_pair_overlap_area': max(intersections, default=0.),
            'sum_pair_overlap_area': overlap, 'height_violation': violation,
            'span_x': span, 'strip_utilization': sum(p.area for p in shapes) / (height * span),
            'strict_valid': overlap == 0. and violation == 0.}


def verify(directory):
    read = lambda name: json.loads((directory / name).read_text(encoding='utf-8'))
    meta, summary, data = read('metadata.json'), read('summary.json'), read('input.json')
    if meta['status'] != 'completed' or not meta['model_executed'] or not meta['input_unchanged'] or meta['model_weights_changed']:
        raise ValueError('Incomplete experiment or changed inputs')
    def source_path(path):
        snapshot = directory / 'frozen_code' / path
        return snapshot if path.endswith('.py') and snapshot.exists() else ROOT / path
    if any(sha256(source_path(path)) != expected for path, expected in meta['input_sha256'].items()):
        raise ValueError('Inputs no longer match the recorded experiment')
    if any(sha256(directory / path) != expected for path, expected in meta.get('output_sha256', {}).items()):
        raise ValueError('Outputs no longer match their recorded hashes')
    trace = read('sampling_trace.json')
    profile = meta.get('sampling_profile', 'paper')
    if profile not in ('paper', 'public-source'):
        raise ValueError('Unknown generation profile')
    expected_schedule = [] if profile == 'public-source' else list(range(97, 129))
    if len(trace) != 128 or [r['step'] for r in trace] != list(range(1,129)) or [r['step'] for r in trace if r['constraint_active']] != expected_schedule:
        raise ValueError('Invalid late-constraint schedule')
    if profile == 'public-source' and (meta['sigma_min'] != 1. or
        meta.get('constraint_integrator') != 'none-public-source' or not meta.get('not_paper_numerical_profile')):
        raise ValueError('Public source settings were incorrectly labelled as paper settings')
    if 'model_training_flag' in meta:
        expected = meta['model_mode'] == 'train'
        if meta['model_training_flag'] != expected or any(r.get('model_training_flag') != expected or r.get('grad_enabled') is not False for r in trace):
            raise ValueError('Runtime model mode or no-grad state disagrees with requested configuration')
    measured = []
    for index in range(summary['candidate_count']):
        saved = read(f'candidate_{index:04d}.json')
        row = measure(data['contours'], saved['actions'], meta['height'])
        recorded = saved['result']['metrics']
        if row['strict_valid'] != recorded['strict_valid']:
            raise ArithmeticError(f'Independent strict-validity disagreement for candidate {index}')
        for name in ('sum_pair_overlap_area', 'max_pair_overlap_area', 'span_x', 'height_violation', 'strip_utilization'):
            if not math.isclose(row[name], recorded[name], rel_tol=1e-9, abs_tol=1e-7):
                raise ArithmeticError((index, name, row[name], recorded[name]))
        measured.append(row)
    valid = [i for i, row in enumerate(measured) if row['strict_valid']]
    chosen = max(valid, key=lambda i: (measured[i]['strip_utilization'], -i)) if valid else None
    if chosen != summary['selected_candidate'] or len(valid) != summary['strict_valid_candidates']:
        raise ArithmeticError('Candidate selection/count mismatch')
    return {'verified': True, 'candidate_count': len(measured), 'strict_valid_candidates': len(valid),
            'selected_candidate': chosen, 'selected_metrics': measured[chosen] if chosen is not None else None,
            'not_a_full_128_sample_benchmark': True, 'independent_metrics': measured}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = verify(args.directory)
    write_json(args.directory / 'verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
