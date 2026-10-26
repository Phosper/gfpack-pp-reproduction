"""Generate deterministic same-candidate legacy repair starts, without inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from freeze_engineering_transfer import validate_transfer_frozen, checked_transfer_case
    args.frozen = args.frozen.resolve()
    args.extension = args.extension.resolve()
    manifest, manifest_sha = validate_transfer_frozen(args.frozen)
    build = read(args.extension/'build.json')
    if (build.get('extension_sha256') != 'd7a28f5a7f3a04f90a983dd9473306903b70e5f2f32c2674f585837617859957'
            or build.get('repair_header_sha256') != 'b422342b149f48c3b8c1044b99e722bbab8c607eb790cec945a54f6862de67ca'
            or build.get('strip_repair_entry_point') != 'collision_probe.overlap_repair_strip'
            or build.get('returncode') != 0 or not build.get('source_unchanged')):
        raise ValueError('Requires the verified round7 strip repair extension')
    sys.path.insert(0, str(args.extension))
    import collision_probe
    extension = Path(collision_probe.__file__).resolve()
    if sha(extension) != build['extension_sha256']:
        raise ValueError('Native extension differs from its recorded build')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'jobs').mkdir()
    (args.output/'source_snapshot').mkdir()
    files = {'scripts/prepare_transfer_seeds.py': Path(__file__),
             'scripts/freeze_engineering_transfer.py': Path(__file__).with_name('freeze_engineering_transfer.py'),
             'probe/build.json': args.extension/'build.json',
             'probe/'+extension.name: extension,
             'frozen/manifest.json': args.frozen/'manifest.json'}
    for entry in manifest['cases']:
        path, _ = checked_transfer_case(args.frozen, entry)
        files['frozen/'+entry['path']] = path
    inputs = {key: sha(path) for key, path in files.items()}
    portable = {}
    for number, (key, path) in enumerate(files.items()):
        relative = 'source_snapshot/%03d_%s' % (number, path.name)
        shutil.copyfile(path, args.output/relative)
        portable[key] = relative
    save(args.output/'inputs.json', inputs)
    save(args.output/'plan.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'planned_inputs': 256, 'height': 1200., 'epsilon': 1.,
        'maximum_iterations': 2000, 'sampling': False, 'training': False,
        'manifest_sha256': manifest_sha, 'extension_sha256': sha(extension),
        'same_candidate_only': True,
        'native_internal_limits': {'seconds': 8., 'evaluations': 100000},
        'timing': 'Per-input native repair wall time, excluding JSON I/O; separate from later solver time'})
    started = time.perf_counter()
    for entry in manifest['cases']:
        _, case = checked_transfer_case(args.frozen, entry)
        for source, field in [('pred', 'raw_actions'), ('gt', 'gt_actions')]:
            original = case[field]
            tick = time.perf_counter()
            result = collision_probe.overlap_repair_strip(
                case['polygon_ids'], [math.atan2(a[3], a[2]) for a in original],
                [a[:2] for a in original], case['polygons'], 1200., 1., 2000)
            elapsed = time.perf_counter()-tick
            seed = [list(xy)+a[2:] for xy, a in zip(result['actions_xy'], original)]
            job = {'sample_index': entry['sample_index'], 'source': source,
                   'candidate_index': case['candidate_index'], 'case_sha256': entry['sha256'],
                   'input_actions': original, 'repair_actions': seed,
                   'native_result': result, 'elapsed_seconds': elapsed,
                   'extension_sha256': sha(extension)}
            name = 'jobs/sample_%03d_%s.json' % (entry['sample_index'], source)
            save(args.output/name, job)
            print(json.dumps({'sample': entry['sample_index'], 'source': source,
                              'seconds': elapsed, 'native_status': result.get('status')}), flush=True)
    changed = [key for key, path in files.items() if sha(path) != inputs[key]]
    save(args.output/'metadata.json', {
        'status': 'completed' if not changed else 'input_changed',
        'changed_inputs': changed, 'input_count': len(inputs),
        'elapsed_seconds': time.perf_counter()-started,
        'source_paths': {key: str(path) for key, path in files.items()},
        'portable_source_paths': portable,
        'output_sha256': {p.relative_to(args.output).as_posix(): sha(p)
                          for p in sorted(args.output.rglob('*')) if p.is_file()}})
    if changed:
        raise RuntimeError('Frozen source changed during seed preparation')


if __name__ == '__main__':
    main()
