"""Finish interrupted postprocessing without generating new neural candidates."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .assets import CHECKPOINT_SHA256, ROOT, sha256, write_json
from .verify import verify


def resume(directory, workers=1):
    directory = Path(directory).resolve()
    read = lambda name: json.loads((directory/name).read_text(encoding='utf-8'))
    generation = read('generation_metadata.json')
    if (generation['status'] != 'sampling_completed' or not generation.get('model_executed')
        or generation.get('replay_only') or not generation.get('model_runtime_parameters_unchanged')):
        raise ValueError('A verified completed neural generation is required; replay-only inputs are insufficient')
    if generation.get('model_weights_changed'):
        raise ValueError('Cannot resume a run with changed weights')
    if any(sha256(ROOT/name) != digest for name,digest in generation['input_sha256'].items()):
        raise ValueError('Generation source/resources changed; use the original checkout to resume')
    if sha256(ROOT/'checkpoints/dental.pth') != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint changed')
    status = directory/'metadata.json'
    if status.exists() and read('metadata.json')['status']=='completed':
        return verify(directory)
    binding = read('postprocess_binding.json')
    if binding['candidate_indices'] != list(range(generation['candidates'])):
        raise ValueError('A complete run must postprocess every generated candidate')
    started = time.perf_counter()
    subprocess.run([sys.executable,'-m','reproduce.postprocess','--directory',str(directory),
                    '--post-policy',binding['post_policy'],'--workers',str(workers),
                    '--relaxation',str(binding['relaxation']),'--post-step',str(binding['post_step']),
                    '--resume'],cwd=ROOT,check=True)
    if status.exists():
        shutil.copyfile(status,directory/f'metadata_before_resume_{time.time_ns()}.json')
        status.unlink()
    metadata = {**generation,'status':'completed',
                'input_unchanged':all(sha256(ROOT/name)==digest for name,digest in generation['input_sha256'].items()),
                'model_weights_changed':sha256(ROOT/'checkpoints/dental.pth')!=CHECKPOINT_SHA256,
                'resumed_postprocessing':True,
                'post_workers':workers,'resume_seconds':time.perf_counter()-started}
    metadata['output_sha256']={p.name:sha256(p) for p in directory.iterdir()
                              if p.is_file() and p.name not in ('initial_metadata.json','verification.json')
                              and not p.name.startswith('metadata_before_resume_')}
    write_json(status,metadata)
    report=verify(directory)
    if not (directory/'verification.json').exists():
        write_json(directory/'verification.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    report=resume(args.directory,args.workers)
    print(json.dumps({k:v for k,v in report.items() if k!='independent_metrics'},indent=2))


if __name__=='__main__':main()
