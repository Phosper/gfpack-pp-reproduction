#!/usr/bin/env python3
"""Three routes on 16 frozen raw representatives plus original GT, CPU only."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from check_collision_saved_cases import validate_frozen, checked_case, checked_build
from postprocess_metrics import geometry, compare, summarize

PATHS = ('gap_only', 'repair_only', 'repair_then_gap')
SOURCES = {'pred': 'raw_actions', 'gt': 'gt_actions'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value, exclusive=True):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def name(index, source, route):
    return 'sample_%03d_%s_%s' % (index, source, route)


def native_metrics(module, case, actions):
    import numpy as np
    ids = case['polygon_ids']
    angles = [math.atan2(a[3], a[2]) for a in actions]
    row = module.cal_util_all([ids], [angles], [[a[:2] for a in actions]], case['polygons'], 50.)[0]
    if len(row) != 7 or not all(math.isfinite(v) for v in row):
        raise ValueError('Invalid native metric output')
    v = np.asarray(row, dtype=np.float32)
    area = (v[5]-v[3])*(v[6]-v[4])
    return {'raw_native_vector': list(row), 'threshold_area': 50.,
            'official_valid': bool(v[1] < 50.),
            'official_fp32_bbox_utilization': float(v[0]/area),
            'official_fp32_thresholded_overlap_percent': float(v[2]/v[0]*100),
            'scope': 'Unmodified calutil, same float32 metric conversion as official evaluator; overlap only'}


def worker(a):
    import numpy as np
    manifest, _ = validate_frozen(a.frozen)
    entry = next(e for e in manifest['cases'] if e['sample_index'] == a.sample_index)
    _, case = checked_case(a.frozen, entry)
    build, _ = checked_build(a.extension)
    sys.path.insert(0, str(a.extension));import collision_probe
    sys.path.insert(0, str(a.native_metrics));import calutil
    if sha(collision_probe.__file__) != build['extension_sha256']:
        raise ValueError('Wrong repair extension')
    original = copy.deepcopy(case[SOURCES[a.source]])
    job = {'sample_index': a.sample_index, 'candidate_index': case['candidate_index'],
           'baseline_group': 'failure' if case['group']=='failure_development' else 'success',
           'source': a.source, 'path': a.route, 'status': 'completed',
           'case_sha256': entry['sha256'], 'input_field': SOURCES[a.source],
           'input_actions': original, 'before': geometry(case['polygons'], case['polygon_ids'], original),
           'before_native': native_metrics(calutil, case, original),
           'repair_native': None, 'repair_actions': None, 'gap_input_actions': None,
           'gap_passes': [], 'stages': {}, 'extension_sha256': sha(collision_probe.__file__),
           'calutil_sha256': sha(calutil.__file__)}
    actions = copy.deepcopy(original);started = time.perf_counter()
    if a.route != 'gap_only':
        r = collision_probe.overlap_repair_strip(case['polygon_ids'],
            [math.atan2(x[3], x[2]) for x in actions], [x[:2] for x in actions],
            case['polygons'], 1200., 1., 2000)
        actions = [list(xy)+x[2:] for x, xy in zip(actions, r['actions_xy'])]
        job['repair_native'] = r;job['repair_actions'] = copy.deepcopy(actions)
    if a.route != 'repair_only':
        job['gap_input_actions'] = copy.deepcopy(actions)
        xy = collision_probe.rm_spacing_all([case['polygon_ids']],
            [[math.atan2(x[3], x[2]) for x in actions]], [[x[:2] for x in actions]],
            case['polygons'], 8000., 1200., 1.)[0]
        actions = [list(v)+x[2:] for x, v in zip(actions, xy)]
        job['gap_passes'] = ['x', 'y', 'x', 'y', 'x']
    for precision, output in [('float64', actions),
                              ('float32', np.asarray(actions, dtype=np.float32).astype(np.float64).tolist())]:
        after = geometry(case['polygons'], case['polygon_ids'], output)
        job['stages'][precision] = {'actions': output, 'after': after,
            'comparison': compare(job['before'], after, original, output),
            'native': native_metrics(calutil, case, output)}
    job['elapsed_seconds'] = time.perf_counter()-started
    save(a.output, job)


def aggregate(rows):
    reports = {}
    for precision in ('float64','float32'):
        items=[]
        for job in rows:
            stage=job.get('stages',{}).get(precision)
            items.append({k:job[k] for k in ('sample_index','source','baseline_group','path','status')} |
                         {'after':stage['after'] if stage else None,
                          'comparison':stage['comparison'] if stage else None,
                          'output_precision':precision})
        reports[precision]=summarize(items)
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--extension',type=Path,required=True)
    p.add_argument('--native-metrics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--sample-index',type=int)
    p.add_argument('--source',choices=SOURCES)
    p.add_argument('--route',choices=PATHS)
    a=p.parse_args()
    for key in ('frozen','extension','native_metrics','output'):
        setattr(a,key,getattr(a,key).resolve())
    if a.worker:
        worker(a);return
    manifest,msha=validate_frozen(a.frozen);build,bsha=checked_build(a.extension)
    if build.get('strip_repair_entry_point')!='collision_probe.overlap_repair_strip':
        raise ValueError('Requires the explicitly unbounded-horizontal strip entrypoint')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'jobs').mkdir();(a.output/'logs').mkdir()
    inputs={'frozen/manifest.json':(a.frozen/'manifest.json',msha),
            'build/build.json':(a.extension/'build.json',bsha)}
    for entry in manifest['cases']:
        path,_=checked_case(a.frozen,entry);inputs['frozen/'+entry['path']]=(path,entry['sha256'])
    for path in sorted(a.extension.rglob('*')):
        if path.is_file():inputs['build/'+path.relative_to(a.extension).as_posix()]=(path,sha(path))
    for f in ('run_postprocess_ablation.py','postprocess_metrics.py','check_collision_saved_cases.py'):
        path=Path(__file__).with_name(f);inputs['scripts/'+f]=(path,sha(path))
    for path in sorted(a.native_metrics.glob('calutil*.so')):
        inputs['native_metrics/'+path.name]=(path,sha(path))
    save(a.output/'frozen_inputs.json',{key:sha for key,(_,sha) in inputs.items()})
    plan={'samples':[e['sample_index'] for e in manifest['cases']], 'sources':SOURCES,
          'paths':list(PATHS),'planned_jobs':96,'outputs_per_job':['float64','float32'],
          'representatives_per_sample':1,'gt_controls':16,'height':1200.,'horizontal_walls':None,
          'repair_epsilon':1.,'repair_max_iterations':2000,'worker_timeout_seconds':15.,
          'input_policy':'Frozen raw representative and original GT, no normalization, clamping or reselection',
          'gap_policy':'Unmodified official rm_spacing_all, x/y/x/y/x; numeric width8000 unused by gap code',
          'rollback_policy':'Repair internal failure rollback retained; no external guard rollback of gap outputs',
          'candidate_selection_caveat':'Representative indices previously selected using enhanced outputs; this is paired development analysis, not an unbiased raw selection or a 512-candidate evaluation',
          'training':False,'new_sampling':False,'manifest_sha256':msha}
    save(a.output/'plan.json',plan)
    rows=[];started=time.perf_counter()
    for entry in manifest['cases']:
        for source in SOURCES:
            for route in PATHS:
                stem=name(entry['sample_index'],source,route);file=a.output/'jobs'/(stem+'.json')
                args=[sys.executable,str(Path(__file__).resolve()),'--worker','--frozen',str(a.frozen),
                      '--extension',str(a.extension),'--native-metrics',str(a.native_metrics),'--output',str(file),
                      '--sample-index',str(entry['sample_index']),'--source',source,'--route',route]
                with (a.output/'logs'/(stem+'.log')).open('w') as log:
                    try:
                        proc=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=15.,
                              env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
                        if proc.returncode:raise RuntimeError('Worker exit '+str(proc.returncode))
                        job=read(file)
                    except (subprocess.TimeoutExpired,RuntimeError) as exc:
                        job={'sample_index':entry['sample_index'],'candidate_index':entry['candidate_index'],
                             'baseline_group':'failure' if entry['group']=='failure_development' else 'success',
                             'source':source,'path':route,'status':'failed','stages':{},
                             'error':type(exc).__name__,'case_sha256':entry['sha256']}
                        save(file,job)
                rows.append(job)
                summary={'status':'running','planned_jobs':96,'completed_jobs':len(rows),
                         'failed_jobs':sum(j['status']=='failed' for j in rows),'rows':rows,
                         'elapsed_seconds':time.perf_counter()-started}
                save(a.output/'summary.json',summary,False)
        print(json.dumps({'sample_index':entry['sample_index'],'jobs_done':len(rows),
                          'failed_jobs':sum(j['status']=='failed' for j in rows)}),flush=True)
    summary.update(status='completed',aggregates=aggregate(rows))
    save(a.output/'summary.json',summary,False)
    changed=[k for k,(path,h) in inputs.items() if sha(path)!=h]
    meta={'status':'completed' if not changed else 'input_integrity_failed','changed_inputs':changed,
          'frozen_input_count':len(inputs),'elapsed_seconds':time.perf_counter()-started,
          'output_sha256':{f.relative_to(a.output).as_posix():sha(f) for f in sorted(a.output.rglob('*')) if f.is_file()}}
    save(a.output/'metadata.json',meta)
    if changed:raise RuntimeError('Frozen inputs changed')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','aggregates')}),flush=True)


if __name__=='__main__':
    main()
