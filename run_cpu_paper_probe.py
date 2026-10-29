#!/usr/bin/env python3
"""Small CPU-only forward/sampling probe with pinned official model and assets.

Unused native evaluation imports are bypassed by compiling seven unchanged
pure geometry/data helpers from the pinned evaluate.py. The neural network,
checkpoint tensors, source graph construction, and named sampler are retained.
No native calutil metric or GPU-equivalence claim is made.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
import types

from run_inference import checked_sample,sha256,write_json,utc_now
from paper_sampler import sample_paper,marginal_prob,profile_settings,PROFILES
from paper_enhancement import field,enhance_chunked
from paper_metrics import sample_metrics,aggregate_metrics
from paper_selection import select_saved_result

PINNED_SOURCE_LF={
    'score_model.py':'b58c75b2c1df71175e31b8a08dc19771aecfda38a3ef1dddd3c7ca1035ed9368',
    'gnn_feature.py':'89145ba747be55ebe5799a8676cd3088176d8c07954f8d87ca3f37a6e9a94344',
    'geometry_utils.py':'3ffa545088a8d98695baf730cf7f9df265599e2f45b073021806a6a58245c632',
    'evaluate.py':'0d6d2b85718372f4de8d0a4c06a7261f8cb5712477d4393c632aa378b2b8eaf8'}
HELPERS=('readAllPolys','genPaddingMask','padData','calculate_angle','compute_node_features','compute_global_features','create_gnn_data')


def pure_helpers(path,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in HELPERS]
    if {n.name for n in functions}!=set(HELPERS):raise ValueError('Missing pinned helper')
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return types.SimpleNamespace(**{name:namespace[name] for name in HELPERS})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sample-index',type=int,default=0)
    p.add_argument('--candidates',type=int,default=1,help='CPU diagnostic only; 1..8, not a paper benchmark')
    p.add_argument('--profile',choices=PROFILES,default='paper_settings')
    p.add_argument('--model-mode',choices=('train','eval'),default='train')
    p.add_argument('--constraints',choices=('on','off'),default='on')
    p.add_argument('--post-iterations',type=int,default=100)
    p.add_argument('--post-policy',choices=('legacy_unit','paper_fixed_step','paper_potential_descent'),default='legacy_unit')
    p.add_argument('--post-step-size',type=float,default=1.)
    p.add_argument('--sampling-noise-seed',type=int)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if not 1<=args.candidates<=8 or not 0<=args.post_iterations<=100 or args.sample_index<0 or args.threads<1:
        p.error('Invalid diagnostic dimensions/budget')
    if (not math.isfinite(args.post_step_size) or args.post_step_size<=0 or
        (args.post_policy=='legacy_unit' and args.post_step_size!=1.)):
        p.error('Invalid fixed/initial post step')
    root,upstream,out=args.root.resolve(),args.upstream.resolve(),args.output.resolve()
    if out.exists() or any(x==out or x in out.parents or out in x.parents for x in (root/'resources',upstream)):
        raise ValueError('Choose a new output outside protected inputs')
    checkpoint=root/'resources/checkpoints/score_model.pth'
    dataset=root/'resources/data/datasets/dataset_dental_vali_128.pkl'
    from run_generation_diagnosis import CHECKPOINT_SHA,DATASET_SHA
    if sha256(checkpoint)!=CHECKPOINT_SHA or sha256(dataset)!=DATASET_SHA:
        raise ValueError('Expected the audited official Dental assets')
    for name,expected in PINNED_SOURCE_LF.items():
        actual=hashlib.sha256((upstream/'src'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        if actual!=expected:raise ValueError('Pinned source mismatch: '+name)
    import numpy as np
    import torch
    import torch_geometric
    from torch_geometric.data import Data,Batch
    from shapely.geometry import Polygon
    sys.path.insert(0,str(upstream))
    from src.score_model import PolygonPackingTransformer
    from src.geometry_utils import Polygon as PolygonGeometry
    torch.set_num_threads(args.threads);torch.manual_seed(3407);np.random.seed(3407);random.seed(3407)
    helpers=pure_helpers(upstream/'src/evaluate.py',{'math':math,'torch':torch,'Data':Data,'Batch':Batch,
        'ShapelyPolygon':Polygon,'PolygonGeometry':PolygonGeometry,'POLYGON_DIR':root/'resources/data/polygons'})
    ids,original_gt,gt,mask,_=checked_sample(torch,helpers,dataset,args.sample_index)
    settings=profile_settings(args.profile);gt[:,2:]*=settings['rotation_feature_scale']/500.
    polygons=helpers.readAllPolys(440);id_list=ids.tolist()
    inputs=[checkpoint,dataset,*[upstream/'src'/n for n in PINNED_SOURCE_LF],
        *sorted((root/'resources/data/polygons').glob('*.txt'))]
    inputs += [Path(__file__).with_name(n) for n in ('run_cpu_paper_probe.py','run_inference.py','paper_sampler.py',
        'paper_sampling_noise.py','paper_enhancement.py','paper_sat_geometry.py','paper_potential_enhancement.py',
        'paper_metrics.py','paper_selection.py','run_paper_alignment.py','run_generation_diagnosis.py')]
    frozen={str(x.resolve()):sha256(x) for x in inputs}
    out.mkdir(parents=True);start=time.perf_counter()
    meta={'status':'running','started_utc':utc_now(),'sample_index':args.sample_index,'planned_candidates':args.candidates,
        'profile':settings,'model_training_flag':args.model_mode=='train','constraints_enabled':args.constraints=='on',
        'post_iterations':args.post_iterations,'post_policy':args.post_policy,'post_step_size':args.post_step_size,
        'geometry_backend':'paper_sat','device':'cpu','seed':3407,'sampling_noise_seed':args.sampling_noise_seed,
        'checkpoint_sha256':sha256(checkpoint),'input_sha256':frozen,'model_executed':False,'weights_loaded':False,
        'paper_full_reproduction':False,'gpu_bitwise_equivalence':False,'native_calutil_executed':False,
        'helper_policy':'seven unchanged pure functions AST-selected to bypass unused native imports',
        'post_input_gauge':'reference geometry min_x, no y reset; no native quantized bbox',
        'software':{'python':sys.version,'torch':str(torch.__version__),'torch_geometric':torch_geometric.__version__}}
    try:
        write_json(out/'input_sample.json',{'sample_index':args.sample_index,'polygon_ids':id_list,
            'gt_actions_original':original_gt.tolist(),'gt_actions_scaled':gt.tolist(),'mask':mask.tolist()})
        write_json(out/'input_geometry.json',{'official_max_contours':polygons,'coordinate_scale':10})
        features=helpers.create_gnn_data(polygons)
        model=PolygonPackingTransformer(marginal_prob_std_func=marginal_prob(args.profile),device=torch.device('cpu'))
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        meta['weights_loaded']=True
        def mark_forward(module,inputs):meta['model_executed']=True
        model.register_forward_pre_hook(mark_forward)
        model.train(args.model_mode=='train');meta['model_parameters']=sum(x.numel() for x in model.parameters())
        noise=None
        if args.sampling_noise_seed is not None:
            from paper_sampling_noise import make_noise_plan
            noise=make_noise_plan(args.candidates,len(ids),args.sampling_noise_seed);torch.save(noise,out/'sampling_noise.pt')
        from paper_sat_geometry import PaperSATGeometry
        native=PaperSATGeometry();callbacks=[]
        def callback(a):
            r,_=field(native,polygons,id_list,a.numpy())
            callbacks.append({'step':96+len(callbacks)+1,'max_pairwise_area':max(r['max_pairwise_area']),
                              'max_height_violation':max(r['height_violation'])})
            if len(callbacks)%8==0:print('constraint callback',len(callbacks),'of32',flush=True)
            return r['repulsion']
        t0=time.perf_counter()
        trajectory,raw,sampler=sample_paper(model,ids,features,mask,args.candidates,
            callback if args.constraints=='on' else None,args.profile,seed=3407,noise_plan=noise)
        meta['sampling_seconds']=time.perf_counter()-t0
        torch.save(raw,out/'predictions_raw.pt');torch.save(trajectory,out/'trajectory.pt')
        write_json(out/'sampler.json',sampler);write_json(out/'constraint_callbacks.json',callbacks)
        from run_paper_alignment import exact_all
        refs=exact_all(polygons,id_list,raw.tolist())
        post_input=raw.numpy().astype(np.float64,copy=True)
        post_input[:,:,0]-=np.array([r['bounds_xyxy'][0] for r in refs])[:,None]
        final,post=enhance_chunked(native,polygons,id_list,post_input,args.post_iterations,
            policy=args.post_policy,post_step_size=args.post_step_size,trace_candidates=True)
        write_json(out/'post.json',post)
        final64=torch.tensor(final,dtype=torch.float64);final32=final64.float()
        torch.save(final64,out/'predictions_enhanced_float64.pt');torch.save(final32,out/'predictions_enhanced_fp32.pt')
        reports={}
        for stage,a in (('raw',raw),('enhanced_float64',final64),('enhanced_fp32',final32)):
            r=exact_all(polygons,id_list,a.tolist())
            reports[stage]=aggregate_metrics([sample_metrics(args.sample_index,None,r,planned_candidate_count=args.candidates)])
            write_json(out/(stage+'_metrics.json'),reports[stage])
        selection=select_saved_result(reports);write_json(out/'selected_result.json',selection)
        meta['selected_result']=selection;meta['status']='completed'
    except Exception as exc:
        meta['status']='failed';meta['error']={'message':str(exc),'traceback':traceback.format_exc()}
        raise
    finally:
        meta['finished_utc']=utc_now();meta['elapsed_seconds']=time.perf_counter()-start
        meta['changed_inputs']=[name for name,digest in frozen.items() if sha256(Path(name))!=digest]
        if meta['changed_inputs']:meta['status']='failed'
        meta['output_sha256']={str(x.relative_to(out)):sha256(x) for x in out.rglob('*') if x.is_file()}
        write_json(out/'metadata.json',meta)
        print(json.dumps({k:meta.get(k) for k in ('status','device','model_executed','elapsed_seconds','changed_inputs')}),flush=True)
    if meta['status']!='completed':raise RuntimeError('Inputs changed')


if __name__=='__main__':main()
