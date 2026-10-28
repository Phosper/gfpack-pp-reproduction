#!/usr/bin/env python3
"""Run isolated, explicitly incomplete paper-mechanism/checkpoint probes.

128 EM updates; last 32 receive Eq9 repulsion; at most 100 post iterations.
The released model/checkpoint remain unchanged. No training or hidden clipping.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from run_inference import checked_sample, native_metrics, sha256, utc_now, write_json, command_snapshot
from paper_sampler import PROFILES, sample_paper, profile_settings, marginal_prob
from paper_enhancement import field, enhance
from paper_metrics import sample_metrics, aggregate_metrics


def exact_all(polygons, ids, candidates):
    from shapely.geometry import Polygon
    rows=[]
    for actions in candidates:
        shapes=[]
        for idx,a in zip(ids,actions):
            theta=math.atan2(a[3],a[2]);c,s=math.cos(theta),math.sin(theta)
            p=Polygon([(x*c-y*s+a[0],x*s+y*c+a[1]) for x,y in polygons[idx]])
            if not p.is_valid or p.area<=0:
                raise ValueError("Invalid transformed contour")
            shapes.append(p)
        areas=[]
        for i,p in enumerate(shapes):
            for q in shapes[i+1:]:
                if p.bounds[2]<q.bounds[0] or q.bounds[2]<p.bounds[0] or p.bounds[3]<q.bounds[1] or q.bounds[3]<p.bounds[1]:
                    continue
                areas.append(p.intersection(q).area)
        rows.append({"status":"measured","max_pairwise_area":max(areas,default=0.),
                     "sum_pairwise_area":sum(areas),"total_polygon_area":sum(p.area for p in shapes),
                     "bounds_xyxy":[min(p.bounds[0] for p in shapes),min(p.bounds[1] for p in shapes),
                                    max(p.bounds[2] for p in shapes),max(p.bounds[3] for p in shapes)]})
    return rows


def plot(path,polygons,ids,actions,title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,4))
    for idx,a in zip(ids,actions):
        theta=math.atan2(a[3],a[2]);c,s=math.cos(theta),math.sin(theta)
        p=[(x*c-y*s+a[0],x*s+y*c+a[1]) for x,y in polygons[idx]]
        p.append(p[0]);ax.plot(*zip(*p),linewidth=.7,color="#326784")
    ax.axhline(0,color="#777777",linestyle="--");ax.axhline(1200,color="#777777",linestyle="--")
    ax.set_aspect("equal");ax.set_title(title);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--extension",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--profile",choices=PROFILES,default="paper_mechanism_release_scale")
    parser.add_argument("--sample-index",type=int,default=0)
    parser.add_argument("--candidates",type=int,default=512)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--no-constraints",action="store_true")
    parser.add_argument("--post-iterations",type=int,default=100)
    parser.add_argument("--post-policy",choices=("legacy_unit","guarded_backtracking"),
                        default="legacy_unit",
                        help="Explicit numerical reconstruction policy; neither is an author-confirmed post solver")
    args=parser.parse_args()
    sys.dont_write_bytecode=True
    if args.candidates<1 or args.candidates>512 or (args.candidates!=512 and not args.smoke):
        parser.error("Use 512 candidates, or explicitly mark a reduced run --smoke")
    if not 0<=args.post_iterations<=100 or args.sample_index<0:
        parser.error("Invalid post budget or sample index")
    root=args.root.resolve();extension=args.extension.resolve();out=args.output.resolve()
    upstream=root/"upstream"
    if out.exists() or upstream==out or upstream in out.parents or out in root.parents:
        raise ValueError("Output must be new and outside the upstream source tree")
    if command_snapshot(["git","rev-parse","HEAD"],upstream).get("stdout")!="ff75815e24ec50cd7fa8b2abdec1acfe0b55be95" or command_snapshot(["git","status","--porcelain"],upstream).get("stdout"):
        raise ValueError("Expected the clean pinned upstream")
    build=json.loads((extension/"build.json").read_text())
    if not build.get("paper_constraints") or not build.get("source_unchanged"):
        raise ValueError("Build lacks the paper constraint entrypoint")
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for model probes")
    sys.path.insert(0,str(upstream));sys.path.insert(0,str(extension))
    import collision_probe
    native_path=Path(collision_probe.__file__).resolve()
    if native_path.parent!=extension or sha256(native_path)!=build["extension_sha256"]:
        raise ValueError("Loaded native extension does not match the build")
    import src
    src.__path__.insert(0,str(root/"extensions"))
    from src import evaluate as evaluation
    from src.score_model import PolygonPackingTransformer
    evaluation.POLYGON_DIR=root/"resources/data/polygons"
    dataset=root/"resources/data/datasets/dataset_dental_vali_128.pkl"
    checkpoint=root/"resources/checkpoints/score_model.pth"
    inputs=[dataset,checkpoint,native_path,extension/"build.json"]
    inputs+=sorted((upstream/"src").glob("*.py"))
    inputs+=sorted((root/"extensions").glob("*.so"))
    inputs+=sorted((root/"resources/data/polygons").glob("*.txt"))
    inputs+=[Path(__file__).resolve().with_name(name) for name in
             ("run_paper_alignment.py","paper_sampler.py","paper_metrics.py","paper_enhancement.py",
              "paper_guarded_enhancement.py","run_inference.py")]
    frozen={str(p):sha256(p) for p in inputs}
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();settings=profile_settings(args.profile)
    meta={"status":"running","started_utc":utc_now(),"profile":settings,"sample_index":args.sample_index,
          "planned_candidates":args.candidates,"smoke":args.smoke,"seed":3407,
          "constraints_enabled":not args.no_constraints,"post_iterations":args.post_iterations,
          "post_policy":args.post_policy,
          "checkpoint_sha256":sha256(checkpoint),"input_sha256":frozen,
          "paper_full_reproduction":False,"checkpoint_training_configuration_verified":False,
          "architecture_boundary_encoder_matches_paper":False,
          "paper_testset_identity_verified":False,
          "sampling_rng_policy":"seed3407 reset after model initialization; paired new-run comparison, not bitwise equality to historical baseline",
          "rotation_scale_scope":"GT input encoding only; Gaussian prior covers all 4 components and generated directions are not reprojected; training scale provenance unverified",
          "model_mode":"release_default_train_dropout; held fixed across probes",
          "software":{"python":sys.version,"pytorch":str(torch.__version__),"cuda":torch.version.cuda},
          "geometry_assumptions":["native Clipper precision2 retained", "convex SAT MTV; concave edge-axis conservative proxy",
              "only translation collision gradients; no documented rotation gradient",
              "height1200, no horizontal right wall",
              ("post unit step then x/y gap; unspecified by paper" if args.post_policy=="legacy_unit" else
               "post same-field backtracking then guarded x/y gap; our numerical reconstruction, unspecified by paper")]}
    try:
        torch.set_num_threads(16);evaluation.setRandomSeed(3407);np.random.seed(3407)
        torch.backends.cudnn.benchmark=False
        ids,original_gt,gt,mask,count=checked_sample(torch,evaluation,dataset,args.sample_index)
        gt[:,2:]*=settings["rotation_feature_scale"]/500.
        polygons=evaluation.readAllPolys(440);id_list=ids.tolist()
        write_json(out/"input_sample.json",{"sample_index":args.sample_index,"polygon_ids":id_list,
                   "gt_actions_original":original_gt.tolist(),"gt_actions_scaled":gt.tolist(),"mask":mask.tolist()})
        write_json(out/"input_geometry.json",{"official_max_contours":polygons,"coordinate_scale":10})
        device=torch.device("cuda",0);features=evaluation.create_gnn_data(polygons).to(device)
        model=PolygonPackingTransformer(marginal_prob_std_func=marginal_prob(args.profile),device=device).to(device)
        state=torch.load(checkpoint,map_location="cpu",weights_only=True);model.load_state_dict(state,strict=True)
        meta["state_dict_elements"]=sum(v.numel() for v in state.values());del state
        meta["model_parameters"]=sum(p.numel() for p in model.parameters())
        meta["model_trainable_parameters"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.train();torch.cuda.reset_peak_memory_stats(0)
        callback_log=[]
        def callback(a):
            r,_=field(collision_probe,polygons,id_list,a.numpy(),1200.)
            callback_log.append({"callback_index":len(callback_log)+1,
                    "max_native_pair_area":max(r["max_pairwise_area"]),
                    "max_height_violation":max(r["height_violation"]),
                    "concave_proxy_pairs":sum(r["concave_proxy_pairs"])})
            return r["repulsion"]
        trajectory,raw,sampler=sample_paper(model,ids,features,mask,args.candidates,
                    None if args.no_constraints else callback,args.profile,seed=3407)
        raw=raw.detach().cpu();torch.save(raw,out/"predictions_raw.pt");torch.save(trajectory.detach().cpu(),out/"trajectory.pt")
        write_json(out/"sampler.json",sampler);write_json(out/"constraint_callbacks.json",callback_log)
        del trajectory
        # Only global x gauge normalization, using release quantized bbox (no y reset).
        raw_native=native_metrics(torch,evaluation,polygons,ids,raw)
        post_input=raw.numpy().astype(np.float64,copy=True)
        post_input[:,:,0]-=np.asarray(raw_native["bounding_boxes_xyxy"])[:,0,None]
        if args.post_iterations:
            final,post=enhance(collision_probe,polygons,id_list,post_input,args.post_iterations,1200.,
                               policy=args.post_policy)
        else:
            final=post_input;post={"iterations_executed":0,"reason":"post disabled","requested_policy":args.post_policy}
        torch.save(torch.tensor(final,dtype=torch.float64),out/"predictions_enhanced_float64.pt")
        # Persist and independently evaluate the FP32 storage conversion as well.
        final32=torch.tensor(final,dtype=torch.float32)
        torch.save(final32,out/"predictions_enhanced_fp32.pt");write_json(out/"post.json",post)
        aggregates={}
        for label,actions in (("raw",raw),("enhanced_float64",torch.tensor(final,dtype=torch.float64)),("enhanced_fp32",final32)):
            native=native_metrics(torch,evaluation,polygons,ids,actions)
            refs=exact_all(polygons,id_list,actions.tolist())
            sample=sample_metrics(args.sample_index,native,refs,planned_candidate_count=args.candidates)
            aggregates[label]=aggregate_metrics([sample]);write_json(out/(label+"_metrics.json"),aggregates[label])
        write_json(out/"gt_geometry.json",exact_all(polygons,id_list,[gt.tolist()]))
        criterion=aggregates["enhanced_float64"]["criteria"]["reference_eps50_fixed_strip"]
        chosen=criterion["sample_selections"][0]["best"]["reference_strip_utilization"]["candidate_index"]
        meta["preview_candidate_index"]=chosen
        preview=chosen if chosen is not None else 0
        plot(out/"enhanced_preview.png",polygons,id_list,final[preview],
             args.profile+" | candidate "+str(preview)+( " | feasible by local eps50+strip" if chosen is not None else " | fallback: NOT feasible"))
        meta["metric_summary"]={stage:{key:{k:v for k,v in item.items() if k not in ("sample_selections","utilization")}
                  for key,item in report["criteria"].items()} for stage,report in aggregates.items()}
        meta["status"]="completed"
    except Exception as exc:
        meta["status"]="failed";meta["error"]={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        if hasattr(exc,"diagnostics"):write_json(out/"sampler_failure.json",exc.diagnostics)
        write_json(out/"failed_metrics.json",aggregate_metrics([sample_metrics(
            args.sample_index,None,planned_candidate_count=args.candidates,status="failed",
            error={"type":type(exc).__name__,"message":str(exc)})]))
        raise
    finally:
        meta["finished_utc"]=utc_now();meta["elapsed_seconds"]=time.perf_counter()-started
        meta["changed_inputs"]=[p for p,h in frozen.items() if sha256(p)!=h]
        meta["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(0)
        meta["output_sha256"]={str(p.relative_to(out)):sha256(p) for p in out.rglob("*") if p.is_file()}
        write_json(out/"metadata.json",meta)
        print(json.dumps({k:meta.get(k) for k in ("status","profile","sample_index","planned_candidates","elapsed_seconds","changed_inputs")}))


if __name__=="__main__":
    main()
