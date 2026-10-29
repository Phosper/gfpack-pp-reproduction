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
from paper_enhancement import field, enhance_chunked
from paper_metrics import sample_metrics, aggregate_metrics
from paper_selection import select_saved_result


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
    parser=argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--extension",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--profile",choices=PROFILES,default="paper_settings",
                        help="Sampling profile; default follows the paper's stated numerical settings")
    parser.add_argument("--sample-index",type=int,default=0)
    parser.add_argument("--candidates",type=int,default=512)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--no-constraints",action="store_true")
    parser.add_argument("--model-mode",choices=("train","eval"),default="train")
    parser.add_argument("--sampling-noise-seed",type=int,
                        help="Diagnostic only: save common Gaussian draws independent of model dropout")
    parser.add_argument("--post-iterations",type=int,default=100)
    parser.add_argument("--result-dtype",choices=("float64","fp32"),default="float64",
                        help="Primary saved output precision; both dtypes are independently evaluated")
    parser.add_argument("--geometry-backend",choices=("paper_sat","legacy_native"),
                        default="paper_sat",
                        help="Reference SAT/minimum-translation geometry, or unchanged historical native proxy")
    parser.add_argument("--post-step-size",type=float,default=1.,
                        help="Fixed post-gradient step; paper does not specify it. Non-unit requires paper_fixed_step")
    parser.add_argument("--post-policy",choices=("legacy_unit","paper_fixed_step","guarded_backtracking"),
                        default="legacy_unit",
                        help="Explicit numerical reconstruction policy; neither is an author-confirmed post solver")
    args=parser.parse_args()
    sys.dont_write_bytecode=True
    if args.candidates<1 or args.candidates>512 or (args.candidates not in (128,512) and not args.smoke):
        parser.error("Use 128 or 512 candidates, or explicitly mark a reduced run --smoke")
    if not 0<=args.post_iterations<=100 or args.sample_index<0:
        parser.error("Invalid post budget or sample index")
    if (not math.isfinite(args.post_step_size) or args.post_step_size<=0 or
            (args.post_policy!="paper_fixed_step" and args.post_step_size!=1.)):
        parser.error("Use a positive finite post step; non-unit steps require --post-policy paper_fixed_step")
    if args.sampling_noise_seed is not None and args.profile=="release_weights_constraints":
        parser.error("Common Gaussian draws require a paper-mechanism profile")
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
              "paper_sat_geometry.py","paper_selection.py",
              "paper_sampling_noise.py",
              "paper_guarded_enhancement.py","run_inference.py")]
    frozen={str(p):sha256(p) for p in inputs}
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();settings=profile_settings(args.profile)
    meta={"status":"running","started_utc":utc_now(),"profile":settings,"sample_index":args.sample_index,
          "planned_candidates":args.candidates,"smoke":args.smoke,"seed":3407,
          "constraints_enabled":not args.no_constraints,"post_iterations":args.post_iterations,
          "post_policy":args.post_policy,
          "post_step_size":args.post_step_size,
          "geometry_backend":args.geometry_backend,
          "primary_result_dtype":args.result_dtype,
          "model_training_flag":args.model_mode=="train","sampling_noise_seed":args.sampling_noise_seed,
          "checkpoint_sha256":sha256(checkpoint),"input_sha256":frozen,
          "paper_full_reproduction":False,"checkpoint_training_configuration_verified":False,
          "architecture_boundary_encoder_matches_paper":False,
          "paper_testset_identity_verified":False,
          "sampling_rng_policy":("saved common Gaussian draws separated from model RNG; not historical RNG replay"
             if args.sampling_noise_seed is not None else
             "seed3407 reset after model initialization; paired new-run comparison, not bitwise equality to historical baseline"),
          "rotation_scale_scope":"GT input encoding only; Gaussian prior covers all 4 components and generated directions are not reprojected; training scale provenance unverified",
          "model_mode":("release_default_train_dropout; held fixed across probes" if args.model_mode=="train" else
                        "eval_dropout_disabled; diagnostic choice, not author-confirmed mode"),
          "software":{"python":sys.version,"pytorch":str(torch.__version__),"cuda":torch.version.cuda},
          "geometry_assumptions":(["float64 GEOS intersection; convex SAT minimum displacement",
              "concave minimum translation via convex decomposition/configuration-space union; author implementation unconfirmed",
              "-x/-y binary search retains a checked feasible endpoint; no unconditional +0.001 shift"]
              if args.geometry_backend=="paper_sat" else
              ["native Clipper precision2 retained", "convex SAT MTV; concave edge-axis conservative proxy"])+[
              "only translation collision gradients; no documented rotation gradient",
              "height1200, no horizontal right wall",
              ("post fixed step then x/y gap; step size/order unspecified by paper" if args.post_policy!="guarded_backtracking" else
               "post same-field backtracking then guarded x/y gap; our numerical reconstruction, unspecified by paper")]}
    try:
        import shapely
        meta["software"].update(shapely=shapely.__version__,geos=shapely.geos_version_string)
        if args.geometry_backend == "paper_sat":
            from paper_sat_geometry import PaperSATGeometry
            geometry_backend = PaperSATGeometry(height=1200.)
        else:
            geometry_backend = collision_probe
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
        model.train(args.model_mode=="train");torch.cuda.reset_peak_memory_stats(0)
        noise_plan=None
        if args.sampling_noise_seed is not None:
            from paper_sampling_noise import make_noise_plan
            noise_plan=make_noise_plan(args.candidates,len(id_list),args.sampling_noise_seed)
            torch.save(noise_plan,out/"sampling_noise.pt")
        callback_log=[]
        def callback(a):
            r,_=field(geometry_backend,polygons,id_list,a.numpy(),1200.)
            callback_log.append({"callback_index":len(callback_log)+1,
                    "geometry_backend":args.geometry_backend,
                    "max_pair_area":max(r["max_pairwise_area"]),
                    **({"max_native_pair_area":max(r["max_pairwise_area"])}
                       if args.geometry_backend=="legacy_native" else
                       {"max_reference_pair_area":max(r["max_pairwise_area"])}),
                    "max_height_violation":max(r["height_violation"]),
                    "concave_mtv_pairs":sum(r.get("concave_mtv_pairs",[])),
                    "concave_proxy_pairs":sum(r["concave_proxy_pairs"])})
            return r["repulsion"]
        trajectory,raw,sampler=sample_paper(model,ids,features,mask,args.candidates,
                    None if args.no_constraints else callback,args.profile,seed=3407,noise_plan=noise_plan)
        raw=raw.detach().cpu();torch.save(raw,out/"predictions_raw.pt");torch.save(trajectory.detach().cpu(),out/"trajectory.pt")
        write_json(out/"sampler.json",sampler);write_json(out/"constraint_callbacks.json",callback_log)
        del trajectory
        # Only global x gauge normalization, using release quantized bbox (no y reset).
        raw_native=native_metrics(torch,evaluation,polygons,ids,raw)
        post_input=raw.numpy().astype(np.float64,copy=True)
        post_input[:,:,0]-=np.asarray(raw_native["bounding_boxes_xyxy"])[:,0,None]
        if args.post_iterations:
            final,post=enhance_chunked(geometry_backend,polygons,id_list,post_input,args.post_iterations,1200.,
                               policy=args.post_policy,trace_candidates=True,post_step_size=args.post_step_size)
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
        selected=select_saved_result(aggregates,args.result_dtype)
        write_json(out/"selected_result.json",selected)
        chosen=selected["candidate_index"]
        meta["selected_result"]=selected
        meta["preview_candidate_index"]=chosen
        preview_actions=final if args.result_dtype=="float64" else final32.tolist()
        if chosen is not None:
            plot(out/"enhanced_preview.png",polygons,id_list,preview_actions[chosen],
                 args.profile+" | candidate "+str(chosen)+" | saved "+args.result_dtype+": strict geometry passed")
        else:
            plot(out/"infeasible_diagnostic_candidate_0.png",polygons,id_list,preview_actions[0],
                 args.profile+" | diagnostic candidate 0 | NO FEASIBLE RESULT")
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
