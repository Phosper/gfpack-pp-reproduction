"""CPU verification of saved generation evidence; does not rerun GPU scores."""
import argparse
import json
import math
from pathlib import Path
import traceback

from run_inference import sha256, write_json
from run_generation_diagnosis import make_plan, CODE, CHECKPOINT_SHA, DATASET_SHA


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition,message):
    if not condition:
        raise ValueError(message)


def verify(run):
    import torch
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.affinity import rotate, translate
    from paper_sampling_noise import validate_noise_plan
    root=Path(run)
    plan=read(root/"plan.json")
    require(plan["candidates_per_run"] in (128,512),"Unexpected candidate denominator")
    expected=make_plan(plan["candidates_per_run"])
    require(plan["asset_sha256"]=={"checkpoint":CHECKPOINT_SHA,"dataset":DATASET_SHA},"Unexpected diagnostic assets")
    for key in expected:
        require(plan[key]==expected[key],"Plan protocol changed: "+key)
    for name in CODE:
        require(sha256(root/"code"/name)==plan["code_sha256"][name],"Snapshot code hash mismatch: "+name)
    def check_file(directory,meta,name):
        path=directory/name
        require(path.is_file() and sha256(path)==meta["output_sha256"].get(name),"Artifact hash mismatch: "+str(path))
        return path
    common={};checkpoint=None;dataset=None;geometry={};summaries=[]
    for item in expected["runs"]:
        directory=root/item["name"]
        m=read(directory/"metadata.json")
        require(m["status"]=="completed" and m["changed_inputs"]==[],"Incomplete/changed run: "+item["name"])
        require(m["sample_index"]==item["sample_index"] and m["planned_candidates"]==plan["candidates_per_run"],"Coverage mismatch")
        require(m["profile"]["profile"]==item["profile"] and m["model_training_flag"]==(item["model_mode"]=="train"),"Profile/model-mode mismatch")
        require(m["constraints_enabled"]==item["constraints"] and m["post_iterations"]==0 and m["geometry_backend"]=="legacy_native","Confounded generation probe")
        require(m["sampling_noise_seed"]==item["sampling_noise_seed"],"Noise seed label mismatch")
        checkpoint=checkpoint or m["checkpoint_sha256"]
        require(m["checkpoint_sha256"]==checkpoint==CHECKPOINT_SHA,"Mixed/unexpected checkpoints")
        inputs={p.replace("\\","/").rsplit("/",1)[-1]:h for p,h in m["input_sha256"].items()}
        dataset=dataset or inputs["dataset_dental_vali_128.pkl"]
        require(inputs["dataset_dental_vali_128.pkl"]==dataset==DATASET_SHA,"Mixed/unexpected datasets")
        for name in CODE:
            if name in inputs:
                require(inputs[name]==plan["code_sha256"][name],"Child code differs from snapshot: "+name)
        sample=read(check_file(directory,m,"input_sample.json"))
        gpath=check_file(directory,m,"input_geometry.json")
        ids=sample["polygon_ids"]
        input_signature=(ids,sample["mask"],sha256(gpath))
        index=item["sample_index"]
        geometry.setdefault(index,input_signature)
        require(geometry[index]==input_signature,"Mixed geometry/IDs/masks within sample")
        noise=torch.load(check_file(directory,m,"sampling_noise.pt"),map_location="cpu",weights_only=True)
        hashes=validate_noise_plan(noise,(plan["candidates_per_run"],len(ids),4))
        common.setdefault(index,hashes)
        require(common[index]==hashes,"Sampling noise is not paired across settings")
        sampler=read(check_file(directory,m,"sampler.json"))
        require(sampler["common_noise_sha256"]==hashes and sampler["status"]=="completed","Sampler does not bind saved noise")
        require(sampler["model_training_flag"]==(item["model_mode"]=="train"),"Sampler mode mismatch")
        steps=sampler["steps"]
        active=[s["step_1based"] for s in steps if s["constraint_active"]]
        require(len(steps)==128 and active==(list(range(97,129)) if item["constraints"] else []),"Wrong sampling/constraint schedule")
        callbacks=read(check_file(directory,m,"constraint_callbacks.json"))
        require(len(callbacks)==(32 if item["constraints"] else 0),"Wrong callback count")
        raw=torch.load(check_file(directory,m,"predictions_raw.pt"),map_location="cpu",weights_only=True)
        require(tuple(raw.shape)==(plan["candidates_per_run"],len(ids),4) and torch.isfinite(raw).all(),"Invalid raw tensor")
        trajectory=torch.load(check_file(directory,m,"trajectory.pt"),map_location="cpu",weights_only=True)
        require(tuple(trajectory.shape)==(len(raw),128,len(ids),4) and torch.equal(trajectory[:,-1],raw),"Trajectory/final state disagree")
        report=read(check_file(directory,m,"raw_metrics.json"))
        recorded=report["samples"][0]["candidates"]
        require(len(recorded)==len(raw) and [r["candidate_index"] for r in recorded]==list(range(len(raw))),"Incomplete candidate records")
        polygons=read(gpath)["official_max_contours"]
        rows=[]
        for candidate, saved in zip(raw.tolist(),recorded):
            shapes=[translate(rotate(Polygon(polygons[idx]),math.atan2(a[3],a[2]),origin=(0,0),use_radians=True),a[0],a[1]) for idx,a in zip(ids,candidate)]
            areas=[p.intersection(q).area for i,p in enumerate(shapes) for q in shapes[i+1:] if p.intersects(q)]
            bounds=[min(p.bounds[0] for p in shapes),min(p.bounds[1] for p in shapes),max(p.bounds[2] for p in shapes),max(p.bounds[3] for p in shapes)]
            ref=saved["reference"]
            declared_valid=(ref["max_pairwise_area"]<=1e-9 and ref["bounds_xyxy"][1]>=-1e-9 and ref["bounds_xyxy"][3]<=1200.+1e-9)
            require(saved["feasible"]["reference_strict_fixed_strip"]==declared_valid,"Strict feasibility flag was changed")
            values={"max_pairwise_area":max(areas,default=0.),"sum_pairwise_area":math.fsum(areas),"total_polygon_area":math.fsum(p.area for p in shapes)}
            for key,value in values.items():
                require(math.isclose(value,ref[key],rel_tol=1e-8,abs_tol=2e-6),"Reference geometry mismatch: "+key)
            require(np.allclose(bounds,ref["bounds_xyxy"],rtol=0,atol=2e-6),"Reference bounds mismatch")
            violation=max(0.,-bounds[1],bounds[3]-1200.)
            rows.append((values["sum_pairwise_area"]/values["total_polygon_area"],values["max_pairwise_area"],violation))
        # Use the saved reference's declared project tolerance flags after the
        # independent numerical comparison; tiny rounding near a threshold is
        # not silently reclassified with the geometric comparison tolerance.
        valid=sum(r["feasible"]["reference_strict_fixed_strip"] is True for r in recorded)
        require(valid==report["criteria"]["reference_strict_fixed_strip"]["feasible_candidates"],"Feasible-count summary mismatch")
        summaries.append({"run":item["name"],"sample_index":index,"candidate_count":len(raw),"strict_valid_candidates":valid,
            "mean_pairwise_overlap_percent":math.fsum(r[0] for r in rows)/len(rows)*100,
            "max_pairwise_area":max(r[1] for r in rows),"max_height_violation":max(r[2] for r in rows)})
    # Constraint-on/off must have identical histories before step97, not only
    # equal seeds. Model-mode comparisons intentionally change dropout.
    for index in (0,45):
        prefix="sample_%03d_"%index
        off=torch.load(root/(prefix+"paper_train_off")/"trajectory.pt",weights_only=True,map_location="cpu")
        on=torch.load(root/(prefix+"paper_train_native_on")/"trajectory.pt",weights_only=True,map_location="cpu")
        require(torch.equal(off[:,:96],on[:,:96]),"Constraint probe changed pre-constraint history")
    return {"status":"verified","paper_full_reproduction":False,"planned_runs":8,"verified_runs":8,
        "checkpoint_sha256":checkpoint,"dataset_sha256":dataset,"summaries":summaries,
        "scope":"Saved hashes, paired draws, schedule, prefix equality, and CPU geometry recomputation; GPU scores not rerun",
        "limits":["Two development samples are not the 128-instance benchmark.","Seed regeneration, training provenance, native metrics and GPU model scores are not independently verified."]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        result=verify(args.run)
    except Exception as exc:
        write_json(args.output,{"status":"failed","error":str(exc),"traceback":traceback.format_exc()})
        raise
    write_json(args.output,result)
    print(json.dumps({"status":"verified","runs":result["verified_runs"]}))


if __name__=="__main__":
    main()
