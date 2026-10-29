#!/usr/bin/env python3
"""One-command GPU generation diagnosis: two inputs, four paired settings, no post solver."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from run_inference import sha256, utc_now, write_json

CELLS = (
    ("paper_train_off", "paper_settings", "train", False),
    ("paper_eval_off", "paper_settings", "eval", False),
    ("sigma1_train_off", "paper_mechanism_release_scale", "train", False),
    ("paper_train_native_on", "paper_settings", "train", True),
)
CODE = ("run_generation_diagnosis.py", "verify_generation_diagnosis.py", "run_paper_alignment.py",
    "run_inference.py", "paper_sampler.py", "paper_sampling_noise.py", "paper_metrics.py",
    "paper_selection.py", "paper_enhancement.py", "paper_guarded_enhancement.py", "paper_sat_geometry.py")
CHECKPOINT_SHA="e3585572100e61c5f3112a8925e7add420b70c8c7a8f23e29be97788ad3c5643"
DATASET_SHA="dffdd191803ddfeb7701d9012b1f38dd88f01ee9333747a19ba86e18870848e0"


def make_plan(candidates=128):
    return {"schema_version":1,"sample_indices":[0,45],"candidates_per_run":candidates,
        "post_iterations":0,"geometry_backend":"legacy_native","paper_full_reproduction":False,
        "scope":"generation ablation only; native constraints reproduce the historical field, not the corrected nonconvex solver",
        "noise_policy":"same saved CPU Gaussian tensors within each sample; dropout cannot consume the sampling noise RNG",
        "cells":[{"name":name,"profile":profile,"model_mode":mode,"constraints":constraints}
                 for name,profile,mode,constraints in CELLS],
        "runs":[{"name":"sample_%03d_%s"%(index,name),"sample_index":index,"cell":name,
                 "profile":profile,"model_mode":mode,"constraints":constraints,"sampling_noise_seed":3407+index}
                for index in (0,45) for name,profile,mode,constraints in CELLS]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--extension",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--candidates",choices=(128,512),type=int,default=128)
    parser.add_argument("--timeout-seconds",type=int,default=900)
    args=parser.parse_args()
    root,extension,out=args.root.resolve(),args.extension.resolve(),args.output.resolve()
    archive=Path(str(out)+".zip")
    if out.exists() or archive.exists() or args.timeout_seconds<1:
        parser.error("Use a new output/ZIP path and a positive timeout")
    protected=[root/"upstream",root/"resources",root/"extensions",extension]
    if any(p==out or p in out.parents or out in p.parents for p in protected):
        parser.error("Output must be outside protected input directories")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required. Run this on the teammate's GPU machine.")
    for path in (root/"resources/checkpoints/score_model.pth",
                 root/"resources/data/datasets/dataset_dental_vali_128.pkl",extension/"build.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path,expected_hash in ((root/"resources/checkpoints/score_model.pth",CHECKPOINT_SHA),
                               (root/"resources/data/datasets/dataset_dental_vali_128.pkl",DATASET_SHA)):
        if sha256(path)!=expected_hash:
            raise ValueError("This diagnostic expects the audited official asset: "+str(path))
    for command,expected_text in ((["git","rev-parse","HEAD"],"ff75815e24ec50cd7fa8b2abdec1acfe0b55be95"),
                                  (["git","status","--porcelain"],"")):
        if subprocess.check_output(command,cwd=root/"upstream",text=True).strip()!=expected_text:
            raise ValueError("Expected clean pinned upstream")
    out.mkdir(parents=True)
    code=out/"code";code.mkdir()
    here=Path(__file__).resolve().parent
    for name in CODE:
        shutil.copyfile(here/name,code/name)
    plan=make_plan(args.candidates)
    plan["code_sha256"]={name:sha256(code/name) for name in CODE}
    plan["created_utc"]=utc_now()
    plan["asset_sha256"]={"checkpoint":CHECKPOINT_SHA,"dataset":DATASET_SHA}
    write_json(out/"plan.json",plan)
    rows=[]
    for run in plan["runs"]:
        command=[sys.executable,"-u",str(code/"run_paper_alignment.py"),"--root",str(root),
                 "--extension",str(extension),"--output",str(out/run["name"]),
                 "--sample-index",str(run["sample_index"]),"--profile",run["profile"],
                 "--candidates",str(args.candidates),"--model-mode",run["model_mode"],
                 "--sampling-noise-seed",str(run["sampling_noise_seed"]),
                 "--geometry-backend","legacy_native","--post-iterations","0"]
        if not run["constraints"]:
            command.append("--no-constraints")
        start=time.perf_counter()
        row={"name":run["name"],"command":command,"status":"failed"}
        try:
            with (out/(run["name"]+".log")).open("w",encoding="utf-8") as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=args.timeout_seconds)
            row.update(returncode=result.returncode,status="completed" if result.returncode==0 else "failed")
        except subprocess.TimeoutExpired:
            row["error"]="per-run timeout; planned candidates remain in the diagnostic denominator"
        row["elapsed_seconds"]=time.perf_counter()-start
        rows.append(row)
        state={"planned_runs":len(plan["runs"]),"runs":rows,"paper_full_reproduction":False}
        temporary=out/"orchestration.tmp"
        temporary.write_text(json.dumps(state,indent=2),encoding="utf-8")
        temporary.replace(out/"orchestration.json")
        print(json.dumps(row),flush=True)
    verification=subprocess.run([sys.executable,str(code/"verify_generation_diagnosis.py"),
                                 "--run",str(out),"--output",str(out/"verification.json")])
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                z.write(path,str(Path(out.name)/path.relative_to(out)))
    print(json.dumps({"archive":str(archive),"verification_exit_code":verification.returncode}))
    if verification.returncode:
        raise SystemExit(verification.returncode)


if __name__=="__main__":
    main()
