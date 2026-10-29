#!/usr/bin/env python3
"""Bounded sequential development probes; no 128-case benchmark claim."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path,value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--extension",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    profiles=[("release_weights_constraints",True),
              ("paper_mechanism_release_scale",False),("paper_settings",False)]
    plan={"samples":[0,45],"candidates":512,"steps":128,"post_iterations":100,
          "seed":3407,"model_mode":"release default dropout active; held constant",
          "geometry_backend":"legacy_native","post_policy":"legacy_unit",
          "profiles":[{"name":x,"no_constraints":y} for x,y in profiles],
          "scope":"Two pre-used development samples, no independent generalization or full paper reproduction",
          "per_run_timeout_seconds":600,"status":"running","rows":[]}
    save(a.output/"plan.json",{k:v for k,v in plan.items() if k not in ("status","rows")})
    started=time.perf_counter()
    for index in plan["samples"]:
        for profile,no_constraints in profiles:
            name="sample_%03d_%s"%(index,profile)
            out=a.output/name
            cmd=[sys.executable,str(Path(__file__).with_name("run_paper_alignment.py")),
                 "--root",str(a.root.resolve()),"--extension",str(a.extension.resolve()),
                 "--output",str(out.resolve()),"--profile",profile,"--sample-index",str(index),
                 "--geometry-backend","legacy_native","--post-policy","legacy_unit"]
            if no_constraints:cmd.append("--no-constraints")
            row={"sample_index":index,"profile":profile,"output":name,"status":"failed"}
            before=time.perf_counter()
            try:
                with (a.output/(name+".log")).open("w") as stream:
                    result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=600,
                      env=dict(os.environ,OMP_NUM_THREADS="16",MPLBACKEND="Agg",PYTHONDONTWRITEBYTECODE="1"))
                row["returncode"]=result.returncode
                if (out/"metadata.json").exists():
                    m=json.loads((out/"metadata.json").read_text())
                    row["status"]=m["status"];row["changed_inputs"]=m["changed_inputs"]
                    row["model_parameters"]=m.get("model_parameters")
                    row["state_dict_elements"]=m.get("state_dict_elements")
                if result.returncode:row["status"]="failed"
            except subprocess.TimeoutExpired:
                row["error"]="600 second timeout; planned 512 candidates remain in denominator"
            row["elapsed_seconds"]=time.perf_counter()-before
            plan["rows"].append(row);plan["elapsed_seconds"]=time.perf_counter()-started
            save(a.output/"summary.json",plan)
            print(json.dumps(row),flush=True)
    plan["status"]="completed";save(a.output/"summary.json",plan)


if __name__=="__main__":main()
