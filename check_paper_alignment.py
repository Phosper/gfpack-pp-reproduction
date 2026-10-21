#!/usr/bin/env python3
"""Independent small controls for paper formula, scheduling and denominators."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True

import numpy as np
import torch

from paper_sampler import sample_paper, profile_settings, marginal_prob, sde_coeff
from paper_enhancement import field, enhance
from paper_metrics import sample_metrics, aggregate_metrics
from run_paper_alignment import exact_all


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension",type=Path,required=True)
    parser.add_argument("--upstream",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    extension=args.extension.resolve();build=json.loads((extension/"build.json").read_text())
    sys.path.insert(0,str(extension));sys.path.insert(0,str(args.upstream.resolve()))
    import collision_probe as native
    assert Path(native.__file__).resolve().parent==extension
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()==build["extension_sha256"]
    cases=[]
    def check(name,passed,details=None):
        cases.append({"name":name,"passed":bool(passed),"details":details})
        if not passed:raise AssertionError(name)
    square=[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]
    rect=[[0.,0.],[2.,0.],[2.,1.],[0.,1.]]
    lshape=[[0.,0.],[3.,0.],[3.,1.],[1.,1.],[1.,3.],[0.,3.]]
    def evaluate(polys,ids,xy,height=1200.):
        a=np.array([[[x,y,1.,0.] for x,y in xy]],dtype=np.float64)
        r,_=field(native,polys,ids,a,height)
        after=a.copy();after[:,:,:2]+=np.array(r["repulsion"])
        before=exact_all(polys,ids,a.tolist())[0];ref=exact_all(polys,ids,after.tolist())[0]
        return a,r,after,before,ref
    a,r,after,before,ref=evaluate([square],[0,0],[[2.,2.],[2.5,2.]])
    check("convex_repulsion_reduces_overlap",before["max_pairwise_area"]>.49 and ref["max_pairwise_area"]==0,r)
    check("correct_signed_pair_displacement",r["repulsion"][0][0][0]<0 and r["repulsion"][0][1][0]>0)
    a,r,after,before,ref=evaluate([square,rect],[0,1],[[2.,2.],[2.5,2.]])
    d=np.array(r["repulsion"])[0]
    check("area_weighted_center_conserved",np.max(np.abs(d[0]+2*d[1]))<1e-12,r)
    _,r,_,_,_=evaluate([square],[0,0],[[2.,2.],[4.,2.]])
    check("disjoint_pair_no_repulsion",np.array_equal(r["repulsion"],[[[0.,0.],[0.,0.]]]))
    _,r,_,_,_=evaluate([lshape,square],[0,1],[[0.,0.],[1.5,1.5]])
    check("concave_void_not_false_collision",np.max(np.abs(r["repulsion"]))==0,r)
    _,r,after,_,ref=evaluate([square],[0],[[10000.,-.5]],1.)
    check("only_vertical_strip_walls",np.array_equal(r["repulsion"],[[[0.,.5]]]) and after[0,0,0]==10000.)
    _,r,_,_,_=evaluate([square],[0],[[0.,0.]],.5)
    check("oversized_piece_retains_boundary_violation",r["height_violation"][0]>.49)
    fullheight=[[0.,0.],[1.,0.],[1.,1200.],[0.,1200.]]
    out,diag=enhance(native,[fullheight],[0],np.array([[[0.,0.,1.,0.]]]),3)
    check("gap_new_height_violation_rolled_back",out[0,0,1]==0 and diag["gap_rollbacks_per_candidate"][0]>0,diag)
    class Fake(torch.nn.Module):
        def __init__(self):super().__init__();self.device=torch.device("cpu")
        def geo_feature(self,x):return x
        def forward(self,actions,*unused):return torch.zeros_like(actions)
    model=Fake();ids=torch.tensor([0,0]);mask=torch.zeros(2)
    plain,base,pd=sample_paper(model,ids,None,mask,2,None,"paper_settings",seed=17)
    seen=[]
    def callback(a):
        seen.append(a.clone());r=torch.zeros(a.shape[0],a.shape[1],2)
        r[:,0,0]=-.1;r[:,1,0]=.1;return r
    trace,constrained,cd=sample_paper(model,ids,None,mask,2,callback,"paper_settings",seed=17)
    check("exact_128_steps_last32_constraints",len(cd["steps"])==128 and len(seen)==32 and
        [s["step_1based"] for s in cd["steps"] if s["constraint_active"]]==list(range(97,129)))
    check("constraint_changes_translation_only",torch.equal(base[:,:,2:],constrained[:,:,2:]))
    check("eq10_sign_outward",torch.all(constrained[:,0,0]<base[:,0,0]) and torch.all(constrained[:,1,0]>base[:,1,0]))
    check("prefix_identical_before_constraints",torch.equal(plain[:,:96],trace[:,:96]))
    from src.sde import pc_sampler_state
    torch.manual_seed(26)
    old_trace,old_final=pc_sampler_state(model,sde_coeff("release_weights_constraints"),2,ids,None,mask,batch_size=2,num_steps=128)
    torch.manual_seed(26)
    new_trace,new_final,_=sample_paper(model,ids,None,mask,2,None,"release_weights_constraints")
    check("unconstrained_release_arithmetic_regression",torch.equal(old_trace,new_trace) and torch.equal(old_final,new_final),
          {"max_abs_difference":float((old_trace-new_trace).abs().max())})
    for profile in ("paper_settings","paper_mechanism_release_scale"):
        check("profile_"+profile,profile_settings(profile)["num_steps"]==128 and profile_settings(profile)["drift_step_multiplier"]==1.)
    native_result={"candidate_count":2,"official_valid":[True,True],"official_valid_count":2,
                   "utilization":[.5,.5],"bounding_boxes_xyxy":[[0.,0.,10000.,1200.],[0.,0.,1.,1205.]],
                   "reported_overlap_area_percent":[0.,0.]}
    refs=[{"bounds_xyxy":[0.,0.,10000.,1200.],"total_polygon_area":6000000.,"max_pairwise_area":.001,"sum_pairwise_area":.001},
          {"bounds_xyxy":[0.,0.,1.,1205.],"total_polygon_area":600.,"max_pairwise_area":0.,"sum_pairwise_area":0.}]
    rows=sample_metrics(0,native_result,refs)
    report=aggregate_metrics([rows,sample_metrics(1,None,planned_candidate_count=2,status="failed")])
    check("paper_denominators_keep_failures",report["planned_candidates"]==4 and report["criteria"]["native_eps50_overlap_only"]["feasible_fraction_all_planned_candidates"]==.5)
    check("strict_boundary_and_overlap_not_native",report["criteria"]["reference_strict_fixed_strip"]["feasible_candidates"]==0)
    check("no_invented_8000_width_wall",report["criteria"]["reference_eps50_fixed_strip"]["feasible_candidates"]==1)
    result={"status":"verified","passed":len(cases),"cases":cases,
            "build_sha256":hashlib.sha256((extension/"build.json").read_bytes()).hexdigest(),
            "scope":"Formula/scheduling/metric controls only; not checkpoint or paper performance validation"}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":"verified","passed":len(cases)}))


if __name__=="__main__":main()
