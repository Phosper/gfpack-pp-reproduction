"""Numerical step selection for the existing area-weighted collision field.

The paper does not specify this line search or the scalar O. We explicitly
choose a squared-separation potential consistent with our reconstructed field:
E = sum_i (S_i/S)*wall_i^2/2 + sum_i<j S_i*S_j/(S*(S_i+S_j))*|d_ij|^2/2.
Away from a change of nearest contact, grad_i E = -(S_i/S)*repulsion_i.
This is a numerical reconstruction probe, NOT an author-confirmed enhancement.
"""
import time
import math
import numpy as np
from paper_metrics import STRICT_AREA_EPS, BOUNDARY_TOLERANCE

ARMIJO_C=1e-4
MAX_TRIALS=21


def enhance_potential(native,polygons,ids,actions,max_iterations=100,height=1200.,*,
                      initial_step=1.,trace_candidates=False):
    from paper_enhancement import field
    if getattr(native,"geometry_backend",None)!="paper_sat":
        raise ValueError("Potential descent requires the paper_sat reference field")
    if type(max_iterations) is not int or not 0<=max_iterations<=100:
        raise ValueError("Post iterations must be in [0,100]")
    if isinstance(initial_step,bool) or not math.isfinite(initial_step) or initial_step<=0:
        raise ValueError("Initial step must be positive and finite")
    a=np.array(actions,dtype=np.float64,copy=True)
    if a.ndim!=3 or not len(a) or a.shape[1:]!=(len(ids),4) or not np.isfinite(a).all():
        raise ValueError("Invalid actions")
    original=a.copy();start=time.perf_counter();batch=len(a)
    statuses=["iteration_limit"]*batch;active=np.ones(batch,dtype=bool)
    rollbacks=np.zeros(batch,dtype=np.int64);log=[]
    def feasible(r):
        return (np.asarray(r["max_pairwise_area"])<=STRICT_AREA_EPS)&(np.asarray(r["height_violation"])<=BOUNDARY_TOLERANCE)
    def rows(r):
        return [{"candidate_index":i,**{k:float(r[k][i]) for k in
                 ("max_pairwise_area","sum_pairwise_area","height_violation","min_x","collision_potential")}}
                for i in range(batch)]
    for iteration in range(max_iterations):
        before=a.copy();r,_=field(native,polygons,ids,a,height)
        energy=np.asarray(r["collision_potential"])
        norm=np.asarray(r["mass_weighted_force_norm_squared"])
        force=np.asarray(r["repulsion"])
        pending=active.copy();accepted=np.zeros(batch,dtype=bool)
        alpha=np.full(batch,initial_step);used=np.zeros(batch);trials=np.zeros(batch,dtype=int)
        for attempt in range(MAX_TRIALS):
            if not pending.any():break
            trial=before.copy();trial[pending,:,:2]+=alpha[pending,None,None]*force[pending]
            checked,_=field(native,polygons,ids,trial,height)
            next_energy=np.asarray(checked["collision_potential"])
            # No geometric threshold is relaxed to make a step pass. At a
            # floating-point floor this may hold an infeasible state and report it.
            good=pending&(next_energy<=energy-ARMIJO_C*alpha*norm)
            a[good]=trial[good];used[good]=alpha[good];accepted|=good
            trials[pending]+=1;pending[good]=False;alpha[pending]*=.5
        collision,_=field(native,polygons,ids,a,height)
        a[active,:,0]-=np.asarray(collision["min_x"])[active,None]
        before_gap,angles=field(native,polygons,ids,a,height)
        gap=a.copy();gap[:,:,:2]=native.paper_gap_xy_all(ids,angles,a[:,:,:2].tolist(),polygons)
        after_gap,_=field(native,polygons,ids,gap,height)
        gap_trace=getattr(native,"last_gap_trace",None)
        # Gap endpoints are independently checked by the backend. Small energy
        # roundoff is logged explicitly; it is not an overlap/boundary tolerance.
        e0=np.asarray(before_gap["collision_potential"])
        e1=np.asarray(after_gap["collision_potential"])
        roundoff=32*np.finfo(float).eps*np.maximum(1.,e0)
        reject=(e1>e0+roundoff)|((np.asarray(after_gap["sum_pairwise_area"])>
            np.asarray(before_gap["sum_pairwise_area"])+STRICT_AREA_EPS))|(
            np.asarray(after_gap["height_violation"])>np.asarray(before_gap["height_violation"])+BOUNDARY_TOLERANCE)
        a[active&~reject]=gap[active&~reject];rollbacks+=active&reject
        a[~active]=before[~active]
        final,_=field(native,polygons,ids,a,height)
        motion=np.linalg.norm(a[:,:,:2]-before[:,:,:2],axis=-1).max(axis=1)
        done=active&feasible(final)&(motion<=1e-3)
        stalled=active&~feasible(final)&(motion==0)
        for i in np.flatnonzero(done):statuses[i]="reference_converged"
        for i in np.flatnonzero(stalled):
            statuses[i]="stationary_infeasible" if norm[i]==0 else "numerically_stalled_infeasible"
        active[done|stalled]=False
        record={"iteration_1based":iteration+1,"active_candidates":int(active.sum()),
                "newly_converged":int(done.sum()),"newly_stalled_infeasible":int(stalled.sum()),
                "gap_rollbacks":int(np.count_nonzero(reject)),"max_pairwise_area":max(final["max_pairwise_area"]),
                "max_height_violation":max(final["height_violation"]),"max_translation_update":float(motion.max()),
                "accepted_collision_step":used.tolist(),"line_search_trials":trials.tolist(),
                "potential_before":energy.tolist(),"potential_after_collision":collision["collision_potential"],
                "potential_after_iteration":final["collision_potential"]}
        if trace_candidates:
            record["candidate_stages"]={"before_collision":rows(r),"after_collision_before_gauge":rows(collision),
                "after_collision_and_x_gauge":rows(before_gap),"after_gap_trial":rows(after_gap),
                "after_accepted_iteration":rows(final),"gap_substages":gap_trace,"gap_rolled_back":reject.tolist()}
        log.append(record)
        if not active.any():break
    return a,{"post_policy":"paper_potential_descent","geometry_backend":"paper_sat",
        "max_iterations":max_iterations,"iterations_executed":len(log),"statuses":statuses,
        "constraint_step_size":float(initial_step),"step_rule":"Armijo along the same reconstructed collision field",
        "armijo_c":ARMIJO_C,"max_trial_evaluations_per_iteration":MAX_TRIALS,
        "gap_rollbacks_per_candidate":rollbacks.tolist(),"trace":log,
        "energy_is_our_field_consistent_surrogate_not_author_confirmed_phi":True,
        "post_step_size_and_order_are_reconstruction_assumptions":True,
        "rotation_components_preserved":bool(np.array_equal(a[:,:,2:],original[:,:,2:])),
        "elapsed_seconds":time.perf_counter()-start}
