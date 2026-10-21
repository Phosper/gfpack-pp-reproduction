#!/usr/bin/env python3
"""Native strip API and exact official gap-pass controls; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from postprocess_metrics import geometry, self_test


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extension',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.extension.resolve()))
    import collision_probe as m
    build=json.loads((a.extension/'build.json').read_text())
    assert hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()==build['extension_sha256']
    sq=[[0.,0.],[10.,0.],[10.,10.],[0.,10.]];cases=[]
    for offset in (-100.,9000.):
        xy=[[offset,0.],[offset+5.,0.]]
        r=m.overlap_repair_strip([0,0],[0.,0.],xy,[sq],10.,1.,2000)
        ref=geometry([sq],[0,0],[list(v)+[1.,0.] for v in r['actions_xy']],10.)
        assert r['status']=='converged' and ref['max_pairwise_area']<=1. and ref['within_strip']
        old=m.overlap_repair([0,0],[0.,0.],xy,[sq],8000.,10.,1.,2000)
        assert old['status']=='infeasible_bounds' and old['actions_xy']==[tuple(v) for v in xy]
        cases.append({'name':'unbounded_x_'+str(offset),'passed':True,'native':r,'reference':ref})
        single=m.overlap_repair_strip([0],[0.],[[offset,0.]],[sq],10.,1.,2000)
        assert single['status']=='converged' and single['actions_xy']==[(offset,0.)]
        cases.append({'name':'legal_unmoved_x_'+str(offset),'passed':True})
    for y in (-1.,1.):
        r=m.overlap_repair_strip([0],[0.],[[0.,y]],[sq],10.,1.,2000)
        assert r['status']=='infeasible_bounds' and r['rolled_back'] and r['actions_xy']==[(0.,y)]
        cases.append({'name':'vertical_outside_'+str(y),'passed':True})
    r=m.overlap_repair_strip([0,0],[0.,0.],[[0.,0.],[5.,0.]],[sq],10.,1.,0)
    assert r['status']=='iteration_limit' and r['rolled_back'] and r['actions_xy']==[(0.,0.),(5.,0.)]
    cases.append({'name':'zero_iterations_rollback','passed':True})
    height=[[0.,0.],[1.,0.],[1.,1200.],[0.,1200.]]
    out=m.rm_spacing_all([[0]],[[0.]],[[[0.,0.]]],[height],8000.,1200.,1.)[0]
    ref=geometry([height],[0],[list(out[0])+[1.,0.]])
    # The original primitive unconditionally adds .001 in y on each y pass.
    assert ref['height_violation']>0
    cases.append({'name':'official_gap_height_regression_visible','passed':True,'reference':ref})
    result={'status':'verified','native_controls':len(cases),'cases':cases,
            'metric_controls':self_test(),'extension_sha256':build['extension_sha256'],
            'scope':'API/geometry controls, not repair efficacy on real candidates'}
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'verified','native_controls':len(cases)}))


if __name__=='__main__':main()
