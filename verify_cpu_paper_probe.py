"""Verify local CPU probe evidence and geometry without rerunning neural scores."""
import argparse
import json
import math
from pathlib import Path

from run_inference import sha256,write_json


def verify(directory):
    import torch
    from shapely.geometry import Polygon
    from shapely.affinity import rotate,translate
    p=Path(directory)
    read=lambda name:json.loads((p/name).read_text(encoding='utf-8'))
    m=read('metadata.json')
    if not (m['status']=='completed' and not m['changed_inputs'] and m['model_executed'] and m['weights_loaded']):
        raise ValueError('Expected an actually executed, unchanged complete CPU probe')
    for name,digest in m['input_sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError('Input hash mismatch: '+name)
    for name,digest in m['output_sha256'].items():
        if sha256(p/name)!=digest:raise ValueError('Output hash mismatch: '+name)
    sample=read('input_sample.json');polys=read('input_geometry.json')['official_max_contours'];ids=sample['polygon_ids']
    raw=torch.load(p/'predictions_raw.pt',weights_only=True,map_location='cpu')
    trajectory=torch.load(p/'trajectory.pt',weights_only=True,map_location='cpu')
    if tuple(raw.shape)!=(m['planned_candidates'],len(ids),4) or not torch.equal(trajectory[:,-1],raw):
        raise ValueError('Saved trajectory/final state disagree')
    sampler=read('sampler.json');steps=sampler['steps']
    active=[x['step_1based'] for x in steps if x['constraint_active']]
    if len(steps)!=128 or active!=(list(range(97,129)) if m['constraints_enabled'] else []):
        raise ValueError('Wrong sampling schedule')
    stages={}
    for stage,name in (('raw','predictions_raw.pt'),('enhanced_float64','predictions_enhanced_float64.pt'),
                       ('enhanced_fp32','predictions_enhanced_fp32.pt')):
        tensor=torch.load(p/name,weights_only=True,map_location='cpu')
        report=read(stage+'_metrics.json');saved=report['samples'][0]['candidates'];rows=[]
        for a,r in zip(tensor.tolist(),saved):
            shapes=[translate(rotate(Polygon(polys[i]),math.atan2(t[3],t[2]),origin=(0,0),use_radians=True),t[0],t[1]) for i,t in zip(ids,a)]
            intersections=[x.intersection(y).area for i,x in enumerate(shapes) for y in shapes[i+1:]]
            bounds=[min(x.bounds[0] for x in shapes),min(x.bounds[1] for x in shapes),
                    max(x.bounds[2] for x in shapes),max(x.bounds[3] for x in shapes)]
            ref=r['reference'];maximum=max(intersections,default=0.);total=math.fsum(intersections)
            for x,y in ((maximum,ref['max_pairwise_area']),(total,ref['sum_pairwise_area']),*zip(bounds,ref['bounds_xyxy'])):
                if not math.isclose(x,y,abs_tol=2e-6,rel_tol=1e-8):raise ValueError('Independent geometry mismatch')
            valid=(ref['max_pairwise_area']<=1e-9 and ref['bounds_xyxy'][1]>=-1e-9 and ref['bounds_xyxy'][3]<=1200.+1e-9)
            if valid!=r['feasible']['reference_strict_fixed_strip']:raise ValueError('Strict validity flag mismatch')
            rows.append({'max_pairwise_area':maximum,'sum_pairwise_area':total,'strict_valid':valid,
                'nominal_strip_utilization':ref['strip_utilization'],
                'height_violation':max(0.,-bounds[1],bounds[3]-1200.)})
        if len(rows)!=m['planned_candidates']:raise ValueError('Missing candidates')
        stages[stage]=rows
    return {'status':'verified','model_executed':True,'candidate_count':len(raw),'stages':stages,
        'scope':'CPU same-machine input/output hashes, saved schedule/trajectory and scalar GEOS geometry; neural scores not rerun',
        'paper_full_reproduction':False,'metadata_sha256':sha256(p/'metadata.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    r=verify(args.run);write_json(args.output,r);print(json.dumps({'status':r['status'],'candidates':r['candidate_count']}))


if __name__=='__main__':main()
