"""Independently compare raw and saved final candidates, without sampling or changing them."""
import argparse
import json
from pathlib import Path
import statistics

from .assets import sha256,write_json
from .selection import select_candidate
from .verify import measure


def diagnose(directory):
    directory=Path(directory).resolve()
    read=lambda name:json.loads((directory/name).read_text(encoding='utf-8'))
    data,raw=read('input.json'),read('raw.json')
    path='generation_metadata.json' if (directory/'generation_metadata.json').exists() else 'metadata.json'
    metadata=read(path)
    height=metadata['height']
    rows=[]
    for index,pose in enumerate(raw):
        before=measure(data['contours'],pose,height)
        record={'candidate':index,'raw':before,'post_saved':False}
        saved=directory/f'candidate_{index:04d}.json'
        if saved.exists():
            payload=read(saved.name)
            if payload['candidate']!=index:
                raise ValueError('Candidate filename and saved ID disagree')
            after=measure(data['contours'],payload['actions'],height)
            record.update(post_saved=True,post=after,span_change=after['span_x']-before['span_x'],
                          utilization_change=after['strip_utilization']-before['strip_utilization'],
                          overlap_increased=after['sum_pair_overlap_area']>before['sum_pair_overlap_area'])
            staged=[t for t in payload['result'].get('trace',[]) if 'sum_pair_overlap_area_after_collision' in t]
            record['collision_stages_recorded']=len(staged)
            record['repair_rounds_increasing_overlap']=(sum(t['sum_pair_overlap_area_after_collision']>
                t['sum_pair_overlap_area_before_collision'] for t in staged) if staged else None)
        rows.append(record)
    saved=[r for r in rows if r['post_saved']]
    best_raw=max(rows,key=lambda r:(r['raw']['strip_utilization'],-r['candidate']))
    selected=select_candidate([r['post'] for r in saved])
    chosen=saved[selected] if selected is not None else None
    configuration={key:metadata.get(key) for key in ('sampling_profile','sigma_min','constraint_integrator',
                    'height','model_mode','post_policy','relaxation','post_initial_step')}
    if (directory/'postprocess_binding.json').exists():
        binding=read('postprocess_binding.json')
        configuration.update({k:binding[k] for k in ('post_policy','relaxation','post_step')})
    return {'sample_index':data.get('sample_index'),'sampling_profile':metadata.get('sampling_profile','unrecorded'),
            'configuration':configuration,
            'sigma_min':metadata.get('sigma_min'),'constraint_integrator':metadata.get('constraint_integrator','em'),
            'candidate_count':len(raw),'post_saved_count':len(saved),'post_complete':len(saved)==len(raw),
            'raw_strict_valid_candidates':sum(r['raw']['strict_valid'] for r in rows),
            'post_strict_valid_candidates':sum(r['post']['strict_valid'] for r in saved),
            'best_raw_nominal_candidate':best_raw['candidate'],'best_raw_nominal_metrics':best_raw['raw'],
            'selected_strict_valid_candidate':chosen['candidate'] if chosen else None,
            'selected_strict_valid_metrics':chosen['post'] if chosen else None,
            'selected_candidate_raw_metrics':chosen['raw'] if chosen else None,
            'mean_span_change':statistics.mean(r['span_change'] for r in saved) if saved else None,
            'post_overlap_increased_candidates':sum(r['overlap_increased'] for r in saved),
            'input_sha256':{'input.json':sha256(directory/'input.json'),'raw.json':sha256(directory/'raw.json')},
            'model_sampling_performed':False,'records':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,help='One instance or benchmark directory')
    parser.add_argument('--output',required=True,type=Path,help='New JSON report, preferably outside original results')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    folders=([args.directory] if (args.directory/'input.json').exists() else
             sorted(p for p in args.directory.glob('sample_*') if (p/'input.json').exists()))
    if not folders:raise ValueError('No saved raw instances found')
    reports=[diagnose(p) for p in folders]
    complete=all(r['post_complete'] for r in reports)
    comparable=len({json.dumps(r['configuration'],sort_keys=True) for r in reports})==1
    scores=[r['selected_strict_valid_metrics']['strip_utilization'] if r['selected_strict_valid_metrics'] else 0.
            for r in reports]
    successes=[v for v,r in zip(scores,reports) if r['selected_strict_valid_candidate'] is not None]
    report={'sample_count':len(reports),'post_complete':complete,'comparable_configuration':comparable,
            'complete_128x512':complete and comparable and len(reports)==128 and
                              sorted(r['sample_index'] for r in reports)==list(range(128)) and
                              all(r['candidate_count']==512 for r in reports),
            'successful_samples':len(successes),
            'mean_selected_utilization_failure_zero':statistics.mean(scores) if complete and comparable else None,
            'mean_selected_utilization_on_successes':statistics.mean(successes) if complete and comparable and successes else None,
            'raw_nominal_scores_are_not_feasible_solution_scores':True,'samples':reports}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
