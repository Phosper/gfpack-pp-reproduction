#!/usr/bin/env python3
"""Draw two fixed development examples from saved evidence; no solver calls."""
import argparse
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Output must be new')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    routes=('gap_only','repair_only','repair_then_gap')
    titles=('Gap only','Repair only','Repair then gap')
    fig,axes=plt.subplots(2,3,figsize=(15,6.5))
    for row,index in enumerate((25,46)):
        case=json.loads((a.frozen/'cases'/('sample_%03d.json'%index)).read_text())
        jobs=[json.loads((a.results/'jobs'/('sample_%03d_pred_%s.json'%(index,route))).read_text()) for route in routes]
        xmin=min(j['stages']['float64']['after']['bounds_xyxy'][0] for j in jobs)
        xmax=max(j['stages']['float64']['after']['bounds_xyxy'][2] for j in jobs)
        for ax,title,job in zip(axes[row],titles,jobs):
            stage=job['stages']['float64'];shapes=[]
            for idx,action in zip(case['polygon_ids'],stage['actions']):
                t=math.atan2(action[3],action[2]);c,s=math.cos(t),math.sin(t)
                shape=Polygon([(x*c-y*s+action[0],x*s+y*c+action[1]) for x,y in case['polygons'][idx]])
                shapes.append(shape);ax.plot(*shape.exterior.xy,color='#28647d',linewidth=.6)
            for i,first in enumerate(shapes):
                for second in shapes[i+1:]:
                    intersection=first.intersection(second)
                    parts=[intersection] if intersection.geom_type=='Polygon' else getattr(intersection,'geoms',[])
                    for part in parts:
                        if part.geom_type=='Polygon' and part.area>0:ax.fill(*part.exterior.xy,color='#d45532',alpha=.7)
            metrics=stage['after']
            ax.axhline(0,color='#888',linestyle='--',linewidth=.6);ax.axhline(1200,color='#888',linestyle='--',linewidth=.6)
            ax.set_xlim(xmin-30,xmax+30);ax.set_ylim(-30,1240);ax.set_aspect('equal')
            ax.set_title('%s | sample %d\nstrip util %.2f%%; max overlap %.5g'%(
                title,index,metrics['strip_utilization']*100,metrics['max_pairwise_area']),fontsize=10)
            ax.tick_params(labelsize=8)
    fig.suptitle('Frozen representatives, FP64 | same horizontal scale within each row | red = intersection area',fontsize=11)
    fig.tight_layout();fig.savefig(a.output,dpi=160);plt.close(fig)


if __name__=='__main__':main()
