"""Explain saved candidate failures without changing layouts or tolerances."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from paper_metrics import STRICT_AREA_EPS, BOUNDARY_TOLERANCE
from paper_sat_geometry import _pair_areas
from run_inference import sha256, write_json


def diagnose_layout(polygons, ids, actions, height=1200.):
    a = np.asarray(actions, dtype=np.float64)
    if (a.shape != (len(ids), 4) or not len(ids) or not np.isfinite(a).all() or
            not math.isfinite(height) or height<=0):
        raise ValueError("Expected a finite nonempty candidate and positive height")
    shapes = []
    for index, t in zip(ids, a):
        angle = math.atan2(t[3], t[2]); c, s = math.cos(angle), math.sin(angle)
        p = Polygon([(x*c-y*s+t[0], x*s+y*c+t[1]) for x, y in polygons[index]])
        if not p.is_valid or p.area<=0:
            raise ValueError("Invalid input/transformed contour")
        shapes.append(p)
    boxes, i, j, areas = _pair_areas(shapes)
    overlap = [{"piece_i": int(ii), "piece_j": int(jj), "polygon_i": ids[ii],
                "polygon_j": ids[jj], "intersection_area": float(area)}
               for ii, jj, area in zip(i, j, areas) if area>STRICT_AREA_EPS]
    boundary, impossible = [], []
    for k, box in enumerate(boxes):
        below, above = max(0., -box[1]), max(0., box[3]-height)
        if max(below, above)>BOUNDARY_TOLERANCE:
            boundary.append({"piece_index": k, "polygon_id": ids[k], "below": float(below),
                             "above": float(above), "bounds_xyxy": box.tolist()})
        if box[3]-box[1]>height+BOUNDARY_TOLERANCE:
            impossible.append(k)
    codes = (["pair_overlap"] if overlap else []) + (["strip_boundary"] if boundary else [])
    if impossible:
        codes.append("fixed_rotation_piece_taller_than_strip")
    return {"feasible": not codes, "reason_codes": codes or ["passes_project_tolerance"],
            "max_pairwise_area": float(max(areas,default=0.)),
            "sum_pairwise_area": math.fsum(float(x) for x in areas),
            "max_height_violation": float(max(0., -boxes[:,1].min(),boxes[:,3].max()-height)),
            "positive_area_pair_count": int(np.count_nonzero(areas>0)),
            "pairs_over_project_tolerance": len(overlap),
            "largest_overlap_pairs": sorted(overlap,key=lambda r:-r["intersection_area"])[:25],
            "boundary_pieces": boundary, "fixed_rotation_impossible_piece_indices": impossible}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run",required=True,type=Path)
    p.add_argument("--result-run",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from replay_paper_geometry import verified_source
    original, frozen = verified_source(args.source_run)
    result=args.result_run.resolve()
    meta=json.loads((result/"metadata.json").read_text(encoding="utf-8"))
    frozen[str(result/"metadata.json")]=sha256(result/"metadata.json")
    if meta.get("status")!="completed" or meta.get("changed_inputs")!=[]:
        raise ValueError("Expected a completed unchanged replay")
    for name in ("input_sample.json","input_geometry.json","predictions_raw.pt"):
        matches=[h for path,h in meta["input_sha256"].items() if path.replace("\\","/").rsplit("/",1)[-1]==name]
        if matches!=[sha256(args.source_run/name)]:
            raise ValueError("Replay is not bound to the supplied source: "+name)
    import torch
    sample=json.loads((args.source_run/"input_sample.json").read_text(encoding="utf-8"))
    polygons=json.loads((args.source_run/"input_geometry.json").read_text(encoding="utf-8"))["official_max_contours"]
    count=meta["planned_candidates"]
    indices=meta["candidate_indices"]
    if len(indices)!=count or len(set(indices))!=count or any(type(i) is not int or not 0<=i<original["planned_candidates"] for i in indices):
        raise ValueError("Invalid replay candidate coverage")
    reports={}
    for dtype in ("float64","fp32"):
        path=result/("predictions_enhanced_"+dtype+".pt")
        if sha256(path)!=meta["output_sha256"].get(path.name):
            raise ValueError("Saved replay output hash mismatch")
        tensor=torch.load(path,map_location="cpu",weights_only=True)
        if tensor.shape!=(count,len(sample["polygon_ids"]),4):
            raise ValueError("Candidate tensor does not match denominator")
        rows=[dict(candidate_index=i,**diagnose_layout(polygons,sample["polygon_ids"],a))
              for i,a in zip(indices,tensor.double().tolist())]
        reports[dtype]={"feasible_count":sum(r["feasible"] for r in rows),
            "reason_counts":{code:sum(code in r["reason_codes"] for r in rows) for code in
                ("pair_overlap","strip_boundary","fixed_rotation_piece_taller_than_strip")},"candidates":rows}
        frozen[str(path)]=sha256(path)
    if any(sha256(Path(path))!=digest for path,digest in frozen.items()):
        raise ValueError("Frozen inputs changed during diagnosis")
    write_json(args.output,{"sample_index":sample["sample_index"],"source_profile":meta["source_profile"],
        "evaluated_candidates":count,"source_planned_candidates":original["planned_candidates"],
        "candidate_indices":indices,"post_step_size":meta.get("post_step_size",1.),
        "area_tolerance":STRICT_AREA_EPS,"boundary_tolerance":BOUNDARY_TOLERANCE,
        "scope":"saved-coordinate failure diagnosis; not a solver or paper-accuracy claim",
        "reports":reports,"input_sha256":frozen})
    print(json.dumps({k:v["reason_counts"] for k,v in reports.items()}))


if __name__=="__main__":
    main()
