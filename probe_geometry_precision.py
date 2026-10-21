#!/usr/bin/env python3
"""Compare the existing native clipper extension with float64 reference squares."""
import argparse
import json
from pathlib import Path
import sys


def main():
    from shapely.geometry import Polygon
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extensions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.extensions.resolve()))
    import calutil
    square = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    rows = []
    for overlap in (0., 0.001, 0.004, 0.006, 0.01):
        dx = 1 - overlap
        native = calutil.cal_util_all([[0, 0]], [[0., 0.]], [[[0., 0.], [dx, 0.]]], [square], 0.)[0]
        reference = Polygon(square).intersection(Polygon([(x+dx, y) for x, y in square])).area
        rows.append({"designed_overlap": overlap, "native_max_intersection": native[1],
                     "float64_reference_intersection": reference})
    # Native Clipper uses quantized coordinates; do not assume its measured
    # 0.01 overlap equals 0.01 to machine precision (that is the question tested).
    if not (rows[0]["native_max_intersection"] == 0 and
            abs(rows[0]["float64_reference_intersection"]) < 1e-12 and
            rows[-1]["native_max_intersection"] > 0 and
            abs(rows[-1]["float64_reference_intersection"] - .01) < 1e-12):
        raise ValueError("Native probe controls failed")
    result = {"scope": "Synthetic unit-square diagnostic; no candidate or metric threshold changes", "rows": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
