"""Float64 reference geometry for the paper's collision/gap mechanism.

Sec. 4.4 specifies an area-weighted minimum separating displacement, approximated
with SAT. SAT is exact for convex contours; applying full-contour projections to
a concave polygon can produce a much larger displacement. Here a concave pair
is decomposed into convex pieces and their translation-obstacle polygons are
unioned. The nearest boundary gives the minimum separating translation of the
*whole* pair, including cavities. This is our explicit completion of the paper's
minimum-displacement definition, NOT confirmed author source code.

The outer sampler, area weights, 100-step budget and post update coefficient are
unchanged. No packing heuristic, resize, backtracking collision policy or extra
clearance is introduced. Binary gap search retains a verified feasible endpoint
and never adds an unchecked +0.001 displacement. GEOS float64 is still numerical
geometry, not exact arithmetic. This CPU reference prioritizes correctness;
512-candidate inference performance has not been established.
"""
from functools import lru_cache
import math

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import nearest_points, unary_union


GAP_DISTANCE_TOLERANCE = 1e-3  # inherited numerical search tolerance, not a paper constant


def _key(points):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("Expected finite polygon coordinates (vertices,2)")
    rows = [tuple(map(float, row)) for row in p]
    if rows and rows[0] == rows[-1]:
        rows.pop()
    rows = [row for i, row in enumerate(rows) if not i or row != rows[i-1]]
    if len(rows) < 3:
        raise ValueError("A polygon requires at least three distinct vertices")
    return tuple(rows)


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _convex(vertices):
    signs = [_cross(vertices[i-1], vertices[i], vertices[(i+1) % len(vertices)])
             for i in range(len(vertices))]
    return all(x >= 0 for x in signs) or all(x <= 0 for x in signs)


def _ear_triangles(poly):
    """Fallback for Shapely <2.1; simple contours only, validated afterwards."""
    vertices = list(poly.exterior.coords[:-1])
    if not poly.exterior.is_ccw:
        vertices.reverse()
    triangles = []
    while len(vertices) > 3:
        clipped = False
        for i in range(len(vertices)):
            a, b, c = vertices[i-1], vertices[i], vertices[(i+1) % len(vertices)]
            turn = _cross(a, b, c)
            if turn == 0:
                vertices.pop(i)
                clipped = True
                break
            if turn < 0:
                continue
            triangle = Polygon([a, b, c])
            if not poly.covers(triangle):
                continue
            others = [v for k, v in enumerate(vertices)
                      if k not in ((i-1) % len(vertices), i, (i+1) % len(vertices))]
            if any(triangle.covers(Point(v)) for v in others):
                continue
            triangles.append(triangle)
            vertices.pop(i)
            clipped = True
            break
        if not clipped:
            raise ValueError("Could not triangulate contour without changing its geometry")
    triangles.append(Polygon(vertices))
    return triangles


@lru_cache(maxsize=512)
def _contour(key):
    poly = Polygon(key)
    if not poly.is_valid or poly.is_empty or poly.area <= 0:
        raise ValueError("Invalid or degenerate polygon; geometry is never silently repaired")
    is_convex = _convex(key)
    if is_convex:
        pieces = (poly,)
    else:
        try:
            from shapely import constrained_delaunay_triangles
        except ImportError:
            pieces = tuple(_ear_triangles(poly))
        else:
            pieces = tuple(constrained_delaunay_triangles(poly).geoms)
        if not pieces or any(p.area <= 0 or not _convex(list(p.exterior.coords[:-1]))
                             for p in pieces):
            raise ValueError("Invalid convex decomposition")
        joined = unary_union(pieces)
        # Triangulation is allowed only if it represents the supplied contour.
        # This round-off check is not a collision acceptance threshold.
        if joined.symmetric_difference(poly).area > 1e-12 * max(1., poly.area):
            raise ValueError("Convex decomposition changed the polygon area")
    return poly, is_convex, pieces


@lru_cache(maxsize=128)
def _obstacle(a_key, a_angle, b_key, b_angle):
    """Translations q = t_a - t_b causing overlap: B_rot + (-A_rot)."""
    a_parts, b_parts = _contour(a_key)[2], _contour(b_key)[2]
    a_parts = [_placed(p, a_angle, (0., 0.)) for p in a_parts]
    b_parts = [_placed(p, b_angle, (0., 0.)) for p in b_parts]
    cells = []
    for a in a_parts:
        av = np.asarray(a.exterior.coords[:-1])
        for b in b_parts:
            bv = np.asarray(b.exterior.coords[:-1])
            differences = (bv[:, None, :] - av[None, :, :]).reshape(-1, 2)
            cells.append(MultiPoint(differences).convex_hull)
    obstacle = unary_union(cells)
    if not obstacle.is_valid or obstacle.is_empty:
        raise ValueError("Invalid configuration-space union")
    return obstacle


def convex_sat_displacement(a, b):
    """Minimum translation of convex a out of b; containment is included."""
    av, bv = (np.asarray(p.exterior.coords[:-1], dtype=np.float64) for p in (a, b))
    best, answer = math.inf, np.zeros(2, dtype=np.float64)
    for vertices in (av, bv):
        edges = np.roll(vertices, -1, axis=0) - vertices
        for edge in edges:
            norm = np.linalg.norm(edge)
            if norm == 0:
                continue
            axis = np.array([-edge[1], edge[0]]) / norm
            ap, bp = av @ axis, bv @ axis
            if ap.max() <= bp.min() or bp.max() <= ap.min():
                return np.zeros(2, dtype=np.float64)
            for shift in (bp.max()-ap.min(), bp.min()-ap.max()):
                if abs(shift) < best:
                    best, answer = abs(shift), axis * shift
    return answer


def _placed(poly, angle, xy):
    # Match the reference evaluation's explicit trigonometry, avoiding geometry
    # rotations with special-case angle rounding in convenience APIs.
    c, s = math.cos(angle), math.sin(angle)
    return Polygon([(x*c-y*s+xy[0], x*s+y*c+xy[1])
                    for x, y in poly.exterior.coords[:-1]])


def _intersection(a, b):
    aa, bb = a.bounds, b.bounds
    if aa[2] <= bb[0] or bb[2] <= aa[0] or aa[3] <= bb[1] or bb[3] <= aa[1]:
        return 0.
    return float(a.intersection(b).area)


def measure_shapes(shapes, height=1200.):
    areas = [_intersection(a, b) for i, a in enumerate(shapes) for b in shapes[i+1:]]
    bounds = [min(p.bounds[0] for p in shapes), min(p.bounds[1] for p in shapes),
              max(p.bounds[2] for p in shapes), max(p.bounds[3] for p in shapes)]
    return {"max_pairwise_area": max(areas, default=0.),
            "sum_pairwise_area": math.fsum(areas), "bounds_xyxy": bounds,
            "height_violation": max(0., -bounds[1], bounds[3]-height),
            "span_x": bounds[2]-bounds[0], "min_x": bounds[0]}


class PaperSATGeometry:
    """Drop-in geometry interface, with explicit per-axis gap diagnostics."""

    geometry_backend = "paper_sat"

    def __init__(self, height=1200.):
        if not math.isfinite(height) or height <= 0:
            raise ValueError("height must be positive and finite")
        self.height = float(height)
        self.last_gap_trace = None

    @staticmethod
    def _validate(ids, rotates, actions, polygons):
        if not ids or any(isinstance(i, bool) or int(i) != i or i < 0 or i >= len(polygons) for i in ids):
            raise ValueError("Invalid polygon IDs")
        xy, angles = np.asarray(actions, dtype=np.float64), np.asarray(rotates, dtype=np.float64)
        if xy.ndim != 3 or xy.shape[1:] != (len(ids), 2) or len(xy) < 1 or angles.shape != xy.shape[:2]:
            raise ValueError("Expected (batch,pieces,2) translations and (batch,pieces) rotations")
        if not np.isfinite(xy).all() or not np.isfinite(angles).all():
            raise ValueError("Nonfinite geometry input")
        keys = [_key(polygons[i]) for i in ids]
        records = [_contour(k) for k in keys]
        return keys, records, angles, xy

    def paper_repulsion_all(self, ids, rotates, actions, polygons, height=1200., collision_area_eps=0.):
        if not math.isfinite(height) or height <= 0 or not math.isfinite(collision_area_eps) or collision_area_eps < 0:
            raise ValueError("Invalid height or collision area threshold")
        self.height = float(height)
        keys, records, angles, xy = self._validate(ids, rotates, actions, polygons)
        result = {key: [] for key in ("repulsion", "max_pairwise_area", "sum_pairwise_area",
                  "colliding_pairs", "concave_proxy_pairs", "concave_mtv_pairs", "height_violation", "min_x")}
        areas = np.asarray([r[0].area for r in records])
        for rotation, translations in zip(angles, xy):
            shapes = [_placed(r[0], angle, t) for r, angle, t in zip(records, rotation, translations)]
            force = np.zeros_like(translations)
            maximum, total, count, concave_count, violation = 0., 0., 0, 0, 0.
            for i, shape in enumerate(shapes):
                below, above = max(0., -shape.bounds[1]), max(0., shape.bounds[3]-height)
                force[i, 1] += below - above
                violation = max(violation, below, above)
                for j in range(i+1, len(ids)):
                    area = _intersection(shape, shapes[j])
                    maximum, total = max(maximum, area), total+area
                    if area <= collision_area_eps:
                        continue
                    count += 1
                    if records[i][1] and records[j][1]:
                        displacement = convex_sat_displacement(shape, shapes[j])
                    else:
                        concave_count += 1
                        obstacle = _obstacle(keys[i], float(rotation[i]), keys[j], float(rotation[j]))
                        relative = translations[i] - translations[j]
                        nearest = nearest_points(Point(relative), obstacle.boundary)[1]
                        displacement = np.asarray(nearest.coords[0]) - relative
                    weight = areas[j] / (areas[i]+areas[j])
                    force[i] += weight * displacement
                    force[j] -= (1.-weight) * displacement
            values = dict(repulsion=force.tolist(), max_pairwise_area=maximum,
                          sum_pairwise_area=total, colliding_pairs=count, concave_proxy_pairs=0,
                          concave_mtv_pairs=concave_count, height_violation=violation,
                          min_x=min(p.bounds[0] for p in shapes))
            for key, value in values.items():
                result[key].append(value)
        result.update(gradient_convention="repulsion=-grad(Phi), add g^2*h*repulsion",
                      mtv_method="convex SAT; validated convex decomposition and translation-obstacle union for concave pairs",
                      geometry_backend="paper_sat_float64_reference",
                      author_implementation_confirmed=False,
                      collision_detection="unquantized GEOS float64 intersection area")
        return result

    def paper_gap_xy_all(self, ids, rotates, actions, polygons):
        keys, records, angles, xy = self._validate(ids, rotates, actions, polygons)
        output = xy.copy()
        traces = []
        for batch, (rotation, translations) in enumerate(zip(angles, output)):
            shapes = [_placed(r[0], angle, t) for r, angle, t in zip(records, rotation, translations)]
            trace = {"candidate_index": batch, "before": measure_shapes(shapes, self.height)}
            for axis, label in ((0, "after_x"), (1, "after_y")):
                skipped_colliding = 0
                for i in sorted(range(len(ids)), key=lambda k: (shapes[k].bounds[axis], k)):
                    # Binary gap search needs a feasible start. It cannot repair
                    # collisions, and an invalid start is never treated as feasible.
                    if any(_intersection(shapes[i], other) > 0 for j, other in enumerate(shapes) if i != j):
                        skipped_colliding += 1
                        continue
                    box = shapes[i].bounds
                    if box[0] < 0 or box[1] < 0 or box[3] > self.height:
                        continue
                    upper = float(box[axis])
                    if upper <= 0:
                        continue
                    start = translations[i].copy()

                    def trial(distance):
                        candidate_xy = start.copy()
                        candidate_xy[axis] -= distance
                        candidate = _placed(records[i][0], rotation[i], candidate_xy)
                        b = candidate.bounds
                        valid = (b[0] >= 0 and b[1] >= 0 and b[3] <= self.height and
                                 not any(_intersection(candidate, other) > 0
                                         for j, other in enumerate(shapes) if i != j))
                        return valid, candidate_xy, candidate

                    low = 0.
                    accepted_xy, accepted_shape = start, shapes[i]
                    valid, candidate_xy, candidate = trial(upper)
                    if valid:
                        accepted_xy, accepted_shape = candidate_xy, candidate
                    else:
                        # Only the feasible endpoint is committed. A final
                        # unvalidated '+eps' is explicitly forbidden here.
                        for _ in range(64):
                            if upper-low <= GAP_DISTANCE_TOLERANCE:
                                break
                            mid = (low+upper)*.5
                            valid, candidate_xy, candidate = trial(mid)
                            if valid:
                                low = mid
                                accepted_xy, accepted_shape = candidate_xy, candidate
                            else:
                                upper = mid
                    translations[i] = accepted_xy
                    shapes[i] = accepted_shape
                trace[label] = measure_shapes(shapes, self.height)
                trace[label+"_skipped_colliding_pieces"] = skipped_colliding
            traces.append(trace)
        self.last_gap_trace = traces
        return output.tolist()


_default = PaperSATGeometry()
paper_repulsion_all = _default.paper_repulsion_all
paper_gap_xy_all = _default.paper_gap_xy_all
