"""Unquantized rigid geometry for the collision field and the paper's x/y enhancement.

SAT is exact for convex pieces. For concave pieces, convex decomposition and
the union of pairwise Minkowski differences define the forbidden translations.
The shortest exit from that union is the nonconvex counterpart of a SAT MTV.
This concave extension and numerical step control are reconstruction choices,
not a claim to possess the authors' unpublished implementation.
"""
from functools import cached_property
import math

import numpy as np
import shapely
from shapely.affinity import translate
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points, unary_union

CONTACT_MARGIN = 1e-7  # Numerical separation in original coordinate units; never an area tolerance.


class SeparationError(ArithmeticError):
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def convex_parts(shape):
    if not shape.is_valid or shape.area <= 0 or shape.interiors:
        raise ValueError('Expected a valid, nonempty simple contour without holes')
    if shape.equals(shape.convex_hull):
        return [shape]
    parts = list(shapely.constrained_delaunay_triangles(shape).geoms)
    # Merge adjacent triangles only when their exact union remains convex.
    changed = True
    while changed:
        changed = False
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                if not parts[i].touches(parts[j]):
                    continue
                merged = parts[i].union(parts[j])
                if merged.geom_type == 'Polygon' and merged.equals(merged.convex_hull):
                    parts[i] = merged
                    parts.pop(j)
                    changed = True
                    break
            if changed:
                break
    if unary_union(parts).symmetric_difference(shape).area > shape.area * 1e-12:
        raise ValueError('Convex decomposition does not reconstruct the contour')
    return parts


def sat_exit(first, second):
    """Minimum translation of convex first to leave convex second, including containment."""
    best = None
    length = math.inf
    for vertices in (first, second):
        edges = np.roll(vertices, -1, axis=0) - vertices
        for edge in edges:
            norm = np.linalg.norm(edge)
            if norm == 0:
                continue
            axis = np.array([-edge[1], edge[0]]) / norm
            a, b = first @ axis, second @ axis
            if a.max() <= b.min() or b.max() <= a.min():
                return np.zeros(2)
            for distance in (b.min() - a.max(), b.max() - a.min()):
                if abs(distance) < length:
                    length, best = abs(distance), distance * axis
    if best is None:
        raise ValueError('Degenerate SAT input')
    return best


class Geometry:
    def __init__(self, contours, height=1200.):
        self.vertices = [np.asarray(p, dtype=np.float64) for p in contours]
        self.local = [Polygon(p) for p in self.vertices]
        if any(not p.is_valid or p.area <= 0 for p in self.local):
            raise ValueError('Invalid original polygon; no automatic shape repair is permitted')
        self.areas = np.array([p.area for p in self.local])
        self.height = float(height)

    @cached_property
    def parts(self):
        return [[np.asarray(q.exterior.coords)[:-1] for q in convex_parts(p)] for p in self.local]

    def oriented(self, actions):
        return Oriented(self, actions)

    def measure(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (len(self.vertices), 4) or not np.isfinite(actions).all():
            raise ValueError('Invalid action dimensions or nonfinite coordinates')
        shapes = self.oriented(actions).shapes(actions[:, :2])
        bounds = np.asarray([p.bounds for p in shapes])
        overlap = []
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                if bounds[i, 0] < bounds[j, 2] and bounds[j, 0] < bounds[i, 2] and bounds[i, 1] < bounds[j, 3] and bounds[j, 1] < bounds[i, 3]:
                    area = shapes[i].intersection(shapes[j]).area
                    if area > 0:
                        overlap.append((i, j, float(area)))
        span = float(bounds[:, 2].max() - bounds[:, 0].min())
        wall = max(0., float(-bounds[:, 1].min()), float(bounds[:, 3].max() - self.height))
        total = float(self.areas.sum())
        return {'max_pair_overlap_area': max((r[2] for r in overlap), default=0.),
                'sum_pair_overlap_area': float(sum(r[2] for r in overlap)),
                'overlap_ratio_percent': sum(r[2] for r in overlap) / total * 100.,
                'overlap_pairs': overlap, 'height_violation': wall, 'span_x': span,
                'strip_utilization': total / (self.height * span), 'part_area': total,
                'strict_valid': not overlap and wall == 0.}


class Oriented:
    """Fixed rotations, with cached pair configuration spaces during enhancement."""
    def __init__(self, geometry, actions):
        self.geometry = geometry
        self.vertices = []
        self.matrices = []
        for vertices, pose in zip(geometry.vertices, actions):
            theta = math.atan2(float(pose[3]), float(pose[2]))
            c, s = math.cos(theta), math.sin(theta)
            matrix = np.array([[c, -s], [s, c]])
            self.matrices.append(matrix)
            self.vertices.append(vertices @ matrix.T)
        self.base = [Polygon(p) for p in self.vertices]
        self.bounds = np.array([p.bounds for p in self.base])
        self.convex = [p.equals(p.convex_hull) for p in self.base]
        self.forbidden_cache = {}
        self.clearance_cache = {}

    def shapes(self, positions):
        return [Polygon(v + p) for v, p in zip(self.vertices, positions)]

    def forbidden(self, i, j):
        if (i, j) not in self.forbidden_cache:
            polygons = []
            for a in self.geometry.parts[i]:
                a = a @ self.matrices[i].T
                for b in self.geometry.parts[j]:
                    b = b @ self.matrices[j].T
                    sums = (b[:, None, :] - a[None, :, :]).reshape(-1, 2)
                    polygons.append(shapely.convex_hull(shapely.multipoints(sums)))
            self.forbidden_cache[i, j] = unary_union(polygons)
        return self.forbidden_cache[i, j]

    def exit_vector(self, i, j, positions, shapes):
        convex = self.convex[i] and self.convex[j]
        base_vector = sat_exit(self.vertices[i] + positions[i], self.vertices[j] + positions[j]) if convex else None
        trials = []
        # At a nonconvex corner, extending the vector radially can enter a second
        # forbidden lobe. Offset the configuration obstacle, not the part contour,
        # then choose its nearest boundary point. Validate on the original parts.
        for multiplier in (1., 4., 16., 64.):
            margin = CONTACT_MARGIN * multiplier
            if convex:
                norm = np.linalg.norm(base_vector)
                if not norm > 0:
                    raise ArithmeticError('Positive real intersection but no separating vector')
                candidate = base_vector * (1 + margin / norm)
            else:
                key = (i, j, multiplier)
                if key not in self.clearance_cache:
                    self.clearance_cache[key] = self.forbidden(i, j).buffer(margin, join_style='mitre')
                relative = positions[i] - positions[j]
                closest = nearest_points(Point(relative), self.clearance_cache[key].boundary)[1]
                candidate = np.asarray(closest.coords[0]) - relative
            remaining = translate(shapes[i], *candidate).intersection(shapes[j]).area
            trials.append({'multiplier': multiplier, 'vector': candidate.tolist(), 'intersection': remaining})
            if remaining == 0:
                return candidate
        raise SeparationError('Minimum separation failed full-contour validation', {
            'pair': [int(i), int(j)], 'convex': [self.convex[i], self.convex[j]],
            'oriented_vertices': [self.vertices[i].tolist(), self.vertices[j].tolist()],
            'positions': [positions[i].tolist(), positions[j].tolist()],
            'trials': trials, 'initial_intersection': shapes[i].intersection(shapes[j]).area})

    def field(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        shapes = self.shapes(positions)
        bounds = self.bounds + np.tile(positions, (1, 2))
        areas = self.geometry.areas
        mass = areas / areas.sum()
        forces = np.zeros_like(positions)
        energy = 0.
        collisions = 0
        for i, bound in enumerate(bounds):
            correction = 0.
            if bound[1] < 0:
                correction = -bound[1] + CONTACT_MARGIN
            if bound[3] > self.geometry.height:
                correction += self.geometry.height - bound[3] - CONTACT_MARGIN
            forces[i, 1] += correction
            energy += mass[i] * correction**2 / 2
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                if bounds[i, 0] >= bounds[j, 2] or bounds[j, 0] >= bounds[i, 2] or bounds[i, 1] >= bounds[j, 3] or bounds[j, 1] >= bounds[i, 3]:
                    continue
                if shapes[i].intersection(shapes[j]).area == 0:
                    continue
                vector = self.exit_vector(i, j, positions, shapes)
                forces[i] += areas[j] / (areas[i] + areas[j]) * vector
                forces[j] -= areas[i] / (areas[i] + areas[j]) * vector
                energy += areas[i] * areas[j] / ((areas[i] + areas[j]) * areas.sum()) * float(vector @ vector) / 2
                collisions += 1
        return forces, float(energy), collisions

    def gap_sweep(self, actions):
        """Try -x/-y moves even from overlap; accept only legal full-contour endpoints."""
        result = np.array(actions, dtype=np.float64, copy=True)
        shapes = self.shapes(result[:, :2])
        bounds = np.asarray([p.bounds for p in shapes])

        def legal(candidate, index):
            box = candidate.bounds
            if box[1] < 0 or box[3] > self.geometry.height:
                return False
            mask = (bounds[:, 0] < box[2]) & (bounds[:, 2] > box[0]) & (bounds[:, 1] < box[3]) & (bounds[:, 3] > box[1])
            mask[index] = False
            return all(candidate.intersection(shapes[j]).area == 0 for j in np.flatnonzero(mask))

        # x=0 is just the global translation gauge of a horizontally unbounded strip.
        shift = bounds[:, 0].min()
        result[:, 0] -= shift
        shapes = self.shapes(result[:, :2])
        bounds = np.asarray([p.bounds for p in shapes])
        moves = 0
        for axis in (0, 1):
            order = np.argsort(bounds[:, axis], kind='stable')
            for i in order:
                # An overlapping start may have a legal compressed endpoint.
                # This is layout optimization, not collision-free path planning.
                # Every accepted endpoint still passes legal() below.
                maximum = max(0., bounds[i, axis])
                if maximum <= CONTACT_MARGIN:
                    continue
                original = result[i, :2].copy()

                def at(distance):
                    position = original.copy()
                    position[axis] -= distance
                    return Polygon(self.vertices[i] + position)

                if legal(at(maximum), i):
                    # Exact wall contact can become a tiny violation under a different
                    # evaluation order. Keep a distance margin, never an area threshold.
                    low = max(0., maximum - CONTACT_MARGIN)
                else:
                    low, high = 0., maximum
                    for _ in range(36):
                        if high - low < CONTACT_MARGIN:
                            break
                        middle = (low + high) / 2
                        if legal(at(middle), i):
                            low = middle
                        else:
                            high = middle
                    low = max(0., low - CONTACT_MARGIN)
                candidate = at(low)
                if low > 0 and legal(candidate, i):
                    result[i, axis] -= low
                    shapes[i] = candidate
                    bounds[i] = candidate.bounds
                    moves += 1
        return result, moves


def enhance(geometry, actions, iterations=100, initial_step=1.):
    """Area-weighted collision descent plus -x/-y enhancement; never a feasibility guarantee."""
    if not 0 <= iterations <= 100 or initial_step <= 0 or not math.isfinite(initial_step):
        raise ValueError('Paper enhancement budget must be in [0,100]; step must be positive')
    result = np.array(actions, dtype=np.float64, copy=True)
    oriented = geometry.oriented(result)
    trace = []
    status = 'iteration_limit'
    mass = geometry.areas / geometry.areas.sum()
    for iteration in range(iterations):
        previous = result[:, :2].copy()
        force, energy, collisions = oriented.field(result[:, :2])
        norm = float(np.sum(mass[:, None] * force**2))
        accepted, step = False, initial_step
        if norm > 0:
            for trial in range(24):
                candidate = result.copy()
                candidate[:, :2] += step * force
                _, trial_energy, _ = oriented.field(candidate[:, :2])
                if trial_energy <= energy - 1e-4 * step * norm:
                    result, accepted = candidate, True
                    break
                step *= .5
        result, moves = oriented.gap_sweep(result)
        metrics = geometry.measure(result)
        trace.append({'iteration': iteration + 1, 'energy_before': energy,
                      'collision_pairs_before': collisions, 'force_norm_squared': norm,
                      'step': step if accepted else 0., 'collision_move_accepted': accepted,
                      'gap_moves': moves, **{k: v for k, v in metrics.items() if k != 'overlap_pairs'}})
        if metrics['strict_valid'] and np.linalg.norm(result[:, :2] - previous, axis=1).max() <= 1e-6:
            status = 'strict_valid'
            break
        if not accepted and moves == 0:
            status = 'stalled_infeasible'
            break
    if geometry.measure(result)['strict_valid']:
        status = 'strict_valid'
    return result, {'status': status, 'iterations': len(trace), 'trace': trace,
                    'metrics': geometry.measure(result)}
