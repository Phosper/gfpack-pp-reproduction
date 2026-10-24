"""Translation-only concave repair and compaction, checked in stored float32.

This is an independent engineering alternative to the released postprocessor,
not a reconstruction of an unpublished paper algorithm. It requires Shapely 2.1.
"""
import math
import time

import numpy as np
import shapely
from shapely.geometry import Polygon

from translation_intervals import decompose, pair_intervals, merge_intervals


class BudgetExpired(Exception):
    pass


class CompactLayout:
    def __init__(self, polygons, ids, actions, height=1200., seconds=45.):
        self.started = time.perf_counter()
        self.seconds = seconds
        self.height = float(height)
        self.actions = np.asarray(actions, dtype=np.float32).astype(np.float64)
        if self.actions.shape != (len(ids), 4) or not len(ids):
            raise ValueError('Expected one four-component action per id')
        if not np.isfinite(self.actions).all() or not math.isfinite(height) or height <= 0:
            raise ValueError('Nonfinite actions or invalid height')
        self.base = []
        for i, pid in enumerate(ids):
            angle = math.atan2(self.actions[i, 3], self.actions[i, 2])
            c, s = math.cos(angle), math.sin(angle)
            p = Polygon([(x*c-y*s, x*s+y*c) for x, y in polygons[pid]])
            if not p.is_valid or p.is_empty or p.area <= 0:
                raise ValueError('Invalid polygon')
            self.base.append(p)
        self.vertices = [np.asarray(p.exterior.coords)[:-1] for p in self.base]
        self.triangles = [decompose(p) for p in self.base]
        self.base_bounds = np.array([p.bounds for p in self.base])
        self.positions = self.actions[:, :2].copy()
        self.y_limits = []
        for p in self.base:
            lo, hi = -p.bounds[1], self.height-p.bounds[3]
            lower, upper = np.float32(lo), np.float32(hi)
            if float(lower) < lo:
                lower = np.nextafter(lower, np.float32(np.inf))
            if float(upper) > hi:
                upper = np.nextafter(upper, np.float32(-np.inf))
            if lower > upper:
                raise ValueError('Piece cannot fit in strip at this stored rotation')
            self.y_limits.append((float(lower), float(upper)))
        self.geoms = np.array([self.placed(i, xy) for i, xy in enumerate(self.positions)], dtype=object)
        self.moves = []
        self.interval_calls = 0
        self.candidate_checks = 0
        self.boundary_moves = 0
        self.repair_status = 'not_started'
        self.compaction_status = 'not_started'

    def check_budget(self):
        if time.perf_counter()-self.started > self.seconds:
            raise BudgetExpired

    def placed(self, i, xy):
        return Polygon(self.vertices[i]+np.asarray(xy))

    def bounds(self):
        return self.base_bounds + np.tile(self.positions, (1, 2))

    def span(self, i=None, xy=None):
        b = self.bounds()
        if i is not None:
            b[i] = self.base_bounds[i]+np.tile(xy, 2)
        return float(b[:, 2].max()-b[:, 0].min())

    def checked_candidate(self, i, xy):
        self.check_budget()
        self.candidate_checks += 1
        xy = np.asarray(xy, dtype=np.float32).astype(np.float64)
        if not np.isfinite(xy).all():
            return None
        lo, hi = self.y_limits[i]
        if not lo <= xy[1] <= hi:
            return None
        geom = self.placed(i, xy)
        if geom.bounds[1] < -1e-9 or geom.bounds[3] > self.height+1e-9:
            return None
        mask = np.arange(len(self.geoms)) != i
        # Validate the exact stored endpoint against every other true contour.
        areas = shapely.area(shapely.intersection(geom, self.geoms[mask]))
        if not np.isfinite(areas).all() or (areas > 1e-9).any():
            return None
        return xy, geom

    def accept(self, i, candidate, phase):
        xy, geom = candidate
        before = self.positions[i].copy()
        old_span = self.span()
        self.positions[i] = xy
        self.geoms[i] = geom
        self.moves.append({'phase': phase, 'piece_index': int(i),
                           'before_xy': before.tolist(), 'after_xy': xy.tolist(),
                           'span_before': old_span, 'span_after': self.span()})

    def overlapping_pieces(self):
        bad = set()
        for i in range(len(self.geoms)):
            self.check_budget()
            areas = shapely.area(shapely.intersection(self.geoms[i], self.geoms[i+1:]))
            for off in np.flatnonzero(areas > 1e-9):
                bad.update((i, i+1+int(off)))
        return sorted(bad)

    def intervals(self, i, direction, lo, hi):
        self.check_budget()
        d = np.asarray(direction, dtype=float)
        tri = self.triangles[i]+self.positions[i]
        b = self.bounds()
        # Conservative swept AABB excludes remote, unreachable obstacles.
        low = b[i, :2]+np.minimum(lo*d, hi*d)
        high = b[i, 2:]+np.maximum(lo*d, hi*d)
        intervals = []
        for j in range(len(b)):
            self.check_budget()
            if j == i or np.any(high < b[j, :2]) or np.any(b[j, 2:] < low):
                continue
            self.interval_calls += 1
            for a, z in pair_intervals(tri, self.triangles[j]+self.positions[j], d):
                if z >= lo and a <= hi:
                    # Clearance covers float32 rounding; final geometry, not
                    # this buffer, decides acceptance. Thresholds stay fixed.
                    gap = max(.002, 8*abs(float(np.spacing(np.float32(
                        max(1., np.abs(self.positions).max()))))))
                    intervals.append((a-gap, z+gap))
        return merge_intervals(intervals)

    def travel(self, i, direction, radius):
        dy = direction[1]
        lo, hi = -radius, radius
        if abs(dy) > 1e-12:
            a, b = [(v-self.positions[i, 1])/dy for v in self.y_limits[i]]
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
        return lo, hi

    def repair(self):
        # Explicitly correct representational boundary violations, then repair
        # any interactions this creates. No tolerance enlargement or clipping.
        for i, (lo, hi) in enumerate(self.y_limits):
            y = self.positions[i, 1]
            if not lo <= y <= hi:
                self.positions[i, 1] = min(hi, max(lo, y))
                self.geoms[i] = self.placed(i, self.positions[i])
                self.boundary_moves += 1
        directions = [(math.cos(a*math.pi/4), math.sin(a*math.pi/4)) for a in range(4)]
        for iteration in range(96):
            bad = self.overlapping_pieces()
            if not bad:
                self.repair_status = 'strict_feasible'
                return
            best = None
            old_span = self.span()
            for radius in (16., 64., 256., 1024., 4096.):
                for i in bad:
                    for d in directions:
                        lo, hi = self.travel(i, d, radius)
                        if lo > hi:
                            continue
                        forbidden = self.intervals(i, d, lo, hi)
                        values = {0.}
                        for a, z in forbidden:
                            if lo <= a <= hi:
                                values.add(a)
                            if lo <= z <= hi:
                                values.add(z)
                        # Only the nearest free endpoints on either side are
                        # needed; skip boundaries still covered by an interval.
                        candidates = []
                        for sign in (-1, 1):
                            ordered = sorted((t for t in values if t*sign >= 0), key=abs)
                            for t in ordered:
                                if any(a < t < z for a, z in forbidden):
                                    continue
                                candidates.append(t)
                                break
                        for t in candidates:
                            if abs(t) < 1e-9:
                                continue
                            candidate = self.checked_candidate(i, self.positions[i]+t*np.asarray(d))
                            if candidate is None:
                                continue
                            growth = max(0., self.span(i, candidate[0])-old_span)
                            motion = float(np.linalg.norm(candidate[0]-self.positions[i]))
                            score = (motion+1000*growth, growth, motion, i)
                            if best is None or score < best[0]:
                                best = score, i, candidate
                if best is not None:
                    break
            if best is None:
                self.repair_status = 'stalled'
                return
            self.accept(best[1], best[2], 'local_repair')
        self.repair_status = 'iteration_limit'

    def compact(self, sweeps=4):
        if self.repair_status != 'strict_feasible':
            self.compaction_status = 'skipped_infeasible'
            return
        for sweep in range(sweeps):
            side = -1 if sweep % 2 == 0 else 1
            b = self.bounds()
            order = np.argsort(b[:, 0] if side < 0 else -b[:, 2], kind='stable')
            for i in order:
                self.check_budget()
                b = self.bounds()
                lo, hi = float(b[:, 0].min()-b[i, 0]), float(b[:, 2].max()-b[i, 2])
                lo, hi = (lo, 0.) if side < 0 else (0., hi)
                forbidden = self.intervals(i, (1., 0.), lo, hi)
                target = lo if side < 0 else hi
                for a, z in (forbidden if side < 0 else reversed(forbidden)):
                    if a <= target <= z:
                        target = z if side < 0 else a
                if not lo <= target <= hi or abs(target) < 1e-6:
                    continue
                old_span = self.span()
                candidate = self.checked_candidate(i, self.positions[i]+[target, 0.])
                if candidate is not None and self.span(i, candidate[0]) <= old_span+1e-9:
                    self.accept(i, candidate, 'compact_left' if side < 0 else 'compact_right')
        self.compaction_status = 'completed'

    def output(self):
        self.actions[:, :2] = self.positions
        return self.actions.tolist()


def optimize(polygons, ids, actions, height=1200., seconds=45., sweeps=4):
    layout = CompactLayout(polygons, ids, actions, height, seconds)
    repair_actions = None
    try:
        layout.repair()
        repair_actions = layout.output()
        layout.compact(sweeps)
    except BudgetExpired:
        if layout.repair_status == 'not_started':
            layout.repair_status = 'time_limit'
        else:
            layout.compaction_status = 'time_limit'
    if repair_actions is None:
        repair_actions = layout.output()
    return {'actions': layout.output(), 'repair_actions': repair_actions,
            'repair_status': layout.repair_status,
            'compaction_status': layout.compaction_status,
            'moves': layout.moves, 'boundary_adjustments': layout.boundary_moves,
            'interval_calls': layout.interval_calls,
            'candidate_checks': layout.candidate_checks,
            'elapsed_seconds': time.perf_counter()-layout.started,
            'storage_precision': 'float32_at_every_accepted_endpoint',
            'rotations': 'single float32 cast, then fixed throughout optimization',
            'failure_policy': 'return measured partial layout with explicit status; never remove input'}
