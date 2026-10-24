"""Continuous forbidden translations for fixed-orientation polygon packing.

An interval ``(lo, hi)`` is open: translating the moving polygon by
``t * direction`` causes *positive area* overlap precisely inside the union
of the returned intervals, apart from ordinary floating-point roundoff.
Boundary contact is allowed.  Concave polygons and holes are represented by
their constrained triangulation, never by a convex hull.  Accepted moves must
still be checked independently against the original polygons by the caller.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.validation import explain_validity


def _polygon(value: Polygon | Sequence[Sequence[float]]) -> Polygon:
    poly = value if isinstance(value, Polygon) else Polygon(value)
    if poly.is_empty or not poly.is_valid or poly.area <= 0:
        raise ValueError(f"Expected a nonempty valid positive-area Polygon: {explain_validity(poly)}")
    coordinates = shapely.get_coordinates(poly)
    if not np.isfinite(coordinates).all():
        raise ValueError("Polygon coordinates must be finite")
    return poly


def decompose(poly: Polygon | Sequence[Sequence[float]]) -> np.ndarray:
    """Return verified constrained triangles as an ``(n, 3, 2)`` array.

    Requires Shapely's constrained Delaunay operation (Shapely >= 2.1 with
    an appropriate GEOS).  Its triangles must cover the exact source polygon,
    including its holes.  Invalid source geometry is rejected, not repaired.
    """
    poly = _polygon(poly)
    if not hasattr(shapely, "constrained_delaunay_triangles"):
        raise RuntimeError("Constrained triangulation requires Shapely >= 2.1")
    collection = shapely.constrained_delaunay_triangles(poly)
    pieces = list(collection.geoms)
    if not pieces:
        raise ValueError("Constrained triangulation produced no triangles")
    triangles = []
    for piece in pieces:
        coords = np.asarray(piece.exterior.coords, dtype=np.float64)[:-1]
        if coords.shape != (3, 2) or piece.interiors or piece.area <= 0:
            raise ValueError("Constrained triangulation produced a non-triangle")
        if not poly.covers(piece):
            raise ValueError("A constrained triangle extends outside the source polygon")
        triangles.append(coords)
    coverage = shapely.union_all(pieces)
    if not coverage.equals(poly):
        raise ValueError("Constrained triangles do not exactly cover the source polygon")
    total = math.fsum(piece.area for piece in pieces)
    # This is only an error bound for area summation.  Geometry coverage above
    # uses topological equality, without changing an overlap/feasibility limit.
    if not math.isclose(total, poly.area, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("Constrained triangles have duplicate or missing area")
    return np.asarray(triangles, dtype=np.float64)


def merge_intervals(intervals: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    """Union overlapping open intervals, preserving isolated contact points.

    In particular, ``(a, b)`` and ``(b, c)`` remain separate: at ``b`` neither
    open interval forbids the translation.  No epsilon joins nearby endpoints.
    """
    ordered = []
    for pair in intervals:
        if len(pair) != 2:
            raise ValueError("Each interval must contain two endpoints")
        low, high = map(float, pair)
        if math.isnan(low) or math.isnan(high):
            raise ValueError("Interval endpoints must not be NaN")
        if low > high:
            raise ValueError("Interval endpoints are reversed")
        if low < high:
            ordered.append((low, high))
    ordered.sort()
    merged: list[tuple[float, float]] = []
    for low, high in ordered:
        if merged and low < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
        else:
            merged.append((low, high))
    return merged


def _triangles(value: np.ndarray) -> np.ndarray:
    triangles = np.asarray(value, dtype=np.float64)
    if triangles.shape == (3, 2):
        triangles = triangles[None, :, :]
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2):
        raise ValueError("Triangles must have shape (3, 2) or (n, 3, 2)")
    if not np.isfinite(triangles).all():
        raise ValueError("Triangle coordinates must be finite")
    if len(triangles):
        edges = triangles[:, 1:, :] - triangles[:, :1, :]
        twice_area = edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
        if np.any(twice_area == 0):
            raise ValueError("Triangles must have positive area")
    return triangles


def _direction(value: Sequence[float]) -> np.ndarray:
    direction = np.asarray(value, dtype=np.float64)
    if direction.shape != (2,) or not np.isfinite(direction).all():
        raise ValueError("Direction must contain two finite values")
    if not math.isclose(float(np.linalg.norm(direction)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Direction must be a unit vector")
    return direction


def pair_intervals(
    tri_a: np.ndarray,
    tri_b: np.ndarray,
    direction: Sequence[float],
    *,
    batch_pairs: int = 32768,
) -> list[tuple[float, float]]:
    """Return open overlap intervals for A translated past stationary B.

    Either argument can contain one triangle or a complete triangulation.
    Every triangle pair uses all six edge normals in the strict separating
    axis test.  Taking their union gives the concave polygon result.  Work is
    vectorized in bounded batches, with no sampling of translation distance.
    """
    a, b, direction = _triangles(tri_a), _triangles(tri_b), _direction(direction)
    if not isinstance(batch_pairs, int) or batch_pairs <= 0:
        raise ValueError("batch_pairs must be a positive integer")
    if not len(a) or not len(b):
        return []
    edges_a = np.roll(a, -1, axis=1) - a
    edges_b = np.roll(b, -1, axis=1) - b
    normals_a = np.stack((-edges_a[:, :, 1], edges_a[:, :, 0]), axis=-1)
    normals_b = np.stack((-edges_b[:, :, 1], edges_b[:, :, 0]), axis=-1)
    perpendicular = np.array([-direction[1], direction[0]])
    perpendicular_a = np.einsum("nvc,c->nv", a, perpendicular)
    perpendicular_b = np.einsum("nvc,c->nv", b, perpendicular)
    sweep_min_a, sweep_max_a = perpendicular_a.min(axis=1), perpendicular_a.max(axis=1)
    sweep_min_b, sweep_max_b = perpendicular_b.min(axis=1), perpendicular_b.max(axis=1)
    intervals: list[tuple[float, float]] = []
    n_pairs = len(a) * len(b)
    for start in range(0, n_pairs, batch_pairs):
        flat = np.arange(start, min(start + batch_pairs, n_pairs))
        ia, ib = flat // len(b), flat % len(b)
        # Translation leaves the perpendicular projection unchanged.  A pair
        # that is disjoint (or only touching) there cannot overlap at any t.
        possible = (sweep_max_a[ia] > sweep_min_b[ib]) & (sweep_max_b[ib] > sweep_min_a[ia])
        ia, ib = ia[possible], ib[possible]
        if not len(ia):
            continue
        # Shift both triangles by the same origin to reduce cancellation in
        # projections.  Relative position and every forbidden t are unchanged.
        origin = a[ia, :1, :]
        aa, bb = a[ia] - origin, b[ib] - origin
        axes = np.concatenate((normals_a[ia], normals_b[ib]), axis=1)
        projected_a = np.einsum("pvc,pnc->pnv", aa, axes)
        projected_b = np.einsum("pvc,pnc->pnv", bb, axes)
        amin, amax = projected_a.min(axis=2), projected_a.max(axis=2)
        bmin, bmax = projected_b.min(axis=2), projected_b.max(axis=2)
        speed = np.einsum("pnc,c->pn", axes, direction)
        moving_axis = speed != 0.0
        # amin + speed*t < bmax AND amax + speed*t > bmin.
        first = np.full_like(speed, -np.inf)
        second = np.full_like(speed, np.inf)
        np.divide(bmin - amax, speed, out=first, where=moving_axis)
        np.divide(bmax - amin, speed, out=second, where=moving_axis)
        low = np.where(moving_axis, np.minimum(first, second), -np.inf)
        high = np.where(moving_axis, np.maximum(first, second), np.inf)
        stationary_separated = (~moving_axis) & ((amax <= bmin) | (bmax <= amin))
        lower, upper = low.max(axis=1), high.min(axis=1)
        valid = (~stationary_separated.any(axis=1)) & (lower < upper)
        intervals.extend(zip(lower[valid].tolist(), upper[valid].tolist()))
    return merge_intervals(intervals)


@dataclass(frozen=True)
class ForbiddenTranslations:
    """Cache source triangulations while polygon translations change."""

    triangles: tuple[np.ndarray, ...]

    @classmethod
    def from_polygons(cls, polygons: Sequence[Polygon | Sequence[Sequence[float]]]) -> "ForbiddenTranslations":
        triangles = tuple(decompose(poly) for poly in polygons)
        for item in triangles:
            item.setflags(write=False)
        return cls(triangles)

    def intervals(
        self,
        moving_index: int,
        direction: Sequence[float],
        positions: np.ndarray | None = None,
        *,
        blocker_indices: Iterable[int] | None = None,
    ) -> list[tuple[float, float]]:
        """Compute forbidden t against selected current fixed polygons.

        ``positions[i]`` is an XY translation relative to the polygon supplied
        to ``from_polygons``.  It is not the polygon's absolute anchor.  There
        are no walls here; strip-height limits belong to the caller.
        """
        n = len(self.triangles)
        if not 0 <= moving_index < n:
            raise IndexError("moving_index is out of range")
        shifts = np.zeros((n, 2)) if positions is None else np.asarray(positions, dtype=np.float64)
        if shifts.shape != (n, 2) or not np.isfinite(shifts).all():
            raise ValueError("positions must be a finite (number_of_polygons, 2) array")
        indices = range(n) if blocker_indices is None else blocker_indices
        moving = self.triangles[moving_index] + shifts[moving_index]
        intervals = []
        for fixed_index in indices:
            if not 0 <= fixed_index < n:
                raise IndexError("A blocker index is out of range")
            if fixed_index != moving_index:
                intervals.extend(pair_intervals(moving, self.triangles[fixed_index] + shifts[fixed_index], direction))
        return merge_intervals(intervals)


def self_test() -> dict[str, int | str]:
    """Bounded controls, including random independent GEOS area comparisons."""
    from shapely.affinity import translate
    from shapely.geometry import box

    controls = 0
    a = decompose(box(0, 0, 1, 1))
    b = decompose(box(2, 0, 3, 1))
    assert pair_intervals(a, b, (1, 0)) == [(1.0, 3.0)]
    assert pair_intervals(a, b, (-1, 0)) == [(-3.0, -1.0)]
    assert pair_intervals(a, decompose(box(0, 2, 1, 3)), (0, 1)) == [(1.0, 3.0)]
    assert pair_intervals(a, decompose(box(2, 1, 3, 2)), (1, 0)) == []
    controls += 4
    assert merge_intervals([(1, 3), (3, 5), (1.5, 2), (2, 2)]) == [(1.0, 3.0), (3.0, 5.0)]
    for t in (1.0, 3.0):
        assert translate(box(0, 0, 1, 1), xoff=t).intersection(box(2, 0, 3, 1)).area == 0.0
    controls += 2
    u = Polygon([(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4)])
    inner = box(1.25, 1.25, 2.75, 2.75)
    nested = pair_intervals(decompose(inner), decompose(u), (1, 0))
    assert nested == [(-2.75, -0.25), (0.25, 2.75)]
    assert inner.intersection(u).area == 0.0
    controls += 1
    hole = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)], holes=[[(1, 1), (1, 4), (4, 4), (4, 1)]])
    assert not any(lo < 0 < hi for lo, hi in pair_intervals(decompose(box(2, 2, 3, 3)), decompose(hole), (1, 0)))
    controls += 1
    cache = ForbiddenTranslations.from_polygons([box(0, 0, 1, 1), box(2, 0, 3, 1), box(5, 0, 6, 1)])
    assert cache.intervals(0, (1, 0)) == [(1.0, 3.0), (4.0, 6.0)]
    assert cache.intervals(0, (1, 0), np.array([[1, 0], [0, 0], [0, 0]])) == [(0.0, 2.0), (3.0, 5.0)]
    controls += 2
    rng = np.random.default_rng(20261024)
    comparisons = 0
    for trial in range(16):
        angles = np.arange(12) * (2 * np.pi / 12)
        radii_a, radii_b = rng.uniform(0.3, 1.8, (2, 12))
        pa = Polygon(np.column_stack((np.cos(angles) * radii_a, np.sin(angles) * radii_a)))
        pb = Polygon(np.column_stack((np.cos(angles) * radii_b + 2.0, np.sin(angles) * radii_b + 0.4)))
        theta = rng.uniform(-np.pi, np.pi)
        direction = (math.cos(theta), math.sin(theta))
        forbidden = pair_intervals(decompose(pa), decompose(pb), direction, batch_pairs=7)
        for t in rng.uniform(-6.0, 6.0, 64):
            area = translate(pa, xoff=t * direction[0], yoff=t * direction[1]).intersection(pb).area
            assert any(lo < t < hi for lo, hi in forbidden) == (area > 0.0), (trial, t, area, forbidden)
            comparisons += 1
    return {"status": "passed", "controls": controls, "random_geos_comparisons": comparisons}


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), sort_keys=True))
