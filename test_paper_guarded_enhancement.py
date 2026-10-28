"""Geometry-based controls for the optional post-generation acceptance guard.

Run with ``python -m unittest -v test_paper_guarded_enhancement``.  The small
adapter below uses actual polygon intersections and the current area-weighted
SAT displacement.  It substitutes for the unavailable compiled extension, not
for GEOS acceptance checks.  These controls are not a Dental benchmark or a
verification of the compiled release's quantization/gap-search implementation.
"""
import math
import unittest

import numpy as np
from shapely.geometry import Polygon

from paper_enhancement import enhance


def rectangle(width=1., height=1.):
    return [[0., 0.], [width, 0.], [width, height], [0., height]]


def actions(*layouts):
    return np.asarray([[[x, y, 1., 0.] for x, y in row]
                       for row in layouts], dtype=np.float64)


def place(polygons, ids, rotations, translations):
    result = []
    for idx, angle, (tx, ty) in zip(ids, rotations, translations):
        c, s = math.cos(angle), math.sin(angle)
        result.append(Polygon([(x*c-y*s+tx, x*s+y*c+ty)
                               for x, y in polygons[idx]]))
    return result


def measure(polygons, ids, layout, height):
    shapes = place(polygons, ids, np.arctan2(layout[:, 3], layout[:, 2]),
                   layout[:, :2])
    overlaps = [p.intersection(q).area for i, p in enumerate(shapes)
                for q in shapes[i+1:]]
    bounds = [min(p.bounds[0] for p in shapes), min(p.bounds[1] for p in shapes),
              max(p.bounds[2] for p in shapes), max(p.bounds[3] for p in shapes)]
    return {"sum_pairwise_area": math.fsum(overlaps),
            "max_pairwise_area": max(overlaps, default=0.),
            "height_violation": max(0., -bounds[1], bounds[3]-height),
            "bounds_xyxy": bounds, "span_x": bounds[2]-bounds[0]}


def sat_displacement(a, b):
    vertices = [np.asarray(p.exterior.coords[:-1], dtype=np.float64) for p in (a, b)]
    axes = [np.array([1., 0.]), np.array([0., 1.])]
    for poly in vertices:
        for first, second in zip(poly, np.roll(poly, -1, axis=0)):
            edge = second-first
            normal = np.array([-edge[1], edge[0]])
            length = np.linalg.norm(normal)
            if length > 1e-12:
                axes.append(normal/length)
    best, vector = math.inf, np.zeros(2)
    for axis in axes:
        p, q = (v @ axis for v in vertices)
        if p.max() <= q.min() or q.max() <= p.min():
            return np.zeros(2)
        for shift in (q.max()-p.min(), q.min()-p.max()):
            if abs(shift) < best:
                best, vector = abs(shift), axis*shift
    return vector


class GeometryNative:
    """Small real-geometry analogue; identity gap isolates collision updates.

    ``miss_below`` models a native intersection detector missing tiny areas;
    ``gap`` optionally proposes intentionally unsafe compression moves.
    """
    def __init__(self, *, miss_below=0., gap=None):
        self.miss_below, self.gap = miss_below, gap

    def paper_repulsion_all(self, ids, rotates, translations, polygons,
                            height=1200., collision_area_eps=0.):
        result = {key: [] for key in ("repulsion", "max_pairwise_area",
                  "sum_pairwise_area", "colliding_pairs", "concave_proxy_pairs",
                  "height_violation", "min_x")}
        for rotation, xy in zip(rotates, translations):
            placed = place(polygons, ids, rotation, xy)
            force = np.zeros((len(ids), 2))
            areas = np.asarray([p.area for p in placed])
            maximum, overlap_sum, count, concave_count, violation = 0., 0., 0, 0, 0.
            for i, shape in enumerate(placed):
                below, above = max(0., -shape.bounds[1]), max(0., shape.bounds[3]-height)
                force[i, 1] += below-above
                violation = max(violation, below, above)
                for j in range(i+1, len(ids)):
                    area = shape.intersection(placed[j]).area
                    if area <= self.miss_below:
                        area = 0.
                    maximum, overlap_sum = max(maximum, area), overlap_sum+area
                    if area <= collision_area_eps:
                        continue
                    count += 1
                    concave_count += int(not shape.equals(shape.convex_hull) or
                                         not placed[j].equals(placed[j].convex_hull))
                    displacement = sat_displacement(shape, placed[j])
                    force[i] += areas[j]/(areas[i]+areas[j])*displacement
                    force[j] -= areas[i]/(areas[i]+areas[j])*displacement
            for key, value in (("repulsion", force.tolist()),
                               ("max_pairwise_area", maximum),
                               ("sum_pairwise_area", overlap_sum),
                               ("colliding_pairs", count),
                               ("concave_proxy_pairs", concave_count),
                               ("height_violation", violation),
                               ("min_x", min(p.bounds[0] for p in placed))):
                result[key].append(value)
        return result

    def paper_gap_xy_all(self, ids, rotates, translations, polygons):
        result = np.array(translations, dtype=np.float64, copy=True)
        if self.gap is not None:
            result = self.gap(result)
        return result.tolist()


class GuardedEnhancementTests(unittest.TestCase):
    def guarded(self, native, polygons, ids, batch, iterations=10, height=1200.):
        return enhance(native, polygons, ids, batch, iterations, height,
                       policy="guarded_backtracking")

    def assert_not_worse(self, before, after):
        for key in ("sum_pairwise_area", "max_pairwise_area", "height_violation"):
            self.assertLessEqual(after[key], before[key]+1e-9, key)

    def test_area_weighting_and_two_piece_separation(self):
        polygons, ids = [rectangle(), rectangle(2.)], [0, 1]
        batch = actions([(2., 2.), (2.5, 2.)])
        original = batch.copy()
        field = GeometryNative().paper_repulsion_all(ids, [[0., 0.]],
                    batch[:, :, :2].tolist(), polygons)
        force = np.asarray(field["repulsion"])[0]
        np.testing.assert_allclose(force[0]+2.*force[1], 0., atol=1e-12)
        result, diagnostic = self.guarded(GeometryNative(), polygons, ids, batch)
        # The 1/3 area weight can leave a 4e-16 roundoff sliver. Preserve the
        # measured value; the project criterion is <=1e-9, not exact arithmetic.
        self.assertLessEqual(measure(polygons, ids, result[0], 1200.)["max_pairwise_area"], 1e-9)
        np.testing.assert_array_equal(batch, original)
        np.testing.assert_array_equal(result[:, :, 2:], original[:, :, 2:])
        self.assertEqual(diagnostic["statuses"], ["reference_converged"])

    def test_backtracking_prevents_boundary_damage_without_inventing_new_direction(self):
        polygons, ids = [rectangle(2.)], [0, 0]
        batch = actions([(0., .2), (0., .3)])
        initial = measure(polygons, ids, batch[0], 1.4)
        legacy, _ = enhance(GeometryNative(), polygons, ids, batch, 1, 1.4)
        self.assertGreater(measure(polygons, ids, legacy[0], 1.4)["height_violation"], .3)
        result, _ = self.guarded(GeometryNative(), polygons, ids, batch, 1, 1.4)
        checked = measure(polygons, ids, result[0], 1.4)
        self.assert_not_worse(initial, checked)
        self.assertLess(checked["sum_pairwise_area"], initial["sum_pairwise_area"])
        self.assertGreater(checked["max_pairwise_area"], 0.)
        np.testing.assert_allclose(result[0, :, 0], 0.)

    def test_all_steps_rejected_keeps_original_and_reports_stall(self):
        polygons, ids = [rectangle(2.)], [0, 0]
        batch = actions([(0., 0.), (0., .1)])
        result, diagnostic = self.guarded(GeometryNative(), polygons, ids, batch, 10, 1.4)
        np.testing.assert_array_equal(result, batch)
        self.assertEqual(diagnostic["statuses"], ["stalled_infeasible"])
        self.assertGreater(measure(polygons, ids, result[0], 1.4)["max_pairwise_area"], 0.)

    def test_multiple_collisions_cannot_hide_a_worse_pair_in_a_better_total(self):
        # Fixed five-square example: full SAT step lowers total overlap from
        # 1.3445 to .4283 but worsens the largest pair from .3448 to .4100.
        # A half step improves BOTH (.7888 total, .3194 largest pair).
        xy = [(0.8467922705606181, 1.2870597600169769),
              (1.1508810047341018, 1.7915284554424962),
              (0.1795317142031081, 0.033737090004868575),
              (0.4772487283235598, 1.826346288561647),
              (0.7989051896265846, 0.47944211928654124)]
        polygons, ids, batch = [rectangle()], [0]*5, actions(xy)
        initial = measure(polygons, ids, batch[0], 1200.)
        field = GeometryNative().paper_repulsion_all(ids, [[0.]*5],
                    batch[:, :, :2].tolist(), polygons)
        full = batch.copy()
        full[:, :, :2] += np.asarray(field["repulsion"])
        full_metric = measure(polygons, ids, full[0], 1200.)
        self.assertLess(full_metric["sum_pairwise_area"], initial["sum_pairwise_area"])
        self.assertGreater(full_metric["max_pairwise_area"], initial["max_pairwise_area"])
        result, _ = self.guarded(GeometryNative(), polygons, ids, batch, 1)
        checked = measure(polygons, ids, result[0], 1200.)
        self.assert_not_worse(initial, checked)
        self.assertAlmostEqual(checked["sum_pairwise_area"], .7888053362179679, places=12)
        self.assertAlmostEqual(checked["max_pairwise_area"], .3194051447332989, places=12)

    def test_candidate_backtracking_is_independent(self):
        polygons, ids = [rectangle(2.)], [0, 0]
        batch = actions([(0., .2), (0., .3)], [(0., 0.), (0., .1)],
                        [(0., .2), (1.8, .2)])
        result, diagnostic = self.guarded(GeometryNative(), polygons, ids, batch, 10, 1.4)
        for index in range(len(batch)):
            individual, local = self.guarded(GeometryNative(), polygons, ids,
                                             batch[index:index+1], 10, 1.4)
            np.testing.assert_array_equal(result[index], individual[0])
            self.assertEqual(diagnostic["statuses"][index], local["statuses"][0])

    def test_native_zero_is_not_reference_success(self):
        polygons, ids = [rectangle()], [0, 0]
        batch = actions([(0., 0.), (1.-1e-5, 0.)])
        result, diagnostic = self.guarded(GeometryNative(miss_below=1e-4), polygons, ids, batch)
        self.assertGreater(measure(polygons, ids, result[0], 1200.)["max_pairwise_area"], 1e-6)
        self.assertEqual(diagnostic["statuses"], ["stalled_infeasible"])
        self.assertGreater(diagnostic["final_reference_geometry"][0]["max_pairwise_area"], 1e-6)

    def test_float32_storage_residual_is_reported(self):
        polygons, ids = [rectangle(.7)], [0, 0]
        batch = actions([(0., 0.), (.7, 0.)])
        _, diagnostic = self.guarded(GeometryNative(), polygons, ids, batch)
        self.assertEqual(diagnostic["final_reference_geometry"][0]["max_pairwise_area"], 0.)
        self.assertGreater(diagnostic["final_fp32_reference_geometry"][0]["max_pairwise_area"], 1e-9)
        self.assertEqual(diagnostic["fp32_reference_feasible"], [False])

    def test_concave_void_has_no_false_collision(self):
        lshape = [[0., 0.], [3., 0.], [3., 1.], [1., 1.], [1., 3.], [0., 3.]]
        polygons, ids = [lshape, rectangle(.5, .5)], [0, 1]
        batch = actions([(0., 0.), (1.5, 1.5)])
        result, diagnostic = self.guarded(GeometryNative(), polygons, ids, batch)
        np.testing.assert_array_equal(result, batch)
        self.assertEqual(diagnostic["statuses"], ["reference_converged"])

    def test_unsafe_gap_is_rolled_back_as_a_whole(self):
        polygons, ids = [rectangle()], [0, 0]
        batch = actions([(0., 0.), (2., 0.)])
        def bad_gap(xy):
            xy[:, 1, 0] = .5
            return xy
        result, diagnostic = self.guarded(GeometryNative(gap=bad_gap), polygons, ids, batch)
        np.testing.assert_array_equal(result, batch)
        self.assertGreater(sum(diagnostic["gap_rollbacks_per_candidate"]), 0)

    def test_gap_that_expands_strip_is_rejected(self):
        polygons, ids = [rectangle()], [0, 0]
        batch = actions([(0., 0.), (2., 0.)])
        def expanding_gap(xy):
            xy[:, 1, 0] += 10.
            return xy
        result, _ = self.guarded(GeometryNative(gap=expanding_gap), polygons, ids, batch)
        np.testing.assert_array_equal(result, batch)

    def test_disabled_post_preserves_actions(self):
        batch = actions([(20., 2.), (20.5, 2.)])
        result, diagnostic = self.guarded(GeometryNative(), [rectangle()], [0, 0], batch, 0)
        np.testing.assert_array_equal(result, batch)
        self.assertEqual(diagnostic["iterations_executed"], 0)

    def test_nontrivial_rotation_components_are_not_normalized(self):
        batch = actions([(10., 10.), (20., 10.)])
        batch[0, 0, 2:] = [500.*math.cos(.37), 500.*math.sin(.37)]
        batch[0, 1, 2:] = [1000.*math.cos(-.8), 1000.*math.sin(-.8)]
        original = batch.copy()
        result, _ = self.guarded(GeometryNative(), [rectangle()], [0, 0], batch)
        np.testing.assert_array_equal(result[:, :, 2:], original[:, :, 2:])
        np.testing.assert_array_equal(batch, original)

    def test_budget_and_policy_are_validated(self):
        batch = actions([(0., 0.)])
        for bad in (-1, 101, 1.5, True):
            with self.assertRaises(ValueError):
                self.guarded(GeometryNative(), [rectangle()], [0], batch, bad)
        with self.assertRaises(ValueError):
            enhance(GeometryNative(), [rectangle()], [0], batch, policy="unknown")

    def test_legacy_default_is_unchanged(self):
        batch = actions([(2., 2.), (2.5, 2.)])
        first, first_log = enhance(GeometryNative(), [rectangle()], [0, 0], batch, 3)
        second, second_log = enhance(GeometryNative(), [rectangle()], [0, 0], batch, 3,
                                     policy="legacy_unit")
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_log["statuses"], second_log["statuses"])
        self.assertEqual(first_log["trace"], second_log["trace"])

    def test_all_accepted_stages_are_monotone_and_traced(self):
        batch = actions([(0., .2), (0., .3)], [(0., 0.), (0., .1)],
                        [(0., .2), (1.8, .2)])
        _, diagnostic = self.guarded(GeometryNative(), [rectangle(2.)], [0, 0],
                                     batch, 10, 1.4)
        self.assertEqual(diagnostic["trace"][0]["repair_step_per_candidate"], [.125, 0., 1.])
        for step in diagnostic["trace"]:
            before = step["before_repair_reference_geometry"]
            repaired = step["after_repair_reference_geometry"]
            compressed = step["after_gap_reference_geometry"]
            for b, r, c in zip(before, repaired, compressed):
                for key in ("sum_pairwise_area", "max_pairwise_area", "height_violation"):
                    self.assertLessEqual(r[key], b[key])
                    self.assertLessEqual(c[key], r[key])
                self.assertLessEqual(c["span_x"], r["span_x"])

    def test_reference_measurement_matches_independent_geometry(self):
        from paper_guarded_enhancement import reference_geometry
        polygons = [rectangle(2., 1.), rectangle(.4, 2.)]
        batch = actions([(3., .2), (3.4, .5)])
        batch[0, 0, 2:] = [500.*math.cos(.37), 500.*math.sin(.37)]
        expected = measure(polygons, [0, 1], batch[0], 1.5)
        actual = reference_geometry(polygons, [0, 1], batch, 1.5)[0]
        for key in ("sum_pairwise_area", "max_pairwise_area", "height_violation", "span_x"):
            self.assertAlmostEqual(actual[key], expected[key], places=12)

    def test_gap_cannot_push_valid_piece_outside_strip(self):
        batch = actions([(0., 0.)])
        def crossing_wall(xy):
            xy[:, :, 1] -= .01
            return xy
        result, diagnostic = self.guarded(GeometryNative(gap=crossing_wall),
                                         [rectangle(1., 1200.)], [0], batch)
        np.testing.assert_array_equal(result, batch)
        self.assertEqual(diagnostic["final_reference_geometry"][0]["height_violation"], 0.)
        self.assertEqual(diagnostic["gap_rollbacks_per_candidate"], [1])


if __name__ == "__main__":
    unittest.main()
