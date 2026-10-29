"""Real-geometry controls, not a Dental benchmark or author-code equivalence."""
import math
import unittest

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

from paper_sat_geometry import PaperSATGeometry, convex_sat_displacement, measure_shapes, _ear_triangles
from paper_enhancement import enhance, enhance_chunked, field


def rectangle(x=0., y=0., w=1., h=1.):
    return [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]


L_SHAPE = [(0.,0.), (3.,0.), (3.,1.), (1.,1.), (1.,3.), (0.,3.)]


class PaperSATGeometryTests(unittest.TestCase):
    def field(self, polygons, xy=None, angles=None, height=1200.):
        xy = [[0.,0.]]*len(polygons) if xy is None else xy
        angles = [0.]*len(polygons) if angles is None else angles
        return PaperSATGeometry(height).paper_repulsion_all(list(range(len(polygons))), [angles], [xy], polygons, height)

    def test_concave_cavity_uses_short_separation_not_full_projection(self):
        polygons = [L_SHAPE, rectangle(.9,1.5), rectangle(1.,3.)]
        result = self.field(polygons)
        forces = np.asarray(result["repulsion"])[0]
        np.testing.assert_allclose(forces[0], [-.1/6, 0.], atol=1e-14)
        np.testing.assert_allclose(forces[1], [.5/6, 0.], atol=1e-14)
        np.testing.assert_array_equal(forces[2], [0.,0.])
        shapes = [affinity.translate(Polygon(p), *d) for p,d in zip(polygons,forces)]
        self.assertLessEqual(measure_shapes(shapes)["sum_pairwise_area"], 1e-14)
        self.assertEqual(result["concave_proxy_pairs"], [0])
        self.assertEqual(result["concave_mtv_pairs"], [1])

    def test_small_overlap_not_quantized_to_zero(self):
        result = self.field([rectangle(), rectangle(.999,0.)])
        self.assertAlmostEqual(result["max_pairwise_area"][0], .001)
        self.assertGreater(np.linalg.norm(result["repulsion"]), 0.)

    def test_convex_containment_and_area_weighting(self):
        outer, inner = Polygon(rectangle(w=4.,h=4.)), Polygon(rectangle(1.,1.))
        displacement = convex_sat_displacement(inner, outer)
        self.assertAlmostEqual(np.linalg.norm(displacement), 2.)
        self.assertEqual(affinity.translate(inner,*displacement).intersection(outer).area, 0.)
        result = self.field([rectangle(), rectangle(.5,0.,2.,1.)])
        force = np.asarray(result["repulsion"])[0]
        np.testing.assert_allclose(force[0]*1.+force[1]*2., 0., atol=1e-15)

    def test_rotated_concave_minimum_is_rotation_equivariant(self):
        angle = .41
        result = self.field([L_SHAPE, rectangle(.9,1.5)], angles=[angle,angle])
        relative = np.diff(np.asarray(result["repulsion"])[0],axis=0)[0]
        np.testing.assert_allclose(relative, [.1*math.cos(angle),.1*math.sin(angle)], atol=1e-12)

    def test_cavity_without_overlap_is_not_repulsed(self):
        result = self.field([L_SHAPE, rectangle(1.2,1.5,.5,.5)])
        self.assertEqual(result["colliding_pairs"], [0])
        self.assertEqual(np.linalg.norm(result["repulsion"]),0.)

    def test_gap_no_positive_epsilon_drift_at_full_height(self):
        native = PaperSATGeometry(height=1.)
        xy = [[[0.,0.]]]
        output = native.paper_gap_xy_all([0],[[0.]],xy,[rectangle()])
        self.assertEqual(output,xy)
        self.assertEqual(native.last_gap_trace[0]["after_y"]["height_violation"],0.)

    def test_gap_stops_at_touching_neighbor_and_keeps_height(self):
        native = PaperSATGeometry(height=1.)
        output = native.paper_gap_xy_all([0,1],[[0.,0.]],[[[0.,0.],[2.,0.]]],[rectangle(),rectangle()])
        self.assertGreaterEqual(output[0][1][0],1.)
        self.assertLess(output[0][1][0],1.001)
        trace = native.last_gap_trace[0]
        self.assertEqual(trace["after_x"]["sum_pairwise_area"],0.)
        self.assertEqual(trace["after_y"]["height_violation"],0.)
        self.assertLess(trace["after_y"]["span_x"],trace["before"]["span_x"])

    def test_gap_does_not_search_from_colliding_start(self):
        native = PaperSATGeometry(height=3.)
        xy = [[[1.,1.],[1.5,1.]]]
        output = native.paper_gap_xy_all([0,1],[[0.,0.]],xy,[rectangle(),rectangle()])
        self.assertEqual(output,xy)
        self.assertEqual(native.last_gap_trace[0]["after_x_skipped_colliding_pieces"],2)

    def test_lower_and_upper_boundary_are_repulsive_only_in_y(self):
        result = self.field([rectangle(),rectangle()],xy=[[-10.,-.25],[10.,1.25]],height=2.)
        np.testing.assert_allclose(result["repulsion"], [[[0.,.25],[0.,-.25]]])

    def test_ear_fallback_covers_concave_polygon(self):
        from shapely.ops import unary_union
        poly = Polygon(L_SHAPE)
        parts = _ear_triangles(poly)
        self.assertEqual(unary_union(parts).symmetric_difference(poly).area,0.)

    def test_invalid_contours_fail_without_repair(self):
        with self.assertRaises(ValueError):
            self.field([[(0.,0.),(1.,1.),(0.,1.),(1.,0.)]])

    def test_negative_y_search_can_enter_strip_from_above(self):
        native = PaperSATGeometry(height=2.)
        xy = [[[0., 1.25]]]
        out = native.paper_gap_xy_all([0], [[0.]], xy, [rectangle()])
        self.assertLess(out[0][0][1], xy[0][0][1])
        self.assertEqual(native.last_gap_trace[0]["after_y"]["height_violation"], 0.)

    def test_above_strip_entry_is_not_committed_through_an_obstacle(self):
        native = PaperSATGeometry(height=2.)
        # The upper piece must move down .25 to enter the strip, but a fixed
        # full-width lower piece occupies that required space.
        xy = [[[0.,0.], [0.,1.25]]]
        out = native.paper_gap_xy_all([0,1], [[0.,0.]], xy,
                                       [rectangle(w=1.,h=1.25),rectangle()])
        self.assertEqual(out, xy)

    def test_measurement_does_not_construct_a_repulsion_field(self):
        native = PaperSATGeometry()
        polygons = [L_SHAPE, rectangle(.9,1.5)]
        a = np.array([[[0.,0.,1.,0.],[0.,0.,1.,0.]]])
        expected,_ = field(native,polygons,[0,1],a)
        measured,_ = field(native,polygons,[0,1],a,compute_repulsion=False)
        for key in measured:
            self.assertEqual(measured[key], expected[key])
        self.assertNotIn("repulsion",measured)

    def test_reference_stops_on_the_same_tolerance_as_result_selection(self):
        class TinyResidual(PaperSATGeometry):
            def paper_repulsion_all(self,*args,**kwargs):
                r = self.paper_measure_all(*args,**kwargs)
                r["repulsion"] = [[[0.,0.]]]
                return r
            def paper_measure_all(self,*unused,**kwargs):
                return {"max_pairwise_area":[1e-13],"sum_pairwise_area":[1e-13],
                        "height_violation":[0.],"min_x":[0.]}
            def paper_gap_xy_all(self,ids,angles,xy,polygons):
                return xy
        _,post = enhance(TinyResidual(),[rectangle()],[0],np.array([[[0.,0.,1.,0.]]]),100)
        self.assertEqual(post["iterations_executed"],1)
        self.assertEqual(post["statuses"],["reference_converged"])
        self.assertEqual(post["overlap_stop_native_area"],1e-9)

    def test_chunking_preserves_candidate_outputs_and_budget(self):
        a=np.array([[[0.,0.,1.,0.],[.5,0.,1.,0.]],
                    [[0.,0.,1.,0.],[2.,0.,1.,0.]]])
        whole, old = enhance(PaperSATGeometry(),[rectangle()],[0,0],a,5)
        chunks,new = enhance_chunked(PaperSATGeometry(),[rectangle()],[0,0],a,5)
        np.testing.assert_array_equal(whole,chunks)
        self.assertEqual(old["statuses"],new["statuses"])
        self.assertEqual(new["iterations_executed"],old["iterations_executed"])
        self.assertEqual([c["candidate_indices"] for c in new["chunks"]],[[0],[1]])
        self.assertTrue(all(c["post"]["iterations_executed"]<=5 for c in new["chunks"]))

    def test_explicit_half_step_changes_only_fixed_gradient_magnitude(self):
        class NoGap(PaperSATGeometry):
            def paper_gap_xy_all(self,ids,rotates,xy,polys):
                return xy
        a=np.array([[[0.,0.,1.,0.],[.5,0.,1.,0.]]])
        out,post=enhance(NoGap(),[rectangle()],[0,0],a,1,
                         policy="paper_fixed_step",post_step_size=.5)
        self.assertAlmostEqual(out[0,1,0]-out[0,0,0],.75)
        self.assertEqual(post["constraint_step_size"],.5)
        with self.assertRaises(ValueError):
            enhance(NoGap(),[rectangle()],[0,0],a,1,post_step_size=.5)

    def test_failure_diagnosis_distinguishes_overlap_boundary_and_oversize(self):
        from paper_failure_diagnostics import diagnose_layout
        overlap=diagnose_layout([rectangle()],[0,0],[[0.,0.,1.,0.],[.5,0.,1.,0.]])
        self.assertEqual(overlap["reason_codes"],["pair_overlap"])
        boundary=diagnose_layout([rectangle()],[0],[[0.,-.1,1.,0.]])
        self.assertEqual(boundary["reason_codes"],["strip_boundary"])
        tall=diagnose_layout([rectangle(h=1201.)],[0],[[0.,0.,1.,0.]])
        self.assertEqual(tall["fixed_rotation_impossible_piece_indices"],[0])


if __name__ == "__main__":
    unittest.main()
