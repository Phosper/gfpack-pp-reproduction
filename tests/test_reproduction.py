"""Geometric counterexamples and schedule/selection checks for the reconstruction."""
import json
from pathlib import Path
import unittest
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from shapely.affinity import translate
from shapely.geometry import Polygon

from reproduce.geometry import Geometry, convex_parts, enhance, sat_exit
from reproduce.pairwise import enhance_pairwise
from reproduce.sampling import sample
from reproduce.verify import measure

SQUARE = [[0., 0.], [2., 0.], [2., 2.], [0., 2.]]
ELBOW = [[0., 0.], [4., 0.], [4., 1.], [1., 1.], [1., 4.], [0., 4.]]


class GeometryTests(unittest.TestCase):
    def test_sat_containment_requires_exit_not_intersection_width(self):
        outer = np.asarray([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
        inner = np.asarray(SQUARE) + 4
        vector = sat_exit(inner, outer)
        self.assertAlmostEqual(np.linalg.norm(vector), 6.)
        self.assertEqual(translate(Polygon(inner), *vector).intersection(Polygon(outer)).area, 0.)

    def test_decomposition_keeps_concave_cavity(self):
        pieces = convex_parts(Polygon(ELBOW))
        self.assertEqual(sum(p.area for p in pieces), 7.)
        geometry = Geometry([ELBOW, [[0, 0], [1, 0], [1, 1], [0, 1]]], 5.)
        poses = np.array([[0., 0., 1., 0.], [2., 2., 1., 0.]])
        force, energy, count = geometry.oriented(poses).field(poses[:, :2])
        self.assertEqual(count, 0)
        self.assertEqual(energy, 0)
        np.testing.assert_array_equal(force, 0)

    def test_concave_exit_is_checked_on_complete_contour(self):
        geometry = Geometry([ELBOW, [[0, 0], [1, 0], [1, 1], [0, 1]]], 5.)
        poses = np.array([[0., 0., 1., 0.], [.5, 2., 1., 0.]])
        oriented = geometry.oriented(poses)
        shapes = oriented.shapes(poses[:, :2])
        vector = oriented.exit_vector(0, 1, poses[:, :2], shapes)
        self.assertLess(np.linalg.norm(vector), .50001)
        self.assertEqual(translate(shapes[0], *vector).intersection(shapes[1]).area, 0.)

    def test_real_concave_corner_does_not_reenter_another_forbidden_lobe(self):
        case = json.loads((Path(__file__).parent / 'fixtures' / 'concave_corner.json').read_text())
        geometry = Geometry(case['contours'], 2000.)
        positions = np.asarray(case['positions'])
        poses = np.column_stack([positions, np.ones(2), np.zeros(2)])
        oriented = geometry.oriented(poses)
        shapes = oriented.shapes(positions)
        self.assertGreater(shapes[0].intersection(shapes[1]).area, 3000.)
        vector = oriented.exit_vector(0, 1, positions, shapes)
        self.assertEqual(translate(shapes[0], *vector).intersection(shapes[1]).area, 0.)

    def test_area_weights_preserve_pair_mass_balance(self):
        small = [[0, 0], [1, 0], [1, 1], [0, 1]]
        geometry = Geometry([small, SQUARE], 8.)
        poses = np.array([[1., 2., 1., 0.], [1., 2., 1., 0.]])
        force, _, pairs = geometry.oriented(poses).field(poses[:, :2])
        self.assertEqual(pairs, 1)
        np.testing.assert_allclose(force[0] + 4 * force[1], 0., atol=1e-12)

    def test_small_real_overlap_is_rejected_without_area_threshold(self):
        geometry = Geometry([SQUARE, SQUARE], 5.)
        poses = [[0, 0, 1, 0], [2-1e-8, 0, 1, 0]]
        measured = geometry.measure(poses)
        self.assertGreater(measured['max_pair_overlap_area'], 0.)
        self.assertFalse(measured['strict_valid'])

    def test_post_repairs_then_serialized_coordinates_stay_exactly_valid(self):
        geometry = Geometry([SQUARE, SQUARE], 5.)
        poses = np.array([[0., 1., 1., 0.], [1., 1., 1., 0.]])
        result, report = enhance(geometry, poses)
        self.assertTrue(report['metrics']['strict_valid'])
        stored = json.loads(json.dumps(result.tolist()))
        independent = measure([SQUARE, SQUARE], stored, 5.)
        self.assertTrue(independent['strict_valid'])
        self.assertEqual(independent['sum_pair_overlap_area'], 0.)
        np.testing.assert_array_equal(result[:, 2:], poses[:, 2:])
        self.assertEqual(report['metrics']['part_area'], 8.)

    def test_gap_pass_cannot_introduce_collisions(self):
        geometry = Geometry([SQUARE, SQUARE, ELBOW], 10.)
        poses = np.array([[0., 3., 1., 0.], [5., 0., 1., 0.], [10., 0., 1., 0.]])
        initial = geometry.measure(poses)
        moved, _ = geometry.oriented(poses).gap_sweep(poses)
        final = geometry.measure(moved)
        self.assertTrue(initial['strict_valid'])
        self.assertTrue(final['strict_valid'])
        self.assertLessEqual(final['span_x'], initial['span_x'])

    def test_rotated_gap_endpoint_has_nonnegative_independent_wall_distance(self):
        triangle = [[.17, .31], [2.77, .43], [.81, 1.93]]
        geometry = Geometry([triangle], 8.)
        for theta in (.317, 1.13, -2.11):
            poses = np.array([[4., 4., np.cos(theta), np.sin(theta)]])
            moved, _ = geometry.oriented(poses).gap_sweep(poses)
            independent = measure([triangle], moved.tolist(), 8.)
            self.assertTrue(independent['strict_valid'])
            self.assertEqual(independent['height_violation'], 0.)

    def test_infeasible_height_cannot_be_reported_as_success(self):
        geometry = Geometry([SQUARE], 1.)
        poses = np.array([[0., -.5, 1., 0.]])
        _, report = enhance(geometry, poses)
        self.assertFalse(report['metrics']['strict_valid'])
        self.assertLessEqual(report['iterations'], 100)

    def test_zero_budget_keeps_invalid_input_invalid(self):
        geometry = Geometry([SQUARE, SQUARE], 5.)
        poses = np.array([[0., 0., 1., 0.], [0., 0., 1., 0.]])
        result, report = enhance(geometry, poses, iterations=0)
        np.testing.assert_array_equal(result, poses)
        self.assertFalse(report['metrics']['strict_valid'])

    def test_pairwise_update_preserves_area_rotations_and_independent_zero(self):
        geometry = Geometry([SQUARE, SQUARE], 5.)
        poses = np.array([[0., 1., 1., 0.], [1., 1., 1., 0.]])
        result, report = enhance_pairwise(geometry, poses)
        self.assertTrue(report['metrics']['strict_valid'])
        self.assertTrue(measure([SQUARE, SQUARE], json.loads(json.dumps(result.tolist())), 5.)['strict_valid'])
        np.testing.assert_array_equal(result[:, 2:], poses[:, 2:])
        self.assertEqual(report['metrics']['part_area'], 8.)
        self.assertLessEqual(report['iterations'], 100)

    def test_pairwise_zero_budget_and_impossible_rotation_never_become_success(self):
        geometry = Geometry([SQUARE, SQUARE], 5.)
        poses = np.array([[0., 0., 1., 0.], [0., 0., 1., 0.]])
        result, report = enhance_pairwise(geometry, poses, iterations=0)
        self.assertFalse(report['metrics']['strict_valid'])
        np.testing.assert_array_equal(result, poses)
        too_high = Geometry([SQUARE], 1.)
        _, report = enhance_pairwise(too_high, poses[:1])
        self.assertFalse(report['metrics']['strict_valid'])
        self.assertEqual(report['status'], 'fixed_orientation_exceeds_height')


class ScheduleTests(unittest.TestCase):
    def test_benchmark_forwards_official_mode_and_explicit_eval_to_child(self):
        from reproduce import benchmark
        for expected, extra in [('train', []), ('eval', ['--model-mode', 'eval'])]:
            with self.subTest(mode=expected), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
                output = Path(tmp) / 'benchmark'
                commands = []
                def fake_run(command, **kwargs):
                    commands.append(command)
                    if command[2] == 'reproduce.run':
                        child = Path(command[command.index('--output') + 1])
                        child.mkdir()
                        (child / 'summary.json').write_text(json.dumps({
                            'sample_success': False, 'selected_metrics': None,
                            'sample_utilization_failure_zero': 0., 'strict_valid_candidates': 0}), encoding='utf-8')
                    return subprocess.CompletedProcess(command, 0)
                argv = ['benchmark', '--count', '1', '--candidates', '1', '--smoke',
                        '--device', 'cpu', '--output', str(output), *extra]
                with patch.object(sys, 'argv', argv), patch.object(benchmark.subprocess, 'run', side_effect=fake_run):
                    benchmark.main()
                generated = commands[0]
                self.assertEqual(generated[generated.index('--model-mode') + 1], expected)
                record = json.loads((output / 'benchmark.json').read_text(encoding='utf-8'))
                self.assertEqual(record['model_mode'], expected)

    def test_high_utilization_invalid_candidate_cannot_win(self):
        from reproduce.run import select_candidate
        self.assertEqual(select_candidate([{'strict_valid': False, 'strip_utilization': .99},
                                           {'strict_valid': True, 'strip_utilization': .7}]), 1)
        self.assertIsNone(select_candidate([{'strict_valid': False, 'strip_utilization': .99}]))

    def test_constraint_runs_only_in_last_32_updates(self):
        class DummyModel:
            device = torch.device('cpu')
            training = True
            def geo_feature(self, data):
                return None
            def __call__(self, states, *args):
                return torch.zeros_like(states)
        class DummyGeometry:
            calls = 0
            def oriented(self, pose):
                return self
            def field(self, xy):
                self.calls += 1
                return np.zeros_like(xy), 0., 0
        geometry = DummyGeometry()
        _, history, trace = sample(DummyModel(), torch.tensor([0]), None, geometry, batch_size=2)
        self.assertEqual(geometry.calls, 64)
        self.assertEqual(history.shape, (2, 129, 1, 4))
        self.assertEqual([r['step'] for r in trace if r['constraint_active']], list(range(97, 129)))
        self.assertTrue(all(r['model_training_flag'] for r in trace))
        self.assertTrue(all(r['grad_enabled'] is False for r in trace))


if __name__ == '__main__':
    unittest.main()
