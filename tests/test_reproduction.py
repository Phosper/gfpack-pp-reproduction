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

from reproduce.geometry import Geometry, SeparationError, convex_parts, enhance, sat_exit
from reproduce.pairwise import enhance_pairwise, validated_pair_update
from reproduce.sampling import sample
from reproduce.verify import measure

SQUARE = [[0., 0.], [2., 0.], [2., 2.], [0., 2.]]
ELBOW = [[0., 0.], [4., 0.], [4., 1.], [1., 1.], [1., 4.], [0., 4.]]
NARROW_CAVITY = [[0., 0.], [2., 0.], [2., 8.], [3.2, 8.],
                 [3.2, 0.], [5.2, 0.], [5.2, 10.], [0., 10.]]


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

    def test_gap_can_rescue_overlapping_start_by_accepting_only_legal_endpoints(self):
        geometry = Geometry([SQUARE] * 3, 5.)
        poses = np.array([[0., 0., 1., 0.], [5., 0., 1., 0.], [6., 0., 1., 0.]])
        initial = geometry.measure(poses)
        moved, count = geometry.oriented(poses).gap_sweep(poses)
        independent = measure([SQUARE] * 3, json.loads(json.dumps(moved.tolist())), 5.)
        self.assertFalse(initial['strict_valid'])
        self.assertTrue(independent['strict_valid'])
        self.assertEqual(independent['sum_pair_overlap_area'], 0.)
        self.assertLess(independent['span_x'], initial['span_x'])
        self.assertGreater(count, 0)
        np.testing.assert_array_equal(moved[:, 2:], poses[:, 2:])

    def test_gap_never_accepts_illegal_endpoint_when_no_negative_move_is_available(self):
        geometry = Geometry([SQUARE] * 2, 5.)
        poses = np.array([[0., 0., 1., 0.], [0., 0., 1., 0.]])
        moved, count = geometry.oriented(poses).gap_sweep(poses)
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(moved, poses)
        self.assertFalse(measure([SQUARE] * 2, moved.tolist(), 5.)['strict_valid'])

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

    def test_overrelaxed_concave_step_falls_back_to_valid_weighted_endpoints(self):
        small = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
        geometry = Geometry([small, NARROW_CAVITY], 20.)
        poses = np.array([[1.1, 3., 1., 0.], [0., 0., 1., 0.]])
        original = poses.copy()
        oriented = geometry.oriented(poses)
        shapes = oriented.shapes(poses[:, :2])
        vector = oriented.exit_vector(0, 1, poses[:, :2], shapes)
        total = geometry.areas.sum()
        unsafe = poses.copy()
        unsafe[0, :2] += geometry.areas[1] / total * 1.75 * vector
        unsafe[1, :2] -= geometry.areas[0] / total * 1.75 * vector
        self.assertGreater(geometry.measure(unsafe)['sum_pair_overlap_area'], .4)
        first, second, a, b, report = validated_pair_update(
            geometry, oriented, poses[:, :2], shapes, 0, 1)
        self.assertEqual(a.intersection(b).area, 0.)
        self.assertTrue(report['minimum_step_fallback'])
        self.assertEqual(report['applied_factor'], 1.)
        updated = poses.copy()
        updated[0, :2], updated[1, :2] = first, second
        self.assertTrue(measure([small, NARROW_CAVITY], updated.tolist(), 20.)['strict_valid'])
        np.testing.assert_allclose(geometry.areas[0] * first + geometry.areas[1] * second,
                                   geometry.areas[0] * poses[0, :2] + geometry.areas[1] * poses[1, :2],
                                   atol=1e-12)
        np.testing.assert_array_equal(poses, original)

    def test_safe_convex_overrelaxation_is_not_unnecessarily_replaced(self):
        geometry = Geometry([SQUARE, SQUARE], 8.)
        poses = np.array([[0., 1., 1., 0.], [1., 1., 1., 0.]])
        oriented = geometry.oriented(poses)
        _, _, first, second, report = validated_pair_update(
            geometry, oriented, poses[:, :2], oriented.shapes(poses[:, :2]), 0, 1)
        self.assertFalse(report['minimum_step_fallback'])
        self.assertEqual(report['applied_factor'], 1.75)
        self.assertEqual(first.intersection(second).area, 0.)

    def test_failed_pair_proposals_do_not_mutate_or_silently_accept(self):
        geometry = Geometry([SQUARE, SQUARE], 8.)
        poses = np.array([[0., 1., 1., 0.], [1., 1., 1., 0.]])
        original = poses.copy()
        oriented = geometry.oriented(poses)
        with patch.object(oriented, 'exit_vector', return_value=np.zeros(2)):
            with self.assertRaises(SeparationError) as caught:
                validated_pair_update(geometry, oriented, poses[:, :2],
                                      oriented.shapes(poses[:, :2]), 0, 1)
        self.assertEqual(len(caught.exception.diagnostic['trials']), 2)
        np.testing.assert_array_equal(poses, original)

    def test_damped_factor_does_not_falsely_report_single_step_separation(self):
        geometry = Geometry([SQUARE, SQUARE], 8.)
        poses = np.array([[0., 1., 1., 0.], [1., 1., 1., 0.]])
        oriented = geometry.oriented(poses)
        _, _, first, second, report = validated_pair_update(
            geometry, oriented, poses[:, :2], oriented.shapes(poses[:, :2]), 0, 1, .5)
        self.assertEqual(report['applied_factor'], .5)
        self.assertGreater(first.intersection(second).area, 0.)
        self.assertGreater(report['remaining_pair_overlap_area'], 0.)

    def test_pairwise_trace_records_rejected_relaxation_and_persists_zero(self):
        small = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
        geometry = Geometry([small, NARROW_CAVITY], 20.)
        poses = np.array([[1.1, 3., 1., 0.], [0., 0., 1., 0.]])
        result, report = enhance_pairwise(geometry, poses, iterations=1)
        self.assertEqual(report['minimum_step_fallbacks'], 1)
        self.assertGreater(report['trace'][0]['max_rejected_relaxed_overlap_area'], .4)
        stored = json.loads(json.dumps(result.tolist()))
        self.assertTrue(measure([small, NARROW_CAVITY], stored, 20.)['strict_valid'])
        np.testing.assert_array_equal(result[:, 2:], poses[:, 2:])


class ScheduleTests(unittest.TestCase):
    def test_frozen_replay_keeps_original_candidate_ids_and_excludes_invalid_winner(self):
        from reproduce import replay
        rectangle = [[0., 0.], [4., 0.], [4., 1.], [0., 1.]]
        raw = [[[0., 0., 0., 1.], [2., 0., 0., 1.]],
               [[0., .1, 1., 0.], [5., .1, 1., 0.]]]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp)
            source = folder / 'source'
            source.mkdir()
            for name, data in [('metadata.json', {'status': 'completed', 'height': 2.}),
                               ('input.json', {'contours': [rectangle, rectangle]}), ('raw.json', raw)]:
                (source / name).write_text(json.dumps(data), encoding='utf-8')
            for name, extra, count in [('all', [], 2), ('subset', ['--candidate-indices', '1'], 1)]:
                with self.subTest(scope=name):
                    output = folder / name
                    argv = ['replay', '--input', str(source), '--output', str(output),
                            '--post-policy', 'pairwise', *extra]
                    with patch.object(sys, 'argv', argv):
                        replay.main()
                    report = json.loads((output / 'diagnostic.json').read_text(encoding='utf-8'))
                    self.assertEqual(report['selected_candidate'], 1)
                    self.assertEqual(report['candidate_count'], count)
                    self.assertEqual(report['replayed_all_saved_candidates'], name == 'all')
                    self.assertEqual(report['strict_valid_candidates'], 1)
                    self.assertEqual(report['selected_metrics']['sum_pair_overlap_area'], 0.)
                    self.assertFalse(report['model_sampling_performed'])
                    self.assertTrue(report['input_unchanged'])

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
