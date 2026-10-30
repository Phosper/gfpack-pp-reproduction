"""Numerical stability and exact-geometry controls, using synthetic contours."""
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reproduce.collision_flow import integrate_collision_field
from reproduce.geometry import Geometry
from reproduce.sampling import sample
from reproduce.postprocess import postprocess
from reproduce.verify import measure
from reproduce.official_sampling import sample_public
from src.sde import init_sde, pc_sampler_state

SQUARE = [[0.,0.],[2.,0.],[2.,2.],[0.,2.]]


class StableGeometryTests(unittest.TestCase):
    def test_long_flow_does_not_turn_separation_time_into_an_oversized_jump(self):
        geometry = Geometry([SQUARE, SQUARE], 8.)
        poses = np.array([[0.,1.,1.,0.],[1.,1.,1.,0.]])
        force, _, _ = geometry.oriented(poses).field(poses[:,:2])
        euler = poses.copy()
        euler[:,:2] += 3.75*force
        result, report = integrate_collision_field(geometry, poses, 3.75)
        final = measure([SQUARE,SQUARE],json.loads(json.dumps(result.tolist())),8.)
        self.assertTrue(final['strict_valid'])
        self.assertLess(final['span_x'], geometry.measure(euler)['span_x']-2.)
        self.assertEqual(report['unintegrated_time'],0.)
        self.assertGreater(report['stationary_time'],0.)
        self.assertTrue(all(t['energy_after'] < t['energy_before'] for t in report['trace']))
        np.testing.assert_array_equal(result[:,2:],poses[:,2:])
        np.testing.assert_allclose(result[:,:2].sum(axis=0),poses[:,:2].sum(axis=0),atol=1e-12)

    def test_float32_rounding_stall_is_reported_without_fake_zero_overlap(self):
        geometry = Geometry([SQUARE,SQUARE],1e8+100.)
        poses = np.array([[1e8,1e8,1.,0.],[1e8,1e8,1.,0.]])
        result, report = integrate_collision_field(geometry,poses,1.,storage_dtype=np.float32)
        self.assertEqual(report['status'],'step_stalled')
        self.assertGreater(report['unintegrated_time'],0.)
        self.assertFalse(geometry.measure(result)['strict_valid'])
        np.testing.assert_array_equal(result,poses)

    def test_zero_time_is_identity_and_bad_time_is_rejected(self):
        geometry = Geometry([SQUARE],8.)
        poses = np.array([[0.,-1.,1.,0.]])
        result, report = integrate_collision_field(geometry,poses,0.)
        np.testing.assert_array_equal(result,poses)
        self.assertEqual(report['substeps'],0)
        for bad in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                integrate_collision_field(geometry,poses,bad)

    def test_substep_budget_exhaustion_reports_unintegrated_time(self):
        geometry = Geometry([SQUARE]*3,8.)
        poses = np.array([[0.,1.,1.,0.],[1.,1.,1.,0.],[2.,1.,1.,0.]])
        result, report = integrate_collision_field(geometry,poses,10.,max_substeps=1)
        self.assertEqual(report['status'],'substep_limit')
        self.assertGreater(report['unintegrated_time'],0.)
        self.assertLess(report['energy_after'],report['energy_before'])
        self.assertFalse(geometry.measure(result)['strict_valid'])

    def test_split_flow_preserves_early_neural_path_and_parallel_order(self):
        class DummyModel:
            device = torch.device('cpu')
            training = True
            def geo_feature(self,data): return None
            def __call__(self,states,*args): return torch.zeros_like(states)
        ids = torch.tensor([0,1])
        _, old_history, _ = sample(DummyModel(),ids,None,Geometry([SQUARE,SQUARE],1000.),batch_size=2)
        _, em_parallel, _ = sample(DummyModel(),ids,None,Geometry([SQUARE,SQUARE],1000.),batch_size=2,
                                  constraint_threads=2)
        torch.testing.assert_close(old_history,em_parallel,rtol=0.,atol=0.)
        _, serial, trace = sample(DummyModel(),ids,None,Geometry([SQUARE,SQUARE],1000.),batch_size=2,
                                  constraint_integrator='split-flow',constraint_threads=1)
        _, parallel, _ = sample(DummyModel(),ids,None,Geometry([SQUARE,SQUARE],1000.),batch_size=2,
                                constraint_integrator='split-flow',constraint_threads=2)
        torch.testing.assert_close(old_history[:,:97],serial[:,:97],rtol=0.,atol=0.)
        torch.testing.assert_close(serial,parallel,rtol=0.,atol=0.)
        self.assertEqual([t['step'] for t in trace if t['constraint_active']],list(range(97,129)))
        self.assertTrue(all(t['constraint_energy_after_sum'] <= t['constraint_energy_before_sum'] for t in trace))
        self.assertTrue(all(not t['grad_enabled'] and t['model_training_flag'] for t in trace))


class ParallelPostTests(unittest.TestCase):
    def test_serial_parallel_resume_and_frozen_input_rejection(self):
        rectangle = [[0.,0.],[4.,0.],[4.,1.],[0.,1.]]
        raw = [[[0.,0.,0.,1.],[2.,0.,0.,1.]], [[0.,.1,1.,0.],[5.,.1,1.,0.]]]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            folders = [root/'serial', root/'parallel']
            summaries = []
            for directory, workers in zip(folders,(1,2)):
                directory.mkdir()
                for name, value in [('input.json', {'sample_index':0,'contours':[rectangle,rectangle]}),
                                    ('raw.json',raw), ('generation_metadata.json',{'status':'sampling_completed','height':2.})]:
                    (directory/name).write_text(json.dumps(value),encoding='utf-8')
                summaries.append(postprocess(directory,workers=workers))
            self.assertEqual(summaries[0]['selected_candidate'],1)
            self.assertEqual(summaries[0]['strict_valid_candidates'],1)
            for index in (0,1):
                first = json.loads((folders[0]/f'candidate_{index:04d}.json').read_text())
                second = json.loads((folders[1]/f'candidate_{index:04d}.json').read_text())
                self.assertEqual(first['actions'],second['actions'])
                self.assertEqual(first['result'],second['result'])
                self.assertNotEqual(second['worker_pid'],os.getpid())
                self.assertFalse(second['worker_imported_torch'])
            saved = folders[1]/'candidate_0001.json'
            timestamp = saved.stat().st_mtime_ns
            with patch('reproduce.postprocess.calculate',side_effect=AssertionError('Cached candidate was recomputed')):
                resumed = postprocess(folders[1],workers=1,resume=True)
            self.assertEqual(resumed['selected_candidate'],1)
            self.assertEqual(saved.stat().st_mtime_ns,timestamp)
            with self.assertRaises(ValueError):
                postprocess(folders[1],policy='simultaneous',resume=True)
            tampered = raw.copy()
            tampered[0] = [[1.,0.,0.,1.],[2.,0.,0.,1.]]
            (folders[1]/'raw.json').write_text(json.dumps(tampered),encoding='utf-8')
            with self.assertRaises(ValueError):
                postprocess(folders[1],resume=True)


class PublicSourceControlTests(unittest.TestCase):
    def test_observational_wrapper_preserves_author_outputs_and_rng_exactly(self):
        class DummyModel(torch.nn.Module):
            device = torch.device('cpu')
            def geo_feature(self, data): return None
            def forward(self, states, *args): return -.000001*states
        model = DummyModel().train()
        ids = torch.tensor([0,1])
        _, _, coefficient, _ = init_sde('ve')
        torch.manual_seed(931)
        expected_history, expected_final = pc_sampler_state(model,coefficient,2,ids,None,
                                                            torch.zeros(2),batch_size=3)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(931)
        final, history, trace = sample_public(model,ids,None,batch_size=3)
        np.testing.assert_array_equal(final,expected_final.double().numpy())
        torch.testing.assert_close(history[:,1:],expected_history,rtol=0.,atol=0.)
        torch.testing.assert_close(torch.get_rng_state(),expected_rng,rtol=0.,atol=0.)
        self.assertEqual(len(trace),128)
        self.assertTrue(all(not t['constraint_active'] and not t['grad_enabled'] for t in trace))
        self.assertTrue(all(t['model_training_flag'] and t['public_drift_multiplier']==14. for t in trace))

    def test_benchmark_forwards_distinct_profiles_without_contradictory_flags(self):
        from reproduce import benchmark
        for profile in ('paper','public-source'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
                root = Path(temporary)/'benchmark'
                commands = []
                def fake_run(command,**kwargs):
                    commands.append(command)
                    if command[2]=='reproduce.run':
                        directory=Path(command[command.index('--output')+1]);directory.mkdir()
                        (directory/'summary.json').write_text(json.dumps({'sample_success':False,
                            'selected_metrics':None,'sample_utilization_failure_zero':0.,'strict_valid_candidates':0}))
                    return subprocess.CompletedProcess(command,0)
                argv=['benchmark','--output',str(root),'--count','1','--sampling-profile',profile]
                with patch.object(sys,'argv',argv),patch.object(benchmark.subprocess,'run',side_effect=fake_run):
                    benchmark.main()
                command=commands[0]
                self.assertEqual(command[command.index('--sampling-profile')+1],profile)
                self.assertEqual(command[command.index('--sigma-min')+1],'0.1' if profile=='paper' else '1.0')
                if profile=='paper':
                    self.assertEqual(command[command.index('--constraint-integrator')+1],'em')
                else:
                    self.assertNotIn('--constraint-integrator',command)

    def test_resume_finalizes_frozen_generation_without_resampling_or_overwriting_candidates(self):
        from reproduce import resume as recovery
        from reproduce.assets import ROOT,sha256,CHECKPOINT_SHA256
        rectangle=[[0.,0.],[4.,0.],[4.,1.],[0.,1.]]
        raw=[[[0.,0.,0.,1.],[2.,0.,0.,1.]],[[0.,.1,1.,0.],[5.,.1,1.,0.]]]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root=Path(temporary)
            source='reproduce/sampling.py'
            snapshot=root/'frozen_code'/source;snapshot.parent.mkdir(parents=True)
            shutil.copyfile(ROOT/source,snapshot)
            metadata={'status':'sampling_completed','height':2.,'candidates':2,
                      'model_executed':True,'model_runtime_parameters_unchanged':True,
                      'model_weights_changed':False,'model_training_flag':True,'model_mode':'train',
                      'sigma_min':1.,'sampling_profile':'public-source','not_paper_numerical_profile':True,
                      'constraint_integrator':'none-public-source','input_sha256':{source:sha256(ROOT/source)}}
            trace=[{'step':i,'constraint_active':False,'model_training_flag':True,'grad_enabled':False}
                   for i in range(1,129)]
            for name,value in [('input.json',{'sample_index':0,'contours':[rectangle,rectangle]}),
                               ('raw.json',raw),('generation_metadata.json',metadata),
                               ('sampling_trace.json',trace),('metadata.json',{'status':'failed','error':'synthetic interruption'})]:
                (root/name).write_text(json.dumps(value),encoding='utf-8')
            postprocess(root)
            stamp=(root/'candidate_0001.json').stat().st_mtime_ns
            real_sha=sha256
            def test_sha(path):
                return CHECKPOINT_SHA256 if Path(path)==ROOT/'checkpoints/dental.pth' else real_sha(path)
            with patch.object(recovery,'sha256',side_effect=test_sha):
                report=recovery.resume(root)
            self.assertTrue(report['verified'])
            self.assertEqual(report['selected_candidate'],1)
            self.assertEqual((root/'candidate_0001.json').stat().st_mtime_ns,stamp)
            self.assertEqual(len(list(root.glob('metadata_before_resume_*.json'))),1)
            finalized=json.loads((root/'metadata.json').read_text())
            self.assertTrue(finalized['resumed_postprocessing'])
            self.assertEqual(finalized['status'],'completed')

    def test_replay_cannot_be_promoted_to_a_fresh_neural_run(self):
        from reproduce.resume import resume
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root=Path(temporary)
            (root/'generation_metadata.json').write_text(json.dumps({'status':'sampling_completed',
                'model_executed':False,'replay_only':True}),encoding='utf-8')
            with self.assertRaises(ValueError):resume(root)


if __name__ == '__main__':
    unittest.main()
