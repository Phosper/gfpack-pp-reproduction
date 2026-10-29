"""CPU controls for paired draws and evidence verification; no real model run."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from paper_sampling_noise import make_noise_plan, validate_noise_plan
from paper_sampler import sample_paper
from paper_metrics import sample_metrics, aggregate_metrics
from run_paper_alignment import exact_all
from run_generation_diagnosis import make_plan, CODE, CHECKPOINT_SHA, DATASET_SHA
from run_inference import sha256, write_json
from verify_generation_diagnosis import verify


class RNGConsumer(torch.nn.Module):
    device=torch.device("cpu")
    def geo_feature(self,features):
        if self.training: torch.rand(19)
        return features
    def forward(self,actions,*unused):
        if self.training: torch.rand(17)
        return torch.zeros_like(actions)


class NoiseTests(unittest.TestCase):
    def test_common_noise_survives_different_dropout_rng_consumption(self):
        plan=make_noise_plan(2,2,77)
        model=RNGConsumer()
        before=torch.random.get_rng_state().clone()
        model.train();_,a,da=sample_paper(model,[0,0],None,[0,0],2,seed=17,noise_plan=plan)
        model.eval();_,b,db=sample_paper(model,[0,0],None,[0,0],2,seed=17,noise_plan=plan)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual(da["common_noise_sha256"],db["common_noise_sha256"])

    def test_constraint_ablation_has_identical_first_96_steps(self):
        plan=make_noise_plan(2,2,77);model=RNGConsumer()
        plain,_,_=sample_paper(model,[0,0],None,[0,0],2,seed=17,noise_plan=plan)
        callback=lambda x:torch.ones(x.shape[0],x.shape[1],2)
        constrained,_,d=sample_paper(model,[0,0],None,[0,0],2,callback,seed=17,noise_plan=plan)
        self.assertTrue(torch.equal(plain[:,:96],constrained[:,:96]))
        self.assertFalse(torch.equal(plain[:,96:],constrained[:,96:]))
        self.assertEqual(d["constraint_callback_calls"],32)

    def test_bad_noise_shape_and_unsupported_profile_are_rejected(self):
        plan=make_noise_plan(1,2,77)
        with self.assertRaises(ValueError):validate_noise_plan(plan,(2,2,4))
        with self.assertRaises(ValueError):sample_paper(RNGConsumer(),[0,0],None,[0,0],1,
            profile="release_weights_constraints",noise_plan=plan)


class EvidenceTests(unittest.TestCase):
    def fixture(self,root):
        code=root/"code";code.mkdir()
        for name in CODE:shutil.copyfile(Path(__file__).parent/name,code/name)
        plan=make_plan(128)
        plan["asset_sha256"]={"checkpoint":CHECKPOINT_SHA,"dataset":DATASET_SHA}
        plan["code_sha256"]={name:sha256(code/name) for name in CODE}
        write_json(root/"plan.json",plan)
        polygon=[[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]]
        a=torch.tensor([[[0.,0.,1.,0.],[2.,0.,1.,0.]]]).repeat(128,1,1)
        refs=exact_all(polygon,[0,0],a.tolist())
        metrics=aggregate_metrics([sample_metrics(0,None,refs,planned_candidate_count=128)])
        for row in plan["runs"]:
            out=root/row["name"];out.mkdir()
            noise=make_noise_plan(128,2,row["sampling_noise_seed"])
            torch.save(noise,out/"sampling_noise.pt");torch.save(a,out/"predictions_raw.pt")
            torch.save(a[:,None].repeat(1,128,1,1),out/"trajectory.pt")
            write_json(out/"input_sample.json",{"polygon_ids":[0,0],"mask":[0.,0.]})
            write_json(out/"input_geometry.json",{"official_max_contours":polygon})
            write_json(out/"sampler.json",{"status":"completed","common_noise_sha256":validate_noise_plan(noise,(128,2,4)),
                "model_training_flag":row["model_mode"]=="train",
                "steps":[{"step_1based":i,"constraint_active":row["constraints"] and i>=97} for i in range(1,129)]})
            write_json(out/"constraint_callbacks.json",[{}]*32 if row["constraints"] else [])
            # Synthetic reports deliberately avoid claiming native measurements.
            report=aggregate_metrics([sample_metrics(row["sample_index"],None,refs,planned_candidate_count=128)])
            write_json(out/"raw_metrics.json",report)
            write_json(out/"metadata.json",{"status":"completed","changed_inputs":[],"sample_index":row["sample_index"],
                "planned_candidates":128,"profile":{"profile":row["profile"]},"model_training_flag":row["model_mode"]=="train",
                "constraints_enabled":row["constraints"],"post_iterations":0,"geometry_backend":"legacy_native",
                "sampling_noise_seed":row["sampling_noise_seed"],"checkpoint_sha256":CHECKPOINT_SHA,
                "input_sha256":{"fixture/dataset_dental_vali_128.pkl":DATASET_SHA},
                "output_sha256":{p.name:sha256(p) for p in out.iterdir()}})
        return root/plan["runs"][0]["name"]

    def test_complete_fixture_verifies_and_missing_run_fails(self):
        with tempfile.TemporaryDirectory(prefix=".generation-test-",dir=Path(__file__).parent) as tmp:
            root=Path(tmp);out=self.fixture(root)
            result=verify(root)
            self.assertEqual(result["verified_runs"],8)
            (out/"metadata.json").unlink()
            with self.assertRaises(FileNotFoundError):verify(root)

    def test_resigned_geometry_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".generation-test-",dir=Path(__file__).parent) as tmp:
            root=Path(tmp);out=self.fixture(root)
            path=out/"raw_metrics.json";r=json.loads(path.read_text(encoding="utf-8"))
            r["samples"][0]["candidates"][0]["reference"]["max_pairwise_area"]=1.
            path.write_text(json.dumps(r),encoding="utf-8")
            meta=out/"metadata.json";m=json.loads(meta.read_text());m["output_sha256"][path.name]=sha256(path)
            meta.write_text(json.dumps(m))
            with self.assertRaises(ValueError):verify(root)


if __name__=="__main__":unittest.main()
