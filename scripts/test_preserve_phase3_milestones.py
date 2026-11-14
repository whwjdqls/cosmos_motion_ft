from pathlib import Path
import json
import tempfile
import unittest
import torch
from scripts.preserve_phase3_milestones import preserve_one,fingerprint,sha

class PreservationTests(unittest.TestCase):
    def fixture(self,root,step=100000):
        run=root/'run';(run/'checkpoints').mkdir(parents=True);(run/'launch/source_snapshot').mkdir(parents=True)
        source=run/'launch/source_snapshot/model.py';source.write_text('# immutable test source\n')
        (run/'launch/source_manifest.json').write_text(json.dumps({'model.py':sha(source)}))
        contract={'recipe':{'video_weight':1.,'trainable_ema':{'decay':.999}}};(run/'contract.json').write_text(json.dumps(contract))
        state=dict(step=step,fingerprint=fingerprint(contract),scheduler={'step':step,'recipe':contract['recipe']},
            next_global_draw=step*32,rng_by_rank=[{},{}],bridges={'gate':torch.ones(2)},
            optimizer={'state':{0:{}},'param_groups':[{'params':[0]}]},
            ema={'num_updates':step,'decay':.999,'state':{'bridges':{'gate':torch.ones(2)}}})
        checkpoint=run/'checkpoints/step_000100000.pt';torch.save(state,checkpoint)
        return {'out':str(run),'expected_recipe':{'video_weight':1.}},checkpoint

    def test_copy_survives_original_unlink_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,source=self.fixture(root);record=preserve_one(model,root/'archive');target=Path(record['path'])
            self.assertNotEqual(source.stat().st_ino,target.stat().st_ino);self.assertEqual(sha(source),sha(target))
            self.assertEqual(preserve_one(model,root/'archive')['sha256'],record['sha256'])
            source.unlink();self.assertEqual(torch.load(target,weights_only=False)['step'],100000)

    def test_wrong_step_cannot_be_promoted_as_100k(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,_=self.fixture(root,step=99990)
            with self.assertRaises(AssertionError):preserve_one(model,root/'archive')
            self.assertFalse((root/'archive/VERIFIED.json').exists())
            self.assertFalse((root/'archive/step_000100000.pt').exists())

if __name__=='__main__':unittest.main()
