"""Behavioral checks for batching, deterministic retries and training isolation."""
import copy
import io
import json
from pathlib import Path
import random
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from .sample import sample, sample_joint, batched_samples
from .settings import RECIPE
from .runtime import configure_execution
from .validation import append_event, preserve_training_state, collective_errors, maybe_validate
from scripts.stream_phase3_wandb import event_metrics, source_finished


@pytest.fixture(autouse=True)
def recipe():
    old = copy.deepcopy(RECIPE)
    yield
    RECIPE.clear()
    RECIPE.update(old)


class Prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.bridge = nn.Parameter(torch.tensor(.1))
        self.inputs = []

    def forward(self, **inputs):
        self.inputs.append(inputs)
        b = len(inputs["motion"])
        assert len(inputs["captions"]) == len(inputs["modes"]) == b
        assert inputs["motion_sigma"].shape == inputs["video_sigma"].shape == (b,)
        # No sample mixing; exercises real UniPC x0/velocity conversion.
        return {"motion": inputs["motion"] * .17 + inputs["neutral"].mean(tuple(range(1, inputs["neutral"].ndim)))[:, None, None],
                "video": inputs["video"] * .13 + inputs["motion"].mean((1, 2))[:, None, None, None, None]}


@pytest.mark.parametrize("mode", ["video2motion", "motimg2video", "joint"])
def test_native_unipc_batch_matches_independent_seeded_cases(mode):
    model = Prediction()
    motion = torch.randn(3, 9, 6)
    neutral = torch.randn(3, 3, 3)
    video = torch.randn(3, 2, 3, 2, 2)
    values = dict(motion=motion, neutral=neutral, video=video)
    batch = sample(model, mode=mode, caption=["a", "b", "c"], seed=[91, 3, 21], steps=5, **values)
    singles = [sample(model, mode=mode, caption=c, seed=s, steps=5,
        **{key: value[i:i + 1] for key, value in values.items()})
        for i, (c, s) in enumerate(zip(["a", "b", "c"], [91, 3, 21]))]
    if isinstance(batch, dict):
        for key in batch:
            torch.testing.assert_close(batch[key], torch.cat([one[key] for one in singles]), rtol=0, atol=0)
        torch.testing.assert_close(batch["video"][:, :, 0], video[:, :, 0], rtol=0, atol=0)
    else:
        torch.testing.assert_close(batch, torch.cat(singles), rtol=0, atol=0)
        if mode == "motimg2video":
            torch.testing.assert_close(batch[:, :, 0], video[:, :, 0], rtol=0, atol=0)


def test_training_rng_modes_and_gradients_restored_after_exception():
    model = Prediction().train()
    model.bridge.grad = torch.tensor(7.)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    hub_directory = torch.hub.get_dir()
    with pytest.raises(RuntimeError):
        with preserve_training_state(model):
            assert not model.training and not torch.is_grad_enabled()
            random.random(), np.random.rand(), torch.rand(3)
            torch.hub.set_dir("/tmp/temporary-validation-hub")
            raise RuntimeError("injected metric failure")
    assert model.training and model.bridge.grad == 7
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    assert torch.hub.get_dir() == hub_directory


def test_validation_events_are_idempotent_and_have_optimizer_step_axis():
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "metrics.jsonl"
        record = dict(step=10000, kind="validation", validation_id="abc",
                      validation=dict(v2m=dict(fid=.7), m2v=dict(dreamsim=.2)))
        with path.open("a", buffering=1) as log:
            append_event(log, path, record)
            append_event(log, path, record)
        assert len(path.read_text().splitlines()) == 1
        assert event_metrics(record) == {"val/step": 10000, "val/v2m/fid": .7, "val/m2v/dreamsim": .2}
        assert event_metrics(dict(step=20, loss=1.)) == {"train/step": 20, "train/loss": 1.}


def test_rank_zero_vae_constructor_scope_restored_on_success_and_failure():
    from .validation_metrics import load_rank_zero_scoring_vae
    from cosmos_framework.model.generator.tokenizers import wan2pt2_vae_4x16x16 as wan
    original = wan.sync_model_states
    def loader(*args):
        assert wan.sync_model_states is not original
        wan.sync_model_states(nn.Linear(2, 2))
        return 'complete_local_vae'
    with patch('torch.distributed.is_initialized', return_value=True), patch('torch.distributed.get_rank', return_value=0):
        assert load_rank_zero_scoring_vae('/unused', loader=loader) == 'complete_local_vae'
        assert wan.sync_model_states is original
        with pytest.raises(RuntimeError, match='injected'):
            load_rank_zero_scoring_vae('/unused', loader=lambda *args: (_ for _ in ()).throw(RuntimeError('injected')))
        assert wan.sync_model_states is original


def test_scoring_vae_rejects_nonzero_rank_and_keeps_single_process_loader():
    from .validation_metrics import load_rank_zero_scoring_vae
    with patch('torch.distributed.is_initialized', return_value=True), patch('torch.distributed.get_rank', return_value=1):
        with pytest.raises(RuntimeError, match='rank zero'):
            load_rank_zero_scoring_vae('/unused', loader=lambda *args: pytest.fail('loader must not run'))
    with patch('torch.distributed.is_initialized', return_value=False):
        assert load_rank_zero_scoring_vae('/unused', loader=lambda *args: args[1:]) == ('256', 97, 'cuda')


def test_execution_configuration_requires_grouping_and_restores_saved_choice():
    from .adapters import configure_checkpoint
    with pytest.raises(ValueError):
        configure_execution("dense", "bf16_bridge")
    configure_execution("grouped", "bf16_bridge")
    assert RECIPE["bridge_execution"] == dict(backend="grouped", precision="bf16_bridge")
    saved = dict(scheduler=dict(recipe=copy.deepcopy(RECIPE)))
    configure_execution()
    assert "bridge_execution" not in RECIPE
    configure_checkpoint(saved)
    assert RECIPE["bridge_execution"]["precision"] == "bf16_bridge"


def test_nonmilestone_never_samples_and_failed_rank_is_fatal():
    RECIPE["validation"] = dict(every=10000)
    maybe_validate(None, None, 9999, "f", 0, 1, None)
    with pytest.raises(RuntimeError, match="rank1"):
        collective_errors("rank1 sampling failed")
    from types import SimpleNamespace
    from .validation import configure_validation
    RECIPE["camera_prediction"] = {"enabled": True}
    with pytest.raises(ValueError, match="camera-aware"):
        configure_validation(SimpleNamespace(validation_config="unused"))


def test_wandb_waits_for_final_validation_and_drains_new_tail():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        metrics = root / "metrics.jsonl"
        metrics.write_text('{"step": 200000}\n')
        position = metrics.stat().st_size
        args = dict(target=200000, latest_step=200000, validation=True)
        (root / "status.json").write_text('{"stage": "validation"}')
        assert not source_finished(root, metrics, position, **args)
        with metrics.open("a") as log:
            log.write('{"step": 200000, "kind": "validation"}\n')
        (root / "status.json").write_text('{"stage": "complete"}')
        assert not source_finished(root, metrics, position, **args)
        assert source_finished(root, metrics, metrics.stat().st_size, **args)
        (root / "status.json").write_text('{"stage": "stopped"}')
        assert source_finished(root, metrics, metrics.stat().st_size,
                               target=200000, latest_step=12100, validation=False)


def test_oom_retry_preserves_order_case_seed_and_prefix():
    records = [dict(id=str(i), caption="", seed=i) for i in range(5)]
    arrays = (np.zeros((9, 6), np.float32), np.zeros((3, 3), np.float32), np.ones((2, 3, 2, 2), np.float32))
    calls = []
    def generate(model, *, seed, video, **kwargs):
        calls.append(list(seed))
        if len(seed) > 2:
            raise torch.cuda.OutOfMemoryError("injected")
        return video + torch.tensor(seed)[:, None, None, None, None]
    with patch("motion_expert_phase3_edge.quantitative.check_bundle", return_value=arrays), \
         patch("motion_expert_phase3_edge_joint.sample.sample", side_effect=generate):
        results = list(batched_samples(None, records, mode="motimg2video", contract={}, batch_size=5, device="cpu"))
    assert calls == [[0, 1, 2, 3, 4], [0, 1], [2, 3], [4]]
    assert [row["id"] for row, _, _ in results] == [str(i) for i in range(5)]
    assert [float(value.mean()) for _, _, value in results] == [1., 2., 3., 4., 5.]


def validation_worker(rank, directory):
    from types import SimpleNamespace
    root = Path(directory)
    dist.init_process_group("gloo", init_method="file://" + str(root / "rendezvous"), rank=rank, world_size=2)
    try:
        policy = dict(every=10000, config="fake", assets={"pin": 7}, sampling_batch_size=32)
        RECIPE["validation"] = policy
        cases = [dict(id=str(i)) for i in range(71)]
        model = Prediction().train()
        model.provenance = dict(phase2_contract={})
        model.bridge.grad = torch.tensor(3.)
        before = model.bridge.detach().clone()
        rng = torch.get_rng_state()
        args = SimpleNamespace(out=root)
        calls = []
        def sampling(model, rows, folder, contract, batch_size, device):
            calls.append(len(rows))
            assert batch_size == 32
            assert not model.training and not torch.is_grad_enabled()
            torch.rand(3)
            (folder / f"sampled{rank}.json").write_text(json.dumps([row["id"] for row in rows]))
            return [32, len(rows) - 32]
        vae_loads = []
        def load_vae(config):
            # Reproduce the native loader's collective requirement. Rank-zero
            # only construction would mismatch/hang against rank one's gather.
            count = torch.tensor(1.)
            dist.all_reduce(count)
            assert count.item() == 2
            vae_loads.append(rank)
            return "collectively_initialized_vae"
        def score(config, rows, folder, contract, policy, *, vae):
            assert vae == "collectively_initialized_vae"
            all_ids = sum([json.loads((folder / f"sampled{i}.json").read_text()) for i in range(2)], [])
            assert sorted(map(int, all_ids)) == list(range(71))
            return dict(v2m=dict(fid=.5), m2v=dict(fvd=200.))
        with (root / "metrics.jsonl").open("a", buffering=1) if rank == 0 else io.StringIO() as log:
            with patch("motion_expert_phase3_edge_joint.validation.read_configuration", return_value=({}, cases, policy["assets"])), \
                 patch("motion_expert_phase3_edge_joint.validation.sample_cases", side_effect=sampling), \
                 patch("motion_expert_phase3_edge_joint.validation_metrics.load_scoring_vae", side_effect=load_vae), \
                 patch("motion_expert_phase3_edge_joint.validation_metrics.score", side_effect=score):
                maybe_validate(args, model, 10000, "fingerprint", rank, 2, log)
                maybe_validate(args, model, 10000, "fingerprint", rank, 2, log)
                assert calls == [36 if rank == 0 else 35]
                assert vae_loads == [rank]  # Successful replay never reloads.
                if rank == 0:
                    assert len((root / "metrics.jsonl").read_text().splitlines()) == 1
                with patch("motion_expert_phase3_edge_joint.validation.sample_cases",
                           side_effect=RuntimeError("injected rank failure") if rank == 1 else sampling):
                    with pytest.raises(RuntimeError, match="rank1"):
                        maybe_validate(args, model, 20000, "fingerprint", rank, 2, log)
        assert model.training and model.bridge.grad == 3
        assert torch.equal(model.bridge, before) and torch.equal(torch.get_rng_state(), rng)
    finally:
        dist.destroy_process_group()


def test_two_rank_validation_coverage_resume_rng_and_collective_failure():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(validation_worker, args=(directory,), nprocs=2, join=True)
