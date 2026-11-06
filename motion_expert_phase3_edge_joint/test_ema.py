"""EMA checkpoint integrity, exact continuation and reversible validation."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from .adapters import (configure_checkpoint, load_trainable_state, new_ema,
                       new_optimizer, restore_ema, trainable_parameters, trainable_state)
from .settings import RECIPE
from .test_adapters import TinyPhase3, setup_recipe, saved_payload, update


@pytest.fixture(autouse=True)
def restore_recipe():
    saved = copy.deepcopy(RECIPE)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    RECIPE.clear()
    RECIPE.update(saved)


@pytest.mark.parametrize("lora", [False, True])
def test_atomic_checkpoint_ema_resume_and_raw_or_ema_inference(tmp_path, lora):
    from .train import atomic_checkpoint
    cfg = setup_recipe(enabled=lora, ema=.9)
    model = TinyPhase3(cfg)
    optimizer, ema = new_optimizer(model), new_ema(model)
    x = torch.randn(3, 8)
    for _ in range(2):
        update(model, optimizer, ema, x)
    payload = saved_payload(model, optimizer, ema)
    payload.update(step=2, next_global_draw=64, schema=2)
    path = tmp_path / "recovery_latest.pt"
    atomic_checkpoint(path, payload)
    loaded = torch.load(path, mmap=True, weights_only=False)
    assert not list(tmp_path.glob("*.tmp.*"))
    assert loaded["ema"]["num_updates"] == loaded["step"] == 2
    restored = TinyPhase3(cfg)
    restored.native.load_state_dict(model.native.state_dict())
    load_trainable_state(restored, loaded, weights="raw")
    ropt, rema = new_optimizer(restored), new_ema(restored)
    ropt.load_state_dict(loaded["optimizer"])
    restore_ema(rema, restored, loaded)
    for _ in range(2):
        update(model, optimizer, ema, x)
        update(restored, ropt, rema, x)
    assert ema.num_updates == rema.num_updates == 4
    for name, state in trainable_state(model).items():
        for key, value in state.items():
            assert torch.equal(value, trainable_state(restored)[name][key])
            assert torch.equal(ema.state[name][key], rema.state[name][key])
    final = saved_payload(model, optimizer, ema)
    final["step"] = 4
    load_trainable_state(restored, final, weights="ema")
    for name, state in trainable_state(restored).items():
        for key, value in state.items():
            assert torch.equal(value, ema.state[name][key])
    # Loading EMA for inference cannot rewrite the raw checkpoint tensors.
    assert any(not torch.equal(final[name][key], ema.state[name][key])
               for name, state in final.items() if name == "bridges" for key in state)


@pytest.mark.parametrize("corruption", ["count", "fraction", "negative", "bool", "decay", "scope", "nan", "dtype"])
def test_malformed_ema_rejected_before_any_weight_change(corruption):
    model = TinyPhase3(setup_recipe(enabled=False))
    optimizer, ema = new_optimizer(model), new_ema(model)
    update(model, optimizer, ema, torch.randn(3, 8))
    payload = saved_payload(model, optimizer, ema)
    payload["step"] = 1
    if corruption in {"count", "fraction", "negative", "bool"}:
        payload["ema"]["num_updates"] = {"count": 2, "fraction": 1.5, "negative": -1, "bool": True}[corruption]
    elif corruption == "decay":
        payload["ema"]["decay"] = .8
    elif corruption == "scope":
        payload["scheduler"]["recipe"]["trainable_ema"]["scope"] = "frozen_specialists"
    else:
        state = payload["ema"]["state"]["bridges"]
        key = next(iter(state))
        if corruption == "nan":
            state[key].fill_(float("nan"))
        else:
            state[key] = state[key].half()
    before = copy.deepcopy(trainable_state(model))
    shadow = copy.deepcopy(ema.state)
    with pytest.raises(ValueError):
        load_trainable_state(model, payload, weights="ema")
    with pytest.raises(ValueError):
        restore_ema(ema, model, payload)
    for name, state in trainable_state(model).items():
        for key, value in state.items():
            assert torch.equal(value, before[name][key])
            assert torch.equal(ema.state[name][key], shadow[name][key])
    # Explicit raw inference remains available independently of damaged EMA.
    load_trainable_state(model, payload, weights="raw")


@pytest.mark.parametrize("fail", [False, True])
def test_validation_swap_restores_raw_gradients_optimizer_and_ema(fail):
    model = TinyPhase3(setup_recipe())
    optimizer, ema = new_optimizer(model), new_ema(model)
    x = torch.randn(3, 8)
    update(model, optimizer, ema, x)
    raw = copy.deepcopy(trainable_state(model))
    gradients = [p.grad.clone() for p in trainable_parameters(model)]
    shadow = copy.deepcopy(ema.state)
    raw_prediction = model(x).detach()
    def validate():
        with ema.average_parameters(model):
            for name, state in trainable_state(model).items():
                for key, value in state.items():
                    assert torch.equal(value, ema.state[name][key])
            if fail:
                raise RuntimeError("injected scoring failure")
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            validate()
    else:
        validate()
    assert torch.equal(raw_prediction, model(x))
    for name, state in trainable_state(model).items():
        for key, value in state.items():
            assert torch.equal(value, raw[name][key])
            assert torch.equal(ema.state[name][key], shadow[name][key])
    assert all(torch.equal(p.grad, g) for p, g in zip(trainable_parameters(model), gradients))
    # Subsequent Adam/EMA updates still work after either exit path.
    update(model, optimizer, ema, x)
    assert ema.num_updates == 2


def test_periodic_validation_samples_ema_and_restores_raw_after_failure(tmp_path):
    from .validation import maybe_validate
    model = TinyPhase3(setup_recipe(enabled=False))
    model.provenance = {"phase2_contract": {}}
    optimizer, ema = new_optimizer(model), new_ema(model)
    update(model, optimizer, ema, torch.randn(3, 8))
    ema.num_updates = 10000
    raw = copy.deepcopy(trainable_state(model))
    policy = dict(every=10000, config="fake", assets={}, sampling_batch_size=32, weights="ema")
    RECIPE["validation"] = policy
    def sampling(*args):
        assert torch.equal(model.bridges.weight, ema.state["bridges"]["weight"])
        raise RuntimeError("injected sampling failure")
    args = SimpleNamespace(out=tmp_path)
    with patch("motion_expert_phase3_edge_joint.validation.read_configuration", return_value=({}, [], {})), \
         patch("motion_expert_phase3_edge_joint.validation.sample_cases", side_effect=sampling):
        with pytest.raises(RuntimeError, match="injected sampling"):
            maybe_validate(args, model, 10000, "fp", 0, 1, io.StringIO(), ema=ema)
    assert model.training
    assert torch.equal(model.bridges.weight, raw["bridges"]["weight"])
    assert json.loads((tmp_path / "validation/step_000010000/identity.json").read_text())["policy"]["weights"] == "ema"


def test_ema_validation_cannot_be_requested_without_ema():
    from .validation import configure_validation
    args = SimpleNamespace(validation_weights="ema", ema_decay=0, validation_config=None)
    with pytest.raises(ValueError, match="requires --ema-decay"):
        configure_validation(args)
    # Legacy checkpoints with no EMA flag still load and configure as raw.
    configure_checkpoint({})
    assert "trainable_ema" not in RECIPE


def ema_ddp_worker(rank, directory):
    from .train import atomic_checkpoint
    from .validation import maybe_validate
    root = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(root / "gloo"), rank=rank, world_size=2)
    try:
        for lora in (False, True):
            cfg = setup_recipe(enabled=lora, ema=.9)
            torch.manual_seed(11 + rank)
            model = TinyPhase3(cfg)
            for p in model.native.parameters():
                dist.broadcast(p, 0)
            ddp = DDP(model, broadcast_buffers=False, find_unused_parameters=False)
            optimizer, ema = new_optimizer(model), new_ema(model)
            x = torch.full((3, 8), float(rank + 1))
            for _ in range(3):
                update(ddp, optimizer, ema, x)
            path = root / f"checkpoint_lora{lora}.pt"
            if rank == 0:
                atomic_checkpoint(path, dict(**trainable_state(model), optimizer=optimizer.state_dict(),
                    scheduler={"recipe": copy.deepcopy(RECIPE)}, ema=ema.state_dict(), step=3))
            dist.barrier()
            payload = torch.load(path, mmap=True, weights_only=False)
            restored = TinyPhase3(cfg)
            restored.native.load_state_dict(model.native.state_dict())
            rddp = DDP(restored, broadcast_buffers=False, find_unused_parameters=False)
            ropt, rema = new_optimizer(restored), new_ema(restored)
            load_trainable_state(restored, payload)
            ropt.load_state_dict(payload["optimizer"])
            restore_ema(rema, restored, payload)
            update(ddp, optimizer, ema, x)
            update(rddp, ropt, rema, x)
            assert ema.num_updates == rema.num_updates == 4
            raw = copy.deepcopy(trainable_state(restored))
            for name, state in trainable_state(model).items():
                for key, value in state.items():
                    assert torch.equal(value, raw[name][key])
                    assert torch.equal(ema.state[name][key], rema.state[name][key])
                    other = rema.state[name][key].clone()
                    dist.broadcast(other, 0)
                    assert torch.equal(other, rema.state[name][key])
            # Simulate a failure preparing EMA on rank1. Rank0 must restore
            # raw weights and every rank must fail before sampling begins.
            RECIPE["validation"] = dict(every=4, config="fake", assets={}, sampling_batch_size=32, weights="ema")
            args = SimpleNamespace(out=root / f"validation_lora{lora}")
            context = patch.object(rema, "average_parameters", side_effect=RuntimeError("injected swap failure"))
            from contextlib import nullcontext
            with patch("motion_expert_phase3_edge_joint.validation.read_configuration", return_value=({}, [], {})), \
                 patch("motion_expert_phase3_edge_joint.validation.sample_cases") as sampling, \
                 (context if rank == 1 else nullcontext()):
                with pytest.raises(RuntimeError, match="injected swap failure"):
                    maybe_validate(args, restored, 4, f"lora{lora}", rank, 2, io.StringIO(), ema=rema)
                sampling.assert_not_called()
            for name, state in trainable_state(restored).items():
                for key, value in state.items():
                    assert torch.equal(value, raw[name][key])
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_real_two_rank_ema_checkpoint_resume_and_collective_swap_failure(tmp_path):
    mp.spawn(ema_ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
