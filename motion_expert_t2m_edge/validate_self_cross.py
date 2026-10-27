"""GPU acceptance checks for the explicitly requested SA/CA experiments."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import torch
import config
import model as model_module
from attention import build_layout, motion_self_attention
from checkpoint import configure_attention_from_checkpoint, load_checkpoint, save_checkpoint, validate_checkpoint_contract
from self_cross import encode_reasoner_memory, motion_cross_attention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--legacy-model", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = model_module.EdgePhase2MotionExpert(device="cuda", dtype=torch.bfloat16)
    model.eval()
    mode = model.attention_mode
    assert mode in {"self_cross_final_hidden", "self_cross_layer_kv"}
    assert not any("reasoner_k_norm_motion" in n for n, _ in model.named_parameters())
    image = torch.zeros(3, 256, 256, dtype=torch.uint8)
    conditions = model.prepare_conditions(
        ["The camera wearer walks forward.", "The camera wearer raises one hand.", ""],
        modes=["text2motion", "textimg2motion", "text2motion"],
        reasoner_images=[None, image, None], image_size=256,
    )
    lengths = [5, 9, 3]
    pad = torch.arange(9, device="cuda")[None] >= torch.tensor(lengths, device="cuda")[:, None]
    inputs = dict(
        reasoner_inputs=conditions, x_sigma=torch.randn(3, 9, 283, device="cuda"),
        sigma=torch.tensor([0.2, 0.7, 1.0], device="cuda"),
        neutral_joints=torch.randn(3, 30, 3, device="cuda") * 0.1,
        motion_pad_mask=pad,
    )
    parts, positions, text_lengths = [], [], []
    for c, n in zip(conditions, lengths):
        parts.extend((c["inputs_embeds"].to(dtype=torch.bfloat16), torch.zeros(n + 1, 2048, device="cuda", dtype=torch.bfloat16)))
        text_lengths.append(c["inputs_embeds"].shape[0])
        positions.extend((c["position_ids"], model_module._motion_positions(n, int(c["next_position_id"]), torch.device("cuda"))))
    layout = build_layout(text_lengths, [n + 1 for n in lengths], "cuda")
    packed = torch.cat(parts)
    cos, sin = model._edge.rope(torch.cat(positions, dim=1))
    with torch.no_grad():
        actual_text, memory = encode_reasoner_memory(model.layers, packed, layout, cos, sin, model.backbone.norm, mode)
        reference = packed
        native_checks = 0
        for index, layer in enumerate(model.layers):
            if layer.has_motion and mode == "self_cross_layer_kv":
                f = layer._frozen
                normalized = f.input_norm(reference.index_select(0, layout.text_idx))
                k = f.k_norm(f.k_proj(normalized).view(-1, 8, 128))
                _, k = layer._rotary(k, k, cos.index_select(0, layout.text_idx), sin.index_select(0, layout.text_idx))
                v = f.v_proj(normalized).view(-1, 8, 128)
                assert torch.equal(k, memory[index][0]) and torch.equal(v, memory[index][1])
                native_checks += 1
            reference = layer._update_reasoner_only(reference, layout, cos, sin)
        assert torch.equal(actual_text, reference.index_select(0, layout.text_idx))
        if mode == "self_cross_final_hidden":
            assert torch.equal(memory, model.backbone.norm(actual_text))
        else:
            assert native_checks == 7

    small = build_layout([3, 4], [5, 3], "cuda")
    q = torch.randn(8, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(7, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cross_before = motion_cross_attention(q, k, v, small)
    k2, v2 = k.clone(), v.clone()
    k2[3:] += 10
    v2[3:] += 10
    cross_after = motion_cross_attention(q, k2, v2, small)
    assert torch.equal(cross_before[:5], cross_after[:5])
    mk, mv = torch.randn(8, 8, 128, device="cuda", dtype=torch.bfloat16), torch.randn(8, 8, 128, device="cuda", dtype=torch.bfloat16)
    sa_before = motion_self_attention(q, mk, mv, small)
    mk[5:] += 10
    mv[5:] += 10
    assert torch.equal(sa_before[:5], motion_self_attention(q, mk, mv, small)[:5])

    prediction = model(**inputs)
    assert torch.isfinite(prediction).all() and torch.count_nonzero(prediction[pad]) == 0
    prediction.square().mean().backward()
    missing = [n for n, p in model.named_trainable_parameters() if p.grad is None]
    nonfinite = [n for n, p in model.named_trainable_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not missing and not nonfinite, (missing, nonfinite)
    assert all(p.grad is None for p in model.frozen_parameters())
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        expected = model(**inputs).clone()
    with tempfile.TemporaryDirectory(prefix="roundtrip_", dir=args.out) as directory:
        path = Path(directory) / "roundtrip.pt"
        save_checkpoint(path, model=model, optimizer=None, step=0, args={"task_weights": config.TASK_WEIGHTS})
        configure_attention_from_checkpoint(path)
        with torch.no_grad():
            next(model.trainable_parameters()).add_(0.125)
        payload = load_checkpoint(path, model=model)
        with torch.no_grad():
            assert torch.equal(expected, model(**inputs))
        bad = copy.deepcopy(payload["contract"])
        bad["motion_attention_mode"] = "self_cross_layer_kv" if mode == "self_cross_final_hidden" else "self_cross_final_hidden"
        try:
            validate_checkpoint_contract({"contract": bad})
        except RuntimeError:
            pass
        else:
            raise AssertionError("incompatible cross checkpoint accepted")
        del payload

    # Compare the unmodified legacy forward against the new default joint
    # branch using exactly the same registered model and frozen backbone.
    spec = importlib.util.spec_from_file_location("legacy_model_reference", args.legacy_model)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    old_mode, old_bypass = config.MOTION_ATTENTION_MODE, config.REASONER_K_NORM_BYPASS
    factory = model_module.FrozenEdgeReasoner
    try:
        config.MOTION_ATTENTION_MODE, config.REASONER_K_NORM_BYPASS = "joint", False
        model_module.FrozenEdgeReasoner = lambda **kwargs: model._edge
        joint = model_module.EdgePhase2MotionExpert(device="cuda", dtype=torch.bfloat16, verbose=False)
        joint.eval()
        with torch.no_grad():
            assert torch.equal(joint(**inputs), legacy.EdgePhase2MotionExpert.forward(joint, **inputs))
        legacy_contract = config.architecture_contract()
        for key in config.attention_mode_contract("joint"):
            legacy_contract.pop(key)
        validate_checkpoint_contract({"contract": legacy_contract})
        del joint
    finally:
        model_module.FrozenEdgeReasoner = factory
        config.MOTION_ATTENTION_MODE, config.REASONER_K_NORM_BYPASS = old_mode, old_bypass
    report = dict(
        status="passed", mode=mode, trainable_parameters=sum(p.numel() for p in model.trainable_parameters()),
        frozen_reasoner_bitwise_equal=True, memory_source_verified=True,
        ragged_padding_and_empty_caption=True, sample_isolation=True,
        all_trainable_gradients_finite=True, frozen_gradients_absent=True,
        checkpoint_roundtrip_bitwise_equal=True, incompatible_resume_rejected=True,
        legacy_joint_forward_bitwise_equal=True, legacy_contract_accepted=True,
    )
    (args.out / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

