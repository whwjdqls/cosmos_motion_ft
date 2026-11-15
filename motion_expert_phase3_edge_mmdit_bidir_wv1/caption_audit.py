"""Native acceptance check for both two-task specialist conditioning paths."""
import torch


@torch.no_grad()
def audit_no_captions(model, raw, noise_batch, device, modes):
    poison = "PRIVATE_DATASET_CAPTION_MUST_NEVER_REACH_EITHER_PHASE3_SPECIALIST"
    prepare, pack = model.motion.prepare_conditions, model.edge.pack_video
    seen = {"motion": [], "video": []}
    training = model.training

    def prepare_empty(captions):
        assert captions == [""], "Dataset caption reached the motion reasoner"
        seen["motion"].append(captions.copy())
        return prepare(captions)

    def pack_empty(captions, task_modes, latents, sigma):
        assert captions == [""], "Dataset caption reached the video reasoner"
        seen["video"].append(captions.copy())
        return pack(captions, task_modes, latents, sigma)

    model.motion.prepare_conditions, model.edge.pack_video = prepare_empty, pack_empty
    model.eval()
    checked = []
    try:
        for mode in modes:
            batch = {**raw, "modes": [mode], "captions": [poison]}
            inputs, _ = noise_batch(batch, device)
            assert inputs["captions"] == [""], "Dataset caption survived Phase3 noising"
            blank = model(**inputs)
            injected = model(**{**inputs, "captions": [poison]})
            for key in ("motion", "video"):
                torch.testing.assert_close(blank[key], injected[key], rtol=0, atol=0)
            checked.append(mode)
            del blank, injected
    finally:
        model.motion.prepare_conditions, model.edge.pack_video = prepare, pack
        model.train(training)
    assert all(len(values) == 2 * len(checked) for values in seen.values())
    return dict(passed=True, modes=checked, specialist_caption_inputs_empty=True,
                poisoned_caption_outputs_bit_exact=True, calls={k: len(v) for k, v in seen.items()})
