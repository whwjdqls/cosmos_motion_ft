"""Audited extensions to the unchanged shared trainer and camera audits."""
import ast
from pathlib import Path
import textwrap


def function_source(module, name, parent=None):
    source = Path(module.__file__).read_text()
    body = ast.parse(source).body
    if parent:
        body = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == parent).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(source, node))


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Shared implementation changed; review scratch overlay: '+old[:100])
    return source.replace(old, new)


def adapt_preflight(source):
    source = replace_once(source, 'reasoner_inputs=model.motion.prepare_conditions(batch["captions"])',
                         'reasoner_inputs=model.motion.prepare_conditions(inputs["captions"])')
    source = source.replace('clip_grad_norm_(model.bridges.parameters(),',
                            'clip_grad_norm_(trainable_parameters(model),')
    # The frozen-specialist diagnostic used peak LR immediately. Fresh motion
    # must be checked under the unchanged production warmup, including resume.
    source = replace_once(source, '    for step in range(12):\n',
        '    for step in range(12):\n'
        '        for group in optimizer.param_groups:\n'
        '            group["lr"] = lr_for_step(step + 1)\n')
    source = replace_once(source,
        '    # One real resumed optimizer update, not just deserialization.\n',
        '    # One real resumed optimizer update, not just deserialization.\n'
        '    for group in optimizer.param_groups:\n'
        '        group["lr"] = lr_for_step(13)\n')
    source = source.replace('model.bridges.state_dict()', 'trainable_state(model)')
    source = replace_once(source, 'model.bridges.load_state_dict(restored["bridges"], strict=True)',
                          'load_trainable_state(model, restored["bridges"])')
    source = source.replace('.contiguous().numpy().tobytes()',
                            '.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()')
    source = source.replace('bit_exact_zero_bridge_phase1_and_phase2_parity',
                            'bit_exact_zero_bridge_phase1_and_fresh_motion_parity')
    source = source.replace('"specialists_frozen"', '"cosmos_frozen_motion_trainable"')
    source = source.replace('DDP_bit_identical_bridge_parameters_and_save_reload_continue',
                            'DDP_bit_identical_bridge_and_motion_parameters_and_save_reload_continue')
    return source


def adapt_main(source):
    source = replace_once(source, '''    if not args.preflight and world not in (2, 4):
        raise ValueError("Production uses two or four allocated GPUs")''',
        '    camera_topology(world, args.microbatch)')
    source = replace_once(source, '''    if (not acceptance.get("passed") or acceptance["fingerprint"] != fingerprint
            or acceptance["world_size"] < 2):
        raise RuntimeError("Missing, stale or non-DDP acceptance gate")''',
        '    scratch_require_acceptance(acceptance, fingerprint, world, args.microbatch)')
    source = replace_once(source, 'int(STOP)',
        'int(STOP or (SCRATCH_PROBE_UPDATES > 0 and step >= start_step + SCRATCH_PROBE_UPDATES))')
    source = source.replace('clip_grad_norm_(model.bridges.parameters(),',
                            'clip_grad_norm_(trainable_parameters(model),')
    source = replace_once(source, 'model.bridges.load_state_dict(payload["bridges"], strict=True)',
        'model.bridges.load_state_dict(payload["bridges"], strict=True)\n'
        '        model.motion.load_motion_state_dict(payload["motion_state"], strict=True)')
    source = replace_once(source, '"bridges": model.bridges.state_dict(), "optimizer": optimizer.state_dict(),',
        '"bridges": model.bridges.state_dict(), "motion_state": model.motion.motion_state_dict(), "optimizer": optimizer.state_dict(),')
    source = source.replace('Non-finite bridge gradient;', 'Non-finite bridge/motion gradient;')
    return source
