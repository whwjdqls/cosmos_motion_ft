"""Pin the actual sanitized evaluation records and native checkpoint contract."""
import argparse
import copy
import json
from pathlib import Path
from .camera import sha256

MODES = {'image2video': 'i2v_input.jsonl', 'inverse_dynamics': 'invdyn_input.jsonl'}
CONTENT_FIELDS = {'name', 'vision_path', 'action_path', 'prompt', 'seed'}


def load_uniform(path, mode):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records or any(r.get('model_mode') != mode for r in records):
        raise ValueError('Evaluation input file has missing/mismatched modes')
    first = {k:v for k,v in records[0].items() if k not in CONTENT_FIELDS}
    if any({k:v for k,v in r.items() if k not in CONTENT_FIELDS} != first for r in records):
        raise ValueError('Select a uniform evaluation cohort/profile')
    return records[0]


def export(checkpoint, evaluation_inputs, out):
    from native_phase_training.run_contract import resolve_eval_contract
    contract, source, checkpoint, run = resolve_eval_contract(Path(checkpoint))
    if contract.model_family != 'edge':
        raise ValueError('This application currently requires an Edge native Phase1 checkpoint')
    templates, evidence = {}, {}
    for mode, filename in MODES.items():
        path = Path(evaluation_inputs) / filename
        template = load_uniform(path, mode)
        if (template['shift'] != contract.inference_shift or template['num_frames'] != 97
                or template['fps'] != 20 or str(template['resolution']) != '256'
                or template['aspect_ratio'] != '1,1'):
            raise ValueError('Use sanitized evaluation inputs matching the resolved checkpoint; legacy shift3 fixtures are not accepted')
        templates[mode] = template
        evidence[mode] = dict(path=str(path.resolve()), sha256=sha256(path))
    metadata = Path(checkpoint) / 'model/.metadata'
    profile = dict(schema=1, checkpoint=str(checkpoint), contract=contract.to_dict(),
        checkpoint_metadata_sha256=sha256(metadata), contract_source=source,
        evaluation_inputs=evidence, templates=templates, sampler='unipc', ema=True,
        config_file='native_phase_training/inference_config.py',
        experiment='world_camera_nymeria_latent_edge', diffusion_cache=True,
        compile=False, cuda_graphs=False, guardrails=False,
        framework_revision=__import__('subprocess').check_output(
            ['git','-C',__import__('os').environ['COSMOS_FRAMEWORK_EDGE_ROOT'],'rev-parse','HEAD'],text=True).strip())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(profile, indent=2) + '\n')
    return profile


def make_record(profile, mode, name, vision_path, prompt, seed):
    from native_phase_training.sanitize_prefix_inference_inputs import sanitize_record
    template = profile['templates'][mode]
    record = copy.deepcopy(template)
    record.update(name=name + '_' + mode, vision_path=str(Path(vision_path).resolve()),
                  prompt=prompt if mode == 'image2video' else '', seed=int(seed))
    record.pop('action_path', None)
    record = sanitize_record(record, mode, model_family='edge', replace_standalone_c=True,
                             standalone_c_subject='camera_wearer')
    for key, value in template.items():
        if key not in CONTENT_FIELDS and record.get(key) != value:
            raise ValueError('Inference setting changed from evaluation: ' + key)
    if mode == 'image2video':
        # The action description is new; every other evaluated prompt field
        # must remain identical, including native duration/resolution metadata.
        reference_prompt = json.loads(template['prompt'])
        actual_prompt = json.loads(record['prompt'])
        reference_prompt['actions'][0]['description'] = '<application action>'
        actual_prompt['actions'][0]['description'] = '<application action>'
        if reference_prompt != actual_prompt:
            raise ValueError('Structured prompt scaffold differs from evaluated I2V template')
    return record


def verify(profile):
    from native_phase_training.run_contract import resolve_eval_contract
    contract, _, checkpoint, _ = resolve_eval_contract(Path(profile['checkpoint']))
    if json.loads(json.dumps(contract.to_dict())) != profile['contract'] or sha256(checkpoint/'model/.metadata') != profile['checkpoint_metadata_sha256']:
        raise ValueError('Frozen checkpoint/contract changed since profile export')
    for mode, source in profile['evaluation_inputs'].items():
        if sha256(source['path']) != source['sha256']:
            raise ValueError('Evaluation reference inputs changed since profile export')
        if load_uniform(source['path'], mode) != profile['templates'][mode]:
            raise ValueError('Profile template differs from evaluation record')
    if profile['sampler'] != 'unipc' or profile['ema'] is not True:
        raise ValueError('Only the evaluated EMA UniPC path is supported')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--evaluation-inputs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    profile = export(args.checkpoint, args.evaluation_inputs, args.out)
    print(json.dumps({mode:{k:v for k,v in r.items() if k not in CONTENT_FIELDS and k != 'negative_prompt'}
                      for mode,r in profile['templates'].items()}, indent=2))


if __name__ == '__main__':
    main()
