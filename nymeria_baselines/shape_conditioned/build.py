"""Build isolated, inspectable shape-input baseline source overlays.

Never modify the upstream trees used by the shape-predicting references.
This performs source-file work only; model/data acceptance runs on compute.
"""
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import json
import shutil


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise RuntimeError(f'Upstream source drift: expected {count} matches: {old[:90]}')
    return text.replace(old, new)


def build(upstream, output):
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, directories in {
        'UniEgoMotion': ['config', 'dataset', 'model', 'module', 'mydiffusion', 'utils'],
        'egoallo': ['src'],
    }.items():
        for directory in directories:
            for source in sorted((upstream / name / directory).rglob('*')):
                if source.suffix not in {'.py', '.yaml', '.yml', '.json', '.typed'} or not source.is_file():
                    continue
                rel = source.relative_to(upstream)
                dest = output / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                files[str(rel)] = hashlib.sha256(source.read_bytes()).hexdigest()
                shutil.copyfile(source, dest)
    patches = []

    def edit(rel, operation):
        p = output / rel
        old = p.read_text()
        new = operation(old)
        ast.parse(new, filename=str(p))
        p.write_text(new)
        patches.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                          fromfile='reference/'+rel, tofile='shape_input/'+rel))

    def uem_model(s):
        s = replace(s, '"soma_body230": 230', '"soma_body230": 230, "soma_body220_shape": 220')
        s = replace(s, '"soma_body230": 18', '"soma_body230": 18, "soma_body220_shape": 18')
        s = replace(s, '        if cond_scale is not None:\n',
                    '        if self.repre_type == "soma_body220_shape":\n'
                    '            if "neutral_joints" not in y or y["neutral_joints"].shape != (x.shape[0], 30, 3):\n'
                    '                raise ValueError("Centered SOMA30 neutral skeleton [B,30,3] is required")\n'
                    '        if cond_scale is not None:\n')
        s = replace(s, '{"valid_frames": y["valid_frames"]})  # unconditional output',
                    '{"valid_frames": y["valid_frames"], **({"neutral_joints": y["neutral_joints"]} if "neutral_joints" in y else {})})  # retain actor shape in CFG null')
        s = replace(s, '        self.output_process = nn.Linear(self.latent_dim, self.input_feats)',
                    '        self.output_process = nn.Linear(self.latent_dim, self.input_feats)\n'
                    '        if self.repre_type == "soma_body220_shape":\n'
                    '            assert not self.cond_betas and self.encoder_tsfm is None\n'
                    '            self.shape2llm = nn.Sequential(nn.Linear(90, self.latent_dim), nn.GELU(),\n'
                    '                nn.LayerNorm(self.latent_dim), nn.Linear(self.latent_dim,self.latent_dim))\n'
                    '            self.shape_type_embed = nn.Parameter(torch.zeros(self.latent_dim))\n'
                    '            self.motion_modality_embed = nn.Parameter(torch.zeros(self.latent_dim))\n'
                    '            std = self.latent_dim ** -0.5\n'
                    '            for layer in self.shape2llm:\n'
                    '                if isinstance(layer, nn.Linear):\n'
                    '                    nn.init.trunc_normal_(layer.weight,std=std,a=-3*std,b=3*std)\n'
                    '                    nn.init.zeros_(layer.bias)\n'
                    '            for parameter in (self.shape_type_embed,self.motion_modality_embed):\n'
                    '                nn.init.trunc_normal_(parameter,std=std,a=-3*std,b=3*std)')
        s = replace(s, '        x = torch.cat((enc_time[:, None], x), axis=1)  # B x (T+1) x D',
                    '        prefix = [enc_time[:,None]]\n'
                    '        if self.repre_type == "soma_body220_shape":\n'
                    '            shape = self.shape2llm(y["neutral_joints"].reshape(B,90).to(x.dtype))\n'
                    '            shape = shape + self.shape_type_embed + self.motion_modality_embed\n'
                    '            prefix.append(shape[:,None])  # clean in-context token; no frame position\n'
                    '        prefix_count = len(prefix)\n'
                    '        x = torch.cat((*prefix,x),dim=1)')
        s = replace(s, 'torch.ones((B, 1), device=mask.device, dtype=mask.dtype)',
                    'torch.ones((B, prefix_count), device=mask.device, dtype=mask.dtype)')
        s = replace(s, 'self.output_process(x[:, 1 : 1 + T])',
                    'self.output_process(x[:, prefix_count : prefix_count + T])')
        return s

    def ego_model(s):
        # Shape is removed from the diffusion state and output heads entirely.
        s = replace(s, '            "betas": 16,\n', '')
        s = replace(s, '        # Helpers for converting between input dimensionality and latent dimensionality.\n',
                    '        self.shape_condition_encoder = nn.Sequential(\n'
                    '            nn.Linear(16, config.d_latent), Activation(),\n'
                    '            nn.Linear(config.d_latent, config.d_latent), Activation(),\n'
                    '            nn.Linear(config.d_latent, config.d_latent))\n\n'
                    '        # Helpers for converting between input dimensionality and latent dimensionality.\n')
        s = replace(s, 'return EgoDenoiseTraj.get_packed_dim(self.config.include_hands, self.config.body_joint_count, self.config.contact_joint_count)',
                    'return EgoDenoiseTraj.get_packed_dim(self.config.include_hands, self.config.body_joint_count, self.config.contact_joint_count) - 16')
        s = replace(s, '        t: Float[Tensor, "batch"],\n        *,\n',
                    '        t: Float[Tensor, "batch"],\n        *,\n        shape_betas: Float[Tensor, "batch 1 16"],\n')
        s = replace(s, '        x_t = EgoDenoiseTraj.unpack(x_t_packed, include_hands=self.config.include_hands, body_joint_count=config.body_joint_count, contact_joint_count=config.contact_joint_count)',
                    '        batch, time, _ = x_t_packed.shape\n'
                    '        if shape_betas.shape != (batch, 1, 16):\n'
                    '            raise ValueError("Clean actor shape [B,1,16] is required")\n'
                    '        x_t = EgoDenoiseTraj.unpack(torch.cat((shape_betas.expand(batch, time, 16), x_t_packed), -1), include_hands=self.config.include_hands, body_joint_count=config.body_joint_count, contact_joint_count=config.contact_joint_count)')
        s = replace(s, '            self.encoders["betas"](x_t.betas.reshape((batch, time, -1)))\n            + self.encoders["body_rotmats"]',
                    '            self.encoders["body_rotmats"]')
        s = replace(s, '        encoder_out = self.latent_from_cond(cond) + pos_enc',
                    '        encoder_out = self.latent_from_cond(cond) + self.shape_condition_encoder(shape_betas) + pos_enc')
        return s

    def ego_loss(s):
        s = replace(s, '        x_0_packed = x_0.pack()', '        x_0_packed = x_0.pack()[..., 16:]  # shape is never diffused')
        s = replace(s, '            x_t_packed=x_t_packed,',
                    '            x_t_packed=x_t_packed,\n            shape_betas=train_batch.betas,')
        s = replace(s, '            x_0_packed_pred, include_hands=unwrapped_model.config.include_hands,',
                    '            torch.cat((train_batch.betas.expand(batch, time, 16), x_0_packed_pred), -1), include_hands=unwrapped_model.config.include_hands,')
        s = replace(s, '            "betas": weight_and_mask_loss(\n                # (b, t, 16)\n                (x_0_pred.betas - x_0.betas) ** 2\n                # (16,)\n                * x_0.betas.new_tensor(self.config.beta_coeff_weights),\n            ),',
                    '            "betas": 0.0,  # supplied condition, no predicted shape or shape objective')
        return s

    edit('UniEgoMotion/model/uniegomotion.py', uem_model)
    edit('egoallo/src/egoallo/network.py', ego_model)
    edit('egoallo/src/egoallo/training_loss.py', ego_loss)
    (output / 'shape_input.patch').write_text(''.join(patches))
    manifest = dict(reference_files=files, output_files={
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob('*')) if p.is_file() and p.suffix in {'.py', '.yaml', '.yml', '.json'}
        and p.name != 'source_manifest.json'})
    (output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'source_files': len(files)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build(a.upstream, a.output)
