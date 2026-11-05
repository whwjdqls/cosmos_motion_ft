"""Render verified evaluation summaries for the requested Notion publication."""
from pathlib import Path
import hashlib
import json

OP=Path('/scratch2/minsu1206/cosmos/operations/egoallo_c2m_fulltest_20261105')
LABEL='egoallo_final200k_camera_shape'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 score=OP/'scores'/LABEL;root=OP/'models'/LABEL
 result=read(score/'summary.json');gt=read(OP/'scores/GT/summary.json');protocol=read(OP/'camera_to_motion_protocol.json')
 assert read(score/'COMPLETE.json')['summary_sha256']==sha(score/'summary.json')
 assert result['cases']==gt['cases']==11911 and result['recordings']==gt['recordings']==71
 assert read(root/'INFERENCE_COMPLETE.json')['cases']==11911
 assert result['input_configuration_sha256']==gt['input_configuration_sha256']==sha(OP/'configuration.json')
 assert result['evaluator_sha256']==gt['evaluator_sha256'] and result['text_cache_sha256']==gt['text_cache_sha256']
 pieces=[read(root/f'GT_extended_rank{i}.json') for i in range(2)]
 assert sum(v['cases'] for v in pieces)==11911
 assert all(v['cases_sha256']==sha(OP/'cases.json') for v in pieces)
 gt['physical'].update({k:sum(v['sums'][k] for v in pieces)/11911 for k in pieces[0]['sums']})
 lines=['## EgoAllo final200k — camera-to-motion results',
  '', 'Complete: all **11,911 annotation/window cases, 11,677 physical windows, 71 recordings**, using the frozen full-test cases and paired seeds. EgoAllo only; final 200,000-update checkpoint.',
  '', 'Inputs: measured camera trajectory plus supplied 16D train-fitted SOMA shape PCA. No RGB or text. This is camera-to-motion; the video-only results above use a different input task.',
  '', 'Native EgoAllo quadratic DDIM sampler, 30 updates; no SMPL guidance or hands. Supplied shape is held fixed throughout denoising and FK. Camera plus training-only camera/head calibration places predictions; no GT motion fit, head inpainting or prediction grounding. Same frozen Body23 completion, Nymeria TMR evaluator, text cache, 20 fps and T97 metrics as the full-test report.',
  '', '| Model | MPJPE cm | PA-MPJPE cm | Root ADE cm | Root FDE cm | UPE cm | LPE cm | sJPE | FID | R@1 % | R@2 % | R@3 % |',
  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
 for name,s in [('GT reference',gt),('EgoAllo supplied shape, final200k',result)]:
  v=s['physical'];r=s['retrieval'];vals=[v[k]*100 for k in ['mpjpe_m','pa_mpjpe_m','root_err_m','root_fde_m','upe_m','lpe_m']]+[v['sjpe'],s['FID']]
  cells=[f'{v:.6f}' for v in vals]+[f"{r[f'R32_R0{i}_mean']:.4f} ± {r[f'R32_R0{i}_std']:.4f}" for i in (1,2,3)]
  lines.append('| '+' | '.join([name,*cells])+' |')
 lines+=['','| Model | Foot skating cm/s | Mean foot penetration cm | Acceleration error m/s² | Head error cm | Noise sJPE | Static sJPE |',
  '| --- | --- | --- | --- | --- | --- | --- |']
 for name,s in [('GT reference',gt),('EgoAllo supplied shape, final200k',result)]:
  v=s['physical'];lines.append('| '+' | '.join([name,*[f'{v[k]:.6f}' for k in ['foot_skate_cm_s','foot_depth_mean_cm','accel_err_m_s2','head_error_cm','sjpe_noise','sjpe_static']]])+' |')
 audits=[read(root/f'acceptance_rank{i}.json') for i in range(2)]
 assert all(x['passed'] and x['replay_max_abs']==0 for x in audits)
 lines+=['','Both GPU acceptance checks passed: native same-seed replay bitwise identical; supplied shape exactly preserved. First-case sampling times: '+', '.join(f"GPU{i} {v['seconds_per_case']:.4f} s/case" for i,v in enumerate(audits))+'.',
  '', 'UPE/LPE use the existing 14 upper / 9 lower joint partition. PA-MPJPE uses whole-clip similarity alignment. sJPE is the existing four-end-effector mean, dimensionless 0–1; its native joint-sum form is four times larger. Retrieval uses the same three seeded R32 groupings (11,904 cases and seven dropped per seed). GT physical and retrieval metrics are measured references.',
  '', f"Artifacts: `{OP}`. Predictions: `models/{LABEL}/predictions/`. Full metrics: `scores/{LABEL}/summary.json`, `per_case.json`, `retrieval_groups.json`, `embeddings.npz`. Complete markers verify full inference and scoring.",
  '', f"Checkpoint SHA256: `{protocol['checkpoint_sha256']}`. Cases SHA256: `{sha(OP/'cases.json')}`. Result SHA256: `{sha(score/'summary.json')}`."]
 content='\n'.join(lines)
 (OP/'notion_results.md').write_text(content+'\n')
 payload=dict(content=content,summary_sha256=sha(score/'summary.json'),cases=11911,checkpoint_sha256=protocol['checkpoint_sha256'])
 (OP/'publication_payload.json').write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps(payload))

if __name__=='__main__':main()
