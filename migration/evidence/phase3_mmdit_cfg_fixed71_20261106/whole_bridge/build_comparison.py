"""Summarize the completed, checkpoint-matched guidance comparison."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
PLAN = json.loads((ROOT / 'plan.json').read_text())

def read(path):
    return json.loads(path.read_text())

def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    summaries, rows, pairs = {}, [], []
    validation = read(Path(PLAN['validation_config']))
    cases_manifest = read(Path(validation['cases']))
    assert len(cases_manifest) == 71 and [r['seed'] for r in cases_manifest] == list(range(71))
    assert len({r['uuid'] for r in cases_manifest}) == 71
    case_receipts = [{key:row[key] for key in
                     ['id','seed','uuid','start','split','bundle_sha256','rgb_sha256','camera_source_sha256']}
                    for row in cases_manifest]
    labels = {'cfg1':'CFG 1', 'edge_cfg2':'Cross-edge CFG 2', 'whole_cfg2':'Whole-bridge CFG 2'}
    for run in PLAN['runs']:
        name = run['attention']
        baseline = Path(run['run_dir']) / 'validation' / f"step_{run['step']:09d}"
        folders = {'cfg1': baseline, 'edge_cfg2':PARENT/name, 'whole_cfg2':ROOT/name}
        assert read(ROOT/name/'acceptance.json')['passed']
        assert read(ROOT/name/'GENERATION_COMPLETE.json')['predictions'] == 142
        assert read(PARENT/name/'cfg1_parity/COMPLETE.json')['matches_cached_native_all_bit_exact']
        for policy, folder in folders.items():
            assert (folder/'COMPLETE.json').exists()
            values = read(folder/'summary.json')
            summaries.setdefault(name, {})[policy] = values
            assert all(np.isfinite(value) for metrics in values.values() for value in metrics.values())
            for mode, metrics in values.items():
                for metric, value in metrics.items():
                    rows.append(dict(model=name, step=run['step'], policy=policy, mode=mode, metric=metric, value=value))
            if policy != 'cfg1':
                assert all(len(list((folder/mode).glob('*.npz'))) == 71 for mode in ('video2motion','motimg2video'))
        ref = read(baseline/'v2m_gt_reference.json')
        for folder in [PARENT/name, ROOT/name]:
            candidate = read(folder/'v2m_gt_reference.json')
            assert candidate.keys() == ref.keys()
            assert all(abs(candidate[k]-ref[k]) < 1e-6 for k in ref)
        evidence = dict(run=run, configuration=read(ROOT/name/'configuration.json'),
                        whole_bridge_acceptance=read(ROOT/name/'acceptance.json'),
                        native_batch32_parity=read(PARENT/name/'cfg1_parity/COMPLETE.json'),
                        summaries=summaries[name], gt_reference=ref, cases=case_receipts,
                        guidance_source=(ROOT/'whole_cfg.py').read_text(),
                        per_case={}, paired_changes=[])
        for mode, filename, keys in [
            ('v2m','v2m_per_case.json',['mpjpe_cm','pa_mpjpe_cm','root_ade_cm','skate_cm_s']),
            ('m2v','m2v_per_case.json',['dreamsim','lpips_alex','psnr_db','ssim'])]:
            cases = {policy:read(folder/filename) for policy,folder in folders.items()}
            assert all(len(value) == 71 for value in cases.values())
            if mode == 'm2v':
                ids = [v['id'] for v in cases['cfg1']]
                assert all([v['id'] for v in values] == ids for values in cases.values())
            evidence['per_case'][mode] = cases
            for policy, reference in [('edge_cfg2','cfg1'),('whole_cfg2','cfg1'),('whole_cfg2','edge_cfg2')]:
                for key in keys:
                    diff = np.array([v[key] for v in cases[policy]]) - np.array([v[key] for v in cases[reference]])
                    assert np.isfinite(diff).all()
                    draws = np.random.default_rng(20261106).integers(0,71,size=(10000,71))
                    ci = np.quantile(diff[draws].mean(1),[.025,.975]).tolist()
                    entry=dict(model=name,mode=mode,metric=key,policy=policy,reference=reference,
                               mean_change=float(diff.mean()),case_bootstrap95=ci)
                    pairs.append(entry)
                    evidence['paired_changes'].append(entry)
        (ROOT/f'{name}_evidence.json').write_text(json.dumps(evidence,separators=(',',':'))+'\n')

    lines=['## Sep-1 no-joint MMDiT 10k: whole-bridge CFG comparison', '',
           'This section evaluates the sep-1 caption-free directional and bidirectional models trained with 50% V2M / 50% M2V / 0% joint objective. These are different checkpoints from the sep-2 joint models elsewhere on this page. Both checkpoints are pinned at step 10,000, the latest saved checkpoints when this evaluation began. All 71 windows per task/model completed under each guidance policy. All model captions are empty.', '',
           'Whole-bridge CFG2 improves directional V2M versus CFG1: MPJPE 35.32→29.84 cm and FID 0.6182→0.5390. Bidirectional MPJPE is 38.99→40.97 cm, with its paired change interval crossing zero. Whole-bridge guidance substantially improves V2M over the earlier cross-edge reference for both models, but worsens M2V DreamSim and FVD versus CFG1 for both. A single guidance reference does not improve both tasks in this experiment.', '',
           '### Video → motion', '',
           '| Model | Guidance | FID ↓ | MPJPE cm ↓ | PA-MPJPE cm ↓ | Root ADE cm ↓ | Skate cm/s | R32 R@1 % ↑ |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, policies in summaries.items():
        for policy, values in policies.items():
            v=values['v2m']
            lines.append(f"| {name} | {labels[policy]} | {v['fid']:.4f} | {v['mpjpe_cm']:.2f} | {v['pa_mpjpe_cm']:.2f} | {v['root_ade_cm']:.2f} | {v['skate_cm_s']:.2f} | {v['R32_R01_mean']:.2f} |")
    lines += ['', '### Motion → video', '',
              '| Model | Guidance | I3D FVD ↓ | DreamSim ↓ | LPIPS-Alex ↓ | PSNR dB ↑ | SSIM ↑ |',
              '|---|---|---:|---:|---:|---:|---:|']
    for name, policies in summaries.items():
        for policy, values in policies.items():
            m=values['m2v']
            lines.append(f"| {name} | {labels[policy]} | {m['fvd']:.2f} | {m['dreamsim']:.4f} | {m['lpips_alex']:.4f} | {m['psnr_db']:.2f} | {m['ssim']:.4f} |")
    lines += ['', 'FVD here reuses the native sep-1 training validator: 426 descriptors (six 16-frame segments per window). The pooled 71-descriptor FVD reported for sep-2 above uses a different aggregation and must not be directly ranked against this table. RGB metrics score future frames 1–96. Lower skating alone does not establish improved reconstruction; GT retrieval reference metrics are included in the evidence files; no new GT physics baseline is computed here.', '',
              '### Paired whole-bridge changes', '',
              '| Model | Reference | Metric | Whole-bridge CFG 2 minus reference | Paired 95% interval |',
              '|---|---|---|---:|---|']
    for item in pairs:
        if item['policy'] == 'whole_cfg2' and item['metric'] in ['mpjpe_cm','dreamsim','lpips_alex']:
            low,high=item['case_bootstrap95']
            lines.append(f"| {item['model']} | {labels[item['reference']]} | {item['mode']} {item['metric']} | {item['mean_change']:+.5f} | [{low:+.5f}, {high:+.5f}] |")
    lines += ['', 'Intervals use 10,000 paired resamples of these 71 cases with a fixed seed. They describe cohort sampling variation, not variation across training seeds. No uncertainty or significance ranking is claimed for FID/FVD. Full quantitative values and all paired changes are attached.', '',
              '### Exact sampling protocol', '',
              '- Same fixed71 manifests, per-case seeds 0–70, bundle order, supplied neutral shapes and pinned frozen Phase1 video-I/O EMA200k + Phase2 200k specialists across policies.',
              '- Native 35-step UniPC; motion shift 3, video shift 10; 97 frames at 20 fps, 256×256. Batch 32 with the same virtual two-rank case/batch positions as the CFG1 validator; no batch-size benchmark is claimed.',
              '- V2M: clean observed video plus supplied neutral shape; motion starts as Gaussian noise in normalized 283-D space. M2V: full GT motion, supplied neutral shape, one initial image; future video starts as Gaussian noise. The first video latent is clamped throughout denoising.',
              '- Whole-bridge CFG 2: each step evaluates the frozen-specialist model with all seven learned bridges enabled, then with `bridges_enabled=False`. The reference skips bridge within-modality attention and cross-modal updates in both streams. Specialist attention and supplied conditioning tensors stay intact.',
              '- Cross-edge CFG 2: the earlier reference removes only target-query→conditioning-key edges. It retains bridge within-modality attention and the reverse cross-modal direction. This is a separate reference policy.',
              '- Both apply `reference + 2 × (conditioned − reference)`. For V2M the implementation blends predicted x0 before native conversion to velocity. Since state and sigma are shared, this is algebraically equivalent to blending the two velocity predictions. M2V blends velocity directly. This is non-text inference extrapolation; a dropped-modality null training distribution is not assumed.',
              '- No caption enters either specialist. External TMR retrieval uses reference captions only for scoring, with the pinned matched evaluator and completion/statistics assets.', '',
              '### Verification and provenance', '',
              'Both tasks/models passed bit-exact native-versus-wrapper CFG1 checks, reference invariance to the removed source, nonzero conditioned/reference differences, finite outputs, and exact initial-video-latent preservation. Hooks assert zero bridge calls in the whole-bridge reference and seven bridge calls per sample per conditioned step. A separate batch32 reproduction matched cached CFG1 outputs bit-for-bit for the first32 cases per task/model. No optimizer update occurs; bridge parameter version counters remain unchanged. Training processes were left active.', '']
    for run in PLAN['runs']:
        config=read(ROOT/run['attention']/'configuration.json')
        lines += [f"{run['attention']} checkpoint SHA256: `{config['checkpoint_sha256']}`; step {run['step']:,}.", '']
    lines += [f'Operation: `{ROOT}`. All sampling, metric and evidence outputs are node-local. The attachments contain the full tables, per-case metrics, acceptance receipts, source/evaluator fingerprints and paired changes. The complete sampled latent arrays remain in the operation directory.', '']
    (ROOT/'comparison.md').write_text('\n'.join(lines))
    dump(ROOT/'comparison.json',summaries)
    dump(ROOT/'paired_changes.json',pairs)
    with (ROOT/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','step','policy','mode','metric','value'])
        writer.writeheader();writer.writerows(rows)
    dump(ROOT/'METRICS_COMPLETE.json',dict(models=2,windows_per_task=71,new_whole_bridge_predictions=284,
                                         policies=['cfg1','edge_cfg2','whole_cfg2']))
    print(json.dumps(summaries,indent=2))

if __name__ == '__main__':
    main()
