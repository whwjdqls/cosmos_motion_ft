"""Score the joint-trained matched71 V2M predictions and publish paired results."""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import runpy
import time
import traceback

import numpy as np
import torch

from nymeria_baselines.body23_tmr_adapter import KEEP, complete_body
from nymeria_baselines.compare_v2m_guidance_tmr import PHYSICAL, physical
from nymeria_baselines.evaluate_reconstruction_tmr import (
    PAGE_ID, PAGE_URL, atomic_npz, fid, retrieval_details, stable_hash,
)
from nymeria_baselines.evaluate_uem_phase3 import read, save, sha
from nymeria_baselines.uem_visual_cache import require_compute
from motion_expert_phase3_edge.settings import digest_json


def progress(args, stage, **values):
    row = dict(stage=stage, time=time.time(), job=os.environ.get('SLURM_JOB_ID'), **values)
    save(args.output / 'scoring_status.json', row)
    print(json.dumps(row), flush=True)


def publish(args, summary):
    helper = runpy.run_path(str(args.notion_helper))
    api = helper['Notion'](helper['read_token']())
    rt = helper['rich_text']
    publication = args.output / 'notion_publication.json'
    if publication.exists():
        receipt = read(publication)
        assert receipt['summary_sha256'] == sha(args.output / 'summary.json')
        blocks = [api.request('GET', '/blocks/'+x) for x in receipt['block_ids']]
        assert all(not b.get('archived') for b in blocks)
        return
    def para(value):
        return dict(object='block', type='paragraph', paragraph=dict(rich_text=rt(value)))
    def table(rows):
        return dict(object='block', type='table', table=dict(table_width=len(rows[0]),
            has_column_header=True, has_row_header=False,
            children=[dict(object='block', type='table_row', table_row=dict(cells=[rt(str(x)) for x in r])) for r in rows]))
    geometry = [['Model / conditioning', 'MPJPE cm ↓', 'PA-MPJPE cm ↓', 'Root ADE cm ↓',
                 'Skate cm/s', 'Penetration cm ↓', 'Airborne %']]
    semantics = [['Model / conditioning', 'FID ↓', 'R@1 % ↑', 'R@2 % ↑', 'R@3 % ↑']]
    for row in summary['rows']:
        p, r = row['physical'], row['retrieval']
        geometry.append([row['label']] + [f'{p[k]:.2f}' for k in
            ('MPJPE cm', 'PA-MPJPE cm', 'Root ADE cm', 'Skate cm/s', 'Penetration cm', 'Airborne %')])
        semantics.append([row['label'], f"{row['FID']:.6f}"] +
            [f"{r[f'R32_R{k:02d}_mean']:.2f} ± {r[f'R32_R{k:02d}_std']:.2f}" for k in (1,2,3)])
    deltas = [['Joint minus original, same setting', 'Δ MPJPE cm', 'Paired 95% CI', 'Δ FID', 'Δ R@3 pp']]
    for row in summary['paired_differences']:
        d = row['physical']['MPJPE cm']
        deltas.append([row['setting'], f"{d['mean']:+.2f}",
                       f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]",
                       f"{row['FID']:+.6f}", f"{row['R3_pp']:+.2f}"])
    title = 'Joint-trained Phase3 200k versus original200k | matched V2M | 2026-10-29'
    # A unique report title prevents accidental duplicate publication after an
    # interrupted write. An existing section without a receipt needs inspection.
    cursor = ''
    while True:
        suffix = '?page_size=100' + ('&start_cursor='+cursor if cursor else '')
        response = api.request('GET', '/blocks/'+PAGE_ID+'/children'+suffix)
        if any(b['type'] == 'heading_2' and helper['plain_text'](b['heading_2']['rich_text']) == title for b in response['results']):
            raise RuntimeError('Report title already exists; inspect before duplicate append')
        if not response.get('has_more'):
            break
        cursor = response['next_cursor']
    blocks = [dict(object='block', type='heading_2', heading_2=dict(rich_text=rt(title))),
        para('Complete: 71/71 fixed T97 windows, one per held-out recording, in each of four prespecified settings (284 new predictions). This evaluates V2M reconstruction of the joint-trained checkpoint, not the quality of its joint video+motion samples. Original results are reused from this page’s exact matched protocol.'),
        table(geometry), table(semantics), table(deltas),
        para(summary['interpretation']),
        para('Protocol: 97 frames at20fps; UniPC35, motion shift3; identical cases, per-case seeds, video latents, captions and supplied neutral shapes. No camera trajectory, GT head inpainting, generated-motion grounding or root alignment. Physical metrics use the unchanged Body23 subset (only PA-MPJPE aligns). Native GT and original-checkpoint evaluator parity checks passed.'),
        para('Guidance2 exactly matches the older table: reference disables entire modality bridges, including their within-motion attention; captions stay in BOTH branches. It is bridge-disabled extrapolation, not a trained missing-video classifier-free null or pure text CFG. Caption-conditioned V2M is an inference intervention for both checkpoints.'),
        para('FID and retrieval use the same validation-selected Nymeria TMR checkpoint, evaluator normalization, training-only Body23-to-SOMA30 completion and20→30fps conversion. FID uses all71. R@1/2/3 uses32 paired candidates and grouping seeds0/1/2 (64 cases per seed,7 dropped), mean ± sample SD. This is a small-sample diagnostic, separate from Phase2 full-test scores. GT is measured with the same completion.'),
        para('Both checkpoints are200k and use the same frozen Phase1 EMA200k / Phase2 Self→cross caption-only B256200k specialists. Original training mixes V2M/M2V50:50; joint training mixes V2M/M2V/joint1:1:1. Thus task exposure and supervision differ despite equal update counts. This comparison alone cannot isolate why performance changes.'),
        para('New checkpoint: '+summary['checkpoint']),
        para('Evidence: '+str(args.output)+' — configuration.json, scoring_configuration.json, summary.json, comparison.csv, per-case physical metrics, embeddings, retrieval groups, paired bootstrap differences and source hashes.')]
    response = api.request('PATCH', '/blocks/'+PAGE_ID+'/children', dict(children=blocks))
    receipt = dict(status='written_pending_verification', url=PAGE_URL,
                   block_ids=[b['id'] for b in response['results']], summary_sha256=sha(args.output/'summary.json'))
    save(publication, receipt)
    for expected, block in zip(blocks, response['results'], strict=True):
        actual = api.request('GET', '/blocks/'+block['id'])
        assert actual['type'] == expected['type'] and not actual.get('archived')
        if actual['type'] == 'table':
            children = api.request('GET', '/blocks/'+block['id']+'/children?page_size=100')['results']
            want = expected['table']['children']
            assert len(children) == len(want)
            for a, b in zip(children, want, strict=True):
                assert [helper['plain_text'](c) for c in a['table_row']['cells']] == [helper['plain_text'](c) for c in b['table_row']['cells']]
        else:
            kind = expected['type']
            assert helper['plain_text'](actual[kind]['rich_text']) == helper['plain_text'](expected[kind]['rich_text'])
    receipt.update(status='published_verified', verified_time=time.time())
    save(publication, receipt)
    status_path = args.output/'notion_progress.json'
    if status_path.exists():
        status = read(status_path)
        message = ('2026-10-29: COMPLETE — joint-trained Phase3 200k matched V2M evaluation: '
                   '71/71 windows in all four settings (284 predictions). Quantitative results, '
                   'GT and original200k comparisons are in the new Joint-trained Phase3 section below. '
                   'Evidence: '+str(args.output))
        api.request('PATCH','/blocks/'+status['block_id'],
                    dict(paragraph=dict(rich_text=rt(message))))
        actual = api.request('GET','/blocks/'+status['block_id'])
        assert helper['plain_text'](actual['paragraph']['rich_text']) == message
        save(status_path,dict(block_id=status['block_id'],text=message,status='complete_verified'))


@torch.inference_mode()
def main(args):
    require_compute()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.15, 0)
    lock = (args.output / 'scorer.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(args.output/'configuration.json')
    base = read(args.base/'configuration.json')
    previous = read(args.previous/'summary.json')
    assert previous['status'] == 'complete'
    cases, ids = config['cases'], [r['id'] for r in config['cases']]
    assert len(ids) == len(set(ids)) == 71
    for a, b in zip(cases, base['cases'], strict=True):
        assert all(a[k] == b[k] for k in ('id','uuid','start','seed','bundle','caption','floor_offset'))
    template = read(args.base/'adapter.json')
    assert sha(args.base/'adapter.json') == base['adapter_sha256']
    assert sha(base['evaluator_checkpoint']) == base['evaluator_sha256']
    assert sha(base['text_cache']) == base['text_cache_sha256']
    contract = config['sources']['phase2_contract']
    for field in ('mean','std'):
        assert sha(contract['motion_stats_'+field]) == contract['motion_'+field]
    mean, std = np.load(contract['motion_stats_mean']), np.load(contract['motion_stats_std'])
    with np.load(args.base/'GT_embeddings.npz', allow_pickle=False) as a:
        saved_gt = a['completed'].copy()
        assert a['case_id'].tolist() == ids and str(a['signature']) == stable_hash(base)
    with np.load(args.base/'text_embeddings.npz', allow_pickle=False) as a:
        text = a['embedding'].copy()
        assert a['case_id'].tolist() == ids
    scoring_config = dict(evaluator=base, adapter_sha256=sha(args.base/'adapter.json'),
        inference_configuration_sha256=sha(args.output/'configuration.json'),
        previous_summary_sha256=sha(args.previous/'summary.json'), source_sha256=sha(__file__),
        GT_embeddings_sha256=sha(args.base/'GT_embeddings.npz'), text_embeddings_sha256=sha(args.base/'text_embeddings.npz'))
    save(args.output/'scoring_configuration.json', scoring_config)
    from decode_uniego_torch import decode_transforms
    from nymeria_eval import NymeriaTMREmbedder
    progress(args, 'loading_evaluator')
    embedder = NymeriaTMREmbedder(base['evaluator_checkpoint'], base['evaluator_stats'], base['text_cache'], 'cuda')
    assert embedder.fps == 30
    recomputed_text = embedder.embed_text([r['caption'] for r in cases]).cpu().numpy()
    text_difference = float(np.max(np.abs(recomputed_text-text)))
    save(args.output/'text_parity.json', dict(max_abs_difference=text_difference,
        gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        bitwise_equal=np.array_equal(recomputed_text,text)))
    # The reference was produced on another GPU architecture. Preserve its
    # exact saved text embeddings for all scoring; audit recomputation within
    # FP32 roundoff instead of claiming cross-architecture bitwise identity.
    assert text_difference < 1e-6, text_difference
    def embed(features, shapes):
        positions, neutrals = [], []
        for raw, neutral in zip(features, shapes, strict=True):
            tr = decode_transforms(torch.from_numpy(raw)[None])[0].numpy()
            j = tr[..., :3, 3][:, KEEP]
            full, rest = complete_body(j, tr[..., :3, :3][:, KEEP], neutral[KEEP], template)
            assert np.array_equal(full[:, KEEP], j)
            positions.append(full)
            neutrals.append(rest)
        values = []
        for start in range(0, len(features), 16):
            j = torch.from_numpy(np.stack(positions[start:start+16]).astype(np.float32))
            n = torch.from_numpy(np.stack(neutrals[start:start+16]).astype(np.float32))
            values.append(embedder.embed_motion(j, n, lengths=[97]*len(j), source_fps=20).cpu().numpy())
        result = np.concatenate(values)
        assert result.shape == (71,256) and np.isfinite(result).all()
        return result
    motions, shapes, input_hashes = [], [], []
    for row in cases:
        with np.load(row['bundle'], allow_pickle=False) as a:
            motions.append(a['motion'].copy())
            shapes.append(a['neutral'].copy())
        input_hashes.append(dict(id=row['id'], sha256=sha(row['bundle'])))
    save(args.output/'input_hashes.json', input_hashes)
    recomputed_gt = embed([m*std+mean for m in motions], shapes)
    gt_parity = float(np.max(np.abs(recomputed_gt-saved_gt)))
    assert gt_parity <= 1e-4, gt_parity
    gt_r, gt_ranks = retrieval_details(recomputed_gt, text, ids)
    gt_row = next(r for r in previous['rows'] if r['key'] == 'GT').copy()
    assert gt_r == gt_row['retrieval']
    # Also reproduce the saved original model's scores with this process.
    old_features = []
    reference = Path(config['reference'])
    for row in cases:
        with np.load(reference/'video_cfg_1'/(row['id']+'.npz'), allow_pickle=False) as a:
            old_features.append(a['generated']*std+mean)
    recomputed_old = embed(old_features, shapes)
    with np.load(args.previous/'video__video_cfg_1/embeddings.npz', allow_pickle=False) as a:
        baseline_parity = float(np.max(np.abs(recomputed_old-a['embedding'])))
    assert baseline_parity <= 1e-4, baseline_parity
    save(args.output/'evaluator_acceptance.json', dict(GT_max_abs_embedding_difference=gt_parity,
        original_max_abs_embedding_difference=baseline_parity, GT_retrieval_exact=True,
        text_embeddings_max_abs_difference=text_difference, saved_text_embeddings_reused_exactly=True,
        retained_body_joints_exact=True))
    atomic_npz(args.output/'GT_embeddings.npz', embedding=recomputed_gt, case_id=np.asarray(ids))
    save(args.output/'GT_retrieval_groups.json', gt_ranks)
    progress(args, 'evaluator_verified', GT_R3=gt_r['R32_R03_mean'], GT_parity=gt_parity, baseline_parity=baseline_parity)
    deadline = time.monotonic()+6*3600
    while not all((args.output/f'shard{s}_COMPLETE.json').exists() for s in range(4)):
        for s in range(4):
            status = args.output/f'shard{s}_status.json'
            if status.exists() and read(status)['stage'] == 'failed':
                raise RuntimeError(f'Inference shard{s} failed')
        if time.monotonic() > deadline:
            raise TimeoutError('Waiting for inference exceeded six hours')
        time.sleep(20)
    rows, differences = [gt_row], []
    rng = np.random.default_rng(20261029)
    resamples = rng.integers(0,71,size=(5000,71))
    for arm in ('video','caption'):
        native = [read(args.output/arm/'per_case'/(r['id']+'.json')) for r in cases]
        for s in range(4):
            assert read(args.output/f'shard{s}_COMPLETE.json')['configuration_sha256'] == sha(args.output/'configuration.json')
        gt_physical = physical(native,'GT')
        assert max(abs(gt_physical[k]-gt_row['physical'][k]) for k in PHYSICAL) < .00011
        gt_row['physical'] = gt_physical
        save(args.output/arm/'per_case.json', native)
        old_folder = reference if arm == 'video' else reference/'video_and_text'
        old_by_id = {r['id']:r for r in read(old_folder/'per_case.json')}
        for scale in (1,2):
            setting = f'video_cfg_{scale}'
            key = arm+'__'+setting
            old_row = next(r for r in previous['rows'] if r['key'] == key).copy()
            old_row['label'] = 'Original200k: '+('video + caption' if arm == 'caption' else 'video')+f', scale{scale}'
            features, source_files = [], []
            for i, row in enumerate(cases):
                path = args.output/arm/setting/(row['id']+'.npz')
                with np.load(path, allow_pickle=False) as a:
                    assert str(a['signature']) == digest_json(config)
                    assert int(a['seed']) == row['seed'] and float(a['video_scale']) == scale
                    assert np.array_equal(a['gt_motion'], motions[i]) and np.array_equal(a['neutral'], shapes[i])
                    assert str(a['caption']) == (row['caption'] if arm == 'caption' else '')
                    features.append(a['generated']*std+mean)
                source_files.append(dict(path=str(path),sha256=sha(path)))
            embedding = embed(features, shapes)
            retrieval, ranks = retrieval_details(embedding,text,ids)
            new = dict(key='joint__'+key,label=old_row['label'].replace('Original200k','Joint200k'),
                training_step=200000, input=old_row['input'], setting=old_row['setting'],
                FID=fid(embedding,saved_gt),retrieval=retrieval,physical=physical(native,setting),
                prediction_hashes=source_files)
            dest = args.output/'scores'/key
            atomic_npz(dest/'embeddings.npz',embedding=embedding,case_id=np.asarray(ids))
            save(dest/'retrieval_groups.json',ranks)
            save(dest/'summary.json',new)
            delta = dict(setting=('video + caption' if arm == 'caption' else 'video')+f', scale{scale}',
                FID=new['FID']-old_row['FID'],R3_pp=retrieval['R32_R03_mean']-old_row['retrieval']['R32_R03_mean'],physical={})
            from nymeria_baselines.compare_v2m_guidance_tmr import METRICS
            for label,(metric,factor) in zip(PHYSICAL,METRICS,strict=True):
                values=np.array([(r[setting][metric]-old_by_id[r['id']][setting][metric])*factor for r in native])
                delta['physical'][label]=dict(mean=float(values.mean()),ci95=np.quantile(values[resamples].mean(1),[.025,.975]).tolist(),
                    per_case=values.tolist())
            rows.extend([old_row,new]); differences.append(delta)
            progress(args,'scored',setting=key,FID=new['FID'],R3=retrieval['R32_R03_mean'],MPJPE=new['physical']['MPJPE cm'])
    native_delta, caption_delta = differences[0], differences[-1]
    interpretation = (f"Joint minus original: native video-only MPJPE {native_delta['physical']['MPJPE cm']['mean']:+.2f}cm, "
        f"FID {native_delta['FID']:+.6f}, R@3 {native_delta['R3_pp']:+.2f}pp; video+caption/scale2 "
        f"MPJPE {caption_delta['physical']['MPJPE cm']['mean']:+.2f}cm, FID {caption_delta['FID']:+.6f}, "
        f"R@3 {caption_delta['R3_pp']:+.2f}pp. Negative error/FID changes and positive retrieval changes are favorable. "
        'Paired physical intervals bootstrap71 recordings5000 times (seed20261029); FID/retrieval differences are descriptive.')
    summary=dict(status='complete',cases=71,new_predictions=284,rows=rows,paired_differences=differences,
        checkpoint=config['checkpoint'],checkpoint_sha256=config['checkpoint_sha256'],interpretation=interpretation,
        evaluator_acceptance=read(args.output/'evaluator_acceptance.json'))
    save(args.output/'summary.json',summary)
    stream=io.StringIO(); writer=csv.writer(stream)
    writer.writerow(['Model',*PHYSICAL,'FID','R1','R1_SD','R2','R2_SD','R3','R3_SD'])
    for row in rows:
        writer.writerow([row['label'],*[row['physical'][k] for k in PHYSICAL],row['FID'],
            *[row['retrieval'][f'R32_R{k:02d}_{s}'] for k in (1,2,3) for s in ('mean','std')]])
    (args.output/'comparison.csv').write_text(stream.getvalue())
    save(args.output/'METRICS_COMPLETE.json',dict(cases=71,summary_sha256=sha(args.output/'summary.json')))
    publish(args,summary)
    save(args.output/'COMPLETE.json',dict(cases=71,predictions=284,notion=PAGE_URL))
    progress(args,'complete',notion=PAGE_URL)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','base','previous','notion-helper'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        main(args)
    except BaseException as error:
        progress(args,'failed',error=str(error),traceback=traceback.format_exc())
        raise
