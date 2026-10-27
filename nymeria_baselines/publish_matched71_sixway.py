"""Publish existing matched-71 predictions, without model loading or resampling."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import json
import multiprocessing
import os
import runpy
import time
from pathlib import Path

BASE = Path('/scratch2/whwjdqls99/cosmos/operations')
REFERENCE = BASE / 'uem_phase3_comparison_20261019'
GUIDANCE = BASE / 'v2m_guidance_uem_tmr_20261023'
TMR = BASE / 'reconstruction_tmr_body23_20261023_axisfix'
VARIANTS = BASE / 'uem_all_input_eval_20261023'
EGO = BASE / 'egoallo_phase3_comparison_20261020'
OURS = BASE / 'v2m_video_cfg_20261021/video_and_text'
PAGE_ID = '3ffd2a44-f44b-806c-a79e-c5987d6fd1f3'
PAGE_URL = 'https://www.notion.so/' + PAGE_ID.replace('-', '')
TITLE = 'Matched 71 | GT, V2M text CFG2, UniEgomotion, EgoAllo | 2026-10-23'
HELPER = BASE / 'phase1_notext_fourway_55k_20261019/upload_notion.py'
KEEP = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29]
PARENTS = [-1, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 3, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21]
LABELS = ['Ground truth', 'Ours: V2M + text, CFG2', 'UniEgomotion: RGB',
          'UniEgomotion: trajectory', 'UniEgomotion: RGB + trajectory', 'EgoAllo: trajectory']
INPUTS = ['Reference motion', 'Observed: RGB + caption', 'Observed: RGB',
          'Observed: camera trajectory', 'Observed: RGB + camera trajectory',
          'Observed: camera trajectory']


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sources(case):
    name = case['id'] + '.npz'
    return [Path(case['bundle']), OURS / 'video_cfg_2' / name,
            VARIANTS / 'RGB_recon/video/predictions' / name,
            REFERENCE / 'predictions' / name,
            VARIANTS / 'recon/video_trajectory/predictions' / name,
            EGO / 'predictions' / name]


def prepare(out):
    ref = read(REFERENCE / 'configuration.json')
    rows = {r['key']: r for r in read(GUIDANCE / 'summary.json')['rows']}
    chosen = [rows[k] for k in ['GT', 'caption__video_cfg_2', 'RGB_recon__video',
                              'traj_recon', 'recon__video_trajectory']]
    ego = next(r for r in read(TMR / 'summary.json')['models'] if r['key'] == 'egoallo')
    agg = read(EGO / 'summary.json')['aggregate']['EgoAllo']
    physical = {label: agg[key]['mean'] * factor for label, key, factor in [
        ('MPJPE cm', 'mpjpe_m', 100), ('PA-MPJPE cm', 'pa_mpjpe_m', 100),
        ('Root ADE cm', 'root_err_m', 100), ('Root FDE cm', 'root_fde_m', 100),
        ('Relative root ADE cm', 'root_relative_ade_m', 100),
        ('Skate cm/s', 'foot_skate_cm_s', 1), ('Penetration cm', 'foot_depth_mean_cm', 1),
        ('Airborne %', 'all_feet_above5cm_frame_fraction', 100)]}
    chosen.append({k: ego[k] for k in ['key', 'label', 'FID', 'retrieval', 'training_step']}
                  | {'physical': physical})
    for row, label in zip(chosen, LABELS):
        row['display_label'] = label
    source_reports = [GUIDANCE / 'summary.json', TMR / 'summary.json', EGO / 'summary.json',
                      REFERENCE / 'configuration.json']
    cases = ref['cases']
    assert len(cases) == len({c['id'] for c in cases}) == 71
    contract = ref['phase3_contract']
    for path_key, hash_key in [('motion_stats_mean', 'motion_mean'), ('motion_stats_std', 'motion_std')]:
        if sha(contract[path_key]) != contract[hash_key]:
            raise RuntimeError('Generator statistics hash mismatch: ' + path_key)
    config = {
        'cases': cases, 'frames': 97, 'fps': 20, 'labels': LABELS, 'inputs': INPUTS,
        'motion_mean': contract['motion_stats_mean'], 'motion_std': contract['motion_stats_std'],
        'motion_mean_sha256': contract['motion_mean'], 'motion_std_sha256': contract['motion_std'],
        'source_reports': [{'path': str(p), 'sha256': sha(p)} for p in source_reports],
        'renderer': 'motion_expert_joint_attention.render_motion; unchanged thin-skeleton/floor/head defaults',
        'render_camera': 'Shared GT-root-following viewport and common scale; display only, no pose translation',
        'body_joints': KEEP, 'parents': PARENTS, 'page_id': PAGE_ID,
        'scope': 'One fixed T97 window per test recording, not every test window',
        'GT_panel': 'Ground truth only; not the privileged oracle-head Phase3 prediction',
        'guidance': 'Video bridge guidance 2; caption retained in both branches; null disables all bridges',
        'alignment': 'Original canonical coordinates; no grounding or prediction alignment',
        'quantitative': 'Reused completed common-Body23 physical metrics and Nymeria-TMR common-completion metrics',
        'retrieval': 'Batch 32, seeds 0/1/2, 64 cases used and 7 omitted per seed; FID uses all 71',
        'shape_caveat': 'Ours uses supplied neutral shape; UEM/EgoAllo predict shape. Input sets differ.',
    }
    save(out / 'configuration.json', config)
    save(out / 'comparison.json', chosen)
    columns = ['Model', 'FID', 'R@1 %', 'R@2 %', 'R@3 %', 'MPJPE cm', 'PA-MPJPE cm',
               'Root ADE cm', 'Root FDE cm', 'Relative root ADE cm', 'Skate cm/s',
               'Penetration cm', 'Airborne %']
    with (out / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in chosen:
            writer.writerow({'Model': row['display_label'], 'FID': row['FID'],
                             **{f'R@{k} %': row['retrieval'][f'R32_R0{k}_mean'] for k in [1, 2, 3]},
                             **row['physical']})
    per_case_sources = {'guidance': OURS / 'per_case.json', 'uem_variants': VARIANTS / 'per_case.json',
                        'uem_trajectory': REFERENCE / 'per_case.json', 'egoallo': EGO / 'per_case.json'}
    records = {name: {r['id']: r for r in read(p)} for name, p in per_case_sources.items()}
    save(out / 'per_case_source_metrics.json', [
        {'id': c['id'], 'sources': {name: data[c['id']] for name, data in records.items()}}
        for c in cases])
    return config, chosen


def render_case(case, config, output):
    import numpy as np
    import imageio.v2 as imageio
    from matplotlib.patches import Rectangle
    from nymeria_baselines.visualize_official_baselines import decode
    from motion_expert_joint_attention import render_motion as r

    out = Path(output)
    gallery = out / 'gallery'
    name = case['id']
    video, poster, sidecar = gallery / f'{name}.mp4', gallery / f'{name}.png', out / 'cases' / f'{name}.json'
    paths = sources(case)
    signature = hashlib.sha256(json.dumps({'config': config, 'sources': [sha(p) for p in paths]},
                                         sort_keys=True).encode()).hexdigest()
    if sidecar.exists() and video.exists() and poster.exists():
        prior = read(sidecar)
        if prior['signature'] == signature and prior['video_sha256'] == sha(video):
            return prior
        raise RuntimeError('Existing rendered case has a different source/configuration: ' + name)
    mean, std = np.load(config['motion_mean']), np.load(config['motion_std'])
    with np.load(paths[0]) as a:
        gt, gt_r = decode(a['motion'] * std + mean, 30)
    arrays, rotations = [gt[:, KEEP]], [gt_r[0, 6]]
    with np.load(paths[1]) as a:
        assert float(a['video_scale']) == 2.0 and int(a['seed']) == case['seed']
        joints, rot = decode(a['generated'] * std + mean, 30)
        arrays.append(joints[:, KEEP]); rotations.append(rot[0, 6])
    for path in paths[2:5]:
        with np.load(path) as a:
            joints, rot = decode(a['raw_body'], 23)
        arrays.append(joints); rotations.append(rot[0, 6])
    with np.load(paths[5]) as a:
        arrays.append(a['joints'].copy()); rotations.append(a['rotations'][0, 6].copy())
        gt_error = float(np.max(np.abs(arrays[0] - a['gt'])))
    if gt_error > 2e-3:
        raise RuntimeError(f'GT coordinate contract mismatch in {name}: {gt_error} m')
    for joints in arrays:
        assert joints.shape == (97, 23, 3) and np.isfinite(joints).all(), name
    displayed = [r._to_display(x) for x in arrays]
    extent = r._world_extent(displayed)
    center = displayed[0][:, 0]
    half = max(r.DEFAULT_VIEW_HALF, max(float(np.max(np.abs(
        x[:, :, [0, 2]] - center[:, None, [0, 2]]))) for x in displayed) + .2)
    fig = r.plt.figure(figsize=(15.36, 10.24), dpi=100)
    temporary = video.with_name(video.stem + '.partial.mp4')
    started = time.time()
    try:
        with imageio.get_writer(str(temporary), fps=20, codec='libx264', macro_block_size=1,
                                ffmpeg_params=['-crf', '22', '-threads', '1']) as writer:
            for frame in range(97):
                fig.clf()
                fig.subplots_adjust(left=.015, right=.985, top=.96, bottom=.055, wspace=.05, hspace=.13)
                for panel, (joints, rotation, label, conditioning) in enumerate(
                        zip(displayed, rotations, LABELS, INPUTS)):
                    color = '#2478ad' if panel == 0 else '#cf8424'
                    ax = fig.add_subplot(2, 3, panel + 1, projection='3d')
                    r._setup_axes(ax, r._viewport_for_center((center[frame, 0], center[frame, 2]),
                                  half, extent['y_top'], extent['y_bottom']), label)
                    r._draw_floor_and_grid(ax, extent)
                    r._draw_trajectory(ax, joints[:, 0], frame, color)
                    r._draw_skel(ax, joints[frame], PARENTS, color, skip_joints=[])
                    r._draw_head_reference(ax, joints[0], rotation)
                    ax.text2D(.5, .97, conditioning, transform=ax.transAxes, ha='center', va='top',
                              color='#236849' if panel else '#2478ad', fontsize=8)
                    box = ax.get_position()
                    fig.add_artist(Rectangle((box.x0, box.y0), box.width, box.height,
                                             transform=fig.transFigure, fill=False,
                                             edgecolor=color, linewidth=1.5))
                fig.text(.5, .018, 'Blue: ground truth | Amber: generated motion | Green labels: observed inputs'
                         ' | Axes: first-frame head | Shared viewport, no grounding correction',
                         ha='center', fontsize=9)
                fig.canvas.draw()
                pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                writer.append_data(pixels)
                if frame == 0:
                    imageio.imwrite(poster, pixels)
    finally:
        r.plt.close(fig)
    temporary.replace(video)
    result = {'id': name, 'uuid': case['uuid'], 'caption': case['caption'], 'seed': case['seed'],
              'signature': signature, 'video': str(video), 'video_sha256': sha(video),
              'poster': str(poster), 'render_seconds': time.time() - started,
              'gt_coordinate_max_error_m': gt_error, 'frames': 97, 'fps': 20,
              'sources': [{'label': label, 'path': str(p), 'sha256': sha(p)}
                          for label, p in zip(LABELS, paths)]}
    save(sidecar, result)
    return result


def rich(value):
    return [{'type': 'text', 'text': {'content': str(value)}}]


def paragraph(value):
    return {'object': 'block', 'type': 'paragraph', 'paragraph': {'rich_text': rich(value)}}


def table(headers, values):
    return {'object': 'block', 'type': 'table', 'table': {
        'table_width': len(headers), 'has_column_header': True, 'has_row_header': False,
        'children': [{'object': 'block', 'type': 'table_row', 'table_row': {'cells': [rich(c) for c in row]}}
                     for row in [headers] + values]}}


class Publisher:
    def __init__(self, output, rows, config):
        import requests
        helper = runpy.run_path(str(HELPER))
        token = helper['read_token']()
        self.api = helper['Notion'](token)
        self.http = requests.Session()
        self.http.headers.update({'Authorization': 'Bearer ' + token, 'Notion-Version': '2026-03-11'})
        self.path = output / 'notion_publication.json'
        self.state = read(self.path) if self.path.exists() else {'page_url': PAGE_URL, 'cases': {}, 'uploads': {}}
        matches = [b for b in self.children(PAGE_ID) if b['type'] == 'toggle'
                   and self.text(b['toggle']['rich_text']) == TITLE]
        if len(matches) > 1:
            raise RuntimeError('Duplicate target toggles; refusing ambiguous publication')
        if matches:
            self.toggle = matches[0]['id']
        else:
            self.toggle = self.append(PAGE_ID, [{'object': 'block', 'type': 'toggle',
                                               'toggle': {'rich_text': rich(TITLE)}}])[0]['id']
        self.state['toggle_id'] = self.toggle
        self.state['toggle_url'] = PAGE_URL + '#' + self.toggle.replace('-', '')
        save(self.path, self.state)
        children = self.children(self.toggle)
        if not children:
            retrieval_rows = [[r['display_label'], f"{r['FID']:.4f}",
                               *[f"{r['retrieval'][f'R32_R0{k}_mean']:.2f}" for k in [1, 2, 3]]]
                              for r in rows]
            physical_keys = ['MPJPE cm', 'PA-MPJPE cm', 'Root ADE cm', 'Skate cm/s', 'Penetration cm', 'Airborne %']
            physical_rows = [[r['display_label'], *[f"{r['physical'][k]:.2f}" for k in physical_keys]] for r in rows]
            children = self.append(self.toggle, [
                paragraph('All 71 fixed test windows, T97 at 20 fps. Six synchronized motion panels per case. '
                          'These are one window per recording, not every window in the 71 recordings. '
                          'Original predictions and completed quantitative evaluations are reused.'),
                table(['Model', 'FID (lower)', 'R@1 %', 'R@2 %', 'R@3 %'], retrieval_rows),
                table(['Model'] + physical_keys, physical_rows),
                paragraph('FID uses all 71 cases. Retrieval uses batches of 32 with three fixed shuffles '
                          '(seeds 0/1/2): 64 cases and 7 omitted per shuffle. The same Body23-to-TMR completion '
                          'is used for all models and GT. Physical metrics use unmodified Body23 joints.'),
                paragraph('Ours: RGB + caption, video-bridge guidance scale 2, 35-step UniPC; no head inpainting. '
                          'The null branch disables all modality bridges and retains the caption, so this is '
                          'not a trained video-null CFG branch. UniEgomotion RGB+trajectory is the recon specialist. '
                          'Ours uses supplied neutral shape, while UniEgomotion and EgoAllo predict shape. '
                          'Conditioning differs across models; this is not an input-controlled ablation.'),
                paragraph('Blue is GT; amber is generated motion; green labels identify observed inputs. '
                          'All panels share a viewport and scale. The first-frame head axes and established '
                          'floor occlusion are retained. No generated pose is translated, grounded, or fit to GT. '
                          'GT foot skating/penetration is reported rather than assumed zero.')])
        self.case_blocks = {self.text(b['toggle']['rich_text']): b['id']
                            for b in children if b['type'] == 'toggle'}
        missing = [c for c in config['cases'] if c['id'] not in self.case_blocks]
        if missing:
            blocks = self.append(self.toggle, [
                {'object': 'block', 'type': 'toggle', 'toggle': {'rich_text': rich(c['id']),
                 'children': [paragraph(c['caption'])]}} for c in missing])
            self.case_blocks.update({self.text(b['toggle']['rich_text']): b['id'] for b in blocks})
        self.state['case_blocks'] = self.case_blocks
        save(self.path, self.state)

    @staticmethod
    def text(items):
        return ''.join(i.get('plain_text', i.get('text', {}).get('content', '')) for i in items)

    def children(self, parent):
        result, cursor = [], None
        while True:
            path = f'/blocks/{parent}/children?page_size=100' + (f'&start_cursor={cursor}' if cursor else '')
            response = self.api.request('GET', path)
            result.extend(b for b in response['results'] if not b.get('archived') and not b.get('in_trash'))
            if not response.get('has_more'):
                return result
            cursor = response['next_cursor']

    def append(self, parent, children):
        return self.api.request('PATCH', f'/blocks/{parent}/children', {'children': children})['results']

    def upload(self, path, mime):
        signature = sha(path)
        prior = self.state['uploads'].get(signature)
        if prior:
            status = self.api.request('GET', '/file_uploads/' + prior['id'])
            if status['status'] == 'uploaded':
                return prior['id']
            if status['status'] != 'pending':
                prior = None
        if not prior:
            prior = self.api.request('POST', '/file_uploads', {'mode': 'single_part',
                                     'filename': path.name, 'content_type': mime})
            self.state['uploads'][signature] = {'id': prior['id'], 'path': str(path)}
            save(self.path, self.state)
        if path.stat().st_size > 20_000_000:
            raise RuntimeError('Single-part upload limit exceeded: ' + str(path))
        with path.open('rb') as stream:
            response = self.http.post('https://api.notion.com/v1/file_uploads/' + prior['id'] + '/send',
                                      files={'file': (path.name, stream, mime)}, timeout=(30, 240))
        response.raise_for_status()
        if response.json()['status'] != 'uploaded':
            raise RuntimeError('File upload did not reach uploaded status')
        return prior['id']

    def publish_case(self, record):
        name, parent = record['id'], self.case_blocks[record['id']]
        caption = name + ' | matched71-sixway | sha256=' + record['video_sha256']
        existing = [b for b in self.children(parent) if b['type'] == 'video']
        matches = [b for b in existing if self.text(b['video'].get('caption', [])) == caption]
        if existing and len(matches) != 1:
            raise RuntimeError('Conflicting video already attached for ' + name)
        if matches:
            block = matches[0]
        else:
            upload_id = self.upload(Path(record['video']), 'video/mp4')
            block = self.append(parent, [{'object': 'block', 'type': 'video', 'video': {
                'type': 'file_upload', 'file_upload': {'id': upload_id}, 'caption': rich(caption)}}])[0]
        if block['type'] != 'video':
            raise RuntimeError('Notion did not return a video block')
        self.state['cases'][name] = {'video_block_id': block['id'], 'video_sha256': record['video_sha256']}
        save(self.path, self.state)


def gallery_index(out, config, results):
    entries = []
    for case in config['cases']:
        if case['id'] not in results:
            continue
        name = case['id']
        entries.append(f'<section><h2>{html.escape(name)}</h2><p>{html.escape(case["caption"])}</p>'
                       f'<video controls loop preload="none" poster="{name}.png" src="{name}.mp4"></video></section>')
    (out / 'gallery/index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Matched71 motion comparison</title>'
        '<style>body{max-width:1536px;margin:24px auto;padding:16px;background:#f3f1ec;color:#202321;'
        'font-family:Georgia,serif}video{width:100%}section{margin:40px 0}h2{font-size:18px}</style>'
        '<h1>GT / Ours V2M text CFG2 / UniEgomotion RGB / UniEgomotion trajectory / '
        'UniEgomotion RGB+trajectory / EgoAllo trajectory</h1>'
        '<p>71 fixed T97 test windows. All panels preserve original coordinates and share a viewport. '
        'No prediction-dependent grounding. <a href="../comparison.csv">Quantitative CSV</a>.</p>'
        + ''.join(entries))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    from nymeria_baselines.uem_visual_cache import require_compute
    require_compute()
    out = args.output
    (out / 'gallery').mkdir(parents=True, exist_ok=True)
    (out / 'cases').mkdir(exist_ok=True)
    started = time.time()
    config, rows = prepare(out)
    publisher = Publisher(out, rows, config) if args.publish else None
    results = {}
    print('PREPARED', json.dumps({'cases': 71, 'workers': args.workers,
          'notion': publisher.state['toggle_url'] if publisher else None}), flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(render_case, case, config, str(out)): case['id'] for case in config['cases']}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            results[record['id']] = record
            if publisher:
                publisher.publish_case(record)
            gallery_index(out, config, results)
            status = {'stage': 'rendering_and_uploading', 'rendered': len(results),
                      'uploaded': len(publisher.state['cases']) if publisher else 0,
                      'total': 71, 'last_case': record['id'], 'elapsed_seconds': time.time() - started,
                      'last_render_seconds': record['render_seconds']}
            save(out / 'progress.json', status)
            print('CASE_COMPLETE', json.dumps(status), flush=True)
    if publisher:
        assert len(publisher.state['cases']) == 71
        publisher.state['status'] = 'complete'
        save(publisher.path, publisher.state)
    save(out / 'COMPLETE.json', {'status': 'complete', 'cases': 71, 'panels': 6,
         'uploaded': len(publisher.state['cases']) if publisher else 0,
         'notion': publisher.state['toggle_url'] if publisher else None,
         'elapsed_seconds': time.time() - started, 'configuration_sha256': sha(out / 'configuration.json')})
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
