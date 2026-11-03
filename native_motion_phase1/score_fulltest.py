"""Incremental GT-matched scoring and fixed qualitative media for native motion."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from native_motion_phase1.evaluate_fulltest import sha,save_json


def aggregate(rows):
    return {k:dict(mean=float(np.mean([r[k] for r in rows])),median=float(np.median([r[k] for r in rows])),
                   std=float(np.std([r[k] for r in rows]))) for k in rows[0]}


def motion_scores(pred,gt,mean,std):
    from eval_motion_recon import recon_metrics,decode_to_joints,joint_metrics
    from phase2_quantitative import floor_metrics
    from kimodo.metrics.foot_skate import FootSkateFromHeight
    from kimodo.skeleton import SOMASkeleton30
    from nymeria_baselines.body_shape import BODY_JOINTS
    joints=decode_to_joints(pred*std+mean);target=decode_to_joints(gt*std+mean)
    assert np.isfinite(joints).all()
    skeleton=SOMASkeleton30();feet=list(skeleton.foot_joint_idx)
    tensor=torch.from_numpy(joints)[None];length=torch.tensor([97])
    result=recon_metrics(pred,gt,mean,std)
    result.update({k:float(v[0]) for k,v in floor_metrics(tensor,length,feet).items()})
    skate=FootSkateFromHeight(skeleton=skeleton,fps=20.)
    result['foot_skate_cm_s']=float(skate(posed_joints=tensor,lengths=length)['foot_skate_from_height'][0])*100
    result['root_fde_m']=float(np.linalg.norm(joints[-1,0]-target[-1,0]))
    relative=(joints[:,0]-joints[0,0])-(target[:,0]-target[0,0])
    result['root_relative_ade_m']=float(np.linalg.norm(relative,axis=-1).mean())
    result['accel_err_m_s2']=result['accel_err']*400
    result['accel_pred_m_s2']=result['accel_pred']*400
    result['head_error_m']=float(np.linalg.norm(joints[:,6]-target[:,6],axis=-1).mean())
    result.update({'body23_'+k:v for k,v in joint_metrics(joints[:,BODY_JOINTS],target[:,BODY_JOINTS]).items()})
    assert all(np.isfinite(v) for v in result.values())
    return result,joints,target


def motion(args):
    cfg=json.loads((args.out/'configuration.json').read_text());rows=json.loads((args.out/'cases.json').read_text())
    mean,std=np.load(cfg['mean']),np.load(cfg['std']);dest=args.out/'metrics';dest.mkdir(exist_ok=True)
    tasks=['video2motion','text2motion','textimg2motion']
    from motion_expert_joint_attention.render_motion import render_sidebyside,write_mp4
    selected=set(cfg['qualitative_indices'])
    while True:
        counts={}
        for task in tasks:
            results=[]
            for index,row in enumerate(rows):
                source=args.out/'predictions'/task/(row['id']+'.npz');output=dest/task/(row['id']+'.json')
                if not source.exists():continue
                if not output.exists():
                    with np.load(row['bundle']) as z:gt=z['motion'].copy()
                    with np.load(source) as z:pred=z['generated'].copy()
                    metrics,joints,target=motion_scores(pred,gt,mean,std)
                    gt_metrics,_,_=motion_scores(gt,gt,mean,std)
                    save_json(output,dict(id=row['id'],uuid=row['uuid'],start=row['start'],model=metrics,GT=gt_metrics))
                    if index in selected:
                        frames=render_sidebyside(target,joints,caption=task+' | GT left, prediction right | '+row['caption'],width=768,height=384)
                        media=args.out/'qualitative'/task/(row['id']+'.mp4');media.parent.mkdir(parents=True,exist_ok=True)
                        write_mp4(frames,str(media),fps=20)
                results.append(json.loads(output.read_text()))
            counts[task]=len(results)
            if results:
                balanced={}
                for label in ['model','GT']:
                    grouped={u:aggregate([r[label] for r in results if r['uuid']==u]) for u in set(r['uuid'] for r in results)}
                    balanced[label]={k:float(np.mean([x[k]['mean'] for x in grouped.values()])) for k in results[0][label]}
                save_json(dest/(task+'_summary.json'),dict(complete=len(results)==len(rows),n=len(results),expected=len(rows),
                    recordings=len(set(r['uuid'] for r in results)),aggregate={k:aggregate([r[k] for r in results]) for k in ['model','GT']},recording_balanced_mean=balanced,
                    tmr={'status':'pending_Nymeria_evaluator_bundle','metrics':['motion_FID','R@1','R@2','R@3','matching_distance','diversity']}))
        save_json(args.out/'motion_metrics_status.json',dict(counts=counts,expected=len(rows),time=time.time()))
        print('MOTION_SCORED',counts,flush=True)
        if all(n==len(rows) for n in counts.values()):
            save_json(args.out/'MOTION_RECON_PHYSICS_COMPLETE.json',dict(counts=counts));return
        if not args.follow:return
        time.sleep(30)


class VideoScorer:
    def __init__(self,args):
        from native_phase_training.evaluate_inverse_forward import LPIPSAlex
        from dreamsim import dreamsim
        self.lpips=LPIPSAlex('cuda',16)
        root=Path('/mnt/local/jinhyung.kim/cosmos_storage/evaluation_models')
        self.dream,self.preprocess=dreamsim(pretrained=True,device='cuda',cache_dir=str(root/'dreamsim'),dreamsim_type='ensemble',normalize_embeds=True)
        code=Path('/mnt/local/jinhyung.kim/cosmos_storage/evaluation_code/content-debiased-fvd');sys.path.insert(0,str(code))
        from cdfvd import fvd
        self.i3d=fvd.cdfvd(model='i3d',n_real='full',n_fake='full',ckpt_path=str(root/'i3d_pretrained_400.pt'),device='cuda',half_precision=False)
        self.vmae=fvd.cdfvd(model='videomae',n_real='full',n_fake='full',ckpt_path=str(root/'cdfvd/vit_g_hybrid_pt_1200e_ssv2_ft.pth'),device='cuda',half_precision=False)
        assert sha(root/'cdfvd/vit_g_hybrid_pt_1200e_ssv2_ft.pth')=='5a210a92f035dff30c53b46157b612e7a1a5d3c99700e1b2d71da5c399ca7e70'
        self.identity=dict(cdfvd_commit='a1e037ab7cb087debd2221d14ae4a001ec054201',videomae_sha256=sha(root/'cdfvd/vit_g_hybrid_pt_1200e_ssv2_ft.pth'),i3d_sha256=sha(root/'i3d_pretrained_400.pt'),feature_dtype='float32',frames='1-96; observed frame0 excluded')
        save_json(args.out/'video_evaluator.json',self.identity)

    @torch.no_grad()
    def score(self,gt,pred):
        from native_phase_training.evaluate_inverse_forward import _frame_metrics
        from native_phase_training.evaluate_forward_dreamsim import _distances
        values=_frame_metrics(gt[1:],pred[1:],self.lpips)
        distances=_distances(self.dream,self.preprocess,gt[1:],pred[1:],torch.device('cuda'),16)
        result={k:float(v.mean()) for k,v in values.items()};result['dreamsim']=float(distances.mean())
        from cdfvd.utils.metric_utils import FeatureStats
        features={}
        for label,video in [('GT',gt),('model',pred)]:
            for metric,evaluator in [('i3d',self.i3d),('videomae',self.vmae)]:
                stats=FeatureStats(capture_mean_cov=True,capture_all=True,max_items=None)
                stats=evaluator.feature_fn(stats,evaluator.model,video[None,1:],batchsize=1,device=evaluator.device,model_dtype=evaluator.model_dtype)
                features[label+'_'+metric]=stats.get_all()
        return result,features


def summarize_video(args,rows,scorer):
    directory=args.out/'metrics/motion2video';results=[];paths=[]
    for row in rows:
        p=directory/(row['id']+'.json');feature=directory/(row['id']+'_features.npz')
        if p.exists() and feature.exists():results.append(json.loads(p.read_text()));paths.append(feature)
    if not results:return
    if getattr(scorer,'last_summary_count',None)==len(results):return
    from cdfvd.utils.metric_utils import FeatureStats
    fvd={}
    for label,evaluator in [('i3d',scorer.i3d),('videomae',scorer.vmae)]:
        real,fake=[FeatureStats(capture_mean_cov=True,capture_all=False,max_items=None) for _ in range(2)]
        for p in paths:
            with np.load(p) as z:real.append(z['GT_'+label]);fake.append(z['model_'+label])
        if len(paths)>=2:fvd[label]=float(evaluator.compute_fvd_from_stats(fake,real))
    save_json(args.out/'metrics/motion2video_summary.json',dict(n=len(results),expected=len(rows),complete=len(results)==len(rows),
        aggregate={'model':aggregate([r['model'] for r in results]),'GT':{'psnr_db':'+infinity (identity reference)','ssim':1.,'lpips_alex':0.,'dreamsim':0.}},
        FVD_I3D=fvd.get('i3d'),CD_FVD_VideoMAE_v2=fvd.get('videomae'),GT_self_FVD=0.,
        evaluator=scorer.identity,note='Distribution scores use the complete processed cohort; partial scores are provisional. Model pixels are lossless decoded uint8, before qualitative MP4 encoding.'))
    scorer.last_summary_count=len(results)


@torch.no_grad()
def video(args):
    from nymeria_camera_dataset import decode_window_pyav
    from native_phase_training.evaluate_inverse_forward import _resize_gt_like_native
    from motion_expert_joint_attention.render_motion import write_mp4
    cfg=json.loads((args.out/'configuration.json').read_text());rows=json.loads((args.out/'cases.json').read_text())
    order=cfg['qualitative_indices']+[i for i in range(len(rows)) if i not in cfg['qualitative_indices']]
    scorer=VideoScorer(args);directory=args.out/'metrics/motion2video';directory.mkdir(parents=True,exist_ok=True)
    newly=0
    while True:
        for index in order:
            row=rows[index];source=args.out/'predictions/motion2video'/(row['id']+'.npz');output=directory/(row['id']+'.json');feature=directory/(row['id']+'_features.npz')
            if not source.exists() or (output.exists() and feature.exists()):continue
            gt=_resize_gt_like_native(decode_window_pyav(row['rgb'],row['start'],97,20.),256,256)
            with np.load(source) as z:pred=z['generated'].copy()
            assert gt.shape==pred.shape==(97,256,256,3)
            metrics,features=scorer.score(gt,pred)
            tmp=feature.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez(f,**features)
            tmp.replace(feature)
            save_json(output,dict(id=row['id'],uuid=row['uuid'],start=row['start'],model=metrics))
            if index in cfg['qualitative_indices']:
                media=args.out/'qualitative/motion2video'/(row['id']+'.mp4');media.parent.mkdir(parents=True,exist_ok=True)
                write_mp4(np.concatenate([gt,pred],axis=2),str(media),fps=20)
            newly+=1
            save_json(args.out/'video_metrics_status.json',dict(newly_scored=newly,last_id=row['id'],time=time.time()))
            print('VIDEO_SCORED',row['id'],json.dumps(metrics),flush=True)
            if newly%256==0:summarize_video(args,rows,scorer)
        summarize_video(args,rows,scorer)
        complete=sum((directory/(r['id']+'.json')).exists() for r in rows)
        if complete==len(rows):save_json(args.out/'VIDEO_METRICS_COMPLETE.json',dict(cases=complete));return
        if not args.follow:return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['motion','video']);p.add_argument('--out',type=Path,required=True);p.add_argument('--follow',action='store_true');a=p.parse_args()
    torch.set_num_threads(1)
    lock=(a.out/(a.mode+'_scoring.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (motion if a.mode=='motion' else video)(a)
