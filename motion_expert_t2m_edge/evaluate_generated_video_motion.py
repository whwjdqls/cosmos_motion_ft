"""Image+caption -> native video without CFG -> Phase3 motion with text-only CFG."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from diagnose_image_conditioning import atomic_json, sha


def prepare(args):
    original=json.loads((args.source_operation/'configuration.json').read_text())
    caption_ref=Path('/scratch2/whwjdqls99/cosmos/operations/v2m_video_cfg_20261021/video_and_text/configuration.json')
    previous=json.loads(caption_ref.read_text())
    assert sha(args.source_operation/'cases.json')==original['cases_sha256']
    payload=torch.load(previous['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    assert payload['step']==200000 and payload['sources']['phase2_contract']==original['checkpoint_contract']
    assert sha(payload['sources']['phase2'])==original['checkpoint_sha256']
    config={**original,'schema':'edge_generated_video_motion_textcfg_v1','batch_cases':1,'shards':1,
        'variants':['generated_video_motion'],'paired_baseline_operation':str(args.source_operation),
        'phase3_checkpoint':previous['checkpoint'],'phase3_sha256':sha(previous['checkpoint']),
        'phase3_sources':payload['sources'],'caption_implementation':previous['caption_implementation'],
        'caption_implementation_sha256':previous['caption_implementation_sha256'],
        'source_sha256':sha(__file__),
        'video_generation':{'specialist':'same Phase1 EMA200k; native generator; bridges disabled',
            'conditioning':'first RGB only plus caption; no motion/camera input','steps':30,'shift':10,'guidance':1,
            'prefix':'first Wan latent pinned every step','video_seed':"stable_seed('generated_video:'+case_id)"},
        'motion_reconstruction':{'model':'original Phase3 baseline200k','steps':35,'shift':3,'text_guidance':2,
            'video_guidance':1,'null':'same generated video and shape; empty caption in both reasoners',
            'motion_seed':"stable_seed('nymeria32:'+case_id), same CPU initial noise as direct Phase2 modes"},
        'video_handoff':'decode native video to lossless uint8 RGB, restore observed first RGB, re-encode with training Wan pipeline; MP4 is a display export',
        'source_motion_input':False,'source_camera_input':False}
    if (args.out/'configuration.json').exists():assert json.loads((args.out/'configuration.json').read_text())==config
    else:
        (args.out/'cases.json').write_bytes((args.source_operation/'cases.json').read_bytes())
        atomic_json(args.out/'configuration.json',config)
        (args.out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
        if not (args.out/'scene_cache').exists(): (args.out/'scene_cache').symlink_to(args.source_operation/'scene_cache',target_is_directory=True)


@torch.inference_mode()
def worker(args):
    import evaluate_generation_modalities as generation
    generation.base.setup()
    from motion_expert_phase3_edge.visualize import configure, load_vae, make_pipeline, encode_window
    from motion_expert_phase3_edge.model import EdgePhase3
    from motion_expert_phase3_edge.sample import sample as specialist_sample
    from nymeria_eval import NymeriaTMREmbedder
    from nymeria_data import NymeriaShapeTMRDataset
    from st_dataset import collate_st
    from shape_tmr_eval_common import stable_seed
    from decode_uniego_torch import decode_joints
    from uniego_layout import ground_features
    from nymeria_baselines.phase2_collision_scene import SceneScorer
    generation.base.setup()
    import eval_phase2_shape_tmr as ref
    from render_motion import write_mp4
    cfg=json.loads((args.out/'configuration.json').read_text());rows=json.loads((args.out/'cases.json').read_text())
    assert sha(__file__)==cfg['source_sha256'] and sha(cfg['phase3_checkpoint'])==cfg['phase3_sha256']
    assert sha(cfg['caption_implementation'])==cfg['caption_implementation_sha256']
    payload=torch.load(cfg['phase3_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    contract=configure(payload)
    print('[loading] frozen Phase1/2 specialists and original Phase3 bridges',flush=True)
    model=EdgePhase3(payload['sources']['phase1'],payload['sources']['phase2'],device='cuda',activation_checkpointing=False)
    assert model.provenance['phase1_metadata_sha256']==payload['sources']['phase1_metadata_sha256']
    assert model.provenance['phase2_contract']==contract
    model.bridges.load_state_dict(payload['bridges'],strict=True);model.requires_grad_(False).eval()
    spec=importlib.util.spec_from_file_location('existing_v2m_caption_context',cfg['caption_implementation'])
    captions=importlib.util.module_from_spec(spec);spec.loader.exec_module(captions)
    import config
    vae=load_vae(os.environ['WAN_VAE_PATH'],'256',97,'cuda');pipe=make_pipeline()
    source=cfg['reference_request']
    assert sha(source['evaluator_checkpoint'])==cfg['evaluator_sha256']
    embedder=NymeriaTMREmbedder(source['evaluator_checkpoint'],source['evaluator_stats'],source['text_cache'],'cuda')
    assert embedder.contract['manifests']['test']==source['evaluator_training_test_manifest_sha256']
    gt_dataset=NymeriaShapeTMRDataset(cfg['manifest'],embedder.rep,train=False,src_fps=20.,rep_fps=embedder.fps)
    skeleton=ref.SOMASkeleton30().cuda()
    mean=torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).cuda().float()
    std=torch.from_numpy(np.load(config.MOTION_STATS_STD)).cuda().float()
    scorer=SceneScorer(args.out)
    # Native generation plus learned Phase1 EMA, with no modality-bridge path.
    class VideoSpecialist:
        def eval(self):return self
        def __call__(self,**kwargs):return model(**kwargs,bridges_enabled=False)
    work=args.out/'shard_0';work.mkdir(exist_ok=True)
    cases_dir=args.out/'case_outputs';cases_dir.mkdir(exist_ok=True)
    (args.out/'claims').mkdir(exist_ok=True)
    (args.out/'collision_cases').mkdir(exist_ok=True)
    beginning=time.monotonic();newly_done=0
    for index,row in enumerate(rows):
        batch_path=work/f'batch_{index:07d}.npz'
        if batch_path.exists():continue
        lock=(args.out/'claims'/f'{index:07d}.lock').open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        if batch_path.exists():lock.close();continue
        folder=cases_dir/row['case_id'];folder.mkdir(exist_ok=True)
        for name,digest in [('features_file','features_sha256'),('neutral_file','neutral_sha256')]:
            assert sha(Path(source['data_dir'])/row[name])==row[digest]
        features=np.load(Path(source['data_dir'])/row['features_file']).astype(np.float32)
        neutral=torch.from_numpy(np.load(Path(source['data_dir'])/row['neutral_file']).astype(np.float32)).cuda()[None]
        neutral=neutral-neutral.mean(1,keepdim=True)
        gt=torch.from_numpy(ground_features(features,row['floor_offset'])).cuda()[None]
        reference_batch=index//4*4
        reference_shard=(reference_batch//4)%cfg['reference_configuration']['shards']
        with np.load(Path(cfg['reference_operation'])/f'shard_{reference_shard}'/f'batch_{reference_batch:07d}.npz') as z:
            expected_hash=z['image_sha256'][index-reference_batch,0]
        assert sha(row['image_file'])==expected_hash
        image=np.load(row['image_file']).transpose(1,2,0)
        # Only the first image exists in this input. Repeating it cannot expose
        # original future frames; all but its first causal latent are discarded.
        if not (folder/'generated_video.npz').exists():
            latent,_=encode_window(vae,pipe,'256',np.repeat(image[None],97,axis=0),'cuda')
            assert latent.shape==(48,25,16,16)
            prefix=torch.zeros((1,48,25,16,16),device='cuda');prefix[:,:,0]=torch.from_numpy(latent[:,0]).cuda()
            template=torch.zeros((1,97,283),device='cuda')
            if index==0:
                test_video=prefix.clone();sigma=torch.ones(1,device='cuda')*.8
                kwargs=dict(neutral=neutral,video=test_video,motion_sigma=sigma*0,video_sigma=sigma,
                            captions=[row['text']],modes=['motimg2video'],bridges_enabled=False)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    a=model(motion=template,**kwargs)['video']
                    b=model(motion=torch.ones_like(template),**kwargs)['video']
                assert torch.equal(a,b),'GT-motion independence failed'
                atomic_json(args.out/'video_motion_independence.json',{'bitwise_equal':True,'bridges_disabled':True})
            with torch.autocast('cuda',dtype=torch.bfloat16):
                video=specialist_sample(VideoSpecialist(),mode='motimg2video',caption=row['text'],motion=template,
                    neutral=neutral,video=prefix,seed=stable_seed('generated_video:'+row['case_id']),steps=30,guidance=1.)
                assert torch.equal(video[:,:,0],prefix[:,:,0])
                pixels=vae.decode(video.float())
            assert pixels.shape==(1,3,97,256,256) and torch.isfinite(pixels).all()
            rgb=((pixels[0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
            transformed=pipe(dict(video=torch.from_numpy(np.repeat(image[None],97,axis=0)).permute(3,0,1,2).contiguous(),
                mode='image2video',ai_caption=''),'256')
            rgb[0]=transformed['video'][:,0].permute(1,2,0).numpy()
            reconstructed,_=encode_window(vae,pipe,'256',rgb,'cuda')
            generation.quant.atomic_npz(folder/'generated_video.npz',native_latents=video[0].half().cpu().numpy(),
                video_for_motion=reconstructed,observed_first_rgb=rgb[0])
            write_mp4(rgb,str(folder/'generated_video.mp4'),fps=20)
            atomic_json(folder/'video_audit.json',{'input_first_image_sha256':sha(row['image_file']),
                'video_guidance':1,'steps':30,'shift':10,'prefix_unchanged':True,'no_GT_video_or_motion':True,
                'motion_video_source':'lossless decoded uint8 RGB, observed first frame restored, Wan re-encoded'})
            del video,pixels,prefix,latent,reconstructed,rgb
        with np.load(folder/'generated_video.npz') as z:video=torch.from_numpy(z['video_for_motion']).cuda().float()[None]
        if not (folder/'motion.npz').exists():
            from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
            scheduler=FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1.,use_dynamic_shifting=False)
            scheduler.set_timesteps(35,device='cuda',shift=3.)
            state=torch.randn((97,283),generator=torch.Generator().manual_seed(stable_seed('nymeria32:'+row['case_id']))).cuda()[None]
            video_before=video.clone();calls_total=0
            for si,timestep in enumerate(scheduler.timesteps):
                sigma=timestep.float().reshape(1)/1000
                kwargs=dict(motion=state,neutral=neutral,video=video,motion_sigma=sigma,
                            video_sigma=sigma*0,captions=[row['text']],modes=['video2motion'],bridges_enabled=True)
                with torch.autocast('cuda',dtype=torch.bfloat16),captions.caption_conditioning(model,row['text']) as cc:
                    conditional=model(**kwargs)['motion'].float()
                with torch.autocast('cuda',dtype=torch.bfloat16),captions.caption_conditioning(model,'') as nc:
                    null=model(**{**kwargs,'captions':['']})['motion'].float()
                assert cc==nc=={'motion_reasoner':1,'video_reasoner':1}
                assert torch.equal(video,video_before)
                if index==0 and si==0:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        native_null=model(**{**kwargs,'captions':['']})['motion'].float()
                    torch.testing.assert_close(null,native_null,atol=1e-5,rtol=1e-5)
                    atomic_json(args.out/'text_cfg_acceptance.json',{
                        'null_matches_native_video2motion':True,'video_preserved_bitwise':True,
                        'bridges_enabled_in_both_branches':True,'no_video_CFG':True,
                        'correct_minus_empty_prediction_rms':float((conditional-null).square().mean().sqrt())})
                prediction=null+2*(conditional-null)
                actual_sigma=scheduler.sigmas[si].to('cuda').float().clamp_min(1e-6)
                state=scheduler.step((state-prediction)/actual_sigma,timestep,state,return_dict=False)[0].float()
                calls_total+=2
            assert torch.isfinite(state).all()
            generation.quant.atomic_npz(folder/'motion.npz',normalized_motion=state[0].cpu().numpy(),
                raw_features=(state*std+mean)[0].cpu().numpy())
            atomic_json(folder/'motion_audit.json',{'text_guidance':2,'video_guidance':1,'steps':35,'shift':3,
                'conditional_and_null_share_video_bitwise':True,'captions_in_both_reasoners':True,
                'conditional_null_calls':calls_total,'same_initial_motion_noise_as_direct_Phase2':True})
        with np.load(folder/'motion.npz') as z:raw=torch.from_numpy(z['raw_features']).cuda()[None]
        posed=decode_joints(raw);length=torch.tensor([97],device='cuda')
        measured=generation.quant.measure(ref,embedder,skeleton,posed,raw[...,279:283]>.5,length,neutral)
        packed=collate_st([gt_dataset[row['index']]])
        gt_embedding=embedder.embed_motion_features(packed['features'],packed['mask'],packed['neutral_joints']).cpu().numpy()
        reference_start=row['index']//64*64
        with np.load(Path(source['reference_dir'])/f'batch_{reference_start:07d}.npz') as z:
            assert np.allclose(gt_embedding[0],z['embedding'][row['index']-reference_start],atol=2e-5,rtol=2e-4)
            gt_metrics={key:np.asarray(z[key][row['index']-reference_start:row['index']-reference_start+1]) for key in z.files if key not in ['index','case_id']}
        values={'index':np.array([index]),'case_id':np.array([row['case_id']]),
            'text_embedding':embedder.embed_text([row['text']]).cpu().numpy(),
            **{'generated_video_motion__'+k:v for k,v in measured.items()},**{'GT__'+k:v for k,v in gt_metrics.items()}}
        collision=scorer.measure(row,raw.cpu().numpy())
        collision['metrics']['generated_video_motion']=collision['metrics'].pop('image_only')
        evidence={key.replace('image_only__','generated_video_motion__'):value for key,value in collision.pop('evidence').items()}
        generation.quant.atomic_npz(args.out/'collision_cases'/(row['case_id']+'.npz'),**evidence)
        atomic_json(args.out/'collision_cases'/(row['case_id']+'.json'),collision)
        generation.quant.atomic_npz(batch_path,**values)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close();newly_done+=1
        complete=len(list(work.glob('batch_*.npz')));elapsed=time.monotonic()-beginning
        progress={'completed':complete,'total':len(rows),'job':os.environ['SLURM_JOB_ID'],
            'cases_per_second':newly_done/elapsed,'eta_hours_single_worker':(len(rows)-complete)*elapsed/newly_done/3600,
            'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'time_unix':time.time()}
        atomic_json(args.out/('progress_'+os.environ['SLURM_JOB_ID']+'.json'),progress)
        print('[pipeline] '+json.dumps(progress),flush=True)
        if index in (0,1):
            from nymeria_baselines.generated_video_motion_report import render_example
            render_example(args.out,row,index)
        if args.max_cases and newly_done>=args.max_cases:break


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','worker','summarize'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-operation',type=Path)
    parser.add_argument('--max-cases',type=int,default=0)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname():raise RuntimeError('Compute allocation required')
    args.out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.stage=='summarize':
        import evaluate_image_conditioning as base
        base.setup();base.VARIANTS=('generated_video_motion',);base.summarize(args)
        from nymeria_baselines.generated_video_motion_report import summarize_pipeline
        summarize_pipeline(args.out)
    else:globals()[args.stage](args)


if __name__=='__main__':main()
