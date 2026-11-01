"""Real-data gradient, shape intervention, reload and sampler acceptance."""
from pathlib import Path
import argparse
import copy
import json
import sys

from .common import require_compute, atomic_json, contract


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['uem','egoallo'], required=True)
    for name in ('upstream','cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device', default='cuda')
    a=p.parse_args()
    require_compute()
    import torch
    import numpy as np
    torch.set_num_threads(4)
    torch.manual_seed(62)
    device=torch.device(a.device)
    contract(a.cache,a.upstream,a.model,64 if a.model=='uem' else 256,'acceptance')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.model=='uem':
        from .uem import components,make_config
        Dataset,Module,_=components(a.upstream)
        from torch.utils.data import default_collate
        cfg=make_config(a.upstream,a.cache,a.output,2)
        ds=Dataset(str(a.cache),'train')
        items=[ds[0],ds[len(ds)//2]]
        batch=default_collate(items)
        y={k:v.to(device) for k,v in batch['y'].items()}
        x=batch['misc']['motion'].to(device)
        model=Module(cfg).to(device)
        assert x.shape==(2,97,220) and y['neutral_joints'].shape==(2,30,3)
        expected=np.array(ds._array(ds.rows[0]['array'],'neutral'),dtype=np.float32,copy=True)
        expected-=expected.mean(axis=0,keepdims=True)
        assert torch.equal(items[0]['y']['neutral_joints'],torch.from_numpy(expected))
        assert 'betas' not in y and not hasattr(model.model,'embed_betas')
        # Prove the motion target/statistics are byte-identical to reference.
        from dataset.nymeria_soma_dataset import NymeriaSOMADataset
        old=NymeriaSOMADataset(str(a.cache),'train')
        assert torch.equal(items[0]['misc']['motion'],old[0]['misc']['motion'][:,:220])
        loss=model.diffusion.training_losses(model.model,x,torch.tensor([100,800],device=device),
                                              model_kwargs={'y':y})['loss'].mean()
        loss.backward()
        grad=model.model.shape2llm[0].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval();noise=torch.randn_like(x);t=torch.tensor([500,500],device=device)
        with torch.no_grad():
            first=model.model(noise,t,y)
            changed={**y,'neutral_joints':y['neutral_joints']*1.1}
            second=model.model(noise,t,changed)
            assert torch.isfinite(first).all() and (first-second).abs().max()>0
            # Native guided null must retain shape, and both output/state stay220.
            guided=model.model(noise,t,y,cond_scale=1.5,diffusion=model.diffusion)
            assert guided.shape==x.shape and torch.isfinite(guided).all()
            for tick in (999,998):
                noise=model.diffusion.p_sample(model.model,noise,torch.full((2,),tick,device=device),
                    clip_denoised=False,model_kwargs={'y':y})['sample']
            assert noise.shape==x.shape and torch.isfinite(noise).all()
            if device.type=='cuda':
                sampled=model.sample({k:v[:1] for k,v in y.items()},B=1,cond_scale=None)
                assert sampled.shape==(1,97,220) and torch.isfinite(sampled).all()
        try:model.model(noise,t,{k:v for k,v in y.items() if k!='neutral_joints'})
        except ValueError:pass
        else:raise AssertionError('Missing shape accepted')
        shape_delta=float((first-second).abs().max())
        grad_norm=float(grad.norm())
        state=a.output/'acceptance_weights.pt'
        torch.save(model.state_dict(),state)
        reload=Module(cfg).to(device).eval()
        reload.load_state_dict(torch.load(state,map_location=device,weights_only=True),strict=True)
        with torch.no_grad():
            assert torch.equal(model.model(noise,t,y),reload.model(noise,t,y))
        dim=220
    else:
        from .egoallo import activate
        activate(a.upstream)
        from egoallo import network,training_loss
        from egoallo.data.nymeria_soma import NymeriaSomaDataset
        from egoallo.data.dataclass import collate_dataclass
        from egoallo.transforms import SE3
        from .sampling import sample_egoallo,egoallo_fk_with_supplied_shape
        ds=NymeriaSomaDataset(a.cache,'train')
        batch=collate_dataclass([ds[0],ds[len(ds)//2]]).to(device)
        cfg=network.EgoDenoiserConfig(include_hands=False,body_joint_count=22,contact_joint_count=4)
        model=network.EgoDenoiser(cfg).to(device)
        assert model.get_d_state()==202 and 'betas' not in model.decoders and 'betas' not in model.encoders
        helper=training_loss.TrainingLossComputer(training_loss.TrainingLossConfig(),device)
        loss,logged=helper.compute_denoising_loss(model,model,batch)
        assert logged['loss_term/betas']==0
        loss.backward()
        grad=model.shape_condition_encoder[0].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval();noise=torch.randn(2,97,202,device=device);t=torch.tensor([500,500],device=device)
        kwargs=dict(T_world_cpf=batch.T_world_cpf,T_cpf_tm1_cpf_t=batch.T_cpf_tm1_cpf_t,
                    project_output_rotmats=False,hand_positions_wrt_cpf=None,mask=batch.mask)
        with torch.no_grad():
            first=model(noise,t,shape_betas=batch.betas,**kwargs)
            second=model(noise,t,shape_betas=batch.betas+1.,**kwargs)
            assert (first-second).abs().max()>0 and torch.isfinite(first).all()
        shape_delta=float((first-second).abs().max());grad_norm=float(grad.norm())
        state=a.output/'acceptance_weights.pt';torch.save(model.state_dict(),state)
        reload=network.EgoDenoiser(cfg).to(device).eval()
        reload.load_state_dict(torch.load(state,map_location=device,weights_only=True),strict=True)
        with torch.no_grad():
            assert torch.equal(first,reload(noise,t,shape_betas=batch.betas,**kwargs))
        cpf=batch.T_world_cpf[0]
        previous=(SE3(cpf[0])@SE3(batch.T_cpf_tm1_cpf_t[0,0]).inverse()).parameters()
        cameras=torch.cat((previous[None],cpf),0)
        pred=sample_egoallo(model,a.upstream,cameras,batch.betas[:1])
        assert pred.body_rotmats.shape==(1,97,22,3,3)
        assert torch.isfinite(pred.pack()).all()
        assert torch.equal(pred.betas,batch.betas[:1].expand(1,97,16))
        with np.load(a.cache/'body_shape16.npz') as basis:
            fk=egoallo_fk_with_supplied_shape(pred.body_rotmats,batch.betas[:1],basis)
            fk_other=egoallo_fk_with_supplied_shape(pred.body_rotmats,batch.betas[:1]+1.,basis)
            assert torch.isfinite(fk).all() and (fk-fk_other).abs().max()>0
        dim=202
    result=dict(passed=True,model=a.model,device=str(device),state_dim=dim,
                loss=float(loss.detach()),shape_gradient_norm=grad_norm,
                shape_intervention_max_change=shape_delta,strict_reload_bitwise=True,
                shape_prediction=False,shape_not_diffused=True,
                sampling='native EgoAllo full30-update sampler' if a.model=='egoallo' else
                    ('native UEM full1000-step sampler + CFG shape retention' if device.type=='cuda'
                     else 'native UEM two sampling transitions + CFG shape retention'))
    atomic_json(a.output/'ACCEPTED.json',result)
    state.unlink()  # disposable acceptance weights, never a trained checkpoint
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
