"""Check the optimizer experiment's scaling controls and state isolation."""
import torch

from scripts.phase3_clip_update_probe import Arm, clip, optimizer_step


def checkpoint(eps=1e-8):
    names = ['3.gate_g', '3.q_g.weight', '3.norm_g.weight']
    params = [torch.nn.Parameter(torch.tensor(.01, dtype=torch.float64)),
              torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64)),
              torch.nn.Parameter(torch.ones(2, dtype=torch.float64))]
    opt = torch.optim.AdamW(params, lr=2e-4, betas=(.9, .99), weight_decay=0, eps=eps)
    for p in params:
        opt.state[p] = {'step': torch.tensor(5.), 'exp_avg': torch.full_like(p, .003),
                        'exp_avg_sq': torch.full_like(p, .00002)}
    return {'bridges': dict(zip(names, [p.detach().clone() for p in params])), 'optimizer': opt.state_dict()}


def gradients(scale=.01):
    return [torch.tensor(100., dtype=torch.float64),
            torch.full((2,2),scale,dtype=torch.float64),torch.full((2,),scale,dtype=torch.float64)]


def test_optimizer_states_do_not_alias():
    p=checkpoint();a=Arm(p,name='a',device='cpu');b=Arm(p,name='b',device='cpu')
    optimizer_step(a,gradients(),'global',2e-4)
    assert int(a.optimizer.state[next(iter(a.parameters.values()))]['step'])==6
    assert int(b.optimizer.state[next(iter(b.parameters.values()))]['step'])==5
    assert int(p['optimizer']['state'][0]['step'])==5
    for name in a.parameters:
        assert a.optimizer.state[a.parameters[name]]['exp_avg'].data_ptr()!=b.optimizer.state[b.parameters[name]]['exp_avg'].data_ptr()


def test_scaled_moments_and_epsilon_restore_scale_invariance():
    p=checkpoint();a=Arm(p,name='a',device='cpu');raw=gradients()
    ca=clip(a.parameters,raw,'global')['coefficients'];cb=clip(a.parameters,raw,'separate')['coefficients']
    b=Arm(p,name='b',device='cpu',scales={k:cb[k]/ca[k] for k in ca},scale_epsilon=True)
    optimizer_step(a,raw,'global',2e-4);optimizer_step(b,raw,'separate',2e-4)
    for name in a.parameters:
        torch.testing.assert_close(a.parameters[name],b.parameters[name],rtol=1e-12,atol=1e-12)


def test_zero_epsilon_cold_adam_is_scale_invariant():
    p=checkpoint(eps=0);a=Arm(p,name='a',device='cpu',cold=True);b=Arm(p,name='b',device='cpu',cold=True)
    for _ in range(4):
        optimizer_step(a,gradients(),'global',2e-4);optimizer_step(b,gradients(),'separate',2e-4)
    for name in a.parameters:
        torch.testing.assert_close(a.parameters[name],b.parameters[name],rtol=1e-12,atol=1e-12)


def test_epsilon_limited_gradients_break_scale_invariance():
    p=checkpoint();a=Arm(p,name='a',device='cpu',cold=True);b=Arm(p,name='b',device='cpu',cold=True)
    x=optimizer_step(a,gradients(1e-10),'global',2e-4)
    y=optimizer_step(b,gradients(1e-10),'separate',2e-4)
    assert y['groups']['qkv']['delta_norm']/x['groups']['qkv']['delta_norm']>90
