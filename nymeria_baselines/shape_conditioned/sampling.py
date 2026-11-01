"""Native EgoAllo sampler with an external, never-diffused actor shape.

The extracted sampler body is unchanged. Its trajectory adapter packs motion
only; shape is supplied when unpacking and at every model invocation. SMPL
guidance is explicitly disabled for this SOMA experiment.
"""
import ast
from pathlib import Path
import time
from types import SimpleNamespace


def sample_egoallo(model, upstream, camera_poses, shape_betas, num_samples=1):
    import numpy as np
    import torch
    from tqdm.auto import tqdm
    from egoallo import network
    from egoallo.transforms import SE3
    from egoallo.tensor_dataclass import TensorDataclass
    device = camera_poses.device
    if shape_betas.shape != (num_samples, 1, 16):
        raise ValueError('Actor shape must be [num_samples,1,16]')

    class MotionTrajectory(network.EgoDenoiseTraj):
        def pack(self):
            return super().pack()[...,16:]

    class Adapter:
        @staticmethod
        def unpack(x, **kwargs):
            result = network.EgoDenoiseTraj.unpack(torch.cat((shape_betas.expand(-1,x.shape[1],-1),x),-1),
                **kwargs, body_joint_count=22, contact_joint_count=4)
            return MotionTrajectory(**vars(result))

    class Denoiser:
        config = model.config
        get_d_state = model.get_d_state
        @staticmethod
        def forward(*args, **kwargs):
            return model(*args, shape_betas=shape_betas, **kwargs)

    path = Path(upstream)/'egoallo/src/egoallo/sampling.py'
    tree = ast.parse(path.read_text())
    names = {'quadratic_ts','CosineNoiseScheduleConstants','run_sampling_with_stitching'}
    nodes = [n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='__future__'
             or isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    env = dict(torch=torch,np=np,time=time,tqdm=tqdm,TensorDataclass=TensorDataclass,
               SE3=SE3,network=SimpleNamespace(EgoDenoiseTraj=Adapter))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    result = env['run_sampling_with_stitching'](Denoiser(),None,'off',False,False,
        camera_poses,0.,None,None,num_samples,device,guidance_verbose=False)
    assert torch.equal(result.betas,shape_betas.expand(-1,result.betas.shape[1],-1))
    return result


def egoallo_fk_with_supplied_shape(local_rotations, shape_betas, shape_basis):
    """Return root-relative body joints using supplied shape, never a prediction."""
    import torch
    from nymeria_baselines.body_shape import PARENTS
    r = local_rotations
    mean = torch.as_tensor(shape_basis['mean'],device=r.device,dtype=r.dtype)
    scale = torch.as_tensor(shape_basis['scale'],device=r.device,dtype=r.dtype)
    components = torch.as_tensor(shape_basis['components'],device=r.device,dtype=r.dtype)
    batch,time = r.shape[:2]
    assert r.shape == (batch,time,22,3,3) and shape_betas.shape == (batch,1,16)
    neutral = (mean+(shape_betas*scale)@components).reshape(batch,1,23,3)
    neutral = neutral-neutral[:,:,:1]
    rotations = [torch.eye(3,device=r.device,dtype=r.dtype).expand(batch,time,3,3)]
    positions = [torch.zeros(batch,time,3,device=r.device,dtype=r.dtype)]
    for j in range(1,23):
        parent=PARENTS[j]
        rotations.append(rotations[parent]@r[:,:,j-1])
        offset=(neutral[:,:,j]-neutral[:,:,parent]).expand(batch,time,3)
        positions.append(positions[parent]+(rotations[parent]@offset[...,None]).squeeze(-1))
    return torch.stack(positions,2)
