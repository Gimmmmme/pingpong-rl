"""Bimanual 6R arm FK and Jacobian from encoder angles only.

Geometry matches the bundled bimanual arm asset and is fixed robot
calibration, not simulator state. All rotations are XYZW. No Isaac/PhysX imports.
The one legitimate live input is q[... ,6], i.e. robot joint encoder angles.

Known mount base_pos/base_quat are supplied once from scene calibration.
Paddle offset .235m is the rigid H-gripper/racket attachment geometry.
"""
from __future__ import annotations
import torch

# origin xyz before each revolute rotation, relative to the preceding link.
ORIGINS=((0.,0.,.0613),(.025,0.,.0542),(-.26,0.,0.),(.2315,0.,.075),(.0825,0.,.085),(.0395,0.,-.0945))
AXES=((0.,0.,1.),(0.,1.,0.),(0.,1.,0.),(0.,1.,0.),(0.,0.,1.),(1.,0.,0.))
ARM_MOUNT=(0.,-.266,.001898154)


def qmul(a,b):
    return torch.cat([a[...,3:4]*b[...,:3]+b[...,3:4]*a[...,:3]+torch.cross(a[...,:3],b[...,:3],dim=-1),a[...,3:4]*b[...,3:4]-(a[...,:3]*b[...,:3]).sum(-1,keepdim=True)],-1)

def qrot(q,v):
    uv=torch.cross(q[...,:3],v,dim=-1);return v+2*(q[...,3:4]*uv+torch.cross(q[...,:3],uv,dim=-1))


def paddle_kinematics(q, base_pos=(-.95,0.,.62), base_quat=(0.,0.,0.,1.), paddle_offset=(.235,0.,0.)):
    """Return paddle pos[... ,3], orientation[... ,4], Jacobian[... ,6,6]."""
    if not isinstance(q,torch.Tensor):q=torch.as_tensor(q,dtype=torch.float32)
    if q.shape[-1]!=6:raise ValueError('Expected six-joint arm encoder angles')
    shape=q.shape[:-1];device=q.device;dtype=q.dtype
    def vec(x,width):return torch.as_tensor(x,device=device,dtype=dtype).expand(*shape,width)
    rot=vec(base_quat,4).clone();pos=vec(base_pos,3).clone()+qrot(rot,vec(ARM_MOUNT,3))
    joint_positions=[];joint_axes=[]
    for i,(origin,axis) in enumerate(zip(ORIGINS,AXES)):
        pos=pos+qrot(rot,vec(origin,3));ax=vec(axis,3)
        joint_positions.append(pos);joint_axes.append(qrot(rot,ax))
        angle=q[...,i:i+1]/2;qr=torch.cat([ax*angle.sin(),angle.cos()],-1);rot=qmul(rot,qr)
    pos=pos+qrot(rot,vec(paddle_offset,3))
    js=torch.stack(joint_positions,-2);axs=torch.stack(joint_axes,-2)
    linear=torch.cross(axs,pos[...,None,:]-js,dim=-1)
    jac=torch.cat([linear.transpose(-2,-1),axs.transpose(-2,-1)],-2)
    return pos,rot,jac


def self_test():
    torch.manual_seed(1);q=torch.rand(16,6,dtype=torch.float64)*.6-.3
    p,rot,j=paddle_kinematics(q)
    eps=1e-6
    for k in range(6):
        qd=q.clone();qd[:,k]+=eps;pd,_,_=paddle_kinematics(qd)
        assert torch.max((((pd-p)/eps)-j[:,:3,k]).abs())<2e-6
    p0,r0,_=paddle_kinematics(torch.zeros(1,6))
    assert torch.allclose(p0,torch.tensor([[-.5965,-.266,.802898154]]),atol=1e-6)
    p1,r1,_=paddle_kinematics(torch.zeros(1,6),base_pos=(.95,0.,.62),base_quat=(0,0,1,0))
    assert torch.allclose(p1,torch.tensor([[.5965,.266,.802898154]]),atol=1e-6)
    print('encoder FK finite-difference Jacobian and mirror tests passed')

if __name__=='__main__':self_test()
