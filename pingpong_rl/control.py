"""Ballistic residual controller + six-DOF damped Jacobian IK.

Only ``estimate`` is used for ball kinematics. No ball scene handle is accepted
or read. It is therefore usable with a calibrated RGB tracker on real robots.
Robot link pose/Jacobian/joints are proprioceptive and used by the local servo.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch


def tensor(x):
    if isinstance(x, torch.Tensor): return x
    if hasattr(x, 'torch'): return x.torch
    import warp as wp
    return wp.to_torch(x)

def qrot(q, p):
    v=q[...,:3]; w=q[...,3:4]; return p+2*(w*torch.cross(v,p,dim=-1)+torch.cross(v,torch.cross(v,p,dim=-1),dim=-1))

def qmul(a,b):
    return torch.cat([a[...,3:4]*b[...,:3]+b[...,3:4]*a[...,:3]+torch.cross(a[...,:3],b[...,:3],dim=-1),a[...,3:4]*b[...,3:4]-(a[...,:3]*b[...,:3]).sum(-1,keepdim=True)],-1)

def qerror(goal,current):
    inv=current.clone(); inv[...,:3]*=-1; rel=qmul(goal,inv); return 2*rel[...,:3]*torch.where(rel[...,3:4]<0,-1.,1.)

def unit(x, eps=1e-7): return x/x.norm(dim=-1,keepdim=True).clamp_min(eps)

def normal_quat(n, local_axis='x'):
    """Quaternion XYZW whose local X is n and local Y stays near world Y."""
    n=unit(n); world_y=torch.zeros_like(n); world_y[...,1]=1
    z=unit(torch.cross(n,world_y,dim=-1)); y=unit(torch.cross(z,n,dim=-1))
    m=torch.stack([n,y,z],-1)
    # Matrix-to-quaternion avoids degeneracy at the right robot's pi rotation.
    # Four candidate formulas are evaluated, and best-conditioned one selected.
    m00,m01,m02=m[...,0,0],m[...,0,1],m[...,0,2]
    m10,m11,m12=m[...,1,0],m[...,1,1],m[...,1,2]
    m20,m21,m22=m[...,2,0],m[...,2,1],m[...,2,2]
    qw=torch.sqrt((1+m00+m11+m22).clamp_min(0))/2
    qx=torch.sqrt((1+m00-m11-m22).clamp_min(0))/2
    qy=torch.sqrt((1-m00+m11-m22).clamp_min(0))/2
    qz=torch.sqrt((1-m00-m11+m22).clamp_min(0))/2
    qx=torch.copysign(qx,m21-m12); qy=torch.copysign(qy,m02-m20); qz=torch.copysign(qz,m10-m01)
    q=unit(torch.stack([qx,qy,qz,qw],-1))
    if local_axis=='x':return q
    if local_axis=='z':
        correction=torch.tensor([0.,-math.sqrt(.5),0.,math.sqrt(.5)],device=q.device).expand_as(q)
        return qmul(q,correction)
    raise ValueError(local_axis)

@dataclass
class ControlConfig:
    table_z: float = .72
    ball_radius: float = .02
    table_half_length: float = .7
    table_half_width: float = .35
    intercept_x: float = .50
    paddle_half_thickness: float = .005
    ball_restitution: float = .88
    return_flight_time: float = .35
    return_landing_x: float = .20
    max_predict: float = 1.0
    backswing: float = .008
    swing_time: float = .07
    max_speed: float = 2.0
    ik_damping: float = .03
    max_dq: float = .16
    max_dpos: float = .035
    max_drot: float = .20
    min_z: float = .85
    max_z: float = 1.12
    max_y: float = .28
    # Residual: xyz metres, outgoing speed scale, normal pitch/yaw radians.
    residual_position: float = .045
    residual_speed: float = .5
    residual_angle: float = .25

class BallisticResidualController:
    """Two bimanual robot sides; RL action [N,2,6] = xyz, speed, pitch, yaw.

    The nominal policy predicts an intercept and a return landing point. RL is
    responsible for residual intercept/face/speed correction at every decision.
    This is not a replayed trajectory: it reacts to the currently supplied ball
    estimate, including velocity changes after physical racket/table impacts.
    """
    def __init__(self, env, config=None, joint_names=None, body_name='right_arm_link6',
                 paddle_offset=(.235,0.,0.), paddle_normal_axis='x'):
        self.env=env; self.robots=env.robots; self.device=torch.device(env.device)
        self.n=int(env.num_envs); self.origins=tensor(env.origins)
        self.cfg=config or ControlConfig(); self.offset=torch.tensor(paddle_offset,device=self.device).repeat(self.n,1)
        self.normal_axis=paddle_normal_axis
        # v2 exposes both arm chains on each bimanual robot. The scene keeps
        # ``active_arms[N,2]`` and the controller resolves the corresponding
        # joint/body IDs on every observation/IK call. This retains the v1
        # [N,2,6] action interface while allowing per-episode arm selection.
        self.joints_by_arm=[]; self.bodies_by_arm=[]
        for robot in self.robots:
            names=list(robot.joint_names); bnames=list(robot.body_names)
            arm_ids=[]; arm_bodies=[]
            for arm in ('left','right'):
                wanted=joint_names if (joint_names and arm=='right') else [f'{arm}_arm_joint{i}' for i in range(1,7)]
                if all(name in names for name in wanted): ids=[names.index(name) for name in wanted]
                else:
                    candidates=[(i,n) for i,n in enumerate(names) if (arm in n and 'joint' in n and not any(k in n.lower() for k in ('finger','grip','knuckle')))]
                    if len(candidates)!=6: raise ValueError(f'Cannot identify {arm} six joints: {names}')
                    ids=[i for i,n in candidates]
                bname=f'{arm}_arm_link6'
                if bname in bnames: body=bnames.index(bname)
                else:
                    cand=[i for i,n in enumerate(bnames) if arm in n and ('link6' in n or 'link_6' in n)]
                    if len(cand)!=1: raise ValueError(f'Cannot identify {bname}: {bnames}')
                    body=cand[0]
                arm_ids.append(ids);arm_bodies.append(body)
            self.joints_by_arm.append(arm_ids); self.bodies_by_arm.append(arm_bodies)
        self._sync_active_ids()
        self.home=[tensor(robot.data.joint_pos)[:,ids].clone() for robot,ids in zip(self.robots,self.joints)]
        self.eye=torch.eye(6,device=self.device).expand(self.n,6,6)
        self.last_targets=torch.stack(self.home,1); self.last_action=torch.zeros(self.n,2,6,device=self.device)
        self.target_pos=torch.zeros(self.n,2,3,device=self.device); self.target_quat=torch.zeros(self.n,2,4,device=self.device)
        self.intercept_time=torch.ones(self.n,2,device=self.device)
        # Per-rally desired landing ordinate.  This is a calibrated task
        # parameter, not simulator state; RGB/proprio deployment receives the
        # same value from the serve/strategy command.
        self.landing_y=torch.zeros(self.n,2,device=self.device)
        self.last_paddle_pos=None; self.paddle_velocity=torch.zeros(self.n,2,3,device=self.device)

    def set_landing_targets(self, y):
        y=torch.as_tensor(y,dtype=torch.float32,device=self.device)
        if tuple(y.shape)!=(self.n,2): raise ValueError(f'landing targets must be [{self.n},2]')
        self.landing_y=y.clamp(-self.cfg.max_y,self.cfg.max_y)

    def _sync_active_ids(self):
        """Resolve selected arm IDs (one pair per environment)."""
        active=getattr(self.env,'active_arms',torch.zeros((self.n,2),device=self.device,dtype=torch.long))
        # IDs are static per robot, while arm selection may differ per env;
        # retain all IDs and gather per-env in the state/Jacobian functions.
        self.joints=[self.joints_by_arm[ri][1] for ri in range(2)]
        self.bodies=[self.bodies_by_arm[ri][1] for ri in range(2)]

    def paddle_state(self):
        self._sync_active_ids()
        active=self.env.active_arms
        ps=[]; qs=[]
        for ri,robot in enumerate(self.robots):
            qall=tensor(robot.data.body_quat_w)[:,self.bodies_by_arm[ri]]
            pall=tensor(robot.data.body_pos_w)[:,self.bodies_by_arm[ri]]-self.origins[:,None,:]
            ix=active[:,ri,None,None].expand(self.n,1,4)
            q=torch.gather(qall,1,ix).squeeze(1)
            ix3=active[:,ri,None,None].expand(self.n,1,3)
            p=torch.gather(pall,1,ix3).squeeze(1)
            ps.append(p+qrot(q,self.offset)); qs.append(q)
        return torch.stack(ps,1),torch.stack(qs,1)

    def proprioception(self):
        self._sync_active_ids();active=self.env.active_arms; out=[]
        for ri,r in enumerate(self.robots):
            qall=torch.stack([tensor(r.data.joint_pos)[:,ids] for ids in self.joints_by_arm[ri]],1)
            ix=active[:,ri,None,None].expand(self.n,1,6)
            out.append(torch.gather(qall,1,ix).squeeze(1))
        return torch.stack(out,1)

    def predict(self, p, v, time):
        """Ballistic position at time with first table bounce, using estimated state.

        Bounce is geometric against the known static table, not simulator contact
        metadata. Court bounds and table calibration are fixed deployment data.
        """
        g=9.81; t=time.clamp(0,self.cfg.max_predict)
        z0=p[:,2]; vz=v[:,2]; floor=self.cfg.table_z+self.cfg.ball_radius
        disc=(vz*vz+2*g*(z0-floor)).clamp_min(0)
        tb=(vz+torch.sqrt(disc))/g
        pb=p+v*tb[:,None]; pb[:,2]=floor
        valid=(tb>1e-4)&(tb<t)&(pb[:,0].abs()<self.cfg.table_half_length)&(pb[:,1].abs()<self.cfg.table_half_width)&(z0>=floor-.015)
        out=p+v*t[:,None]; out[:,2]-=.5*g*t*t
        after=(t-tb).clamp_min(0); vv=v.clone(); vv[:,2]=-(vz-g*tb)*self.cfg.ball_restitution
        bounced=pb+vv*after[:,None]; bounced[:,2]-=.5*g*after*after
        out=torch.where(valid[:,None],bounced,out)
        vel=v.clone(); vel[:,2]-=g*t; bv=vv.clone(); bv[:,2]-=g*after
        vel=torch.where(valid[:,None],bv,vel)
        return out,vel

    def goals(self, estimated_pos, estimated_vel, residual, confidence=None):
        p=torch.as_tensor(estimated_pos,dtype=torch.float32,device=self.device)
        v=torch.as_tensor(estimated_vel,dtype=torch.float32,device=self.device)
        a=torch.as_tensor(residual,dtype=torch.float32,device=self.device).reshape(self.n,2,6).clamp(-1,1)
        targets=[]; orientations=[]; impacts=[]; phases=[]
        cfg=self.cfg
        for side in range(2):
            s=-1. if side==0 else 1.; inward=-s
            vx=v[:,0]; incoming=(vx*s>.08)
            plane=torch.full((self.n,),s*cfg.intercept_x,device=self.device)
            time=(plane-p[:,0])/torch.where(vx.abs()>.03,vx,torch.full_like(vx,.03)*s)
            time=time.clamp(0,cfg.max_predict)
            contact,vin=self.predict(p,v,time)
            contact[:,0]=plane
            contact[:,1]+=a[:,side,1]*cfg.residual_position
            contact[:,2]+=a[:,side,2]*cfg.residual_position
            contact[:,1].clamp_(-cfg.max_y,cfg.max_y); contact[:,2].clamp_(cfg.min_z,cfg.max_z)
            flight=cfg.return_flight_time*(1-.18*a[:,side,3])
            desired=torch.zeros_like(p)
            desired[:,0]=(inward*cfg.return_landing_x-contact[:,0])/flight
            desired[:,1]=(self.landing_y[:,side]-contact[:,1])/flight
            desired[:,2]=(cfg.table_z+cfg.ball_radius-contact[:,2]+.5*9.81*flight**2)/flight
            # Desired collision normal follows the impulse vector. Tangential
            # sphere velocity is preserved by the smooth rigid paddle surface.
            normal=unit(desired-vin)
            normal[:,0]=normal[:,0].abs()*inward
            # Actor can correct pitch/yaw due to unmodelled spin/contact/latency.
            pitch=a[:,side,4]*cfg.residual_angle; yaw=a[:,side,5]*cfg.residual_angle
            px=normal[:,0]*torch.cos(pitch)-normal[:,2]*torch.sin(pitch)
            pz=normal[:,0]*torch.sin(pitch)+normal[:,2]*torch.cos(pitch)
            nx=px*torch.cos(yaw)-normal[:,1]*torch.sin(yaw)
            ny=px*torch.sin(yaw)+normal[:,1]*torch.cos(yaw)
            normal=unit(torch.stack([nx,ny,pz],-1))
            # Required normal paddle speed from Newton's restitution law.
            speed=((desired*normal).sum(-1)+cfg.ball_restitution*(vin*normal).sum(-1))/(1+cfg.ball_restitution)
            speed=speed.clamp(-.3,cfg.max_speed)
            # At contact t=0 the paddle surface reaches the ball. Farther away
            # it is cocked backwards; repeated feedback advances the stroke.
            stroke=(cfg.swing_time-time).clamp(-cfg.swing_time,cfg.swing_time)
            base=contact-normal*(cfg.ball_radius+cfg.paddle_half_thickness)
            base=base+normal*(stroke*speed-cfg.backswing*(time/cfg.swing_time).clamp(0,1))[:,None]
            base[:,0]+=a[:,side,0]*cfg.residual_position
            ready=torch.stack([plane-inward*.015,torch.zeros_like(plane),torch.full_like(plane,.95)],-1)
            active=incoming&(time<cfg.max_predict)&(p[:,2]>.5)
            if confidence is not None:active &= torch.as_tensor(confidence,device=self.device).reshape(self.n)>.05
            base=torch.where(active[:,None],base,ready)
            ready_n=torch.zeros_like(normal); ready_n[:,0]=inward; ready_n[:,2]=.16
            normal=torch.where(active[:,None],normal,unit(ready_n))
            targets.append(base); orientations.append(normal_quat(normal,self.normal_axis)); impacts.append(time); phases.append(active.float())
        self.target_pos=torch.stack(targets,1); self.target_quat=torch.stack(orientations,1); self.intercept_time=torch.stack(impacts,1)
        self.last_action=a; return self.target_pos,self.target_quat,torch.stack(phases,1)

    def ik(self, target_pos, target_quat):
        ps,qs=self.paddle_state(); results=[]; cfg=self.cfg
        active=self.env.active_arms
        for side,robot in enumerate(self.robots):
            allj=tensor(robot.data.body_link_jacobian_w)
            qparts=[];jparts=[];lparts=[]
            for ai in (0,1):
                ids=self.joints_by_arm[side][ai]; body=self.bodies_by_arm[side][ai]
                qparts.append(tensor(robot.data.joint_pos)[:,ids])
                brow=body-int(robot.is_fixed_base); cols=[j+int(robot.num_base_dofs) for j in ids]
                jparts.append(allj[:,brow,:,:][:,:,cols].clone())
                lim=tensor(robot.data.soft_joint_pos_limits)[:,ids]
                lparts.append(lim)
            qi=torch.stack(qparts,1);ji=torch.stack(jparts,1);li=torch.stack(lparts,1)
            ix=active[:,side,None,None].expand(self.n,1,6);q=torch.gather(qi,1,ix).squeeze(1)
            ixj=active[:,side,None,None,None].expand(self.n,1,6,6);jac=torch.gather(ji,1,ixj).squeeze(1)
            ixl=active[:,side,None,None,None].expand(self.n,1,6,2);limits=torch.gather(li,1,ixl).squeeze(1)
            r=qrot(qs[:,side],self.offset)
            jac[:,:3]+=torch.cross(jac[:,3:].transpose(1,2),r[:,None,:].expand(-1,6,-1),dim=-1).transpose(1,2)
            dp=target_pos[:,side]-ps[:,side]; dp*=torch.clamp(cfg.max_dpos/dp.norm(dim=-1,keepdim=True).clamp_min(1e-7),max=1)
            axis=torch.zeros((self.n,3),device=self.device);axis[:,0 if self.normal_axis=='x' else 2]=1
            current_n=qrot(qs[:,side],axis);goal_n=qrot(target_quat[:,side],axis)
            # Disk roll is unconstrained: only its physical face normal matters.
            dr=torch.cross(current_n,goal_n,dim=-1); dr*=torch.clamp(cfg.max_drot/dr.norm(dim=-1,keepdim=True).clamp_min(1e-7),max=1)
            error=torch.cat([dp,dr],-1); dq=(jac.transpose(1,2)@torch.linalg.solve(jac@jac.transpose(1,2)+cfg.ik_damping**2*self.eye,error[:,:,None])).squeeze(-1)
            dq=dq.clamp(-cfg.max_dq,cfg.max_dq)
            results.append(torch.clamp(q+dq,min=limits[:,:,0]+.015,max=limits[:,:,1]-.015))
        self.last_targets=torch.stack(results,1); return self.last_targets

    def observation(self, estimated_pos, estimated_vel, confidence=None):
        """[N,2,36], canonical player-frame observations for one shared actor.

        Robot 1 is rotated pi around world Z to match Robot 0. Static table
        calibration and proprioception are permitted at deployment time.
        Call goals/step first if the freshest intercept prediction is needed.
        """
        p=torch.as_tensor(estimated_pos,dtype=torch.float32,device=self.device)
        v=torch.as_tensor(estimated_vel,dtype=torch.float32,device=self.device)
        pp,_=self.paddle_state(); q=self.proprioception()
        active=self.env.active_arms; qvparts=[]
        for ri,r in enumerate(self.robots):
            allqv=torch.stack([tensor(r.data.joint_vel)[:,ids] for ids in self.joints_by_arm[ri]],1)
            ix=active[:,ri,None,None].expand(self.n,1,6)
            qvparts.append(torch.gather(allqv,1,ix).squeeze(1))
        qv=torch.stack(qvparts,1)
        conf=torch.ones(self.n,device=self.device) if confidence is None else torch.as_tensor(confidence,device=self.device)
        obs=[]
        for side in range(2):
            flip=1. if side==0 else -1.
            axes=torch.tensor([flip,flip,1.],device=self.device)
            bp=p.clone();bp[:,2]-=self.cfg.table_z
            own=pp[:,side].clone();own[:,2]-=self.cfg.table_z
            opp=pp[:,1-side].clone();opp[:,2]-=self.cfg.table_z
            target=self.target_pos[:,side].clone();target[:,2]-=self.cfg.table_z
            incoming=(v[:,0]*(-flip)>.08).float()
            last=self.last_action[:,side].clone()
            # World xyz residual is mirrored with the observation; pitch and yaw
            # are local policy outputs, transformed by canonical_action below.
            last[:,:2]*=flip
            last[:,4]*=flip
            # Explicitly expose the selected-hand identity; the shared actor
            # can learn mirrored trajectories while retaining arm-specific
            # reachability/proprioception.  Two one-hot values are stable in
            # deployment and are included in the checkpoint observation hash.
            hand=torch.nn.functional.one_hot(active[:,side],num_classes=2).to(p.dtype)
            land=(self.landing_y[:,side,None]*flip).clamp(-1,1)
            ob=torch.cat([bp*axes,v*axes/3.,own*axes,opp*axes,q[:,side]/math.pi,qv[:,side]/6.,target*axes,self.intercept_time[:,side,None],conf[:,None],incoming[:,None],land,hand,last],-1)
            obs.append(ob.clamp(-5,5))
        return torch.stack(obs,1)

    @staticmethod
    def canonical_action(action):
        """Map shared-policy player-local xyz residuals to the world controller."""
        world=action.clone() if isinstance(action,torch.Tensor) else torch.as_tensor(action).clone()
        world[:,1,:2]*=-1
        world[:,1,4]*=-1  # local +pitch lifts either paddle normal
        # Yaw is about world/local Z and is unchanged by the pi frame rotation.
        return world

    def step(self, estimated_pos, estimated_vel, residual, confidence=None):
        pos,quat,phase=self.goals(estimated_pos,estimated_vel,residual,confidence)
        return self.ik(pos,quat)


def math_self_test():
    normals=torch.tensor([[1.,0.,0.],[-1.,0.,0.],[.9,.1,.2],[-.9,.1,.2]])
    normals=unit(normals); q=normal_quat(normals); x=torch.zeros_like(normals);x[:,0]=1
    assert torch.allclose(qrot(q,x),normals,atol=2e-5),(qrot(q,x),normals)
    assert torch.isfinite(q).all(); assert qerror(q,q).abs().max()<1e-5
    print('pp_control quaternion/normal self-test passed')

if __name__=='__main__':math_self_test()
