"""RGB-only asymmetric actor--critic for the two-arm ping-pong task.

The actor receives only quantities that can be obtained at deployment time:
tracked RGB ball position/velocity, two paddle poses, confidence, arm proprio
and the contact-phase features.  During simulation training the critic may also
receive a privileged vector (true ball state and contact impulse); the actor
checkpoint is still consumable with the RGB vector alone.

This module intentionally has a small dependency surface and can be imported by
an Isaac Lab project without modifying the simulator.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

ACTOR_OBS_DIM = 39
CRITIC_OBS_DIM = ACTOR_OBS_DIM
PRIVILEGED_OBS_DIM = ACTOR_OBS_DIM + 9
ACTION_DIM = 6  # per-player xyz + speed + pitch + yaw residual


def _mlp(inp: int, hidden: int, out: int, gain: float = 1.0):
    net = nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, math.sqrt(2.0)); nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(net[-1].weight, gain)
    return net

@dataclass
class PPOConfig:
    hidden: int = 192
    lr: float = 3e-4
    gamma: float = 0.985
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.002
    epochs: int = 6
    minibatch: int = 512
    max_grad_norm: float = 0.7
    initial_std: float = 0.35
    min_std: float = 0.035
    max_std: float = 0.8

class AsymmetricPPO(nn.Module):
    """PPO with separate RGB actor and optionally privileged critic."""
    def __init__(self, actor_dim=ACTOR_OBS_DIM, action_dim=ACTION_DIM,
                 critic_dim=CRITIC_OBS_DIM, device='cpu', seed=0, **kwargs):
        super().__init__()
        self.actor_dim, self.action_dim, self.critic_dim = actor_dim, action_dim, critic_dim
        self.device = torch.device(device); self.seed = int(seed); torch.manual_seed(seed)
        self.cfg = PPOConfig(**kwargs)
        self.actor = _mlp(actor_dim, self.cfg.hidden, action_dim, 0.01)
        self.critic = _mlp(critic_dim, self.cfg.hidden, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(self.cfg.initial_std)))
        self.updates = 0; self.transitions = 0
        self.to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, eps=1e-5)

    def _t(self, x, dim=None):
        y = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if dim and (y.ndim != 2 or y.shape[-1] != dim): raise ValueError((y.shape, dim))
        if not torch.isfinite(y).all(): raise ValueError('non-finite observation')
        return y
    def _dist(self, obs):
        mean = 3.0 * torch.tanh(self.actor(obs) / 3.0)
        std = self.log_std.clamp(math.log(self.cfg.min_std), math.log(self.cfg.max_std)).exp()
        return Normal(mean, std)
    @staticmethod
    def _logp(dist, action):
        a = action.clamp(-1 + 1e-6, 1 - 1e-6); z = torch.atanh(a)
        return (dist.log_prob(z) - torch.log1p(-a.square())).sum(-1)
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        ob = self._t(obs, self.actor_dim); d = self._dist(ob)
        z = d.mean if deterministic else d.rsample(); a = z.tanh()
        return a.cpu().numpy(), self._logp(d, a).cpu().numpy()
    @torch.no_grad()
    def act_with_value(self, obs, critic_obs=None, deterministic=False):
        ob = self._t(obs, self.actor_dim); d = self._dist(ob)
        z = d.mean if deterministic else d.sample(); a = z.tanh()
        co = self._t(critic_obs if critic_obs is not None else obs, self.critic_dim)
        return a.cpu().numpy(), self._logp(d, a).cpu().numpy(), self.critic(co).squeeze(-1).cpu().numpy()
    @torch.no_grad()
    def deterministic(self, obs):
        return self._dist(self._t(obs, self.actor_dim)).mean.tanh().cpu().numpy()
    def update(self, roll: dict[str, Any]):
        # obs/actions/logprob/value/reward/done [T,N], critic_obs [T,N,C], last_critic_obs [N,C]
        obs = self._t(roll['obs']).reshape(-1, self.actor_dim)
        actions = self._t(roll['action']).reshape(-1, self.action_dim)
        old_lp = torch.as_tensor(roll['logprob'], dtype=torch.float32, device=self.device).reshape(-1)
        oldv = torch.as_tensor(roll['value'], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(roll['reward'], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(roll['done'], dtype=torch.float32, device=self.device)
        critic_obs = self._t(roll['critic_obs']).reshape(-1, self.critic_dim)
        tsteps, nenv = rewards.shape
        if roll['last_critic_obs'] is None: raise ValueError('last_critic_obs required')
        last = self.critic(self._t(roll['last_critic_obs'], self.critic_dim)).squeeze(-1)
        with torch.no_grad():
            adv = torch.zeros_like(rewards); gae = torch.zeros(nenv, device=self.device); nv = last
            for t in reversed(range(tsteps)):
                cont = 1 - dones[t]; delta = rewards[t] + self.cfg.gamma * nv * cont - oldv[t]
                gae = delta + self.cfg.gamma * self.cfg.gae_lambda * cont * gae; adv[t] = gae; nv = oldv[t]
            ret = adv + oldv; af = ((adv.flatten()-adv.mean())/(adv.std(unbiased=False)+1e-8)).detach(); rf = ret.flatten().detach()
        total = obs.shape[0]; mets = {k:0.0 for k in ('policy_loss','value_loss','entropy','kl','clip_fraction')}; count=0
        for _ in range(self.cfg.epochs):
            permutation = torch.randperm(total, device=self.device)
            for start in range(0,total,self.cfg.minibatch):
                ix = permutation[start:start+self.cfg.minibatch]
                d=self._dist(obs[ix]); lp=self._logp(d,actions[ix]); ratio=(lp-old_lp[ix]).exp(); clipr=ratio.clamp(1-self.cfg.clip,1+self.cfg.clip)
                pl=-torch.minimum(ratio*af[ix],clipr*af[ix]).mean(); vl=0.5*(self.critic(critic_obs[ix]).squeeze(-1)-rf[ix]).square().mean(); ent=d.entropy().sum(-1).mean()
                loss=pl+self.cfg.value_coef*vl-self.cfg.entropy_coef*ent; self.opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(self.parameters(),self.cfg.max_grad_norm); self.opt.step()
                with torch.no_grad(): kl=((ratio-1)-torch.log(ratio.clamp_min(1e-8))).mean(); cf=((ratio-1).abs()>self.cfg.clip).float().mean(); self.log_std.clamp_(math.log(self.cfg.min_std),math.log(self.cfg.max_std))
                k=len(ix); count+=k
                for name,val in (('policy_loss',pl),('value_loss',vl),('entropy',ent),('kl',kl),('clip_fraction',cf)): mets[name]+=float(val.detach())*k
        self.updates += 1; self.transitions += total
        mets={k:v/max(count,1) for k,v in mets.items()}; mets.update(mean_reward=float(rewards.mean()), updates=self.updates, transitions=self.transitions, std=float(self.log_std.detach().exp().mean())); return mets
    def save(self, path, metadata=None):
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix('.tmp')
        torch.save({'format':'pingpong_asymmetric_ppo_v1','actor_dim':self.actor_dim,'critic_dim':self.critic_dim,'action_dim':self.action_dim,'config':asdict(self.cfg),'actor':self.actor.state_dict(),'critic':self.critic.state_dict(),'log_std':self.log_std.detach().cpu(),'updates':self.updates,'transitions':self.transitions,'metadata':metadata or {}},tmp); tmp.replace(p)
    @classmethod
    def load(cls, path, device='cpu'):
        ck=torch.load(path,map_location=device,weights_only=False); obj=cls(ck['actor_dim'],ck['action_dim'],ck['critic_dim'],device=device,**ck['config']); obj.actor.load_state_dict(ck['actor']); obj.critic.load_state_dict(ck['critic']); obj.log_std.data.copy_(ck['log_std'].to(obj.device)); obj.updates=ck.get('updates',0); obj.transitions=ck.get('transitions',0); return obj


def _feature_vector(value, size, name):
    out = np.asarray(value, dtype=np.float32).reshape(-1)
    if out.size != size or not np.isfinite(out).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return out


def actor_observation(ball_xyz, ball_vxyz, own_paddle, opponent_paddle, joints,
                      joint_vel, target_xyz, time_to_impact, confidence, incoming,
                      previous_action, *, landing_y, active_arm):
    """Build the checkpoint's 39 features from one player's canonical frame.

    Positions are relative to the table surface. Velocities are metres/second;
    encoders and their velocities are radians and radians/second. The caller
    mirrors world coordinates and the previous action into the player frame.
    ``active_arm`` is 0 for the left hand and 1 for the right hand.
    """
    if active_arm not in (0, 1):
        raise ValueError('active_arm must be 0 or 1')
    chunks = [
        _feature_vector(ball_xyz, 3, 'ball_xyz'),
        _feature_vector(ball_vxyz, 3, 'ball_vxyz') / 3.0,
        _feature_vector(own_paddle, 3, 'own_paddle'),
        _feature_vector(opponent_paddle, 3, 'opponent_paddle'),
        _feature_vector(joints, 6, 'joints') / np.pi,
        _feature_vector(joint_vel, 6, 'joint_vel') / 6.0,
        _feature_vector(target_xyz, 3, 'target_xyz'),
        _feature_vector([time_to_impact, confidence, incoming], 3, 'phase'),
        np.clip(_feature_vector(landing_y, 1, 'landing_y'), -1.0, 1.0),
        np.eye(2, dtype=np.float32)[int(active_arm)],
        _feature_vector(previous_action, ACTION_DIM, 'previous_action'),
    ]
    return np.clip(np.concatenate(chunks), -5.0, 5.0)


def privileged_observation(actor_obs, true_ball_xyz, true_ball_vxyz, contact_impulse):
    """Optional 48-feature critic input for ``critic_dim=PRIVILEGED_OBS_DIM``.

    Released checkpoints use the 39 actor features for their critic as well.
    These additional values are for training a separate privileged critic.
    """
    return np.concatenate([
        _feature_vector(actor_obs, ACTOR_OBS_DIM, 'actor_obs'),
        _feature_vector(true_ball_xyz, 3, 'true_ball_xyz'),
        _feature_vector(true_ball_vxyz, 3, 'true_ball_vxyz'),
        _feature_vector(contact_impulse, 3, 'contact_impulse'),
    ])

if __name__ == '__main__':
    # CPU smoke: check dimensions, train a few PPO updates with random rollouts,
    # and ensure deployment path never asks the critic for a privileged vector.
    p=AsymmetricPPO(seed=7,hidden=32,epochs=2,minibatch=64)
    rng=np.random.default_rng(7); t,n=8,16
    obs=rng.normal(size=(t,n,ACTOR_OBS_DIM)).astype('f4'); action,lp,v=p.act_with_value(obs.reshape(t*n,ACTOR_OBS_DIM)); action=action.reshape(t,n,ACTION_DIM); lp=lp.reshape(t,n); v=v.reshape(t,n)
    roll={'obs':obs,'action':action,'logprob':lp,'value':v,'reward':rng.normal(size=(t,n)).astype('f4'),'done':np.zeros((t,n),'f4'),'critic_obs':obs.copy(),'last_critic_obs':obs[-1].copy()}
    p.update(roll); q=Path('/tmp/pingpong_ppo.pt'); p.save(q); z=AsymmetricPPO.load(q); assert z.deterministic(obs[0,:2]).shape==(2,ACTION_DIM); print('policy self-test passed',p.updates,q)
