"""Encoder-only deployment controller.

This module is the deployment boundary for the learned controller.  It copies
only joint encoders, joint velocities, joint/name metadata, and static joint
limits from an Isaac Lab articulation.  Paddle pose and Jacobians are rebuilt
from :mod:`pingpong_rl.kinematics`; no simulator body poses, simulator
Jacobians, ball state, or contact state are read.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from .control import BallisticResidualController, qrot
from .kinematics import paddle_kinematics


_KNOWN_JOINT_NAMES = tuple(
    [f"left_arm_joint{i}" for i in range(1, 7)]
    + ["left_arm_gripper"]
    + [f"right_arm_joint{i}" for i in range(1, 7)]
    + ["right_arm_gripper"]
)
_KNOWN_BODY_NAMES = ("left_arm_link6", "right_arm_link6")


def _copy_tensor(value: Any, *, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    value = value if isinstance(value, torch.Tensor) else getattr(value, "torch", value)
    return torch.as_tensor(value, device=device, dtype=dtype).detach().clone()


@dataclass
class _RobotSnapshot:
    """A deliberately small, read-only robot view used by the controller."""

    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    data: Any
    num_base_dofs: int = 0
    is_fixed_base: bool = True


class RobotProprioception:
    """Copy encoder-only robot data from an environment.

    ``sync`` is the only method that touches the source environment.  The
    resulting object contains no scene, body, ball, contact, or simulator
    handles, so it is safe to pass to policy/control code.
    """

    def __init__(self, env: Any):
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.device = torch.device(getattr(env, "device", "cpu"))
        self.origins = _copy_tensor(getattr(env, "origins", torch.zeros(self.num_envs, 3)), device=self.device)
        self._source = None
        self.active_arms = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.long)
        self.robots: list[_RobotSnapshot] = []
        self.base_positions, self.base_quaternions = self._static_bases(env)
        self.sync(env)

    @staticmethod
    def _static_bases(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
        params = getattr(env, "scene_parameters", {}) or {}
        d = float(params.get("base_distance", 0.95))
        y = float(params.get("base_y", 0.0))
        z = float(params.get("base_height", 0.62))
        pos = torch.tensor(((-d, y, z), (d, -y, z)), dtype=torch.float32)
        quat = torch.tensor(((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0)), dtype=torch.float32)
        return pos, quat

    def _joint_names(self, robot: Any) -> tuple[str, ...]:
        names = getattr(robot, "joint_names", None)
        return tuple(str(n) for n in names) if names else _KNOWN_JOINT_NAMES

    def sync(self, env: Any | None = None) -> "RobotProprioception":
        """Refresh encoder values and active-hand selection from ``env``."""
        if env is None:
            raise RuntimeError("RobotProprioception.sync requires an environment")
        active_arms = torch.as_tensor(
            getattr(env, "active_arms", torch.zeros((self.num_envs, 2))),
            device=self.device,
            dtype=torch.long,
        ).detach().clone()
        if active_arms.shape != (self.num_envs, 2) or bool(((active_arms < 0) | (active_arms > 1)).any()):
            raise ValueError("active arms must have shape [N,2] with values 0/1")
        robots = list(getattr(env, "robots"))
        if len(robots) != 2:
            raise ValueError("encoder deployment expects exactly two robot sides")
        snapshots: list[_RobotSnapshot] = []
        for side, robot in enumerate(robots):
            data = getattr(robot, "data")
            q = _copy_tensor(getattr(data, "joint_pos"), device=self.device)
            qv = _copy_tensor(getattr(data, "joint_vel"), device=self.device)
            names = self._joint_names(robot)
            if q.shape != (self.num_envs, len(names)) or qv.shape != q.shape:
                raise ValueError("encoder positions and velocities must match [N,joint_names]")
            if self.robots and names != self.robots[side].joint_names:
                raise ValueError("joint names and ordering cannot change after controller calibration")
            limits_value = getattr(data, "soft_joint_pos_limits", None)
            if limits_value is None:
                limits_value = getattr(data, "joint_pos_limits", None)
            if limits_value is None:
                limits = torch.full((*q.shape, 2), 3.14, device=self.device)
                limits[..., 0] *= -1
            else:
                limits = _copy_tensor(limits_value, device=self.device)
                if limits.shape == (len(names), 2):
                    limits = limits.expand(self.num_envs, -1, -1).clone()
                if limits.shape != (*q.shape, 2):
                    raise ValueError("joint limits must have shape [N,joints,2] or [joints,2]")
            # Preserve only encoder arrays and static metadata in the facade.
            snap_data = SimpleNamespace(joint_pos=q, joint_vel=qv, soft_joint_pos_limits=limits)
            snapshots.append(
                _RobotSnapshot(
                    joint_names=names,
                    body_names=_KNOWN_BODY_NAMES,
                    data=snap_data,
                )
            )
        # The base controller keeps this list. Preserve its identity so every
        # encoder consumer sees the new snapshots after a sync.
        self.robots[:] = snapshots
        self.active_arms = active_arms
        return self

    def attach(self, env: Any) -> "RobotProprioception":
        """Alias for a later sync when a caller wants to reuse the facade."""
        return self.sync(env)


class EncoderResidualController(BallisticResidualController):
    """Ballistic residual controller driven by RGB estimates and encoders only.

    Construct with ``EncoderResidualController(env)``.  Call ``sync(env)``
    after each simulator control step (and after changing active arms); the
    controller then uses FK for paddle state and FK Jacobians for IK.
    """

    def __init__(self, env: Any, *args, **kwargs):
        self.proprio = RobotProprioception(env)
        super().__init__(self.proprio, *args, **kwargs)
        # ``super`` keeps a reference to the safe facade, never to ``env``.
        self.env = self.proprio
        self._base_positions = self.proprio.base_positions.to(self.device)
        self._base_quaternions = self.proprio.base_quaternions.to(self.device)
        self._arm_mount_delta = torch.tensor((0.0, 0.532, 0.0), device=self.device)

    def sync(self, env: Any) -> "EncoderResidualController":
        self.proprio.sync(env)
        self.env = self.proprio
        self.robots = self.proprio.robots
        return self

    def reset(self) -> None:
        """Reset temporal control state at the beginning of a rally."""
        self.last_paddle_pos = None
        self.paddle_velocity.zero_()
        self.last_action.zero_()
        self.last_targets = self.proprioception().clone()
        self.target_pos.zero_()
        self.target_quat.zero_()
        self.intercept_time.fill_(1.0)
        self.landing_y.zero_()

    def _fk_all(self, robot_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        robot = self.robots[robot_index]
        q_all = robot.data.joint_pos
        positions, quaternions, jacobians = [], [], []
        base = self._base_positions[robot_index].to(self.device)
        quat = self._base_quaternions[robot_index].to(self.device)
        base = base.expand(self.n, 3) + self.origins
        quat = quat.expand(self.n, 4)
        # USD has identical chain origins; the left chain is the +.532 local
        # counterpart of the canonical ARM_MOUNT=(0,-.266,.001898154).
        for arm, ids in enumerate(self.joints_by_arm[robot_index]):
            q = q_all[:, ids]
            arm_base = base
            if arm == 0:
                arm_base = arm_base + qrot(quat, self._arm_mount_delta.expand(self.n, 3))
            p, r, j = paddle_kinematics(q, base_pos=arm_base, base_quat=quat, paddle_offset=self.offset)
            positions.append(p - self.origins)
            quaternions.append(r)
            jacobians.append(j)
        return torch.stack(positions, 1), torch.stack(quaternions, 1), torch.stack(jacobians, 1)

    def _all_state(self):
        states = [self._fk_all(i) for i in range(2)]
        return (
            torch.stack([s[0] for s in states], 1),
            torch.stack([s[1] for s in states], 1),
            torch.stack([s[2] for s in states], 1),
        )

    def paddle_state(self):
        positions, quaternions, _ = self._all_state()
        ix3 = self.env.active_arms[:, :, None, None].expand(self.n, 2, 1, 3)
        ix4 = self.env.active_arms[:, :, None, None].expand(self.n, 2, 1, 4)
        return positions.gather(2, ix3).squeeze(2), quaternions.gather(2, ix4).squeeze(2)

    def proprioception(self):
        values = []
        for ri, robot in enumerate(self.robots):
            arms = [robot.data.joint_pos[:, ids] for ids in self.joints_by_arm[ri]]
            values.append(torch.stack(arms, 1).gather(1, self.env.active_arms[:, ri, None, None].expand(self.n, 1, 6)).squeeze(1))
        return torch.stack(values, 1)

    def ik(self, target_pos, target_quat):
        positions, quaternions, jacobians = self._all_state()
        active = self.env.active_arms
        current_p = positions.gather(2, active[:, :, None, None].expand(self.n, 2, 1, 3)).squeeze(2)
        current_q = quaternions.gather(2, active[:, :, None, None].expand(self.n, 2, 1, 4)).squeeze(2)
        result = []
        cfg = self.cfg
        for side, robot in enumerate(self.robots):
            q_all = torch.stack([robot.data.joint_pos[:, ids] for ids in self.joints_by_arm[side]], 1)
            j_all = jacobians[:, side]
            limits_all = []
            for ids in self.joints_by_arm[side]:
                limits_all.append(robot.data.soft_joint_pos_limits[:, ids])
            limits_all = torch.stack(limits_all, 1)
            idx6 = active[:, side, None, None].expand(self.n, 1, 6)
            q = q_all.gather(1, idx6).squeeze(1)
            jac = j_all.gather(1, active[:, side, None, None, None].expand(self.n, 1, 6, 6)).squeeze(1)
            limits = limits_all.gather(1, active[:, side, None, None, None].expand(self.n, 1, 6, 2)).squeeze(1)
            dp = target_pos[:, side] - current_p[:, side]
            dp *= torch.clamp(cfg.max_dpos / dp.norm(dim=-1, keepdim=True).clamp_min(1e-7), max=1)
            axis = torch.zeros((self.n, 3), device=self.device)
            axis[:, 0 if self.normal_axis == "x" else 2] = 1
            current_n = qrot(current_q[:, side], axis)
            goal_n = qrot(target_quat[:, side], axis)
            dr = torch.cross(current_n, goal_n, dim=-1)
            dr *= torch.clamp(cfg.max_drot / dr.norm(dim=-1, keepdim=True).clamp_min(1e-7), max=1)
            error = torch.cat([dp, dr], -1)
            eye = torch.eye(6, device=self.device).expand(self.n, 6, 6)
            dq = (jac.transpose(1, 2) @ torch.linalg.solve(jac @ jac.transpose(1, 2) + cfg.ik_damping**2 * eye, error[:, :, None])).squeeze(-1)
            dq = dq.clamp(-cfg.max_dq, cfg.max_dq)
            result.append(torch.clamp(q + dq, min=limits[:, :, 0] + .015, max=limits[:, :, 1] - .015))
        self.last_targets = torch.stack(result, 1)
        return self.last_targets
