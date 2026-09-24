"""Standalone Isaac Lab ping-pong environment.

The public API intentionally mirrors the previous policy adapter:
``reset``, ``step``, ``serve``, active-arm selection, RGB camera reads,
proprioception and evaluation-only privileged state.  The policy action never
reads simulator ball state; the latter is exposed only by ``privileged_state``
for training/evaluation diagnostics.
"""
from __future__ import annotations

from typing import Any

import torch

try:
    from .scene import (
        ARM_NAMES,
        BALL_MASS,
        BALL_RADIUS,
        CAMERA_EYES,
        CAMERA_NAMES,
        CAMERA_TARGET,
        DT,
        PADDLE_OFFSET,
        PADDLE_RADIUS,
        TABLE_SIZE,
        TABLE_TOP,
        NET_HEIGHT,
        make_scene_cfg,
    )
except ImportError:
    from scene import (
        ARM_NAMES,
        BALL_MASS,
        BALL_RADIUS,
        CAMERA_EYES,
        CAMERA_NAMES,
        CAMERA_TARGET,
        DT,
        PADDLE_OFFSET,
        PADDLE_RADIUS,
        TABLE_SIZE,
        TABLE_TOP,
        NET_HEIGHT,
        make_scene_cfg,
    )


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


# Compatibility name used by the policy modules.
tensor = _tensor


class PingPongEnv:
    """Two fixed-base bimanual robots, a netted table and a physical ball."""

    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cuda:0",
        cameras: bool = False,
        dt: float = DT,
        base_distance: float = 0.95,
        base_y: float = 0.29,
        base_height: float = 0.62,
    ):
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab_physx.sim.schemas import (
            PhysxCollisionPropertiesCfg,
        )
        from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
        import isaaclab.sim as sim
        import omni.usd
        try:
            from .paddle import attach_v2_all_links
        except ImportError:
            from paddle import attach_v2_all_links

        self.num_envs = int(num_envs)
        self.device = device
        self.dt = float(dt)
        self.step_dt = self.dt
        self.common_step_counter = 0
        self.cameras = bool(cameras)
        self.active_arms = torch.zeros((self.num_envs, 2), device=device, dtype=torch.long)
        self.scene_parameters = {
            "table_size": TABLE_SIZE,
            "table_top": TABLE_TOP,
            "net_height": NET_HEIGHT,
            "base_distance": base_distance,
            "base_y": base_y,
            "base_height": base_height,
            "ball_radius": BALL_RADIUS,
            "ball_mass": BALL_MASS,
            "paddle_offset": PADDLE_OFFSET,
            "paddle_radius": PADDLE_RADIUS,
            "paddle_thickness": 0.010,
            "physics_dt": self.dt,
            "static_friction": 0.01,
            "dynamic_friction": 0.01,
            "restitution": 0.88,
            "gravity": [0.0, 0.0, -9.81],
            "linear_damping": 0.015,
            "angular_damping": 0.01,
            "air_model": "linear damping only; no aerodynamic drag or Magnus force",
            "ball_write_contract": "reset/serve initialization only; step writes joint targets",
        }

        physics = PhysxCfg(
            min_position_iteration_count=16,
            min_velocity_iteration_count=4,
            bounce_threshold_velocity=0.05,
        )
        self.sim = SimulationContext(
            SimulationCfg(device=device, dt=self.dt, render_interval=4, physics=physics)
        )
        self.scene = __import__("isaaclab.scene", fromlist=["InteractiveScene"]).InteractiveScene(
            make_scene_cfg(
                self.num_envs,
                base_distance=base_distance,
                base_y=base_y,
                base_height=base_height,
                cameras=self.cameras,
            )
        )
        self.robots = [self.scene["robot_left"], self.scene["robot_right"]]
        self.ball = self.scene["ball"]
        self.origins = _tensor(self.scene.env_origins)

        collision = PhysxCollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)
        material = PhysxRigidBodyMaterialCfg(
            static_friction=0.01,
            dynamic_friction=0.01,
            restitution=0.88,
            restitution_combine_mode="max",
            friction_combine_mode="min",
        )
        stage = omni.usd.get_context().get_stage()
        self.paddle_paths = attach_v2_all_links(
            stage,
            sim=sim,
            collision=collision,
            material=material,
            num_envs=self.num_envs,
            roots=("RobotLeft", "RobotRight"),
            arms=ARM_NAMES,
        )
        self.paddle_metadata = {
            "model": "elliptical_table_tennis_paddle_v2",
            "blade_width_m": 0.160,
            "blade_height_m": 0.170,
            "blade_thickness_m": 0.010,
            "handle_length_m": 0.120,
            "handle_width_m": 0.030,
            "handle_thickness_m": 0.018,
            "paddle_offset_m": list(PADDLE_OFFSET),
            "all_robot_arms": 4,
            "collision_primitive": "convex hull ellipse mesh",
            "visual_skins": ["red_front", "black_back", "wood_edge"],
        }

        self.sim.reset()
        self.scene.reset()
        self.camera_poses = dict(CAMERA_EYES)
        if self.cameras:
            self._configure_cameras()

        self.joint_ids_by_arm: list[list[list[int]]] = []
        self.body_ids_by_arm: list[list[int]] = []
        for robot in self.robots:
            self.joint_ids_by_arm.append(
                [
                    robot.find_joints(
                        [f"{arm}_arm_joint{i}" for i in range(1, 7)],
                        preserve_order=True,
                    )[0]
                    for arm in ARM_NAMES
                ]
            )
            self.body_ids_by_arm.append(
                [robot.find_bodies(f"{arm}_arm_link6")[0][0] for arm in ARM_NAMES]
            )
        # Backward-compatible aliases select the right arm on each robot.
        self.joint_ids = [arms[1] for arms in self.joint_ids_by_arm]
        self.body_ids = [arms[1] for arms in self.body_ids_by_arm]
        self.home = [_tensor(robot.data.default_joint_pos).clone() for robot in self.robots]
        self.targets = self._home_targets()
        self.reset()

    def _home_targets(self) -> torch.Tensor:
        return torch.stack(
            [
                torch.stack([home[:, ids] for ids in arm_ids], dim=1)
                for home, arm_ids in zip(self.home, self.joint_ids_by_arm)
            ],
            dim=1,
        )

    def _configure_cameras(self) -> None:
        eyes = torch.tensor(
            [CAMERA_EYES[name] for name in CAMERA_NAMES],
            device=self.device,
            dtype=torch.float32,
        )
        targets = torch.tensor(CAMERA_TARGET, device=self.device, dtype=torch.float32).expand(3, 3)
        for index, name in enumerate(CAMERA_NAMES):
            eye = eyes[index].expand(self.num_envs, 3) + self.origins
            target = targets[index].expand(self.num_envs, 3) + self.origins
            self.scene[name].set_world_poses_from_view(eye, target)
        for _ in range(4):
            self.sim.render()

    def set_active_arms(self, active: Any) -> None:
        values = torch.as_tensor(active, device=self.device, dtype=torch.long)
        if tuple(values.shape) != (self.num_envs, 2) or bool(((values < 0) | (values > 1)).any()):
            raise ValueError(f"active arms must have shape [{self.num_envs}, 2] with values 0/1")
        self.active_arms = values

    def all_paddle_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        from isaaclab.utils.math import quat_apply

        positions, normals = [], []
        offset = torch.tensor(PADDLE_OFFSET, device=self.device, dtype=torch.float32)
        axis = torch.tensor((1.0, 0.0, 0.0), device=self.device, dtype=torch.float32)
        for robot, body_ids in zip(self.robots, self.body_ids_by_arm):
            robot_positions, robot_normals = [], []
            for body_id in body_ids:
                quat = _tensor(robot.data.body_quat_w)[:, body_id]
                robot_positions.append(
                    _tensor(robot.data.body_pos_w)[:, body_id]
                    + quat_apply(quat, offset.expand(self.num_envs, 3))
                    - self.origins
                )
                robot_normals.append(quat_apply(quat, axis.expand(self.num_envs, 3)))
            positions.append(torch.stack(robot_positions, dim=1))
            normals.append(torch.stack(robot_normals, dim=1))
        return torch.stack(positions, dim=1), torch.stack(normals, dim=1)

    def active_paddle_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions, normals = self.all_paddle_state()
        gather_index = self.active_arms[:, :, None, None].expand(self.num_envs, 2, 1, 3)
        return (
            positions.gather(2, gather_index).squeeze(2),
            normals.gather(2, gather_index).squeeze(2),
        )

    def paddle_state(self):
        return self.active_paddle_state()

    def reset(self, env_ids: Any = None):
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        for robot, home in zip(self.robots, self.home):
            position = home[ids]
            robot.write_joint_state_to_sim(position, torch.zeros_like(position), env_ids=ids)
            robot.set_joint_position_target(position, env_ids=ids)
        self.targets = self._home_targets()
        self.serve((-.35, 0.0, 0.96), (1.7, 0.0, 1.15), env_ids=ids)
        self.scene.write_data_to_sim()
        self.scene.update(self.dt)
        return None

    def serve(self, position, velocity, env_ids=None, angular_velocity=None):
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        p = torch.as_tensor(position, device=self.device, dtype=torch.float32).expand(len(ids), 3)
        p = p + self.origins[ids]
        v = torch.as_tensor(velocity, device=self.device, dtype=torch.float32).expand(len(ids), 3)
        quat = torch.tensor((0.0, 0.0, 0.0, 1.0), device=self.device).expand(len(ids), 4)
        self.ball.write_root_pose_to_sim(torch.cat((p, quat), dim=-1), env_ids=ids)
        if angular_velocity is None:
            angular_velocity = torch.zeros_like(v)
        w = torch.as_tensor(angular_velocity, device=self.device, dtype=torch.float32).expand(len(ids), 3)
        self.ball.write_root_velocity_to_sim(torch.cat((v, w), dim=-1), env_ids=ids)
        self.ball.reset(env_ids=ids)

    def step(self, joint_targets=None, *, render: bool = False, substeps: int = 1):
        if joint_targets is not None:
            targets = torch.as_tensor(joint_targets, device=self.device, dtype=torch.float32)
            if tuple(targets.shape) == (self.num_envs, 2, 6):
                full = self.targets.clone()
                env_index = torch.arange(self.num_envs, device=self.device)
                for robot_index in range(2):
                    full[env_index, robot_index, self.active_arms[:, robot_index]] = targets[:, robot_index]
                self.targets = full
            elif tuple(targets.shape) == (self.num_envs, 2, 2, 6):
                self.targets = targets
            else:
                raise ValueError("targets must have shape [N,2,6] or [N,2,2,6]")
        for robot_index, (robot, home, arm_ids) in enumerate(
            zip(self.robots, self.home, self.joint_ids_by_arm)
        ):
            full = home.clone()
            for arm_index, joint_ids in enumerate(arm_ids):
                full[:, joint_ids] = self.targets[:, robot_index, arm_index]
            robot.set_joint_position_target(full)
        for _ in range(int(substeps)):
            self.scene.write_data_to_sim()
            self.sim.step(render=render)
            self.scene.update(self.dt)
            self.common_step_counter += 1
        return None

    def camera_rgb(self):
        if not self.cameras:
            raise RuntimeError("RGB cameras were not enabled")
        return {
            name: _tensor(self.scene[name].data.output["rgb"])[..., :3]
            for name in CAMERA_NAMES
        }

    def camera_calibration(self) -> dict[str, dict]:
        """Return fixed camera intrinsics/extrinsics for the vision module."""
        import numpy as np
        from isaaclab.utils.math import matrix_from_quat

        result = {}
        for name in CAMERA_NAMES:
            camera = self.scene[name]
            K = _tensor(camera.data.intrinsic_matrices)[0].detach().cpu().numpy()
            position = _tensor(camera.data.pos_w)[0].detach().cpu().numpy()
            quat = _tensor(camera.data.quat_w_ros)[0].detach().cpu().numpy()
            R_wc = matrix_from_quat(_tensor(camera.data.quat_w_ros))[0].detach().cpu().numpy()
            result[name] = {
                "K": np.asarray(K).tolist(),
                "position_world": np.asarray(position).tolist(),
                "quat_xyzw_ros": np.asarray(quat).tolist(),
                "R_wc": np.asarray(R_wc).tolist(),
                "width": 640,
                "height": 480,
                "calibration_source": "Isaac Lab camera sensor pose and intrinsics",
            }
        return result

    def proprio_state(self) -> dict[str, torch.Tensor]:
        paddle_pos, paddle_normal = self.paddle_state()
        return {
            "joint_pos": torch.stack(
                [_tensor(robot.data.joint_pos)[:, ids] for robot, ids in zip(self.robots, self.joint_ids)],
                dim=1,
            ),
            "paddle_pos": paddle_pos,
            "paddle_normal": paddle_normal,
            "active_arms": self.active_arms.clone(),
        }

    def evaluation_contacts(self):
        return _tensor(self.scene["ball_contacts"].data.net_forces_w).clone()

    def privileged_state(self) -> dict[str, torch.Tensor]:
        """Return simulator state for evaluator/training code only."""
        from isaaclab.utils.math import quat_apply

        paddle_pos, paddle_normal = self.paddle_state()
        all_pos, all_normals = self.all_paddle_state()
        tangents = []
        y_axis = torch.tensor((0.0, 1.0, 0.0), device=self.device).expand(self.num_envs, 3)
        for robot, body_ids in zip(self.robots, self.body_ids_by_arm):
            row = []
            for body_id in body_ids:
                quat = _tensor(robot.data.body_quat_w)[:, body_id]
                row.append(quat_apply(quat, y_axis))
            tangents.append(torch.stack(row, dim=1))
        return {
            "ball_pos": _tensor(self.ball.data.root_pos_w) - self.origins,
            "ball_vel": _tensor(self.ball.data.root_lin_vel_w),
            "joint_pos": torch.stack(
                [_tensor(robot.data.joint_pos)[:, ids] for robot, ids in zip(self.robots, self.joint_ids)],
                dim=1,
            ),
            "paddle_pos": paddle_pos,
            "paddle_normal": paddle_normal,
            "all_paddle_pos": all_pos,
            "all_paddle_normal": all_normals,
            "all_paddle_tangent_y": torch.stack(tangents, dim=1),
            "active_arms": self.active_arms.clone(),
        }

    def close(self):
        self.sim.stop()
        self.sim.clear_instance()
