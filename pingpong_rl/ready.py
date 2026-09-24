"""Drive all four arms to repeatable ready postures before each rally."""

from __future__ import annotations

import torch

from .control import tensor

# Joint-angle seeds select the reachable IK branch. The servo then calibrates
# each ready pose against the loaded robot; these are initialization guesses,
# not replayed motion or a live-rally state override.
_SEEDS = {
    (0.29, 0): (-1.297442, 2.881076, -2.712191, -0.691525, 1.270389, 1.640898),
    (0.29, 1): (0.0, 0.791755, -0.281850, -0.668561, 0.0, 0.458614),
    (0.0, 0): (-1.056255, 1.662377, -0.764682, -1.212030, 1.034458, -1.449326),
    (0.0, 1): (1.056255, 1.662377, -0.764682, -1.212030, -1.034458, -0.113180),
}


def _clear_filter(controller):
    controller.last_paddle_pos = None
    base = getattr(controller, "base", None)
    if base is not None:
        base.last_paddle_pos = None


def prepare_ready_homes(env, controller, callback=None, steps: int = 450):
    """Calibrate each hand using joint drives; return parked and ready joints."""
    if steps < 1:
        raise ValueError("ready calibration needs at least one step")
    parked = [tensor(robot.data.default_joint_pos).clone() for robot in env.robots]
    for robot_index in range(2):
        for arm, joint_ids in enumerate(env.joint_ids_by_arm[robot_index]):
            parked[robot_index][:, joint_ids[0]] = 0.9 if arm == 0 else -0.9

    ready = []
    confidence = torch.zeros(env.num_envs, device=env.device)
    action = torch.zeros(env.num_envs, 2, 6, device=env.device)
    ball = torch.tensor([[0.0, 0.0, 0.95]], device=env.device).expand(env.num_envs, 3)
    for arm in range(2):
        env.home = [joint.clone() for joint in parked]
        env.set_active_arms(
            torch.full((env.num_envs, 2), arm, device=env.device, dtype=torch.long)
        )
        if callback is not None:
            callback()
        for robot_index, robot in enumerate(env.robots):
            mount_y = round(abs(float(robot.cfg.init_state.pos[1])), 2)
            # Use the closest known branch seed, then solve the actual target.
            seed_y = min((0.0, 0.29), key=lambda y: abs(y - mount_y))
            joint_ids = env.joint_ids_by_arm[robot_index][arm]
            env.home[robot_index][:, joint_ids] = torch.tensor(
                _SEEDS[seed_y, arm], device=env.device
            )
        env.reset()
        _clear_filter(controller)
        for step in range(steps):
            if hasattr(controller, "sync"):
                controller.sync(env)
            targets = controller.step(ball, torch.zeros_like(ball), action, confidence)
            env.step(targets, render=env.cameras and step % 4 == 3)
        ready.append([tensor(robot.data.joint_pos).clone() for robot in env.robots])
    return parked, ready


def reset_ready(env, calibration, active, callback=None):
    """Reset each arm to its calibrated active pose or parked inactive pose."""
    parked, ready = calibration
    env.set_active_arms(active)
    if callback is not None:
        callback()
    env.home = [joint.clone() for joint in parked]
    for robot in range(2):
        for arm, joint_ids in enumerate(env.joint_ids_by_arm[robot]):
            selected = env.active_arms[:, robot] == arm
            env.home[robot][:, joint_ids] = torch.where(
                selected[:, None],
                ready[arm][robot][:, joint_ids],
                parked[robot][:, joint_ids],
            )
    env.reset()
