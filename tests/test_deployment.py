"""Encoder deployment tests with a robot facade that forbids body-state reads."""
from __future__ import annotations

import torch
import pytest

from pingpong_rl.deployment import EncoderResidualController, RobotProprioception


JOINT_NAMES = tuple(
    [f"left_arm_joint{i}" for i in range(1, 7)]
    + ["left_arm_gripper"]
    + [f"right_arm_joint{i}" for i in range(1, 7)]
    + ["right_arm_gripper"]
)


class _EncoderData:
    def __init__(self):
        self.joint_pos = torch.zeros(1, 14)
        self.joint_vel = torch.zeros(1, 14)
        self.soft_joint_pos_limits = torch.tensor(
            [[-3.14, 3.14]] * 14, dtype=torch.float32
        ).unsqueeze(0)

    def __getattr__(self, name):
        if name in {"body_pos_w", "body_quat_w", "body_link_jacobian_w", "root_pos_w"}:
            raise AssertionError(f"encoder controller read forbidden simulator field: {name}")
        raise AttributeError(name)


class _Robot:
    joint_names = JOINT_NAMES
    body_names = ()

    def __init__(self):
        self.data = _EncoderData()


class _FakeEnv:
    num_envs = 1
    device = "cpu"
    origins = torch.zeros(1, 3)
    scene_parameters = {"base_distance": 0.95, "base_y": 0.29, "base_height": 0.62}

    def __init__(self):
        self.active_arms = torch.tensor([[1, 1]], dtype=torch.long)
        self.robots = [_Robot(), _Robot()]


def test_facade_contains_only_encoder_arrays():
    env = _FakeEnv()
    facade = RobotProprioception(env)
    assert facade._source is None
    assert not hasattr(facade.robots[0].data, "body_pos_w")
    assert tuple(facade.robots[0].data.joint_pos.shape) == (1, 14)
    env.robots[0].data.joint_pos[:, 0] = 0.2
    facade.sync(env)
    assert abs(float(facade.robots[0].data.joint_pos[0, 0]) - 0.2) < 1e-6
    assert facade._source is None


def test_fk_observation_and_ik_never_read_body_state():
    env = _FakeEnv()
    controller = EncoderResidualController(env)
    p, q = controller.paddle_state()
    # With the inward right arm on each rotated base, the FK is symmetric.
    torch.testing.assert_close(p, torch.tensor([[[-0.5965, 0.024, 0.802898154], [0.5965, -0.024, 0.802898154]]]), atol=1e-5, rtol=0)
    obs = controller.observation(torch.tensor([[0.0, 0.0, 0.95]]), torch.zeros(1, 3))
    assert obs.shape == (1, 2, 39)
    targets = controller.step(
        torch.tensor([[0.0, 0.0, 0.95]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.zeros(1, 2, 6),
    )
    assert targets.shape == (1, 2, 6)
    assert torch.isfinite(targets).all()


def test_active_arm_switch_and_reset_clear_temporal_state():
    env = _FakeEnv()
    controller = EncoderResidualController(env)
    controller.last_action.fill_(1.0)
    controller.target_pos.fill_(1.0)
    controller.intercept_time.zero_()
    controller.reset()
    assert torch.count_nonzero(controller.last_action) == 0
    assert torch.count_nonzero(controller.target_pos) == 0
    assert torch.all(controller.intercept_time == 1)

    env.active_arms = torch.tensor([[0, 0]])
    controller.sync(env)
    p, _ = controller.paddle_state()
    # Left-chain mount uses canonical ARM_MOUNT plus local +.532 m.
    torch.testing.assert_close(p[0, 0, 1], torch.tensor(0.556), atol=1e-5, rtol=0)
    torch.testing.assert_close(p[0, 1, 1], torch.tensor(-0.556), atol=1e-5, rtol=0)


def test_sync_refreshes_fk_observation_and_ik_from_new_encoders():
    env = _FakeEnv()
    controller = EncoderResidualController(env)
    old_position, _ = controller.paddle_state()
    snapshot_list = controller.robots
    right_ids = list(range(7, 13))
    new_angles = torch.tensor([[0.2, 0.1, -0.3, 0.15, -0.1, 0.2]])
    new_velocity = torch.tensor([[0.1, -0.4, 0.2, 0.3, 0.5, -0.2]])
    env.robots[0].data.joint_pos[:, right_ids] = new_angles
    env.robots[0].data.joint_vel[:, right_ids] = new_velocity
    # Source tensors are copied, so an unsynchronized write is invisible.
    torch.testing.assert_close(controller.paddle_state()[0], old_position)

    controller.sync(env)
    assert controller.robots is snapshot_list is controller.proprio.robots
    current_position, current_orientation = controller.paddle_state()
    assert not torch.allclose(current_position[:, 0], old_position[:, 0])
    torch.testing.assert_close(current_position[:, 1], old_position[:, 1])
    torch.testing.assert_close(controller.proprioception()[:, 0], new_angles)
    observation = controller.observation(torch.tensor([[0.0, 0.0, 0.95]]), torch.zeros(1, 3))
    torch.testing.assert_close(observation[:, 0, 12:18], new_angles / torch.pi)
    torch.testing.assert_close(observation[:, 0, 18:24], new_velocity / 6)
    own_position = current_position[:, 0].clone()
    own_position[:, 2] -= controller.cfg.table_z
    torch.testing.assert_close(observation[:, 0, 6:9], own_position)
    # An IK goal at the measured pose must retain the refreshed encoders.
    targets = controller.ik(current_position, current_orientation)
    torch.testing.assert_close(targets[:, 0], new_angles, atol=1e-6, rtol=0)
    controller.reset()
    torch.testing.assert_close(controller.last_targets[:, 0], new_angles)


def test_failed_sync_does_not_keep_source_or_publish_partial_snapshot():
    env = _FakeEnv()
    facade = RobotProprioception(env)
    old_robots = list(facade.robots)
    env.robots[1].joint_names = tuple(reversed(JOINT_NAMES))
    with pytest.raises(ValueError, match="joint names and ordering"):
        facade.sync(env)
    assert facade._source is None
    assert all(actual is old for actual, old in zip(facade.robots, old_robots))


def test_batched_encoders_and_per_environment_arm_selection():
    env = _FakeEnv()
    env.num_envs = 2
    env.origins = torch.tensor([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    env.active_arms = torch.tensor([[0, 1], [1, 0]])
    for robot in env.robots:
        robot.data.joint_pos = robot.data.joint_pos.repeat(2, 1)
        robot.data.joint_vel = robot.data.joint_vel.repeat(2, 1)
        # Static limits may be unbatched on a physical robot adapter.
        robot.data.soft_joint_pos_limits = robot.data.soft_joint_pos_limits[0]
    controller = EncoderResidualController(env)
    env.robots[0].data.joint_pos[1, 7] = 0.25
    env.robots[1].data.joint_pos[1, 0] = -0.3
    controller.sync(env)
    positions, orientations = controller.paddle_state()
    assert positions.shape == (2, 2, 3)
    targets = controller.ik(positions, orientations)
    torch.testing.assert_close(targets, controller.proprioception(), atol=1e-6, rtol=0)
    assert positions[1].abs().max() < 2  # Replica origin cancels in table coordinates.
    torch.testing.assert_close(targets[1, :, 0], torch.tensor([0.25, -0.3]), atol=1e-6, rtol=0)
