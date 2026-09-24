"""Encoder kinematics checks independent of any physics engine."""

import math

import pytest
import torch

from pingpong_rl.kinematics import paddle_kinematics, qmul


@pytest.mark.parametrize("yaw", [0.0, math.pi, 0.7])
def test_encoder_jacobian_matches_position_and_rotation_finite_differences(yaw):
    generator = torch.Generator().manual_seed(13)
    joints = torch.rand((16, 6), generator=generator, dtype=torch.float64) * 1.2 - 0.6
    base_quaternion = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    position, orientation, jacobian = paddle_kinematics(joints, base_quat=base_quaternion)
    assert torch.allclose(orientation.norm(dim=-1), torch.ones(16, dtype=torch.float64))
    epsilon = 1e-6
    for joint in range(6):
        plus, minus = joints.clone(), joints.clone()
        plus[:, joint] += epsilon
        minus[:, joint] -= epsilon
        p_plus, q_plus, _ = paddle_kinematics(plus, base_quat=base_quaternion)
        p_minus, q_minus, _ = paddle_kinematics(minus, base_quat=base_quaternion)
        linear = (p_plus - p_minus) / (2 * epsilon)
        q_minus_inverse = q_minus.clone()
        q_minus_inverse[:, :3] *= -1
        delta = qmul(q_plus, q_minus_inverse)
        angular = delta[:, :3] * torch.sign(delta[:, 3:4]) / epsilon
        torch.testing.assert_close(linear, jacobian[:, :3, joint], rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(angular, jacobian[:, 3:, joint], rtol=1e-7, atol=1e-8)
    assert torch.isfinite(position).all()


def test_mirrored_mount_has_mirrored_paddle_position():
    joints = torch.zeros((1, 6), dtype=torch.float64)
    left, _, _ = paddle_kinematics(joints, base_pos=(-0.95, 0.29, 0.62))
    right, _, _ = paddle_kinematics(
        joints, base_pos=(0.95, -0.29, 0.62), base_quat=(0.0, 0.0, 1.0, 0.0)
    )
    torch.testing.assert_close(left, torch.tensor([[-0.5965, 0.024, 0.802898154]], dtype=torch.float64))
    torch.testing.assert_close(right, left * torch.tensor([-1.0, -1.0, 1.0]))


def test_encoder_dimension_is_validated():
    with pytest.raises(ValueError, match="six-joint"):
        paddle_kinematics(torch.zeros(1, 5))
