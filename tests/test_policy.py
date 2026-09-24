"""Checkpoint, observation contract, and learning checks on the CPU."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pingpong_rl.control import BallisticResidualController, ControlConfig
from pingpong_rl.policy import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    PRIVILEGED_OBS_DIM,
    AsymmetricPPO,
    actor_observation,
    privileged_observation,
)

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


@pytest.mark.parametrize(
    ("name", "updates", "transitions"),
    [("policy.pt", 35, 10330), ("initial_checkpoint.pt", 0, 0)],
)
def test_checkpoint_preserves_actor_output_on_roundtrip(name, updates, transitions, tmp_path):
    policy = AsymmetricPPO.load(CHECKPOINT_DIR / name)
    assert (policy.actor_dim, policy.critic_dim, policy.action_dim) == (39, 39, 6)
    assert (policy.updates, policy.transitions) == (updates, transitions)
    observation = np.random.default_rng(31).normal(size=(12, 39)).astype(np.float32)
    expected = policy.deterministic(observation)
    assert expected.shape == (12, 6)
    assert np.isfinite(expected).all()
    assert np.max(np.abs(expected)) <= 1.0

    output = tmp_path / "policy.pt"
    policy.save(output)
    restored = AsymmetricPPO.load(output)
    np.testing.assert_array_equal(restored.deterministic(observation), expected)
    for original, recovered in zip(policy.parameters(), restored.parameters()):
        assert torch.equal(original, recovered)


def test_actor_inference_never_invokes_critic(monkeypatch):
    policy = AsymmetricPPO.load(CHECKPOINT_DIR / "policy.pt")

    def reject_critic(*args, **kwargs):
        raise AssertionError("Actor inference requested the critic")

    monkeypatch.setattr(policy.critic, "forward", reject_critic)
    action = policy.deterministic(np.zeros((2, ACTOR_OBS_DIM), dtype=np.float32))
    assert action.shape == (2, ACTION_DIM)
    with pytest.raises(ValueError):
        policy.deterministic(np.zeros((2, 36)))
    with pytest.raises(ValueError, match="non-finite"):
        policy.deterministic(np.full((2, ACTOR_OBS_DIM), np.nan))


def test_ppo_update_changes_parameters_and_can_reload(tmp_path):
    policy = AsymmetricPPO(seed=9, hidden=24, epochs=2, minibatch=16)
    assert policy.critic_dim == CRITIC_OBS_DIM == 39
    rng = np.random.default_rng(9)
    steps, environments = 5, 8
    observation = rng.normal(size=(steps, environments, ACTOR_OBS_DIM)).astype(np.float32)
    action, logprob, value = policy.act_with_value(observation.reshape(-1, ACTOR_OBS_DIM))
    action = action.reshape(steps, environments, ACTION_DIM)
    initial_actor = [p.detach().clone() for p in policy.actor.parameters()]
    initial_critic = [p.detach().clone() for p in policy.critic.parameters()]
    done = np.zeros((steps, environments), dtype=np.float32)
    done[-1] = 1.0
    rollout = {
        "obs": observation,
        "critic_obs": observation.copy(),
        "action": action,
        "logprob": logprob.reshape(steps, environments),
        "value": value.reshape(steps, environments),
        "reward": (action[..., 0] + 0.1 * observation[..., 0]).astype(np.float32),
        "done": done,
        "last_critic_obs": observation[-1].copy(),
    }
    metrics = policy.update(rollout)
    assert (policy.updates, policy.transitions) == (1, steps * environments)
    assert np.isfinite(list(metrics.values())).all()
    assert any(not torch.equal(before, after) for before, after in zip(initial_actor, policy.actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(initial_critic, policy.critic.parameters()))
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())

    output = tmp_path / "updated.pt"
    policy.save(output)
    restored = AsymmetricPPO.load(output)
    np.testing.assert_array_equal(restored.deterministic(observation[0]), policy.deterministic(observation[0]))


def test_observation_helper_matches_both_player_feature_frames():
    """Compare the public helper to the controller with encoder-only fixtures."""
    controller = BallisticResidualController.__new__(BallisticResidualController)
    controller.device = torch.device("cpu")
    controller.n = 1
    controller.cfg = ControlConfig()
    controller.env = SimpleNamespace(active_arms=torch.tensor([[0, 1]]))
    controller.joints_by_arm = [[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]] * 2
    velocities = [torch.linspace(-2.0, 3.5, 12)[None], torch.linspace(1.0, 6.5, 12)[None]]
    controller.robots = [SimpleNamespace(data=SimpleNamespace(joint_vel=v)) for v in velocities]
    joints = torch.linspace(-0.9, 0.9, 12).reshape(1, 2, 6)
    paddles = torch.tensor([[[-0.5, 0.12, 0.94], [0.5, -0.1, 0.99]]])
    controller.paddle_state = lambda: (paddles, None)
    controller.proprioception = lambda: joints
    controller.target_pos = torch.tensor([[[-0.48, 0.03, 0.91], [0.48, -0.04, 0.92]]])
    controller.intercept_time = torch.tensor([[0.2, 0.4]])
    controller.last_action = torch.linspace(-0.6, 0.5, 12).reshape(1, 2, 6)
    controller.landing_y = torch.tensor([[0.15, -0.2]])
    ball = torch.tensor([[0.1, 0.02, 1.04]])
    velocity = torch.tensor([[2.1, -0.3, -1.2]])
    confidence = torch.tensor([0.8])
    observed = controller.observation(ball, velocity, confidence).numpy()
    assert observed.shape == (1, 2, ACTOR_OBS_DIM)

    for side, flip in enumerate((1.0, -1.0)):
        axes = np.array([flip, flip, 1.0], dtype=np.float32)

        def table_frame(position):
            out = position.numpy().copy()
            out[2] -= controller.cfg.table_z
            return out * axes

        previous = controller.last_action[0, side].numpy().copy()
        previous[[0, 1, 4]] *= flip
        hand = int(controller.env.active_arms[0, side])
        selected_velocity = velocities[side][0, controller.joints_by_arm[side][hand]].numpy()
        expected = actor_observation(
            table_frame(ball[0]), velocity[0].numpy() * axes,
            table_frame(paddles[0, side]), table_frame(paddles[0, 1 - side]),
            joints[0, side].numpy(), selected_velocity,
            table_frame(controller.target_pos[0, side]),
            float(controller.intercept_time[0, side]), float(confidence[0]),
            float(velocity[0, 0] * -flip > 0.08), previous,
            landing_y=float(controller.landing_y[0, side]) * flip, active_arm=hand,
        )
        np.testing.assert_allclose(observed[0, side], expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(expected[31:33], [1 - hand, hand])


def test_optional_privileged_critic_has_separate_dimension():
    actor = np.arange(ACTOR_OBS_DIM, dtype=np.float32) / ACTOR_OBS_DIM
    privileged = privileged_observation(actor, [1, 2, 3], [4, 5, 6], [7, 8, 9])
    assert privileged.shape == (PRIVILEGED_OBS_DIM,) == (48,)
    np.testing.assert_array_equal(privileged[:ACTOR_OBS_DIM], actor)
    policy = AsymmetricPPO(critic_dim=PRIVILEGED_OBS_DIM, hidden=16)
    action, logprob, value = policy.act_with_value(actor[None], privileged[None])
    assert action.shape == (1, ACTION_DIM)
    assert np.isfinite(logprob).all() and np.isfinite(value).all()
    with pytest.raises(ValueError):
        privileged_observation(actor[:-1], [1, 2, 3], [4, 5, 6], [7, 8, 9])
