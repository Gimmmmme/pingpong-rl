"""Evaluate the released actor checkpoint without simulator-only observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .policy import ACTOR_OBS_DIM, ACTION_DIM, AsymmetricPPO


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/policy.pt"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--sim", action="store_true", help="run the RGB camera simulation rollout")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.self_test:
        policy = AsymmetricPPO(seed=3, hidden=16)
        action = policy.deterministic(np.zeros((2, ACTOR_OBS_DIM), dtype=np.float32))
        assert action.shape == (2, ACTION_DIM)
        print(json.dumps({"self_test": "passed", "actor_obs_dim": ACTOR_OBS_DIM, "action_dim": ACTION_DIM}))
        return 0
    policy = AsymmetricPPO.load(args.checkpoint, device="cpu" if not args.sim else args.device)
    if (policy.actor_dim, policy.action_dim) != (ACTOR_OBS_DIM, ACTION_DIM):
        raise ValueError("checkpoint does not match the 39D actor contract")
    if not args.sim:
        actions = policy.deterministic(np.zeros((2, ACTOR_OBS_DIM), dtype=np.float32))
        print(json.dumps({"checkpoint": str(args.checkpoint), "actor_obs_dim": ACTOR_OBS_DIM,
                          "action_dim": ACTION_DIM, "deterministic_zero_action": actions.tolist()}))
        return 0

    if args.steps < 1:
        raise ValueError("steps must be positive")
    from .runtime import make_app
    app = make_app(args.device, cameras=True)
    env = None
    try:
        import torch
        from .control import BallisticResidualController
        from .environment import PingPongEnv
        from .vision import RGBBallObserver

        env = PingPongEnv(num_envs=1, device=args.device, cameras=True)
        env.set_active_arms([[1, 0]])
        env.reset()
        tracker = RGBBallObserver(env)
        controller = BallisticResidualController(env)
        target = torch.zeros((1, 2, 6), device=args.device)
        p = torch.tensor([[0.0, 0.0, 0.95]], device=args.device)
        v = torch.zeros_like(p)
        visible = 0
        for step in range(args.steps):
            render = step % 4 == 3
            env.step(target, render=render)
            if render:
                estimate = tracker.update(env.common_step_counter * env.dt)
                prediction = tracker.predict(env.common_step_counter * env.dt)
                if prediction is not None:
                    p = torch.as_tensor(prediction[0], dtype=torch.float32, device=args.device).reshape(1, 3)
                    v = torch.as_tensor(prediction[1], dtype=torch.float32, device=args.device).reshape(1, 3)
                    visible += 1
                obs = controller.observation(p, v)
                actions = policy.deterministic(obs[:, 0].detach().cpu().numpy())
                # A shared actor is evaluated for both player frames; control
                # receives only the actor output and RGB-derived state.
                actions_both = np.stack([actions, policy.deterministic(obs[:, 1].detach().cpu().numpy())], axis=1)
                target = controller.step(p, v, torch.as_tensor(actions_both, device=args.device))
        print(json.dumps({"checkpoint": str(args.checkpoint), "steps": args.steps,
                          "rgb_updates": visible, "actor_obs_dim": ACTOR_OBS_DIM,
                          "privileged_state_used": False}))
    finally:
        if env is not None:
            env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
