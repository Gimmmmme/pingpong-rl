#!/usr/bin/env python3
"""Headless Isaac Lab reset/step smoke test for the standalone scene."""
from __future__ import annotations
import argparse
import json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--cameras", action="store_true")
    args = p.parse_args()
    from pingpong_rl.runtime import make_app
    app = make_app(args.device, cameras=args.cameras)
    env = None
    try:
        import torch
        from pingpong_rl.environment import PingPongEnv
        env = PingPongEnv(args.num_envs, args.device, cameras=args.cameras)
        active = torch.tensor([[1, 0]] * args.num_envs, device=args.device, dtype=torch.long)
        env.set_active_arms(active)
        env.reset()
        target = torch.zeros((args.num_envs, 2, 6), device=args.device)
        env.step(target, render=args.cameras, substeps=2)
        result = {"status": "passed", "num_envs": args.num_envs, "cameras": args.cameras,
                  "ball_shape": list(env.privileged_state()["ball_pos"].shape)}
        if args.cameras:
            frames = env.camera_rgb()
            calibration = env.camera_calibration()
            result.update({"camera_names": sorted(frames), "camera_shapes": {k: list(v.shape) for k, v in frames.items()},
                           "calibration_names": sorted(calibration)})
        print(json.dumps(result))
    finally:
        if env is not None:
            env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
