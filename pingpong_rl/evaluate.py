"""Visible RGB and encoder deployment of the released table-tennis policy."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .policy import ACTOR_OBS_DIM, ACTION_DIM, AsymmetricPPO


class StrokeDecision:
    """Debounce RGB velocity reversals and choose one action per stroke."""

    def __init__(self, stable_frames=3):
        self.stable_frames = stable_frames
        self.reset()

    def reset(self):
        self.side = self.candidate = None
        self.count = 0

    def update(self, velocity_x: float) -> int | None:
        if abs(velocity_x) <= 0.15:
            self.candidate, self.count = None, 0
            return None
        side = int(velocity_x > 0)
        self.count = self.count + 1 if side == self.candidate else 1
        self.candidate = side
        if side != self.side and self.count >= self.stable_frames:
            self.side = side
            return side
        return None


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/policy.pt"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seconds", type=float, default=0,
                   help="total simulation seconds; 0 keeps Kit running until closed")
    p.add_argument("--episodes", type=int, default=0,
                   help="maximum rallies; 0 keeps serving after every drop")
    p.add_argument("--rally-seconds", type=float, default=30)
    p.add_argument("--headless", action="store_true", help="disable the Kit window")
    p.add_argument("--base-y", type=float, default=0.0)
    p.add_argument("--arm-pair", choices=["cycle", "00", "01", "10", "11"], default="00")
    p.add_argument("--ready-steps", type=int, default=450)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--diversity", type=float, default=0.4)
    p.add_argument("--no-resample-targets", action="store_true")
    p.add_argument("--output", type=Path, help="save a JSON session summary")
    p.add_argument("--video", type=Path, help="record the overview camera at 30 fps")
    p.add_argument("--audit", action="store_true",
                   help="score physical contacts after actions; never feed them to control")
    p.add_argument("--sim", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--steps", type=int, help=argparse.SUPPRESS)
    p.add_argument("--self-test", action="store_true")
    return p


def _serve(rng, device, diversity):
    """Generate the release validation's randomized serve command."""
    import torch

    direction = 1.0 if rng.integers(0, 2) else -1.0
    position = [-0.35 * direction, rng.uniform(-0.25, 0.25) * diversity,
                1.0 + (rng.uniform(0.91, 1.10) - 1.0) * diversity]
    velocity = [direction * (2.0 + (rng.uniform(1.55, 2.45) - 2.0) * diversity),
                rng.uniform(-0.55, 0.55) * diversity,
                0.2 + (rng.uniform(-0.35, 0.55) - 0.2) * diversity]
    spin = rng.uniform(-14.0, 14.0, (1, 3)) * diversity
    return tuple(torch.as_tensor(a, device=device, dtype=torch.float32).reshape(1, 3)
                 for a in (position, velocity, spin))


def _run(args):
    if args.seconds < 0 or args.episodes < 0 or args.ready_steps < 1 or args.rally_seconds <= 0:
        raise ValueError("invalid duration, episode count, or ready calibration steps")
    if not 0 <= args.diversity <= 1:
        raise ValueError("diversity must be in [0, 1]")
    if args.headless and not (args.seconds or args.episodes or args.steps):
        raise ValueError("headless runs need --seconds or --episodes")

    import torch
    from .runtime import make_app

    app = make_app(args.device, cameras=True, headless=args.headless)
    env = writer = None
    report = {"rallies": [], "actor_input": "stereo RGB and joint encoders",
              "seed": args.seed, "checkpoint": str(args.checkpoint)}
    try:
        from .deployment import EncoderResidualController
        from .environment import PingPongEnv
        from .ready import prepare_ready_homes, reset_ready
        from .vision import RGBBallObserver
        from .scoring import RallyScorer

        torch.set_num_threads(4)
        rng = np.random.default_rng(args.seed)
        policy = AsymmetricPPO.load(args.checkpoint, device=args.device)
        if (policy.actor_dim, policy.action_dim) != (ACTOR_OBS_DIM, ACTION_DIM):
            raise ValueError("checkpoint must use the 39D observation / 6D action contract")
        env = PingPongEnv(1, args.device, cameras=True, base_y=args.base_y)
        if not args.headless:
            env.sim.set_camera_view((2.1, -2.1, 1.9), (0.0, 0.0, 0.8))
        controller = EncoderResidualController(env)
        print("Calibrating ready poses for both hands...", flush=True)
        homes = prepare_ready_homes(env, controller, steps=args.ready_steps)
        observer = RGBBallObserver(env)
        scorer = RallyScorer(1, args.device) if args.audit else None
        decision = StrokeDecision()
        actions = torch.zeros((1, 2, ACTION_DIM), device=args.device)
        confidence = torch.zeros(1, device=args.device)
        max_steps = args.steps or (round(args.seconds / env.dt) if args.seconds else None)
        total_steps = 0
        start_wall = time.monotonic()

        if args.video:
            import cv2
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     30, (640, 480))
            if not writer.isOpened():
                raise RuntimeError(f"cannot open video output: {args.video}")

        print(json.dumps({"event": "ready", "kit_gui": not args.headless,
                          "checkpoint_updates": policy.updates,
                          "checkpoint_transitions": policy.transitions}), flush=True)
        rally = 0
        while app.is_running() and (not args.episodes or rally < args.episodes):
            if max_steps is not None and total_steps >= max_steps:
                break
            pair = [rally // 2 % 2, rally % 2] if args.arm_pair == "cycle" else list(map(int, args.arm_pair))
            reset_ready(env, homes, [pair])
            controller.sync(env).reset()
            observer.reset()
            decision.reset()
            actions.zero_()
            confidence.zero_()
            landing = torch.as_tensor(rng.uniform(-.26, .26, (1, 2)) * args.diversity,
                                      device=args.device, dtype=torch.float32)
            controller.set_landing_targets(landing)
            p, v, spin = _serve(rng, args.device, args.diversity)
            serve_p, serve_v = p.clone(), v.clone()
            env.serve(p, v, angular_velocity=spin)
            if scorer:
                scorer.reset(env.privileged_state())
            target = controller.proprioception().clone()
            measured = captures = decisions = 0
            hits = returns = 0
            reason = "time_limit"
            rally_steps = 0
            limit = max(4, round(args.rally_seconds / env.dt))
            for step in range(limit):
                if not app.is_running():
                    reason = "window_closed"
                    break
                if max_steps is not None and total_steps >= max_steps:
                    reason = "session_limit"
                    break
                now = env.common_step_counter * env.dt
                # Four physical steps after a serve flush stale camera frames.
                if step >= 4 and step % 4 == 0:
                    estimate = observer.update(now)
                    captures += 1
                    measured += int(estimate.visible)
                if step % 2 == 0:
                    controller.sync(env)
                    prediction = observer.predict(now)
                    if prediction is None:
                        confidence.zero_()
                        # Until the first valid stereo pair, the known serve
                        # command is the only non-privileged ball prior.  Once
                        # tracking expires, returning to that prior is safer
                        # than holding stale RGB coordinates indefinitely.
                        p, v = serve_p, serve_v
                    else:
                        p, v = (torch.as_tensor(a, device=args.device, dtype=torch.float32).reshape(1, 3)
                                for a in prediction)
                        latest = observer.tracker.last_estimate
                        confidence.fill_(float(latest.confidence) if latest is not None else 0.0)
                        side = decision.update(float(v[0, 0]))
                        if side is not None:
                            if not args.no_resample_targets:
                                landing[0, side] = rng.uniform(-.26, .26) * args.diversity
                                controller.set_landing_targets(landing)
                            controller.goals(p, v, controller.canonical_action(actions), confidence)
                            observation = controller.observation(p, v, confidence)[:, side]
                            actions[:, side] = torch.as_tensor(
                                policy.deterministic(observation), device=args.device)
                            decisions += 1
                    target = controller.step(p, v, controller.canonical_action(actions), confidence)
                # Identical sensing cadence in Kit and headless execution.
                env.step(target, render=(step % 4 == 3))
                total_steps += 1
                rally_steps += 1
                if scorer:
                    state = env.privileged_state()
                    state["contact_force"] = env.evaluation_contacts().reshape(1, 3)
                    events = scorer.update(state, env.dt)
                    hits = int(events["legal_hits"][0])
                    returns = int(events["legal_returns"][0])
                if writer is not None and step % 8 == 7:
                    frame = env.camera_rgb()["camera_demo"][0].detach().cpu().numpy()
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if bool(env.ball_needs_reset()[0]):
                    reason = "ball_left_court"
                    break
            row = {"rally": rally + 1, "active_arms": pair, "reason": reason,
                   "duration_s": round(rally_steps * env.dt, 4), "rgb_frames": captures,
                   "rgb_visible_frames": measured, "policy_decisions": decisions}
            if scorer:
                row.update(legal_hits=hits, legal_returns=returns)
            report["rallies"].append(row)
            print(json.dumps({"event": "rally_complete", **row}), flush=True)
            rally += 1
        report.update(physics_steps=total_steps,
                      simulation_seconds=round(total_steps * env.dt, 4),
                      wall_seconds=round(time.monotonic() - start_wall, 3))
    except KeyboardInterrupt:
        print("Inference stopped.", flush=True)
    finally:
        if writer is not None:
            writer.release()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        if env is not None:
            env.close()
        app.close()
    return 0


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.self_test:
        policy = AsymmetricPPO(seed=3, hidden=16)
        assert policy.deterministic(np.zeros((2, ACTOR_OBS_DIM), "f4")).shape == (2, ACTION_DIM)
        print("Policy self-test passed: 39 observations, 6 actions.")
        return 0
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
