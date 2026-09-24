"""Shared PPO on every incoming physical stroke, using four-arm geometry.

Training ball estimates are noisy geometric surrogates derived from simulated
ball state. Deployment uses RGB tracking. Contact forces are confined to reward
calculation and evaluation. During a rally only joint targets are written. Each
stable velocity reversal creates a new actor decision and PPO transition.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np


def event_gae(trajectories, gamma=0.985, lam=0.95):
    """GAE on alternating player decisions in a finite cooperative rally."""
    transitions = []
    for trajectory in trajectories:
        gae = 0.0
        next_value = 0.0
        for item in reversed(trajectory):
            delta = item["reward"] + gamma * next_value - item["value"]
            gae = delta + gamma * lam * gae
            item["lambda_return"] = gae + item["value"]
            next_value = item["value"]
        transitions.extend(trajectory)
    if not transitions:
        raise ValueError("No physical stroke decisions collected")

    def arr(k):
        return np.stack([x[k] for x in transitions]).astype("f4")[None]

    obs = arr("obs")
    n = len(transitions)
    # The PPO implementation performs its own GAE. A terminal one-step batch of precomputed
    # lambda returns gives exactly those advantages without double discounting.
    return {
        "obs": obs,
        "critic_obs": obs.copy(),
        "action": arr("action"),
        "logprob": arr("logprob"),
        "value": arr("value"),
        "reward": arr("lambda_return"),
        "done": np.ones((1, n), "f4"),
        "last_critic_obs": obs[0].copy(),
    }


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/train"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--episodes-per-update", type=int, default=2)
    p.add_argument("--min-transitions", type=int, default=128)
    p.add_argument("--max-episodes-per-update", type=int, default=8)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=712)
    p.add_argument("--initial-std", type=float, default=0.12)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--position-noise", type=float, default=0.003)
    p.add_argument("--velocity-noise", type=float, default=0.05)
    p.add_argument("--control-decimation", type=int, default=2)
    p.add_argument(
        "--ready-steps",
        type=int,
        default=450,
        help="Joint-drive steps for each ready-pose calibration",
    )
    p.add_argument("--diversity", type=float, default=0.25)
    p.add_argument("--diversity-end", type=float)
    p.add_argument("--base-y", type=float, default=0.0)
    p.add_argument(
        "--arm-pair", choices=["cycle", "00", "01", "10", "11"], default="cycle"
    )
    p.add_argument(
        "--resample-targets",
        action="store_true",
        help="Randomize the selected player landing ordinate at each new stroke",
    )
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--evaluation-episodes", type=int, default=2)
    p.add_argument("--validation-seed", type=int, default=19003)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--zero-residual", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _ppo_self_test():
    """Exercise a real PPO gradient update without starting the simulator."""
    from .policy import AsymmetricPPO

    policy = AsymmetricPPO(
        actor_dim=39,
        action_dim=6,
        critic_dim=39,
        device="cpu",
        seed=7,
        hidden=32,
        epochs=1,
        minibatch=8,
    )
    observations = np.random.default_rng(7).normal(size=(8, 39)).astype("f4")
    actions, logprob, values = policy.act_with_value(observations, observations)
    trajectories = [
        [
            {
                "obs": observations[i],
                "action": actions[i],
                "logprob": logprob[i],
                "value": values[i],
                "reward": float(i % 3 - 1),
            }
            for i in range(8)
        ]
    ]
    metrics = policy.update(event_gae(trajectories))
    if policy.updates != 1 or policy.transitions != 8:
        raise AssertionError("PPO update did not consume the expected transitions")
    if not all(np.isfinite(value) for value in metrics.values()):
        raise AssertionError("PPO update produced non-finite metrics")


def main(argv=None):
    args = arguments(argv)
    if args.self_test:
        rec = lambda r, v: {
            "obs": np.zeros(2),
            "action": np.zeros(1),
            "logprob": 0.0,
            "value": v,
            "reward": r,
        }
        out = event_gae([[rec(1, 0.3), rec(2, 0.1)], [rec(-1, 0.4)]], 1.0, 1.0)
        assert np.allclose(out["reward"], [[3, 2, -1]])
        assert out["obs"].shape == (1, 3, 2)
        _ppo_self_test()
        print("Event GAE and PPO update self-test passed")
        return
    if (
        args.num_envs < 1
        or args.seconds <= 0
        or args.control_decimation < 1
        or args.ready_steps < 1
    ):
        raise ValueError("Rollout dimensions and calibration steps must be positive")
    if (
        args.updates < 1
        or args.episodes_per_update < 1
        or args.max_episodes_per_update < args.episodes_per_update
    ):
        raise ValueError(
            "Use positive updates and max-episodes-per-update >= episodes-per-update"
        )
    if args.min_transitions < 1 or args.ppo_epochs < 1 or args.evaluation_episodes < 1:
        raise ValueError("Transition, epoch, and evaluation counts must be positive")
    if args.position_noise < 0 or args.velocity_noise < 0 or args.learning_rate <= 0:
        raise ValueError("Noise must be non-negative and learning rate positive")
    if not 0 <= args.diversity <= 1:
        raise ValueError("diversity must be in [0,1]")
    if args.diversity_end is not None and not 0 <= args.diversity_end <= 1:
        raise ValueError("diversity-end must be in [0,1]")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    log = (args.output / "metrics.jsonl").open("a", buffering=1)
    start = time.monotonic()

    def record(event, **kw):
        row = {
            "event": event,
            "elapsed_s": round(time.monotonic() - start, 3),
            "timestamp": time.time(),
            **kw,
        }
        log.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    from .runtime import make_app

    app = make_app(args.device, cameras=False)
    env = None
    event_log = None
    try:
        import torch
        from .environment import PingPongEnv
        from .control import BallisticResidualController
        from .ready import prepare_ready_homes, reset_ready
        from .scoring import RallyScorer
        from .policy import AsymmetricPPO

        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        env = PingPongEnv(args.num_envs, args.device, base_y=args.base_y)
        ctrl = BallisticResidualController(env)
        scorer = RallyScorer(args.num_envs, args.device)
        n = args.num_envs
        device = env.device
        actions = torch.zeros(n, 2, 6, device=device)
        homes = prepare_ready_homes(env, ctrl, steps=args.ready_steps)
        initial = env.privileged_state()
        ctrl.goals(initial["ball_pos"], initial["ball_vel"], actions)
        obs_dim = ctrl.observation(initial["ball_pos"], initial["ball_vel"]).shape[-1]
        if obs_dim != 39:
            raise ValueError(f"Expected 39 actor inputs, got {obs_dim}")
        policy = (
            AsymmetricPPO.load(args.checkpoint, device)
            if args.checkpoint
            else AsymmetricPPO(obs_dim, 6, obs_dim, device, seed=args.seed)
        )
        if (policy.actor_dim, policy.critic_dim, policy.action_dim) != (39, 39, 6):
            raise ValueError(
                f"Checkpoint dimensions {(policy.actor_dim,policy.critic_dim,policy.action_dim)} != (39,39,6)"
            )
        if args.initial_std > 0:
            policy.log_std.data.fill_(float(np.log(args.initial_std)))
        policy.cfg.lr = args.learning_rate
        policy.cfg.epochs = args.ppo_epochs
        for group in policy.opt.param_groups:
            group["lr"] = args.learning_rate
        # Loading a checkpoint constructs a module and initializes its RNG.
        # Restore the requested rollout seed after construction.
        torch.manual_seed(args.seed)
        record(
            "start",
            config=config,
            observation_dim=obs_dim,
            policy_updates=policy.updates,
            actor_input="noisy geometric ball-state surrogate; no contact or reward fields",
            task="four-arm physical rally, shared PPO on every stable incoming stroke",
            ready_paddle_pos=env.paddle_state()[0][0].tolist(),
        )

        def episode(number, sample, diversity):
            codes = (number + np.arange(n)) % 4
            pair = np.stack([codes // 2, codes % 2], axis=1)
            if args.arm_pair != "cycle":
                pair[:] = [int(x) for x in args.arm_pair]
            reset_ready(
                env, homes, torch.as_tensor(pair, device=device, dtype=torch.long)
            )
            actions.zero_()
            ctrl.last_action.zero_()
            ctrl.last_paddle_pos = None
            side = torch.as_tensor(
                rng.integers(0, 2, n), device=device, dtype=torch.long
            )
            direction = torch.where(side == 1, 1.0, -1.0)
            p = torch.zeros(n, 3, device=device)
            v = torch.zeros_like(p)
            p[:, 0] = -0.35 * direction
            p[:, 1] = torch.as_tensor(
                rng.uniform(-0.18, 0.18, n) * diversity, device=device
            )
            p[:, 2] = torch.as_tensor(
                1.0 + (rng.uniform(0.91, 1.10, n) - 1.0) * diversity, device=device
            )
            speed = torch.as_tensor(
                2.0 + (rng.uniform(1.55, 2.45, n) - 2.0) * diversity, device=device
            )
            v[:, 0] = direction * speed
            v[:, 1] = torch.as_tensor(
                rng.uniform(-0.35, 0.35, n) * diversity, device=device
            )
            v[:, 2] = torch.as_tensor(
                0.2 + (rng.uniform(-0.35, 0.55, n) - 0.2) * diversity, device=device
            )
            spin = torch.as_tensor(
                rng.uniform(-8.0, 8.0, (n, 3)) * diversity, device=device, dtype=p.dtype
            )
            landing_y = torch.as_tensor(
                rng.uniform(-0.26, 0.26, (n, 2)) * diversity,
                device=device,
                dtype=p.dtype,
            )
            ctrl.set_landing_targets(landing_y)
            env.serve(p, v, angular_velocity=spin)
            state = env.privileged_state()
            scorer.reset(state)
            pbias = torch.randn_like(p) * args.position_noise * 0.5
            vbias = torch.randn_like(v) * args.velocity_noise * 0.5

            def estimate(state):
                return (
                    state["ball_pos"]
                    + pbias
                    + torch.randn_like(p) * args.position_noise * 0.5,
                    state["ball_vel"]
                    + vbias
                    + torch.randn_like(v) * args.velocity_noise * 0.5,
                )

            pe, ve = estimate(state)
            ctrl.goals(pe, ve, ctrl.canonical_action(actions))
            trajectories = [[] for _ in range(n)]
            active = np.ones(n, dtype=bool)
            pending = [None] * n
            last_hit_record = [None] * n
            side_now = side.clone()
            candidate = side.clone()
            stable_count = torch.zeros(n, device=device, dtype=torch.long)

            def decide(mask, resample=False):
                inds = torch.where(mask)[0]
                if not len(inds):
                    return
                if resample and args.resample_targets:
                    landing_y[inds, side_now[inds]] = torch.as_tensor(
                        rng.uniform(-0.26, 0.26, len(inds)) * diversity,
                        device=device,
                        dtype=p.dtype,
                    )
                    ctrl.set_landing_targets(landing_y)
                    ctrl.goals(pe, ve, ctrl.canonical_action(actions))
                observations = ctrl.observation(pe, ve)[inds, side_now[inds]]
                if sample:
                    a, lp, val = policy.act_with_value(observations, observations)
                else:
                    a = (
                        np.zeros((len(inds), 6), "f4")
                        if args.zero_residual
                        else policy.deterministic(observations)
                    )
                    lp = np.zeros(len(inds), "f4")
                    val = np.zeros(len(inds), "f4")
                actions[inds, side_now[inds]] = torch.as_tensor(a, device=device)
                ob = observations.cpu().numpy()
                for k, e in enumerate(inds.cpu().tolist()):
                    pending[e] = {
                        "obs": ob[k].copy(),
                        "action": a[k].copy(),
                        "logprob": float(lp[k]),
                        "value": float(val[k]),
                        "reward": -0.03 * float(np.square(a[k]).sum()),
                        "episode": number,
                        "env_index": e,
                        "player": int(side_now[e]),
                        "arm": int(env.active_arms[e, side_now[e]]),
                        "decision_step": int(env.common_step_counter),
                        "landing_y": float(landing_y[e, side_now[e]]),
                        "legal_hits": 0,
                        "legal_returns": 0,
                    }

            decide(torch.ones(n, device=device, dtype=torch.bool))
            max_steps = max(1, round(args.seconds / env.dt))
            j = ctrl.step(pe, ve, ctrl.canonical_action(actions))
            duration = np.full(n, args.seconds)
            episode_returns = np.zeros(n, dtype=int)
            episode_hits = np.zeros(n, dtype=int)
            for step in range(max_steps):
                if step % args.control_decimation == 0:
                    pe, ve = estimate(state)
                    observed = torch.where(ve[:, 0] > 0, 1, 0)
                    same = observed == candidate
                    stable_count = torch.where(
                        same, stable_count + 1, torch.ones_like(stable_count)
                    )
                    candidate = observed
                    changed = (
                        (observed != side_now)
                        & (stable_count >= 3)
                        & (ve[:, 0].abs() > 0.15)
                        & torch.as_tensor(active, device=device)
                    )
                    if bool(changed.any()):
                        for e in torch.where(changed)[0].cpu().tolist():
                            if pending[e] is not None:
                                trajectories[e].append(pending[e])
                                pending[e] = None
                        side_now = torch.where(changed, observed, side_now)
                        ctrl.goals(pe, ve, ctrl.canonical_action(actions))
                        decide(changed, resample=True)
                    j = ctrl.step(pe, ve, ctrl.canonical_action(actions))
                env.step(j)
                state = env.privileged_state()
                state["contact_force"] = env.evaluation_contacts().reshape(n, 3)
                events = scorer.update(state, env.dt)
                legal_hit = events["legal_hit"].cpu().numpy()
                legal_return = events["legal_return"].cpu().numpy()
                bad = (events["out"] | events["invalid"]).cpu().numpy()
                wrong_bounce = events["wrong_bounce"].cpu().numpy()
                hit_side = events["hit_side"].cpu().numpy()
                new_bad = bad & active
                for e in np.flatnonzero(active):
                    if pending[e] is not None:
                        pending[e]["reward"] -= 0.0001
                    if legal_hit[e]:
                        # Event detector intentionally lags by two control frames,
                        # so the striking player's record is still pending here.
                        target = pending[e]
                        if target is not None and target["player"] == int(hit_side[e]):
                            target["reward"] += 2.0
                            target["legal_hits"] += 1
                            last_hit_record[e] = target
                    if legal_return[e] and last_hit_record[e] is not None:
                        # Credit the striker, even if its record has already been
                        # appended when the observed ball reversed direction.
                        target = last_hit_record[e]
                        target["reward"] += 3.0
                        target["legal_returns"] += 1
                        landing_error = (
                            float(state["ball_pos"][e, 1]) - target["landing_y"]
                        )
                        target["reward"] += float(np.exp(-(landing_error**2) / 0.01))
                    if new_bad[e]:
                        target = (
                            last_hit_record[e]
                            if wrong_bounce[e] and last_hit_record[e] is not None
                            else pending[e]
                        )
                        if target is not None:
                            target["reward"] -= 7.0
                        if pending[e] is not None:
                            trajectories[e].append(pending[e])
                            pending[e] = None
                        episode_returns[e] = int(events["legal_returns"][e])
                        episode_hits[e] = int(events["legal_hits"][e])
                        active[e] = False
                        duration[e] = (step + 1) * env.dt
                if not active.any():
                    break
            for e in np.flatnonzero(active):
                if pending[e] is not None:
                    trajectories[e].append(pending[e])
                    pending[e] = None
                episode_returns[e] = int(scorer.legal_returns[e])
                episode_hits[e] = int(scorer.legal_hits[e])
            event_count = sum(map(len, trajectories))
            record(
                "episode",
                episode=number,
                sampled=sample,
                diversity=diversity,
                legal_returns=episode_returns.tolist(),
                legal_hits=episode_hits.tolist(),
                active_arm_pair=pair.tolist(),
                survived_full_rally=active.tolist(),
                mean_duration=float(duration.mean()),
                event_count=event_count,
                mean_return=float(episode_returns.mean()),
                max_return=int(episode_returns.max()),
                physics_steps=step + 1,
                stroke_samples_per_physics_step=event_count / max(step + 1, 1),
                serve_y_range=[float(p[:, 1].min()), float(p[:, 1].max())],
                serve_speed_range=[float(speed.min()), float(speed.max())],
                landing_y_range=[float(landing_y.min()), float(landing_y.max())],
            )
            return trajectories, float(episode_returns.mean()), float(duration.mean())

        def validate(update, diversity):
            nonlocal rng
            training_rng = rng
            rng = np.random.default_rng(args.validation_seed)
            scores = []
            durations = []
            cuda_devices = (
                [torch.device(device).index or 0]
                if torch.device(device).type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(args.validation_seed)
                for ep in range(args.evaluation_episodes):
                    _, score, duration = episode(ep, False, diversity)
                    scores.append(score)
                    durations.append(duration)
            rng = training_rng
            score = float(np.mean(scores))
            duration = float(np.mean(durations))
            record(
                "deterministic_evaluation",
                rally_update=update,
                validation_seed=args.validation_seed,
                mean_legal_returns=score,
                mean_duration=duration,
                diversity=diversity,
            )
            return score, duration

        if args.eval_only:
            validate(0, args.diversity)
            record("complete")
            return
        event_log = (args.output / "shot_events.jsonl").open("a", buffering=1)
        best = (-float("inf"), -float("inf"))
        episode_index = 0
        initial_transitions = policy.transitions
        for update in range(args.updates):
            frac = update / max(args.updates - 1, 1)
            diversity = (
                args.diversity
                if args.diversity_end is None
                else args.diversity + (args.diversity_end - args.diversity) * frac
            )
            trajectories = []
            scores = []
            samples = 0
            episodes = 0
            while episodes < max(
                args.episodes_per_update, args.max_episodes_per_update
            ):
                tr, score, _ = episode(episode_index, True, diversity)
                episode_index += 1
                trajectories.extend(tr)
                scores.append(score)
                samples += sum(map(len, tr))
                episodes += 1
                if (
                    episodes >= args.episodes_per_update
                    and samples >= args.min_transitions
                ):
                    break
            roll = event_gae(trajectories, policy.cfg.gamma, policy.cfg.gae_lambda)
            for tr in trajectories:
                for ev in tr:
                    event_log.write(
                        json.dumps(
                            {
                                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in ev.items()
                            }
                        )
                        + "\n"
                    )
            metrics = policy.update(roll)
            metrics["mean_lambda_return_target"] = metrics.pop("mean_reward")
            record(
                "ppo_update",
                rally_update=update + 1,
                **metrics,
                new_transitions=policy.transitions - initial_transitions,
                rollout_episodes=episodes,
                rollout_samples=samples,
                mean_legal_returns=float(np.mean(scores)),
                diversity=diversity,
                actual_event_reward=float(
                    np.mean([x["reward"] for tr in trajectories for x in tr])
                ),
            )
            meta = {
                "task": "physical alternating-stroke cooperative PPO",
                "training_config": config,
                "observation_dim": obs_dim,
                "current_diversity": diversity,
            }
            policy.save(args.output / "latest.pt", meta)
            if args.eval_every and (update + 1) % args.eval_every == 0:
                result = validate(
                    update + 1,
                    (
                        args.diversity
                        if args.diversity_end is None
                        else args.diversity_end
                    ),
                )
                if result > best:
                    best = result
                    policy.save(args.output / "best.pt", meta)
            if (update + 1) % 5 == 0:
                policy.save(args.output / f"update_{update+1:03d}.pt", meta)
        policy.save(args.output / "final.pt", meta)
        record("complete")
    finally:
        if event_log is not None:
            event_log.close()
        log.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
