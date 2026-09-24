# Algorithm

The task is a two-sided table-tennis rally with one physical ball, a net, two
fixed-base bimanual robot articulations, and one realistic paddle attached to
each arm link. The simulator handles rigid-body contacts and gravity. The
policy controls Cartesian paddle targets through a damped Jacobian controller;
joint targets are then sent to the Isaac Lab articulation actuators.

## Observation and deployment boundary

The actor receives a compact vector built from:

- the RGB tracker estimate of ball position and velocity;
- the selected paddle and opposing paddle pose;
- arm joint position and velocity from proprioception;
- the predicted intercept/landing target and time to impact;
- ball-track confidence, incoming direction, hand identity, and the previous
  residual action.

The exact vector size is validated by the policy module and recorded with each
checkpoint. Camera pixels are processed by `pingpong_rl.vision`; the actor does
not read depth, simulator object handles, contact sensors, reward values, or
privileged ball state.

During training, an asymmetric critic may consume physical ball state and
contact diagnostics to reduce variance. This information is kept in the critic
path and is unavailable to `evaluate.py`.

## Action and control

The PPO action is a bounded residual for each player: intercept position,
return-speed adjustment, and paddle-face pitch/yaw adjustment. The nominal
controller predicts a ballistic intercept, accounts for one table bounce, and
selects a legal return landing point. The residual policy corrects model error,
latency, and changing incoming trajectories. A canonical player frame mirrors
the two table sides, so a shared actor can control left/right sides and switch
between the two gripper-mounted paddles.

The paddle collision geometry is intentionally stricter than a disk: a thin
elliptical blade and a separate handle are attached to every terminal arm link.
The scorer accepts a hit only when the ball reverses direction at the blade
plane, lies within the blade footprint, and has physical contact evidence when
that signal is available. Contacts with a handle, inactive arm, robot body, or
an out-of-bounds ball are invalid.

## Reward and evaluation

The rollout code assigns event rewards to the policy decision that preceded an
observed stroke. It rewards a legal blade hit and a legal one-bounce return,
adds a smooth landing-target term, applies a small action/step cost, and
penalizes illegal contact, wrong-side bounces, and out-of-bounds termination.
The event scorer is an evaluator and audit component; its ground-truth fields
are not copied into actor observations.

Training logs should include seed, environment count, physics step, camera
mode, checkpoint checksum, observation dimensions, and legal-hit/return counts.
Evaluation should report deterministic episodes separately from sampled
training rollouts. A short smoke or demo is not a claim of generalization or
physical-robot readiness.

## Reproducibility

Use the same Isaac Lab commit, Sim wheel version, seed, table geometry, camera
calibration, and checkpoint configuration when comparing runs. Save the
configuration and SHA-256 checksum next to every checkpoint. Keep RGB-only
policy evaluation separate from privileged diagnostics so that an accidental
simulator-state dependency is visible in review.
