"""Isaac Lab application bootstrap for the standalone ping-pong project."""
from __future__ import annotations

import argparse


def make_app(device: str = "cuda:0", *, cameras: bool = False, headless: bool = True):
    """Start Isaac Lab Sim 6 and return its ``SimulationApp``.

    The caller must keep the returned app alive until :meth:`PingPongEnv.close`
    has run.  No benchmark-specific packages or environment variables are
    required; the robot USD is resolved relative to this project.
    """
    import torch  # Load the runtime tensor library before Kit initializes extensions.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--viz", "none" if headless else "kit"])
    args.device = device
    args.enable_cameras = bool(cameras)
    # Isaac Lab 6 accepts this compatibility field and selects the rendering
    # experience automatically when cameras are enabled.
    args.headless = bool(headless)
    args.livestream = -1
    return AppLauncher(args).app
