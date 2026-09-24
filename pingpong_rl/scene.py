"""Isaac Lab scene configuration for the standalone ping-pong environment.

The scene is built from the public bimanual robot USD shipped with
ManipArena-Sim.  This module imports only Isaac Lab and PhysX; it does not
import the benchmark package or an application-specific runtime.
"""
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.backend_utils import get_default_renderer_cfg
from isaaclab_physx.sim.schemas import (
    PhysxCollisionPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

HERE = Path(__file__).resolve().parent
ROBOT_USD = HERE.parent / "assets" / "bimanual_robot" / "bimanual_robot.usd"

DT = 1.0 / 240.0
TABLE_SIZE = (1.40, 0.70, 0.04)
TABLE_TOP = 0.72
NET_HEIGHT = 0.075
BALL_RADIUS = 0.020
BALL_MASS = 0.0027
PADDLE_OFFSET = (0.235, 0.0, 0.0)
PADDLE_RADIUS = 0.10
ARM_NAMES = ("left", "right")
CAMERA_NAMES = ("camera_left", "camera_right", "camera_demo")
CAMERA_EYES = {
    "camera_left": (-0.65, -1.65, 1.65),
    "camera_right": (0.65, -1.65, 1.65),
    "camera_demo": (2.1, -2.1, 1.9),
}
CAMERA_TARGET = (0.0, 0.0, 0.90)


def _robot_cfg(prim_path: str, *, pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> ArticulationCfg:
    """Create one fixed-base bimanual robot articulation from the public USD."""
    if not ROBOT_USD.is_file():
        raise FileNotFoundError(f"robot asset is missing: {ROBOT_USD}")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim.UsdFileCfg(
            usd_path=str(ROBOT_USD),
            activate_contact_sensors=True,
            rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos={".*": 0.0},
        ),
        actuators={
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint[1-6]"],
                effort_limit_sim=200.0,
                stiffness=1500.0,
                damping=80.0,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["right_arm_joint[1-6]"],
                effort_limit_sim=200.0,
                stiffness=1500.0,
                damping=80.0,
            ),
            "left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_gripper"],
                effort_limit_sim=200.0,
                stiffness=200.0,
                damping=30.0,
            ),
            "right_gripper": ImplicitActuatorCfg(
                joint_names_expr=["right_arm_gripper"],
                effort_limit_sim=200.0,
                stiffness=200.0,
                damping=30.0,
            ),
        },
    )


def make_scene_cfg(
    num_envs: int = 1,
    *,
    env_spacing: float = 3.5,
    base_distance: float = 0.95,
    base_y: float = 0.29,
    base_height: float = 0.62,
    cameras: bool = False,
) -> InteractiveSceneCfg:
    """Build the complete table, ball, net, camera and robot scene config."""
    material = PhysxRigidBodyMaterialCfg(
        static_friction=0.01,
        dynamic_friction=0.01,
        restitution=0.88,
        restitution_combine_mode="max",
        friction_combine_mode="min",
    )
    collision = PhysxCollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)
    cfg = InteractiveSceneCfg(
        num_envs=int(num_envs),
        env_spacing=env_spacing,
        replicate_physics=True,
    )

    cfg.robot_left = _robot_cfg(
        "{ENV_REGEX_NS}/RobotLeft",
        pos=(-base_distance, base_y, base_height),
        rot=(0.0, 0.0, 0.0, 1.0),
    )
    cfg.robot_right = _robot_cfg(
        "{ENV_REGEX_NS}/RobotRight",
        pos=(base_distance, -base_y, base_height),
        rot=(0.0, 0.0, 1.0, 0.0),
    )
    cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=collision,
            physics_material=material,
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.035, 0.15, 0.28)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_TOP - TABLE_SIZE[2] / 2)
        ),
    )

    for i, (x, y) in enumerate(((-0.52, -0.25), (-0.52, 0.25), (0.52, -0.25), (0.52, 0.25))):
        setattr(
            cfg,
            f"table_leg_{i}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/TableLeg{i}",
                spawn=sim.CuboidCfg(
                    size=(0.045, 0.045, TABLE_TOP - 0.04),
                    visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.12, 0.13, 0.14)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(x, y, (TABLE_TOP - 0.04) / 2)
                ),
            ),
        )
    for i, (x, y) in enumerate(((-base_distance, base_y), (base_distance, -base_y))):
        setattr(
            cfg,
            f"robot_mount_{i}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/RobotMount{i}",
                spawn=sim.CuboidCfg(
                    size=(0.18, 0.18, base_height - 0.007),
                    visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.23, 0.24, 0.25)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(x, y, (base_height - 0.007) / 2)
                ),
            ),
        )

    cfg.net = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Net",
        spawn=sim.CuboidCfg(
            size=(0.006, 0.74, NET_HEIGHT),
            collision_props=collision,
            physics_material=PhysxRigidBodyMaterialCfg(restitution=0.1),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.80, 0.80, 0.80), opacity=0.18),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_TOP + NET_HEIGHT / 2)),
    )

    court_lines = [
        ((1.40, 0.007, 0.001), (0.0, -0.346, TABLE_TOP + 0.0006)),
        ((1.40, 0.007, 0.001), (0.0, 0.346, TABLE_TOP + 0.0006)),
        ((0.007, 0.70, 0.001), (-0.696, 0.0, TABLE_TOP + 0.0006)),
        ((0.007, 0.70, 0.001), (0.696, 0.0, TABLE_TOP + 0.0006)),
        ((1.40, 0.003, 0.001), (0.0, 0.0, TABLE_TOP + 0.0006)),
    ]
    for i, (size, pos) in enumerate(court_lines):
        setattr(
            cfg,
            f"court_line_{i}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/CourtLine{i}",
                spawn=sim.CuboidCfg(
                    size=size,
                    visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ),
        )
    for i in range(25):
        setattr(
            cfg,
            f"net_vertical_{i}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/NetVertical{i}",
                spawn=sim.CuboidCfg(
                    size=(0.001, 0.0015, NET_HEIGHT),
                    visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.2, 0.22, 0.25)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(0.0, -0.36 + i * 0.03, TABLE_TOP + NET_HEIGHT / 2)
                ),
            ),
        )
    for i in range(5):
        setattr(
            cfg,
            f"net_horizontal_{i}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/NetHorizontal{i}",
                spawn=sim.CuboidCfg(
                    size=(0.001, 0.74, 0.0015),
                    visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.2, 0.22, 0.25)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(0.0, 0.0, TABLE_TOP + i * NET_HEIGHT / 4)
                ),
            ),
        )

    cfg.ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim.SphereCfg(
            radius=BALL_RADIUS,
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.015,
                angular_damping=0.01,
                max_linear_velocity=20.0,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim.MassPropertiesCfg(mass=BALL_MASS),
            collision_props=collision,
            physics_material=material,
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(1.0, 0.33, 0.015)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.30, 0.0, 1.10)),
    )
    cfg.ball_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        update_period=0.0,
        history_length=2,
        track_air_time=False,
    )

    if cameras:
        for name in CAMERA_NAMES:
            setattr(
                cfg,
                name,
                CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    update_period=1.0 / 60.0,
                    height=480,
                    width=640,
                    data_types=["rgb"],
                    spawn=sim.PinholeCameraCfg(
                        focal_length=18.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.03, 20.0),
                    ),
                    renderer_cfg=get_default_renderer_cfg(),
                ),
            )

    cfg.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim.GroundPlaneCfg(color=(0.14, 0.15, 0.17)),
    )
    cfg.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim.DomeLightCfg(intensity=1500.0),
    )
    return cfg
