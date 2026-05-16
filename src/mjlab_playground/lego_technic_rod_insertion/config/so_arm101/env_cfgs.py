"""SO-101 LEGO Technic rod insertion environment configuration."""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_playground.asset_zoo.objects.lego_technic import (
  get_axle_4l_spec,
  get_holder_32064_spec,
)
from mjlab_playground.asset_zoo.robots.so_arm101.so_arm101_constants import (
  get_so_arm101_robot_cfg,
)
from mjlab_playground.asset_zoo.robots.so_arm101.so_arm101_constants import (
  get_spec as get_so_arm101_spec,
)
from mjlab_playground.lego_technic_rod_insertion import mdp

_GRASPED_AXLE_PREFIX = "grasped_"
_GRIPPER_FRAME_POS = (-0.0079, -0.000218121, -0.0981274)
_GRIPPER_FRAME_QUAT = (0.0, 0.0, 1.0, 0.0)

_GRIPPER_CLOSED_POS = 0.8
_INSERTION_JOINT_POS = {
  "shoulder_pan": 0.0,
  "shoulder_lift": -1.0,
  "elbow_flex": 1.3,
  "wrist_flex": -0.5,
  "wrist_roll": 0.0,
  "gripper": _GRIPPER_CLOSED_POS,
}

# The holder hole starts near the rod tip; insertion success is based on the rod center.
_HOLDER_POS = (0.08755264, -0.05820378, 0.09578617)
_HOLDER_QUAT = (0.70508539, -0.05344602, 0.05344908, 0.70508248)

_HOLDER_ROD_CONTACT_SENSOR = "holder_rod_contact"
_ROBOT_SELF_COLLISION_SENSOR = "robot_self_collision"


def _so_arm101_with_grasped_axle_spec() -> mujoco.MjSpec:
  """Return SO-101 with a fixed LEGO axle attached at the gripper frame."""
  spec = get_so_arm101_spec()
  for key in list(spec.keys):
    spec.delete(key)

  axle_spec = get_axle_4l_spec()

  for joint in list(axle_spec.joints):
    if joint.type == mujoco.mjtJoint.mjJNT_FREE:
      axle_spec.delete(joint)

  gripper_body = spec.body("gripper_link")
  frame = gripper_body.add_frame()
  frame.pos[:] = _GRIPPER_FRAME_POS
  frame.quat[:] = _GRIPPER_FRAME_QUAT
  spec.attach(axle_spec, prefix=_GRASPED_AXLE_PREFIX, frame=frame)
  return spec


def _so_arm101_with_grasped_axle_cfg() -> EntityCfg:
  robot_cfg = get_so_arm101_robot_cfg()
  robot_cfg.spec_fn = _so_arm101_with_grasped_axle_spec
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    joint_pos=_INSERTION_JOINT_POS,
    joint_vel={".*": 0.0},
  )
  return robot_cfg


def _holder_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=_HOLDER_POS,
      rot=_HOLDER_QUAT,
      joint_pos={},
      joint_vel={},
    ),
    spec_fn=get_holder_32064_spec,
  )


def so_arm101_lego_technic_rod_insertion_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the SO-101 LEGO Technic rod insertion task configuration."""
  robot_all_joints = SceneEntityCfg("robot", joint_names=(".*",))
  gripper_frame = SceneEntityCfg("robot", site_names=("gripper_frame",))
  rod_tip = SceneEntityCfg("robot", site_names=("grasped_rod_tip",))
  rod_back = SceneEntityCfg("robot", site_names=("grasped_rod_back",))
  hole_center = SceneEntityCfg("holder", site_names=("hole_center",))
  hole_axis_end = SceneEntityCfg("holder", site_names=("hole_axis_end",))
  holder_asset = SceneEntityCfg("holder")
  rod_hole_params = {
    "rod_tip_site_cfg": rod_tip,
    "rod_back_site_cfg": rod_back,
    "hole_center_site_cfg": hole_center,
    "hole_axis_end_site_cfg": hole_axis_end,
  }
  holder_rod_force_params = {
    **rod_hole_params,
    "sensor_name": _HOLDER_ROD_CONTACT_SENSOR,
  }

  sensors = (
    ContactSensorCfg(
      name=_HOLDER_ROD_CONTACT_SENSOR,
      primary=ContactMatch(
        mode="geom",
        pattern="holder_32064_.*_collision",
        entity="holder",
      ),
      secondary=ContactMatch(
        mode="geom",
        pattern="grasped_axle_4l_collision",
        entity="robot",
      ),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    ),
    ContactSensorCfg(
      name=_ROBOT_SELF_COLLISION_SENSOR,
      primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
      fields=("found",),
      reduce="maxforce",
      num_slots=1,
    ),
  )

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": robot_all_joints, "biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": robot_all_joints},
    ),
    "ee_pose": ObservationTermCfg(
      func=mdp.end_effector_pose,
      params={"ee_site_cfg": gripper_frame},
    ),
    "rod_tip_pos": ObservationTermCfg(
      func=mdp.rod_tip_position,
      params={"rod_tip_site_cfg": rod_tip},
    ),
    "rod_axis": ObservationTermCfg(
      func=mdp.rod_axis,
      params={
        "rod_tip_site_cfg": rod_tip,
        "rod_back_site_cfg": rod_back,
      },
    ),
    "hole_center_pos": ObservationTermCfg(
      func=mdp.hole_center_position,
      params={"hole_center_site_cfg": hole_center},
    ),
    "hole_axis": ObservationTermCfg(
      func=mdp.hole_axis,
      params={
        "hole_center_site_cfg": hole_center,
        "hole_axis_end_site_cfg": hole_axis_end,
      },
    ),
    "tip_to_hole": ObservationTermCfg(func=mdp.tip_to_hole, params=rod_hole_params),
    "lateral_error": ObservationTermCfg(
      func=mdp.lateral_error, params=rod_hole_params
    ),
    "axis_angle_error": ObservationTermCfg(
      func=mdp.axis_angle_error, params=rod_hole_params
    ),
    "insertion_depth": ObservationTermCfg(
      func=mdp.insertion_depth, params=rod_hole_params
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "ee_delta_pose": envs_mdp.DifferentialIKActionCfg(
      entity_name="robot",
      actuator_names=(
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
      ),
      frame_type="site",
      frame_name="gripper_frame",
      use_relative_mode=True,
      delta_pos_scale=0.015,
      delta_ori_scale=0.2,
      damping=0.05,
      max_dq=0.2,
      position_weight=1.0,
      orientation_weight=0.25,
      joint_limit_weight=0.05,
      posture_weight=0.02,
      posture_target=_INSERTION_JOINT_POS,
    ),
  }

  events = {
    "reset_robot_root": EventTermCfg(
      mode="reset",
      func=envs_mdp.reset_root_state_uniform,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "pose_range": {},
      },
    ),
    "reset_holder_root": EventTermCfg(
      mode="reset",
      func=envs_mdp.reset_root_state_uniform,
      params={
        "asset_cfg": SceneEntityCfg("holder"),
        "pose_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      mode="reset",
      func=mdp.reset_joints_to_default,
      params={"asset_cfg": robot_all_joints},
    ),
  }

  rewards = {
    "lateral_alignment": RewardTermCfg(
      func=mdp.lateral_alignment_reward,
      weight=1.5,
      params={
        **rod_hole_params,
        "std": 0.0035,
      },
    ),
    "rod_axis_alignment": RewardTermCfg(
      func=mdp.rod_axis_alignment_reward,
      weight=0.5,
      params=rod_hole_params,
    ),
    "insertion_depth_progress": RewardTermCfg(
      func=mdp.insertion_depth_progress_reward,
      weight=2.0,
      params={
        **rod_hole_params,
        "start_depth": -0.020,
        "target_depth": 0.004,
      },
    ),
    "success_bonus": RewardTermCfg(
      func=mdp.stable_success_bonus,
      weight=5.0,
      params={
        **rod_hole_params,
        "lateral_error_threshold": 0.0035,
        "axis_angle_threshold": 0.15,
        "insertion_depth_threshold": 0.004,
        "stable_steps": 5,
      },
    ),
    "action_l2": RewardTermCfg(func=mdp.action_l2, weight=-0.02),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
    "lateral_contact_force": RewardTermCfg(
      func=mdp.lateral_contact_force_cost,
      weight=-0.5,
      params={
        **holder_rod_force_params,
        "force_scale": 12.0,
      },
    ),
    "holder_displacement": RewardTermCfg(
      func=mdp.holder_displacement_reward_cost,
      weight=-0.5,
      params={
        "asset_cfg": holder_asset,
        "position_scale": 0.005,
        "angle_scale": 0.10,
      },
    ),
    "jam": RewardTermCfg(
      func=mdp.jam_cost,
      weight=-2.0,
      params={
        **holder_rod_force_params,
        "lateral_force_threshold": 12.0,
        "lateral_error_threshold": 0.005,
        "insertion_depth_threshold": 0.004,
        "insertion_depth_margin": 0.002,
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "success": TerminationTermCfg(
      func=mdp.stable_success,
      params={
        **rod_hole_params,
        "lateral_error_threshold": 0.0035,
        "axis_angle_threshold": 0.15,
        "insertion_depth_threshold": 0.004,
        "stable_steps": 5,
      },
    ),
    "dropped_rod": TerminationTermCfg(
      func=mdp.dropped_rod_termination,
      params={
        **rod_hole_params,
        "max_tip_to_hole_distance": 0.12,
      },
    ),
    "holder_displaced": TerminationTermCfg(
      func=mdp.holder_displaced_termination,
      params={
        "asset_cfg": holder_asset,
        "position_threshold": 0.005,
        "angle_threshold": 0.10,
      },
    ),
    "excessive_force": TerminationTermCfg(
      func=mdp.excessive_force_termination,
      params={
        "sensor_name": _HOLDER_ROD_CONTACT_SENSOR,
        "force_threshold": 25.0,
      },
    ),
    "jammed": TerminationTermCfg(
      func=mdp.jammed_termination,
      params={
        **holder_rod_force_params,
        "lateral_force_threshold": 12.0,
        "lateral_error_threshold": 0.005,
        "insertion_depth_threshold": 0.004,
        "insertion_depth_margin": 0.002,
      },
    ),
    "self_collision": TerminationTermCfg(
      func=mdp.self_collision_termination,
      params={"sensor_name": _ROBOT_SELF_COLLISION_SENSOR},
    ),
  }

  metrics = {
    "lateral_error": MetricsTermCfg(
      func=mdp.lateral_error_metric,
      params=rod_hole_params,
    ),
    "axis_angle_error": MetricsTermCfg(
      func=mdp.axis_angle_error_metric,
      params=rod_hole_params,
    ),
    "insertion_depth": MetricsTermCfg(
      func=mdp.insertion_depth_metric,
      params=rod_hole_params,
    ),
    "contact_force": MetricsTermCfg(
      func=mdp.contact_force_metric,
      params={"sensor_name": _HOLDER_ROD_CONTACT_SENSOR},
    ),
    "lateral_force": MetricsTermCfg(
      func=mdp.lateral_force_metric,
      params=holder_rod_force_params,
    ),
    "holder_displacement": MetricsTermCfg(
      func=mdp.holder_displacement_metric,
      params={"asset_cfg": holder_asset},
    ),
    "success_ready": MetricsTermCfg(
      func=mdp.success_ready_metric,
      reduce="last",
      params={
        **rod_hole_params,
        "lateral_error_threshold": 0.0035,
        "axis_angle_threshold": 0.15,
        "insertion_depth_threshold": 0.004,
      },
    ),
    "stable_success": MetricsTermCfg(
      func=mdp.stable_success_metric,
      reduce="last",
      params={"termination_name": "success"},
    ),
    "success_rate": MetricsTermCfg(
      func=mdp.stable_success_metric,
      reduce="last",
      params={"termination_name": "success"},
    ),
    "dropped": MetricsTermCfg(
      func=mdp.dropped_metric,
      reduce="last",
      params={
        **rod_hole_params,
        "max_tip_to_hole_distance": 0.12,
      },
    ),
    "jammed": MetricsTermCfg(
      func=mdp.jammed_metric,
      reduce="last",
      params={
        **holder_rod_force_params,
        "lateral_force_threshold": 12.0,
        "lateral_error_threshold": 0.005,
        "insertion_depth_threshold": 0.004,
        "insertion_depth_margin": 0.002,
      },
    ),
    "excessive_force": MetricsTermCfg(
      func=mdp.excessive_force_metric,
      reduce="last",
      params={
        "sensor_name": _HOLDER_ROD_CONTACT_SENSOR,
        "force_threshold": 25.0,
      },
    ),
    "self_collision": MetricsTermCfg(
      func=mdp.self_collision_metric,
      reduce="last",
      params={"sensor_name": _ROBOT_SELF_COLLISION_SENSOR},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={
        "robot": _so_arm101_with_grasped_axle_cfg(),
        "holder": _holder_cfg(),
      },
      sensors=sensors,
      num_envs=1 if play else 4096,
      extent=0.4,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="gripper_link",
      distance=0.45,
      elevation=-20.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      njmax=120,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,
    episode_length_s=3.0,
  )
