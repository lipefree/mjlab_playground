"""Shared task signals for LEGO Technic rod insertion rewards and dones."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_playground.lego_technic_rod_insertion.mdp.geometry import (
  contact_force_magnitude,
  dropped_rod_from_geometry,
  excessive_contact_force_from_magnitude,
  holder_displaced_from_error,
  holder_pose_error,
  jammed_from_signals,
  lateral_force_magnitude,
  rod_hole_geometry,
  success_ready_from_geometry,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Return raw contact-force tensor from a configured contact sensor."""
  sensor = env.scene.sensors[sensor_name]
  force = sensor.data.force
  if force is None:
    raise ValueError(f"Contact sensor {sensor_name!r} does not expose force data.")
  return force


def contact_force_norm(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Return total contact force magnitude from a configured contact sensor."""
  return contact_force_magnitude(contact_force(env, sensor_name))


def lateral_contact_force_norm(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return contact force magnitude perpendicular to the holder hole axis."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return lateral_force_magnitude(contact_force(env, sensor_name), geometry.hole_axis)


def holder_displacement_errors(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return holder position and angular displacement from the reset pose."""
  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  default_pos = default_root_state[:, 0:3] + env.scene.env_origins
  default_quat = default_root_state[:, 3:7]
  return holder_pose_error(
    current_pos=asset.data.root_link_pos_w,
    current_quat=asset.data.root_link_quat_w,
    default_pos=default_pos,
    default_quat=default_quat,
  )


def holder_displacement_norm(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return holder translational displacement from the reset pose."""
  position_error, _ = holder_displacement_errors(env, asset_cfg)
  return position_error


def holder_displacement_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  position_scale: float = 0.005,
  angle_scale: float = 0.10,
) -> torch.Tensor:
  """Return normalized holder displacement cost."""
  position_error, angle_error = holder_displacement_errors(env, asset_cfg)
  position_cost = position_error / position_scale
  angle_cost = angle_error / angle_scale
  return torch.maximum(position_cost, angle_cost)


def success_ready(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
  lateral_error_threshold: float = 0.0035,
  axis_angle_threshold: float = 0.15,
  insertion_depth_threshold: float = 0.004,
) -> torch.Tensor:
  """Return geometric success readiness before applying the stability window."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return success_ready_from_geometry(
    geometry,
    lateral_error_threshold=lateral_error_threshold,
    axis_angle_threshold=axis_angle_threshold,
    insertion_depth_threshold=insertion_depth_threshold,
  )


def dropped_rod(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
  max_tip_to_hole_distance: float = 0.12,
) -> torch.Tensor:
  """Return environments where the rod tip is too far from the target hole."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return dropped_rod_from_geometry(
    geometry,
    max_tip_to_hole_distance=max_tip_to_hole_distance,
  )


def holder_displaced(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  position_threshold: float = 0.005,
  angle_threshold: float = 0.10,
) -> torch.Tensor:
  """Return environments where the holder moved beyond allowed thresholds."""
  position_error, angle_error = holder_displacement_errors(env, asset_cfg)
  return holder_displaced_from_error(
    position_error,
    angle_error,
    position_threshold=position_threshold,
    angle_threshold=angle_threshold,
  )


def excessive_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 25.0,
) -> torch.Tensor:
  """Return environments exceeding the allowed contact-force threshold."""
  return excessive_contact_force_from_magnitude(
    contact_force_norm(env, sensor_name),
    force_threshold=force_threshold,
  )


def jammed(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
  lateral_force_threshold: float = 12.0,
  lateral_error_threshold: float = 0.005,
  insertion_depth_threshold: float = 0.004,
  insertion_depth_margin: float = 0.002,
) -> torch.Tensor:
  """Return environments where contact suggests sideways jamming."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  lateral_force = lateral_force_magnitude(
    contact_force(env, sensor_name),
    geometry.hole_axis,
  )
  return jammed_from_signals(
    lateral_force,
    geometry.lateral_error,
    geometry.insertion_depth,
    lateral_force_threshold=lateral_force_threshold,
    lateral_error_threshold=lateral_error_threshold,
    insertion_depth_threshold=insertion_depth_threshold,
    insertion_depth_margin=insertion_depth_margin,
  )


def self_collision(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Return environments where the configured self-collision sensor is active."""
  sensor = env.scene.sensors[sensor_name]
  found = sensor.data.found
  if found is None:
    raise ValueError(f"Contact sensor {sensor_name!r} does not expose found data.")
  if found.ndim == 1:
    return found > 0.0
  return torch.any(found > 0.0, dim=1)
