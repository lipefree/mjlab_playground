"""Metrics for LEGO Technic rod insertion diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_playground.lego_technic_rod_insertion.mdp.geometry import rod_hole_geometry
from mjlab_playground.lego_technic_rod_insertion.mdp.signals import (
  contact_force_norm,
  dropped_rod,
  excessive_contact_force,
  holder_displacement_norm,
  jammed,
  lateral_contact_force_norm,
  self_collision,
  success_ready,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def lateral_error_metric(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Current lateral rod-tip error from the holder hole axis."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return geometry.lateral_error.squeeze(-1)


def axis_angle_error_metric(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Current rod-axis angle error from the holder hole axis."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return geometry.axis_angle_error.squeeze(-1)


def insertion_depth_metric(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Current signed rod-tip insertion depth along the holder hole axis."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return geometry.insertion_depth.squeeze(-1)


def contact_force_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Current holder-rod contact-force magnitude."""
  return contact_force_norm(env, sensor_name)


def lateral_force_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Current holder-rod lateral-force magnitude."""
  return lateral_contact_force_norm(
    env,
    sensor_name,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )


def holder_displacement_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Current holder translational displacement from reset pose."""
  return holder_displacement_norm(env, asset_cfg)


def success_ready_metric(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
  lateral_error_threshold: float = 0.0035,
  axis_angle_threshold: float = 0.15,
  insertion_depth_threshold: float = 0.004,
) -> torch.Tensor:
  """Whether the current rod pose satisfies the geometric success gate."""
  return success_ready(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
    lateral_error_threshold=lateral_error_threshold,
    axis_angle_threshold=axis_angle_threshold,
    insertion_depth_threshold=insertion_depth_threshold,
  ).float()


def stable_success_metric(
  env: ManagerBasedRlEnv,
  termination_name: str = "success",
) -> torch.Tensor:
  """Whether the stable-success termination fired on the current step."""
  return env.termination_manager.get_term(termination_name).float()


def dropped_metric(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
  max_tip_to_hole_distance: float = 0.12,
) -> torch.Tensor:
  """Whether the rod is considered dropped or lost."""
  return dropped_rod(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
    max_tip_to_hole_distance=max_tip_to_hole_distance,
  ).float()


def jammed_metric(
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
  """Whether force and geometry indicate a jam."""
  return jammed(
    env,
    sensor_name,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
    lateral_force_threshold=lateral_force_threshold,
    lateral_error_threshold=lateral_error_threshold,
    insertion_depth_threshold=insertion_depth_threshold,
    insertion_depth_margin=insertion_depth_margin,
  ).float()


def excessive_force_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 25.0,
) -> torch.Tensor:
  """Whether holder-rod contact force exceeds the allowed threshold."""
  return excessive_contact_force(
    env,
    sensor_name,
    force_threshold=force_threshold,
  ).float()


def self_collision_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Whether the robot self-collision sensor is active."""
  return self_collision(env, sensor_name).float()
