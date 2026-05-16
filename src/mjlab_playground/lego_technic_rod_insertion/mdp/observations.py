"""Observation terms for the LEGO Technic rod insertion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_playground.lego_technic_rod_insertion.mdp.geometry import (
  rod_hole_geometry,
  site_axis,
  site_midpoint,
  site_pos,
  site_pose,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def end_effector_pose(
  env: ManagerBasedRlEnv,
  ee_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """End-effector site pose in world frame as xyz + wxyz."""
  return site_pose(env, ee_site_cfg)


def rod_tip_position(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Rod tip position in world frame."""
  return site_pos(env, rod_tip_site_cfg)


def rod_axis(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Rod axis direction from tip to back in world frame."""
  return site_axis(env, rod_tip_site_cfg, rod_back_site_cfg)


def hole_center_position(
  env: ManagerBasedRlEnv,
  hole_center_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Hole center position in world frame."""
  return site_pos(env, hole_center_site_cfg)


def hole_axis(
  env: ManagerBasedRlEnv,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Hole axis direction from center to axis end in world frame."""
  return site_axis(env, hole_center_site_cfg, hole_axis_end_site_cfg)


def tip_to_hole(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Vector from rod tip to hole center in world frame."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return geometry.tip_to_hole


def lateral_error(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Rod-tip distance from the hole axis."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return geometry.lateral_error


def axis_angle_error(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Angle between rod axis and hole axis in radians."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return geometry.axis_angle_error


def insertion_depth(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Signed rod-tip depth along the hole axis."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return geometry.insertion_depth


def site_position_error(
  env: ManagerBasedRlEnv,
  source_site_cfg: SceneEntityCfg,
  target_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Vector from a source site to a target site in world frame."""
  return site_pos(env, target_site_cfg) - site_pos(env, source_site_cfg)


def rod_center_position_error(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  target_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Vector from the rod lengthwise center to a target site in world frame."""
  rod_center = site_midpoint(env, rod_tip_site_cfg, rod_back_site_cfg)
  return site_pos(env, target_site_cfg) - rod_center


def site_axis_alignment(
  env: ManagerBasedRlEnv,
  source_start_site_cfg: SceneEntityCfg,
  source_end_site_cfg: SceneEntityCfg,
  target_start_site_cfg: SceneEntityCfg,
  target_end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Cosine alignment between source and target site axes."""
  source_axis = site_axis(env, source_start_site_cfg, source_end_site_cfg)
  target_axis = site_axis(env, target_start_site_cfg, target_end_site_cfg)
  return torch.sum(source_axis * target_axis, dim=-1, keepdim=True)
