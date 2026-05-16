"""Termination terms for the LEGO Technic rod insertion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_playground.lego_technic_rod_insertion.mdp.geometry import (
  site_midpoint,
  site_pos,
)
from mjlab_playground.lego_technic_rod_insertion.mdp.signals import (
  dropped_rod,
  excessive_contact_force,
  holder_displaced,
  jammed,
  self_collision,
  success_ready,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def rod_inserted(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  distance_threshold: float = 0.004,
) -> torch.Tensor:
  """Terminate when the rod lengthwise center reaches the holder hole center."""
  rod_center = site_midpoint(env, rod_tip_site_cfg, rod_back_site_cfg)
  distance = torch.linalg.norm(
    rod_center - site_pos(env, hole_center_site_cfg),
    dim=-1,
  )
  return distance < distance_threshold


class stable_success:
  """Terminate once geometric success remains true for a short window."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._stable_steps = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._stable_steps[:] = 0
    else:
      self._stable_steps[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    rod_tip_site_cfg: SceneEntityCfg,
    rod_back_site_cfg: SceneEntityCfg,
    hole_center_site_cfg: SceneEntityCfg,
    hole_axis_end_site_cfg: SceneEntityCfg,
    lateral_error_threshold: float = 0.0035,
    axis_angle_threshold: float = 0.15,
    insertion_depth_threshold: float = 0.004,
    stable_steps: int = 5,
  ) -> torch.Tensor:
    ready = success_ready(
      env,
      rod_tip_site_cfg,
      rod_back_site_cfg,
      hole_center_site_cfg,
      hole_axis_end_site_cfg,
      lateral_error_threshold=lateral_error_threshold,
      axis_angle_threshold=axis_angle_threshold,
      insertion_depth_threshold=insertion_depth_threshold,
    )
    self._stable_steps = torch.where(
      ready,
      self._stable_steps + 1,
      torch.zeros_like(self._stable_steps),
    )
    return self._stable_steps >= stable_steps


def dropped_rod_termination(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
  max_tip_to_hole_distance: float = 0.12,
) -> torch.Tensor:
  """Terminate when the rod tip is too far from the target hole."""
  return dropped_rod(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
    max_tip_to_hole_distance=max_tip_to_hole_distance,
  )


def holder_displaced_termination(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  position_threshold: float = 0.005,
  angle_threshold: float = 0.10,
) -> torch.Tensor:
  """Terminate when the holder has shifted too far from its reset pose."""
  return holder_displaced(
    env,
    asset_cfg,
    position_threshold=position_threshold,
    angle_threshold=angle_threshold,
  )


def excessive_force_termination(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 25.0,
) -> torch.Tensor:
  """Terminate when holder-rod contact force exceeds the allowed threshold."""
  return excessive_contact_force(
    env,
    sensor_name,
    force_threshold=force_threshold,
  )


def jammed_termination(
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
  """Terminate when force and geometry indicate a jam."""
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
  )


def self_collision_termination(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Terminate when the robot self-collision sensor is active."""
  return self_collision(env, sensor_name)
