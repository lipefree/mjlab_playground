"""Reward terms for the LEGO Technic rod insertion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_playground.lego_technic_rod_insertion.mdp.geometry import (
  rod_hole_geometry,
  site_midpoint,
  site_pos,
)
from mjlab_playground.lego_technic_rod_insertion.mdp.signals import (
  holder_displacement_cost,
  jammed,
  lateral_contact_force_norm,
  success_ready,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def rod_center_to_hole_reward(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  std: float = 0.02,
) -> torch.Tensor:
  """Dense reward for inserting the rod center into the holder hole center."""
  rod_center = site_midpoint(env, rod_tip_site_cfg, rod_back_site_cfg)
  distance = torch.linalg.norm(
    rod_center - site_pos(env, hole_center_site_cfg),
    dim=-1,
  )
  return torch.exp(-torch.square(distance / std))


def lateral_alignment_reward(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
  std: float = 0.0035,
) -> torch.Tensor:
  """Dense reward for keeping the rod tip near the holder hole axis."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return torch.exp(-torch.square(geometry.lateral_error.squeeze(-1) / std))


def rod_axis_alignment_reward(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward coaxial rod and holder-hole alignment."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return torch.sum(geometry.rod_axis * geometry.hole_axis, dim=-1).clamp(min=0.0)


def insertion_depth_progress_reward(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
  start_depth: float = -0.020,
  target_depth: float = 0.004,
) -> torch.Tensor:
  """Reward normalized rod-tip insertion progress along the hole axis."""
  geometry = rod_hole_geometry(
    env,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  progress = (geometry.insertion_depth.squeeze(-1) - start_depth) / (
    target_depth - start_depth
  )
  return progress.clamp(min=0.0, max=1.0)


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize excessive raw policy action magnitude."""
  return torch.sum(torch.square(env.action_manager.action), dim=1)


def lateral_contact_force_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
  force_scale: float = 12.0,
) -> torch.Tensor:
  """Penalize contact force perpendicular to the insertion axis."""
  lateral_force = lateral_contact_force_norm(
    env,
    sensor_name,
    rod_tip_site_cfg,
    rod_back_site_cfg,
    hole_center_site_cfg,
    hole_axis_end_site_cfg,
  )
  return torch.square(lateral_force / force_scale)


def holder_displacement_reward_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  position_scale: float = 0.005,
  angle_scale: float = 0.10,
) -> torch.Tensor:
  """Penalize holder displacement from its reset pose."""
  return holder_displacement_cost(
    env,
    asset_cfg,
    position_scale=position_scale,
    angle_scale=angle_scale,
  )


def jam_cost(
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
  """Penalize sideways jamming behavior."""
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


class stable_success_bonus:
  """Binary bonus once geometric success remains true for a short window."""

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
    return (self._stable_steps >= stable_steps).float()


class rod_inserted_metric:
  """Binary metric that latches once the rod center reaches the holder hole center."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._inserted = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._inserted[:] = 0.0
    else:
      self._inserted[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    rod_tip_site_cfg: SceneEntityCfg,
    rod_back_site_cfg: SceneEntityCfg,
    hole_center_site_cfg: SceneEntityCfg,
    distance_threshold: float = 0.004,
  ) -> torch.Tensor:
    rod_center = site_midpoint(env, rod_tip_site_cfg, rod_back_site_cfg)
    distance = torch.linalg.norm(
      rod_center - site_pos(env, hole_center_site_cfg),
      dim=-1,
    )
    inserted = (distance < distance_threshold).float()
    self._inserted = torch.maximum(self._inserted, inserted)
    return self._inserted
