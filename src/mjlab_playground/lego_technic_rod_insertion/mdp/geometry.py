"""Reusable rod-hole geometry helpers for LEGO Technic insertion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class RodHoleGeometry:
  """Batched rod and hole geometry in world frame."""

  rod_tip_pos: torch.Tensor
  rod_axis: torch.Tensor
  rod_center_pos: torch.Tensor
  hole_center_pos: torch.Tensor
  hole_axis: torch.Tensor
  tip_to_hole: torch.Tensor
  lateral_error: torch.Tensor
  axis_angle_error: torch.Tensor
  insertion_depth: torch.Tensor


def success_ready_from_geometry(
  geometry: RodHoleGeometry,
  lateral_error_threshold: float = 0.0035,
  axis_angle_threshold: float = 0.15,
  insertion_depth_threshold: float = 0.004,
) -> torch.Tensor:
  """Return environments whose rod pose satisfies the geometric success gate."""
  return (
    (geometry.lateral_error.squeeze(-1) <= lateral_error_threshold)
    & (geometry.axis_angle_error.squeeze(-1) <= axis_angle_threshold)
    & (geometry.insertion_depth.squeeze(-1) >= insertion_depth_threshold)
  )


def dropped_rod_from_geometry(
  geometry: RodHoleGeometry,
  max_tip_to_hole_distance: float = 0.12,
) -> torch.Tensor:
  """Return environments where the rod tip is too far from the target hole."""
  tip_to_hole_distance = torch.linalg.norm(geometry.tip_to_hole, dim=-1)
  return tip_to_hole_distance > max_tip_to_hole_distance


def contact_force_vector(contact_force: torch.Tensor) -> torch.Tensor:
  """Collapse per-contact-slot forces to one force vector per environment."""
  if contact_force.ndim == 3:
    return torch.sum(contact_force, dim=1)
  if contact_force.ndim == 2:
    return contact_force
  raise ValueError(
    "Expected contact force tensor with shape [num_envs, 3] or "
    "[num_envs, num_slots, 3]."
  )


def contact_force_magnitude(contact_force: torch.Tensor) -> torch.Tensor:
  """Return total contact force magnitude per environment."""
  return torch.linalg.norm(contact_force_vector(contact_force), dim=-1)


def lateral_force_magnitude(
  contact_force: torch.Tensor,
  axis: torch.Tensor,
) -> torch.Tensor:
  """Return contact force magnitude perpendicular to the provided axis."""
  force = contact_force_vector(contact_force)
  axis = torch.nn.functional.normalize(axis, dim=-1)
  axial_force = torch.sum(force * axis, dim=-1, keepdim=True) * axis
  return torch.linalg.norm(force - axial_force, dim=-1)


def excessive_contact_force_from_magnitude(
  force_magnitude: torch.Tensor,
  force_threshold: float = 25.0,
) -> torch.Tensor:
  """Return environments exceeding the allowed contact force."""
  return force_magnitude > force_threshold


def jammed_from_signals(
  lateral_force: torch.Tensor,
  lateral_error: torch.Tensor,
  insertion_depth: torch.Tensor,
  lateral_force_threshold: float = 12.0,
  lateral_error_threshold: float = 0.005,
  insertion_depth_threshold: float = 0.004,
  insertion_depth_margin: float = 0.002,
) -> torch.Tensor:
  """Return environments exhibiting sideways ramming without enough depth."""
  return (
    (lateral_force > lateral_force_threshold)
    & (lateral_error.squeeze(-1) > lateral_error_threshold)
    & (
      insertion_depth.squeeze(-1)
      < insertion_depth_threshold - insertion_depth_margin
    )
  )


def holder_pose_error(
  current_pos: torch.Tensor,
  current_quat: torch.Tensor,
  default_pos: torch.Tensor,
  default_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return holder position and angular displacement from its reset pose."""
  position_error = torch.linalg.norm(current_pos - default_pos, dim=-1)
  current_quat = torch.nn.functional.normalize(current_quat, dim=-1)
  default_quat = torch.nn.functional.normalize(default_quat, dim=-1)
  quat_dot = torch.abs(torch.sum(current_quat * default_quat, dim=-1))
  angle_error = 2.0 * torch.acos(quat_dot.clamp(min=-1.0, max=1.0))
  return position_error, angle_error


def holder_displaced_from_error(
  position_error: torch.Tensor,
  angle_error: torch.Tensor,
  position_threshold: float = 0.005,
  angle_threshold: float = 0.10,
) -> torch.Tensor:
  """Return environments where the holder moved beyond allowed thresholds."""
  return (position_error > position_threshold) | (angle_error > angle_threshold)


def site_pos(env: ManagerBasedRlEnv, site_cfg: SceneEntityCfg) -> torch.Tensor:
  """Return a single resolved site position in world frame."""
  entity: Entity = env.scene[site_cfg.name]
  assert isinstance(site_cfg.site_ids, list)
  assert len(site_cfg.site_ids) == 1
  return entity.data.site_pos_w[:, site_cfg.site_ids[0]]


def site_pose(env: ManagerBasedRlEnv, site_cfg: SceneEntityCfg) -> torch.Tensor:
  """Return a single resolved site pose in world frame as xyz + wxyz."""
  entity: Entity = env.scene[site_cfg.name]
  assert isinstance(site_cfg.site_ids, list)
  assert len(site_cfg.site_ids) == 1
  return entity.data.site_pose_w[:, site_cfg.site_ids[0]]


def site_axis(
  env: ManagerBasedRlEnv,
  start_site_cfg: SceneEntityCfg,
  end_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return the normalized axis from a start site to an end site."""
  axis = site_pos(env, end_site_cfg) - site_pos(env, start_site_cfg)
  return torch.nn.functional.normalize(axis, dim=-1)


def site_midpoint(
  env: ManagerBasedRlEnv,
  first_site_cfg: SceneEntityCfg,
  second_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return the midpoint between two resolved sites."""
  return 0.5 * (site_pos(env, first_site_cfg) + site_pos(env, second_site_cfg))


def compute_rod_hole_geometry(
  rod_tip_pos: torch.Tensor,
  rod_back_pos: torch.Tensor,
  hole_center_pos: torch.Tensor,
  hole_axis_end_pos: torch.Tensor,
) -> RodHoleGeometry:
  """Compute insertion geometry from batched world-frame keypoints."""
  rod_axis = torch.nn.functional.normalize(rod_back_pos - rod_tip_pos, dim=-1)
  rod_center_pos = 0.5 * (rod_tip_pos + rod_back_pos)
  hole_axis = torch.nn.functional.normalize(
    hole_axis_end_pos - hole_center_pos, dim=-1
  )
  tip_to_hole = hole_center_pos - rod_tip_pos
  tip_from_hole = rod_tip_pos - hole_center_pos
  insertion_depth = torch.sum(tip_from_hole * hole_axis, dim=-1, keepdim=True)
  lateral_vector = tip_from_hole - insertion_depth * hole_axis
  lateral_error = torch.linalg.norm(lateral_vector, dim=-1, keepdim=True)
  axis_alignment = torch.sum(rod_axis * hole_axis, dim=-1, keepdim=True)
  axis_angle_error = torch.acos(axis_alignment.clamp(min=-1.0, max=1.0))

  return RodHoleGeometry(
    rod_tip_pos=rod_tip_pos,
    rod_axis=rod_axis,
    rod_center_pos=rod_center_pos,
    hole_center_pos=hole_center_pos,
    hole_axis=hole_axis,
    tip_to_hole=tip_to_hole,
    lateral_error=lateral_error,
    axis_angle_error=axis_angle_error,
    insertion_depth=insertion_depth,
  )


def rod_hole_geometry(
  env: ManagerBasedRlEnv,
  rod_tip_site_cfg: SceneEntityCfg,
  rod_back_site_cfg: SceneEntityCfg,
  hole_center_site_cfg: SceneEntityCfg,
  hole_axis_end_site_cfg: SceneEntityCfg,
) -> RodHoleGeometry:
  """Compute insertion geometry from resolved environment sites."""
  return compute_rod_hole_geometry(
    rod_tip_pos=site_pos(env, rod_tip_site_cfg),
    rod_back_pos=site_pos(env, rod_back_site_cfg),
    hole_center_pos=site_pos(env, hole_center_site_cfg),
    hole_axis_end_pos=site_pos(env, hole_axis_end_site_cfg),
  )
