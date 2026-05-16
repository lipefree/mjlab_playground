"""Reset events for the LEGO Technic rod insertion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot", joint_names=(".*",))


def reset_joints_to_default(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
  """Reset selected joints and hold their position targets at the default pose."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  joint_pos = default_joint_pos[env_ids][:, joint_ids].clone()
  joint_vel = default_joint_vel[env_ids][:, joint_ids].clone()

  asset.write_joint_state_to_sim(
    joint_pos,
    joint_vel,
    joint_ids=joint_ids,
    env_ids=env_ids,
  )
  asset.set_joint_position_target(joint_pos, joint_ids=joint_ids, env_ids=env_ids)

