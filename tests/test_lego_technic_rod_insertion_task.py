"""Tests for the LEGO Technic rod insertion task scaffold."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab_playground.lego_technic_rod_insertion.config.so_arm101.env_cfgs import (
  so_arm101_lego_technic_rod_insertion_env_cfg,
)
from mjlab_playground.lego_technic_rod_insertion.config.so_arm101.rl_cfg import (
  so_arm101_lego_technic_rod_insertion_ppo_runner_cfg,
)
from mjlab_playground.lego_technic_rod_insertion.mdp import (
  terminations as insertion_terminations,
)
from mjlab_playground.lego_technic_rod_insertion.mdp.geometry import (
  compute_rod_hole_geometry,
  contact_force_magnitude,
  dropped_rod_from_geometry,
  excessive_contact_force_from_magnitude,
  holder_displaced_from_error,
  jammed_from_signals,
  lateral_force_magnitude,
  success_ready_from_geometry,
)


def test_rod_hole_geometry_aligned_known_pose() -> None:
  geometry = compute_rod_hole_geometry(
    rod_tip_pos=torch.tensor([[0.0, 0.0, -0.25]]),
    rod_back_pos=torch.tensor([[0.0, 0.0, 0.75]]),
    hole_center_pos=torch.tensor([[0.0, 0.0, 0.0]]),
    hole_axis_end_pos=torch.tensor([[0.0, 0.0, 1.0]]),
  )

  torch.testing.assert_close(geometry.rod_axis, torch.tensor([[0.0, 0.0, 1.0]]))
  torch.testing.assert_close(geometry.hole_axis, torch.tensor([[0.0, 0.0, 1.0]]))
  torch.testing.assert_close(
    geometry.tip_to_hole, torch.tensor([[0.0, 0.0, 0.25]])
  )
  torch.testing.assert_close(geometry.lateral_error, torch.tensor([[0.0]]))
  torch.testing.assert_close(geometry.axis_angle_error, torch.tensor([[0.0]]))
  torch.testing.assert_close(geometry.insertion_depth, torch.tensor([[-0.25]]))


def test_rod_hole_geometry_lateral_depth_and_angle_known_poses() -> None:
  lateral_geometry = compute_rod_hole_geometry(
    rod_tip_pos=torch.tensor([[0.003, 0.004, 0.2]]),
    rod_back_pos=torch.tensor([[0.003, 0.004, 1.2]]),
    hole_center_pos=torch.tensor([[0.0, 0.0, 0.0]]),
    hole_axis_end_pos=torch.tensor([[0.0, 0.0, 1.0]]),
  )
  torch.testing.assert_close(lateral_geometry.lateral_error, torch.tensor([[0.005]]))
  torch.testing.assert_close(lateral_geometry.insertion_depth, torch.tensor([[0.2]]))

  perpendicular_geometry = compute_rod_hole_geometry(
    rod_tip_pos=torch.tensor([[0.0, 0.0, 0.0]]),
    rod_back_pos=torch.tensor([[1.0, 0.0, 0.0]]),
    hole_center_pos=torch.tensor([[0.0, 0.0, 0.0]]),
    hole_axis_end_pos=torch.tensor([[0.0, 0.0, 1.0]]),
  )
  torch.testing.assert_close(
    perpendicular_geometry.axis_angle_error,
    torch.tensor([[math.pi / 2.0]]),
  )


def test_rod_hole_geometry_batched_shapes() -> None:
  geometry = compute_rod_hole_geometry(
    rod_tip_pos=torch.tensor([[0.0, 0.0, -0.1], [0.01, 0.0, 0.2]]),
    rod_back_pos=torch.tensor([[0.0, 0.0, 0.9], [0.01, 0.0, 1.2]]),
    hole_center_pos=torch.zeros(2, 3),
    hole_axis_end_pos=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
  )

  assert geometry.rod_tip_pos.shape == (2, 3)
  assert geometry.rod_axis.shape == (2, 3)
  assert geometry.rod_center_pos.shape == (2, 3)
  assert geometry.hole_center_pos.shape == (2, 3)
  assert geometry.hole_axis.shape == (2, 3)
  assert geometry.tip_to_hole.shape == (2, 3)
  assert geometry.lateral_error.shape == (2, 1)
  assert geometry.axis_angle_error.shape == (2, 1)
  assert geometry.insertion_depth.shape == (2, 1)


def test_success_gate_checks_lateral_angle_depth() -> None:
  success_geometry = compute_rod_hole_geometry(
    rod_tip_pos=torch.tensor([[0.0, 0.0, 0.004]]),
    rod_back_pos=torch.tensor([[0.0, 0.0, 0.036]]),
    hole_center_pos=torch.zeros(1, 3),
    hole_axis_end_pos=torch.tensor([[0.0, 0.0, 1.0]]),
  )
  assert success_ready_from_geometry(success_geometry).tolist() == [True]

  lateral_geometry = compute_rod_hole_geometry(
    rod_tip_pos=torch.tensor([[0.004, 0.0, 0.004]]),
    rod_back_pos=torch.tensor([[0.004, 0.0, 0.036]]),
    hole_center_pos=torch.zeros(1, 3),
    hole_axis_end_pos=torch.tensor([[0.0, 0.0, 1.0]]),
  )
  assert success_ready_from_geometry(lateral_geometry).tolist() == [False]

  angled_geometry = compute_rod_hole_geometry(
    rod_tip_pos=torch.tensor([[0.0, 0.0, 0.004]]),
    rod_back_pos=torch.tensor([[0.008, 0.0, 0.036]]),
    hole_center_pos=torch.zeros(1, 3),
    hole_axis_end_pos=torch.tensor([[0.0, 0.0, 1.0]]),
  )
  assert success_ready_from_geometry(angled_geometry).tolist() == [False]

  shallow_geometry = compute_rod_hole_geometry(
    rod_tip_pos=torch.tensor([[0.0, 0.0, 0.003]]),
    rod_back_pos=torch.tensor([[0.0, 0.0, 0.035]]),
    hole_center_pos=torch.zeros(1, 3),
    hole_axis_end_pos=torch.tensor([[0.0, 0.0, 1.0]]),
  )
  assert success_ready_from_geometry(shallow_geometry).tolist() == [False]


def test_stable_success_requires_consecutive_window(monkeypatch) -> None:
  env = cast(ManagerBasedRlEnv, SimpleNamespace(num_envs=1, device="cpu"))
  dummy_site = cast(Any, None)
  term = insertion_terminations.stable_success(None, env)

  ready = torch.tensor([True])

  def _fake_success_ready(*args, **kwargs):
    del args, kwargs
    return ready

  monkeypatch.setattr(insertion_terminations, "success_ready", _fake_success_ready)
  for _ in range(4):
    assert not term(
      env, dummy_site, dummy_site, dummy_site, dummy_site, stable_steps=5
    ).item()
  assert term(
    env, dummy_site, dummy_site, dummy_site, dummy_site, stable_steps=5
  ).item()

  ready = torch.tensor([False])
  assert not term(
    env, dummy_site, dummy_site, dummy_site, dummy_site, stable_steps=5
  ).item()


def test_force_projection_and_failure_predicates() -> None:
  contact_force = torch.tensor([[[3.0, 4.0, 5.0], [0.0, 0.0, 2.0]]])
  hole_axis = torch.tensor([[0.0, 0.0, 1.0]])

  torch.testing.assert_close(
    contact_force_magnitude(contact_force),
    torch.tensor([math.sqrt(74.0)]),
  )
  torch.testing.assert_close(
    lateral_force_magnitude(contact_force, hole_axis),
    torch.tensor([5.0]),
  )

  dropped_geometry = compute_rod_hole_geometry(
    rod_tip_pos=torch.tensor([[0.0, 0.0, -0.13]]),
    rod_back_pos=torch.tensor([[0.0, 0.0, -0.10]]),
    hole_center_pos=torch.zeros(1, 3),
    hole_axis_end_pos=torch.tensor([[0.0, 0.0, 1.0]]),
  )
  assert dropped_rod_from_geometry(dropped_geometry).tolist() == [True]

  assert excessive_contact_force_from_magnitude(
    torch.tensor([25.1]), force_threshold=25.0
  ).tolist() == [True]
  assert holder_displaced_from_error(
    torch.tensor([0.006]), torch.tensor([0.0])
  ).tolist() == [True]
  assert jammed_from_signals(
    lateral_force=torch.tensor([13.0]),
    lateral_error=torch.tensor([[0.006]]),
    insertion_depth=torch.tensor([[0.001]]),
  ).tolist() == [True]
  assert jammed_from_signals(
    lateral_force=torch.tensor([13.0]),
    lateral_error=torch.tensor([[0.006]]),
    insertion_depth=torch.tensor([[0.004]]),
  ).tolist() == [False]


def test_lego_technic_rod_insertion_task_registered() -> None:
  import mjlab_playground  # noqa: F401

  assert "LegoTechnicRodInsertion-v0" in list_tasks()


def test_lego_technic_rod_insertion_state_ppo_baseline_cfg() -> None:
  cfg = so_arm101_lego_technic_rod_insertion_ppo_runner_cfg()

  assert cfg.num_steps_per_env == 64
  assert cfg.actor.hidden_dims == (256, 256, 128)
  assert cfg.critic.hidden_dims == (256, 256, 128)
  assert cfg.actor.activation == "elu"
  assert cfg.critic.activation == "elu"
  assert cfg.actor.obs_normalization
  assert cfg.critic.obs_normalization
  assert cfg.algorithm.learning_rate == 3.0e-4
  assert cfg.algorithm.gamma == 0.99
  assert cfg.algorithm.lam == 0.95
  assert cfg.algorithm.clip_param == 0.2
  assert not cfg.algorithm.normalize_advantage_per_mini_batch


def test_lego_technic_rod_insertion_train_and_play_defaults_registered() -> None:
  import mjlab_playground  # noqa: F401

  train_env_cfg = load_env_cfg("LegoTechnicRodInsertion-v0")
  play_env_cfg = load_env_cfg("LegoTechnicRodInsertion-v0", play=True)
  rl_cfg = load_rl_cfg("LegoTechnicRodInsertion-v0")

  assert train_env_cfg.scene.num_envs == 4096
  assert play_env_cfg.scene.num_envs == 1
  assert rl_cfg.num_steps_per_env == 64


def test_lego_technic_rod_insertion_env_resets() -> None:
  cfg = so_arm101_lego_technic_rod_insertion_env_cfg()
  cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg, device="cpu")
  obs, _ = env.reset()

  assert "actor" in obs
  actor_obs = obs["actor"]
  assert isinstance(actor_obs, torch.Tensor)
  assert actor_obs.shape == (1, 43)
  critic_obs = obs["critic"]
  assert isinstance(critic_obs, torch.Tensor)
  assert critic_obs.shape == (1, 43)

  expected_terms = [
    "joint_pos",
    "joint_vel",
    "ee_pose",
    "rod_tip_pos",
    "rod_axis",
    "hole_center_pos",
    "hole_axis",
    "tip_to_hole",
    "lateral_error",
    "axis_angle_error",
    "insertion_depth",
    "actions",
  ]
  expected_dims = [
    (6,),
    (6,),
    (7,),
    (3,),
    (3,),
    (3,),
    (3,),
    (3,),
    (1,),
    (1,),
    (1,),
    (6,),
  ]
  assert env.observation_manager.active_terms["actor"] == expected_terms
  assert env.observation_manager.active_terms["critic"] == expected_terms
  assert env.observation_manager.group_obs_term_dim["actor"] == expected_dims
  assert env.observation_manager.group_obs_term_dim["critic"] == expected_dims
  assert env.observation_manager.group_obs_dim["actor"] == (43,)
  assert env.observation_manager.group_obs_dim["critic"] == (43,)

  assert env.reward_manager.active_terms == [
    "lateral_alignment",
    "rod_axis_alignment",
    "insertion_depth_progress",
    "success_bonus",
    "action_l2",
    "action_rate_l2",
    "lateral_contact_force",
    "holder_displacement",
    "jam",
  ]
  assert env.termination_manager.active_terms == [
    "time_out",
    "success",
    "dropped_rod",
    "holder_displaced",
    "excessive_force",
    "jammed",
    "self_collision",
  ]
  assert env.metrics_manager.active_terms == [
    "lateral_error",
    "axis_angle_error",
    "insertion_depth",
    "contact_force",
    "lateral_force",
    "holder_displacement",
    "success_ready",
    "stable_success",
    "success_rate",
    "dropped",
    "jammed",
    "excessive_force",
    "self_collision",
  ]

  holder_rod_contact = env.scene.sensors["holder_rod_contact"]
  holder_rod_data = cast(Any, holder_rod_contact.data)
  assert holder_rod_data.found.shape == (1, 4)
  assert holder_rod_data.force.shape == (1, 4, 3)
  robot_self_collision = env.scene.sensors["robot_self_collision"]
  robot_self_collision_data = cast(Any, robot_self_collision.data)
  assert robot_self_collision_data.found.shape == (1, 1)

  robot = env.scene["robot"]
  holder = env.scene["holder"]

  assert holder.is_fixed_base
  assert holder.is_mocap

  gripper_joint_ids, _ = robot.find_joints(("gripper",))
  assert len(gripper_joint_ids) == 1
  gripper_joint_id = gripper_joint_ids[0]
  expected_gripper_pos = torch.tensor(0.8, device=env.device)
  assert torch.isclose(
    robot.data.joint_pos[0, gripper_joint_id], expected_gripper_pos
  )
  assert torch.isclose(
    robot.data.joint_pos_target[0, gripper_joint_id], expected_gripper_pos
  )

  rod_tip_ids, _ = robot.find_sites(("grasped_rod_tip",))
  rod_back_ids, _ = robot.find_sites(("grasped_rod_back",))
  hole_center_ids, _ = holder.find_sites(("hole_center",))
  hole_axis_end_ids, _ = holder.find_sites(("hole_axis_end",))

  rod_tip = robot.data.site_pos_w[:, rod_tip_ids[0]]
  rod_back = robot.data.site_pos_w[:, rod_back_ids[0]]
  hole_center = holder.data.site_pos_w[:, hole_center_ids[0]]
  hole_axis_end = holder.data.site_pos_w[:, hole_axis_end_ids[0]]

  rod_center = 0.5 * (rod_tip + rod_back)
  center_to_hole = torch.linalg.norm(rod_center - hole_center, dim=-1)
  tip_to_hole = torch.linalg.norm(rod_tip - hole_center, dim=-1)
  assert torch.all(center_to_hole < 0.02)
  assert torch.all(tip_to_hole > center_to_hole)

  rod_axis = torch.nn.functional.normalize(rod_back - rod_tip, dim=-1)
  hole_axis = torch.nn.functional.normalize(hole_axis_end - hole_center, dim=-1)
  assert torch.all(torch.sum(rod_axis * hole_axis, dim=-1) > 0.99)
