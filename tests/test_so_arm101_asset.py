"""Tests for the SO-101 asset-zoo entry."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.entity.entity import Entity
from mjlab_playground.asset_zoo.robots.so_arm101.so_arm101_constants import (
  SO_ARM101_ARTICULATION,
  SO_ARM101_SCENE_XML,
  SO_ARM101_XML,
  get_so_arm101_robot_cfg,
)

_EXPECTED_JOINTS = {
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
  "gripper",
}

_EXPECTED_MESHES = {
  "base_motor_holder_so101_v1.stl",
  "base_so101_v2.stl",
  "sts3215_03a_v1.stl",
  "waveshare_mounting_plate_so101_v2.stl",
  "motor_holder_so101_base_v1.stl",
  "rotation_pitch_so101_v1.stl",
  "upper_arm_so101_v1.stl",
  "under_arm_so101_v1.stl",
  "motor_holder_so101_wrist_v1.stl",
  "sts3215_03a_no_horn_v1.stl",
  "wrist_roll_pitch_so101_v2.stl",
  "wrist_roll_follower_so101_v1.stl",
  "moving_jaw_so101_v1.stl",
}


def _joint_names(model: mujoco.MjModel) -> set[str]:
  return {model.joint(i).name for i in range(model.njnt)}


def _actuator_names(model: mujoco.MjModel) -> set[str]:
  return {model.actuator(i).name for i in range(model.nu)}


def _geom_names(model: mujoco.MjModel) -> set[str]:
  return {model.geom(i).name for i in range(model.ngeom)}


def test_so_arm101_assets_exist() -> None:
  assert SO_ARM101_XML.exists()
  assert SO_ARM101_SCENE_XML.exists()

  asset_dir = Path(__file__).parents[1] / (
    "src/mjlab_playground/asset_zoo/robots/so_arm101/xmls/assets"
  )
  assert {path.name for path in asset_dir.glob("*.stl")} == _EXPECTED_MESHES
  assert not list(asset_dir.glob("*.part"))


def test_so_arm101_scene_xml_compiles() -> None:
  model = mujoco.MjModel.from_xml_path(str(SO_ARM101_SCENE_XML))

  assert _EXPECTED_JOINTS <= _joint_names(model)
  assert {"fixed_fingertip_collision", "moving_fingertip_collision"} <= _geom_names(
    model
  )


def test_so_arm101_entity_compiles_with_actuators() -> None:
  entity = Entity(get_so_arm101_robot_cfg())
  model = entity.spec.compile()

  expected_actuators = {
    name
    for actuator_cfg in SO_ARM101_ARTICULATION.actuators
    for name in actuator_cfg.target_names_expr
  }
  assert _EXPECTED_JOINTS <= _joint_names(model)
  assert expected_actuators <= _actuator_names(model)

  for geom_name in ("fixed_fingertip_collision", "moving_fingertip_collision"):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    assert geom_id >= 0
    assert model.geom_priority[geom_id] == 1
    assert model.geom_friction[geom_id, 0] == 1.0
