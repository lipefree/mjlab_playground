"""Tests for primitive LEGO Technic object assets."""

from __future__ import annotations

import mujoco
from mjlab_playground.asset_zoo.objects.lego_technic import (
  LEGO_TECHNIC_AXLE_4L_XML,
  LEGO_TECHNIC_HOLDER_32064_XML,
)


def _geom_names(model: mujoco.MjModel) -> set[str]:
  return {model.geom(i).name for i in range(model.ngeom)}


def _site_names(model: mujoco.MjModel) -> set[str]:
  return {model.site(i).name for i in range(model.nsite)}


def _geom_id(model: mujoco.MjModel, name: str) -> int:
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def test_lego_technic_xml_files_exist() -> None:
  assert LEGO_TECHNIC_AXLE_4L_XML.exists()
  assert LEGO_TECHNIC_HOLDER_32064_XML.exists()


def test_axle_4l_xml_compiles() -> None:
  model = mujoco.MjModel.from_xml_path(str(LEGO_TECHNIC_AXLE_4L_XML))

  assert model.njnt == 1
  assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
  assert {"rod_tip", "rod_back"} <= _site_names(model)
  assert {"axle_4l_collision"} <= _geom_names(model)
  assert model.nmesh == 0
  assert all(
    geom_type != mujoco.mjtGeom.mjGEOM_MESH for geom_type in model.geom_type
  )


def test_holder_32064_xml_compiles() -> None:
  model = mujoco.MjModel.from_xml_path(str(LEGO_TECHNIC_HOLDER_32064_XML))

  collision_geoms = {
    "holder_32064_left_collision",
    "holder_32064_right_collision",
    "holder_32064_front_collision",
    "holder_32064_back_collision",
  }
  visual_geoms = {
    "holder_32064_left_visual",
    "holder_32064_right_visual",
    "holder_32064_front_visual",
    "holder_32064_back_visual",
  }

  assert model.njnt == 0
  assert {"hole_center", "hole_axis_end"} <= _site_names(model)
  assert "holder_32064_collision" not in _geom_names(model)
  assert collision_geoms | visual_geoms <= _geom_names(model)

  for geom_name in collision_geoms:
    geom_id = _geom_id(model, geom_name)
    assert geom_id >= 0
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom_contype[geom_id] == 1
    assert model.geom_conaffinity[geom_id] == 1

  for geom_name in visual_geoms:
    geom_id = _geom_id(model, geom_name)
    assert geom_id >= 0
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom_contype[geom_id] == 0
    assert model.geom_conaffinity[geom_id] == 0

  assert model.nmesh == 0
  assert all(
    geom_type != mujoco.mjtGeom.mjGEOM_MESH for geom_type in model.geom_type
  )
