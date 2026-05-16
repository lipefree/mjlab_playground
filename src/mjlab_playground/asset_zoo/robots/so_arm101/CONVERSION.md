# SO-101 / SO-ARM101 MJCF Asset Notes

This asset is derived from the RobotStudio SO-101 URDF:

- Source URDF: `Simulation/SO101/so101_new_calib.urdf`
- Source meshes: `Simulation/SO101/assets/*.stl`
- Upstream repository: https://github.com/TheRobotStudio/SO-ARM100
- License: Apache-2.0

Only the STL files referenced by the SO-101 URDF are vendored in
`xmls/assets/`; the Onshape `.part` files are intentionally omitted.

## Cleanup Choices

- The MJCF follows the upstream `mjlab.asset_zoo` layout used by `i2rt_yam`:
  constants beside an `xmls/` folder, with mesh files under `xmls/assets/`.
- Joint names are kept from the RobotStudio URDF: `shoulder_pan`,
  `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, and `gripper`.
- Visual geometry uses the RobotStudio STL meshes for inspection in the viewer.
- Physical contact is intentionally limited to simplified fingertip box geoms:
  `fixed_fingertip_collision` and `moving_fingertip_collision`.
- The Menagerie `low_cost_robot_arm` model was used as a cleanup reference for
  MuJoCo defaults, scene setup, and gripper pad-style contact geometry.

## Validation Targets

- `mujoco.MjSpec.from_file("xmls/so_arm101.xml").compile()` succeeds.
- `mujoco.MjModel.from_xml_path("xmls/scene.xml")` succeeds for standalone
  viewer inspection.
- `mjlab.entity.Entity(get_so_arm101_robot_cfg()).spec.compile()` succeeds and
  creates readable position actuators for the six SO-101 joints.
