# LEGO Technic Primitive MJCF Asset Notes

These v0 assets are hand-authored MJCF primitives for the LEGO Technic
rod-in-hole benchmark. They intentionally do not vendor LDraw files or generated
meshes, and collision does not rely on mesh geometry.

## Source Dimensions

- LEGO Technic Axle 4L / design ID 3705, LDraw `parts/3705.dat`.
- LEGO Technic Brick 1 x 2 with Axle Hole / design ID 32064 family, LDraw
  `parts/32064a.dat` or `parts/32064c.dat`.
- LDraw units: `1 LDU = 0.4 mm`.
- Standard LEGO dimensions: `1 stud = 20 LDU = 8 mm`; `1 brick height = 24 LDU
  = 9.6 mm`.

## Cleanup Choices

- `axle_4l.xml` models the 4L axle as a single capsule along local Z, with
  length `32 mm` and collision radius `2.3 mm`. The radius is intentionally
  slightly undersized for stable v0 insertion behavior.
- `holder_32064.xml` models the fixed holder as a static box with dimensions
  `16 x 8 x 9.6 mm`. It does not cut a physical axle hole into the collision
  geometry.
- The holder uses local +Z as the abstract insertion axis for observations and
  success checks. `hole_center` marks the target origin and `hole_axis_end`
  marks the positive insertion-axis direction.
- Mesh visuals can be added later from LDraw conversion, but v0 runtime assets
  are primitive-only.

## Validation Targets

- `mujoco.MjModel.from_xml_path("xmls/axle_4l.xml")` succeeds.
- `mujoco.MjModel.from_xml_path("xmls/holder_32064.xml")` succeeds.
- The axle exposes `rod_tip` and `rod_back` sites.
- The holder exposes `hole_center` and `hole_axis_end` sites.

