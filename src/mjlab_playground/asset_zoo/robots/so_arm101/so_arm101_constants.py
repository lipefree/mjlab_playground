"""SO-101 / SO-ARM101 constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

SO_ARM101_XML: Path = Path(__file__).parent / "xmls" / "so_arm101.xml"
SO_ARM101_SCENE_XML: Path = Path(__file__).parent / "xmls" / "scene.xml"
assert SO_ARM101_XML.exists()
assert SO_ARM101_SCENE_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(SO_ARM101_XML))


##
# Actuator config.
##

# SO-101 uses STS3215 servos. These gains are intentionally conservative for a
# first asset baseline; task-specific tuning can tighten them later.
SO_ARM101_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=("shoulder_pan", "shoulder_lift", "elbow_flex"),
  stiffness=60.0,
  damping=6.0,
  effort_limit=10.0,
  armature=1.0e-3,
)

SO_ARM101_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=("wrist_flex", "wrist_roll"),
  stiffness=30.0,
  damping=3.0,
  effort_limit=10.0,
  armature=5.0e-4,
)

SO_ARM101_ACTUATOR_GRIPPER = BuiltinPositionActuatorCfg(
  target_names_expr=("gripper",),
  stiffness=10.0,
  damping=1.0,
  effort_limit=5.0,
  armature=1.0e-4,
)


##
# Keyframes.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.8,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

GRIPPER_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={".*_fingertip_collision": 6, ".*_collision": 3},
  friction={".*_fingertip_collision": (1.0, 5e-3, 5e-4), ".*_collision": (0.6,)},
  solref={".*_fingertip_collision": (0.004, 1)},
  priority={".*_collision": 1},
)


##
# Final config.
##

SO_ARM101_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    SO_ARM101_ACTUATOR_ARM,
    SO_ARM101_ACTUATOR_WRIST,
    SO_ARM101_ACTUATOR_GRIPPER,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_so_arm101_robot_cfg() -> EntityCfg:
  """Get a fresh SO-101 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(GRIPPER_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=SO_ARM101_ARTICULATION,
  )


SO_ARM101_ACTION_SCALE: dict[str, float] = {}
for a in SO_ARM101_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    SO_ARM101_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_so_arm101_robot_cfg())

  viewer.launch(robot.spec.compile())
