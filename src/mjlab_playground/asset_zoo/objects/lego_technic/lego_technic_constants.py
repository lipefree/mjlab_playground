"""LEGO Technic primitive object asset constants."""

from pathlib import Path

import mujoco

LEGO_TECHNIC_XML_DIR: Path = Path(__file__).parent / "xmls"
LEGO_TECHNIC_AXLE_4L_XML: Path = LEGO_TECHNIC_XML_DIR / "axle_4l.xml"
LEGO_TECHNIC_HOLDER_32064_XML: Path = LEGO_TECHNIC_XML_DIR / "holder_32064.xml"

assert LEGO_TECHNIC_AXLE_4L_XML.exists()
assert LEGO_TECHNIC_HOLDER_32064_XML.exists()


def get_axle_4l_spec() -> mujoco.MjSpec:
  """Get the primitive LEGO Technic Axle 4L spec."""
  return mujoco.MjSpec.from_file(str(LEGO_TECHNIC_AXLE_4L_XML))


def get_holder_32064_spec() -> mujoco.MjSpec:
  """Get the primitive LEGO Technic 32064 holder spec."""
  return mujoco.MjSpec.from_file(str(LEGO_TECHNIC_HOLDER_32064_XML))

