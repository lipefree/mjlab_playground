from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import so_arm101_lego_technic_rod_insertion_env_cfg
from .rl_cfg import so_arm101_lego_technic_rod_insertion_ppo_runner_cfg

register_mjlab_task(
  task_id="LegoTechnicRodInsertion-v0",
  env_cfg=so_arm101_lego_technic_rod_insertion_env_cfg(),
  play_env_cfg=so_arm101_lego_technic_rod_insertion_env_cfg(play=True),
  rl_cfg=so_arm101_lego_technic_rod_insertion_ppo_runner_cfg(),
)

