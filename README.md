# mjlab playground

A collection of tasks built with [mjlab](https://github.com/mujocolab/mjlab), starting with ports from [MuJoCo Playground](https://playground.mujoco.org/).

## Tasks

| Task ID | Robot | Description | Preview |
|---------|-------|-------------|---------|
| **Getup** | | | |
| `Mjlab-Getup-Flat-Unitree-Go1` | Unitree Go1 | Fall recovery on flat terrain | <img src="https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/go1_getup_teaser.gif" width="200"/> |
| `Mjlab-Getup-Flat-Booster-T1` | Booster T1 | Fall recovery on flat terrain | <img src="https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/t1_getup_teaser.gif" width="200"/> |
| **Manipulation** | | | |
| `LegoTechnicRodInsertion-v0` | SO-101 | State-based LEGO Technic axle insertion | |

## Getting Started

```bash
git clone https://github.com/mujocolab/mjlab_playground.git && cd mjlab_playground
uv sync
```

Train a task:

```bash
uv run train <task-id> --env.scene.num-envs 4096
```

Play back a trained policy:

```bash
uv run play <task-id>
```

### Getup training

On a single NVIDIA 5090, the Go1 getup task converges in ~2 minutes and T1 in ~8 minutes, but we continue training with a curriculum that progressively tightens action rate, joint velocity, and power penalties to produce smoother, safer policies.

<p align="center">
  <img src="https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/training_curves.png" width="80%"/>
</p>

### LEGO Technic rod insertion training

`LegoTechnicRodInsertion-v0` registers a state-based PPO baseline. The default
training environment uses 4096 parallel environments through
`env.scene.num_envs`, and the rollout horizon is configured separately as
`agent.num_steps_per_env=64`. Override either from the CLI for local smoke runs:

```bash
uv run train LegoTechnicRodInsertion-v0 \
  --env.scene.num-envs 4 \
  --agent.max-iterations 1 \
  --agent.num-steps-per-env 32 \
  --agent.logger tensorboard \
  --agent.upload-model False
```

The baseline uses privileged state observations for actor and critic, an
end-effector differential IK delta-pose action, ELU MLPs with hidden dimensions
`(256, 256, 128)`, observation normalization, and RSL-RL's full-rollout
advantage normalization. mjlab logs jam and force penalties as reward terms
(`Episode_Reward/jam` and `Episode_Reward/lateral_contact_force`), alongside
episode metrics such as `success_rate`, `insertion_depth`, `lateral_error`,
`axis_angle_error`, `contact_force`, and `lateral_force`.

## Citation

If you use this repository in your research, consider citing mjlab:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

This repository is released under an [Apache-2.0 License](LICENSE).
