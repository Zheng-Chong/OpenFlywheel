# OpenFlywheel — Contributor & Agent Guide

This repository is **OpenFlywheel**, an open-source Real2Sim simulation, data generation, and policy evaluation codebase for academic research and teaching (built on RoboDojo + Isaac Sim / MuJoCo).

## Key Conventions

- **School lab scene**: `real2sim/scenes/real_scene4_school/` (`env_cfg/g1d_school.yml`) is the primary workspace scene.
- **Upstream compatibility paths**: `task/RoboDojo/`, `Assets/Object/RoboDojo/`, and `Assets/Eval_Layout/RoboDojo/` are preserved for compatibility with upstream RoboDojo layouts and tasks.
- **Data contract**: G1-D trajectories follow `scripts/datagen/CONTRACT.md` (30 Hz, 16-dim joint + gripper actions `[L7, LG, R7, RG]`).
- **Simulation backends**: Default is `isaacsim` (Isaac Sim 5.1 + PhysX); CPU `mujoco` backend is available for rigid-body debugging (`env/backends/README.md`).

## Verification

Before committing changes, run the CPU checks:

```bash
bash scripts/check.sh
```
