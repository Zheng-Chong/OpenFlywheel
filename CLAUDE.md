# OpenFlywheel — Project Guide

See `README.md` for setup and architecture, and `AGENTS.md` for contributor conventions.

## Core Modules

- `env/` & `env_cfg/`: Simulation environments, robot configs (`g1d_school.yml`), and backends (`isaacsim`, `mujoco`).
- `real2sim/scenes/real_scene4_school/`: School lab tabletop scene.
- `real2sim/objects/`: Curated Rigid, Clutter, and Garment assets for tabletop manipulation.
- `task/RoboDojo/tasks/`: Task logic and reward evaluators.
- `scripts/datagen/`: Scripted expert collection, replay, and LeRobot v3 export (`CONTRACT.md`).
- `src/eval_client/`: Closed-loop policy evaluation client.

## Checks

Run `bash scripts/check.sh` (linting via `ruff` + CPU unit tests via `pytest`).
