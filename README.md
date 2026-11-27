# OpenFlywheel

**OpenFlywheel** is an open-source Real2Sim robot simulation, scripted data generation, and policy evaluation framework built for academic research and teaching.

Built on top of [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) and NVIDIA Isaac Sim (with optional CPU MuJoCo backend), OpenFlywheel provides:

- **Unitree G1-D** bimanual manipulation setup (`env_cfg/g1d_school.yml`, `env_cfg/g1d_wb.yml`) with Dex1-1 parallel grippers (16-DoF tabletop & 19-DoF fixed-base wholebody with waist/lift) and optional dexterous hand configurations (including 61-DoF Sharpa Wave wholebody pipelines).
- **School Laboratory Scene** (`real2sim/scenes/real_scene4_school/`): a calibrated tabletop workspace tailored for university lab experiments.
- **Object Affordance & Public Asset Adapters** (`scripts/objects/`): automated Stage 1–3 object affordance annotation (`affordance.py`, `affordance_vlm.py`, `affordance_dexhand.py`, `affordance_2d.py`), Isaac Sim physical grasp verification, and adapters for public datasets (RoboCasa, SimReady, GSO, Poly Haven, MolmoSpaces THOR/Objaverse, Palatial, USDCraft-10K, ProjectSim, ArtVIP, PartNet-Mobility, FMB, Sand3D).
- **Scripted Data Generation & Evaluation**: reproducible layout generation, scripted & affordance-guided experts (`scripts/g1d/`, `scripts/dexhands/`), opt-in depth/intrinsics/extrinsics & 2D affordance recording (`env/observation_manager/vision_utils.py`, `scripts/datagen/collect.py`), twitch motion filtering (`scripts/datagen/twitch.py`), dynamic layout pooling, HDF5/LeRobot v3 export (`scripts/datagen/export.py`), and closed-loop policy evaluation (`src/eval_client/`).

---

## Repository Structure

```text
OpenFlywheel/
├── env/                   # Simulation environment & backends (Isaac Sim / MuJoCo)
├── env_cfg/               # Robot, camera, sim, and scene configs (e.g. g1d_school.yml)
├── real2sim/
│   ├── scenes/
│   │   └── real_scene4_school/   # School lab green-screen tabletop scene
│   ├── objects/                  # Curated Rigid, Clutter, and Garment objects
│   ├── robot/                    # G1-D robot & end-effector descriptions
│   ├── grippers/                 # Gripper assets & specs
│   ├── embodiments/              # Embodiment metadata
│   └── tasks/                    # Task specs & benchmarks
├── task/RoboDojo/         # Task implementations & reward definitions
├── src/
│   ├── eval_client/       # Policy evaluation client & server interface
│   └── real2sim_deformables/     # Deformable object runtime helpers
├── scripts/
│   ├── datagen/           # Scripted expert collection, replay, and LeRobot export
│   ├── g1d/               # G1-D asset builder, scripted demos, and task utilities
│   ├── objects/           # Object asset adapters (blocks, cups, articulated, etc.)
│   ├── scenes/            # Scene building, camera overlay, and appearance tools
│   └── check.sh           # CPU unit tests & lint checks
├── tests/                 # Physics & contact unit tests
└── utils/                 # Shared utilities
```

---

## Quick Start

### 1. Environment Setup

Install NVIDIA Isaac Sim 5.1+ (or use the CPU MuJoCo backend for rigid-body debugging) and Python 3.11+:

```bash
pip install -r scripts/requirements.txt
# Optional: CPU MuJoCo backend
pip install -r scripts/requirements-mujoco.txt
```

Initialize submodules if you need `XPolicyLab` / `IsaacLab` / `curobo`:

```bash
git submodule update --init --recursive
```

### 2. School Lab Scene & G1-D Config

The default school lab environment configuration is [`env_cfg/g1d_school.yml`](env_cfg/g1d_school.yml) backed by [`real2sim/scenes/real_scene4_school/`](real2sim/scenes/real_scene4_school/).

- Robot: Unitree G1-D with Dex1-1 grippers (16-DoF action space: `[L7, LG, R7, RG]` at 30 Hz).
- Cameras: `cam_head`, `cam_left_wrist`, `cam_right_wrist`.
- Data contract: see [`scripts/datagen/CONTRACT.md`](scripts/datagen/CONTRACT.md).

### 3. Running CPU Checks

```bash
bash scripts/check.sh
```

### 4. Scripted Data Generation & Evaluation

See [`scripts/datagen/SKILL.md`](scripts/datagen/SKILL.md) and [`scripts/datagen/CONTRACT.md`](scripts/datagen/CONTRACT.md) for layout generation, scripted expert collection (`scripts/datagen/collect.py`), batch runs (`scripts/datagen/run_batch.py`), and LeRobot v3 export (`scripts/datagen/export.py`).

---

## Included Objects

`real2sim/objects/` includes a clean set of basic manipulation objects for lab experiments:

- **Rigid**: `basket`, `ceramic_cup`, `measuring_cup`, `plastic_box`, `plastic_cup`, `plastic_mug`, `red_bean`, `soda_can`, `wave_curve_cup`, `wooden_arch_block`
- **Clutter**: `metal_cup`, `paper_cup`, `plastic_bottle`
- **Garment**: `microfiber_towel`

---

## Changelog & Releases

See [CHANGELOG.md](CHANGELOG.md) and [GitHub Releases](https://github.com/Zheng-Chong/OpenFlywheel/releases) for version history and release notes.

---

## License & Acknowledgements

OpenFlywheel is released under the [MIT License](LICENSE).
Upstream simulation architecture and benchmark compatibility paths (`task/RoboDojo/`, `Assets/Object/RoboDojo/`, `Assets/Eval_Layout/RoboDojo/`) originate from [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo).
Third-party object libraries (e.g., RoboCasa, ArtVIP, PartNet-Mobility, FMB, Sand3D) remain subject to their respective upstream licenses.
