# Changelog

All notable changes to **OpenFlywheel** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), using calendar release tags (`vYYYY.MM.DD`).

## [v2026.11.27] - 2026-11-27

### Added
- **Object Affordance Pipeline (M1–M3)**:
  - Offline Stage 1 geometric antipodal grasp sampling, Stage 2 Isaac Sim physical verification, and Stage 3 multi-view VLM/3D part semantic tagging (`scripts/objects/affordance.py`, `affordance_vlm.py`, `verify_affordance_sim.py`).
  - Downstream `part_tag`-aware grasp selection and task planners (`scripts/g1d/affordance_grasp.py`, `scripts/datagen/plugins/g1d_{pick,place}_anything_affordance.py`).
- **Sharpa Wave Dexterous Hand Affordance & Wholebody Pipelines**:
  - 22-DoF/hand multi-primitive grasp annotation, tactile closed-loop verification, and online planner (`scripts/objects/affordance_dexhand.py`, `verify_affordance_dexhand_sim.py`, `scripts/g1d/affordance_dexhand.py`, `g1d_pick_anything_sharpa_wave.py`).
  - 61-DoF fixed-base wholebody Sharpa Wave pour-beans and X-Anything experts, plugins, and configs (`scripts/dexhands/{pour_beans,x_anything}_sharpa.py`, `env_cfg/g1d_sharpa_{beans,x_anything}_wb.yml`).
- **19-DoF Fixed-Base Wholebody (`g1d_wb19`) Support**:
  - Decoupled waist-pitch + torso-lift wholebody IK, rate limiting, and X-Anything experts (`scripts/g1d/wholebody_demo.py`, `x_anything_wb.py`, `env_cfg/g1d_wb.yml`, `env_cfg/g1d_beans_rigid_wb.yml`, `tests/test_g1d_wholebody.py`) with 19D LeRobot v3 export support.
- **Opt-In Depth & 2D Affordance Vision Recording**:
  - Zero-overhead-when-off depth, camera intrinsics/extrinsics, and occlusion-aware 2D affordance projection in `ObsManager`, `TiledCaptureManager`, `env/observation_manager/vision_utils.py`, `scripts/datagen/collect.py`, and `export.py`.
- **Expanded Public Commercial Asset Adapters**:
  - Added adapters and CPU unit tests for Poly Haven CC0, MolmoSpaces (THOR & Objaverse), Palatial, USDCraft-10K, ProjectSim, SimReady upstream, and Video2Data (`scripts/objects/`), plus expanded bilingual captions in `object_names.json`.
- **Data Generation & Quality Utilities**:
  - Trajectory twitch diagnostics and export filtering (`scripts/datagen/twitch.py`), `flock`-backed dynamic layout pool (`layout_pool.py`), v4.4 `PATTERN_SETTLE` stall-break experts (`place_v44.py`, `sort_v44.py`), and `place_anything_release.py`.

### Changed
- Optimized simulation startup (`setup_scene()` pre-render reduction), rejected-episode storage (skip full image HDF5), `pick_anything_layouts.py` round-based exclusion (`round_pool()`), and AV1 export quality (`preset=8` with 16×16 block residual gate).
- Updated `scripts/check.sh` to cover all newly added CPU test suites with `--import-mode=importlib`.
- Removed obsolete non-school-lab configuration and unused test files.

## [v2026.11.20] - 2026-11-20

### Added
- Initial public release of **OpenFlywheel** (school lab edition), built on RoboDojo and NVIDIA Isaac Sim with optional CPU MuJoCo backend.
- Unitree G1-D bimanual manipulation configs (`env_cfg/g1d_school.yml`), calibrated school lab scene (`real2sim/scenes/real_scene4_school/`), curated basic object set, scripted experts, LeRobot v3 export, and closed-loop evaluation client.
