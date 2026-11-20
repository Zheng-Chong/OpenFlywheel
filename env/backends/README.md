# Simulation backends

The tasks, managers, layouts, reward checks and the eval client are the same on every backend; only the simulator
under them changes. The backend is chosen once per process (`env/sim_backend.py`):

1. `--sim_backend` of `src/eval_client/main.py` (and of `scripts/g1d/scripted_demo.py`, which runs it), or
   `bash scripts/robodojo.sh <command> --sim-backend <name> ...` for every robodojo command;
2. else `ROBODOJO_SIM_BACKEND`;
3. else `backend:` of the sim config (`env_cfg/sim/*.yml`);
4. else `isaacsim`.

| Backend | Simulator | Rendering | Runs on |
| :-- | :-- | :-- | :-- |
| `isaacsim` (default) | Isaac Sim 5.1 + IsaacLab 2.3, PhysX | RTX | the reference: benchmark scores, datasets |
| `isaaclab_newton` | Isaac Sim 6.2 + IsaacLab 3.0, Newton 1.6 with the MuJoCo-Warp solver | RTX | an Isaac Sim 6 environment |
| `mujoco` | MuJoCo 4.0 (CPU), no Kit app | `mujoco.Renderer` (OpenGL) | any machine with Python 3.11+ |

`env/environment/task_env.py` gives every task the backend's `TaskEnv`; the shared episode flow is
`env/environment/task_common.py`. The Isaac code path of `isaacsim` is unchanged: a process that does not select another
backend imports and runs exactly what it ran before.

## mujoco

`env/backends/mujoco/`: per env one MuJoCo model (`world.py`) assembled from the robot's URDF and `RobotSpec`
(`env/robot_manager/robot_spec.py`, the G1-D's in `g1d_spec.py`) and the layout's USD objects through MuJoCo 4's USD
importer (`usd_assets.py`). `articulation.py` implements the part of IsaacLab's `Articulation` / `ContactSensor` API
the robot manager, the observations and the experts use (including `root_physx_view.get_jacobians()`), so
`RobotManager`, `LayoutManager`, the reward manager and the eval client run unchanged.

```bash
python -m venv .venv && .venv/bin/pip install -r scripts/requirements-mujoco.txt
ROBODOJO_SIM_BACKEND=mujoco .venv/bin/python scripts/datagen/collect.py --task g1d_sort_blocks_by_color \
    --env-cfg g1d --seed 0 --layout-ids 0
.venv/bin/python -m pytest env/backends/mujoco   # CPU tests; the last one needs Assets/ (G1-D, a sorting layout)
```

Physics choices that differ from MuJoCo's defaults, each because a light grasped object failed without it:

- The physics step is at most 1/600 s (`sim_env.MAX_TIMESTEP`): the sim config's `dt` is split into substeps. At
  1/300 s an IK-servoed pinch of a light block on the table oscillated and pressed the block out of the fingers.
- Contacts are stiff (`world.CONTACT_SOLIMP`, `solref` = two steps, critically damped): MuJoCo's contacts are soft and
  mass-normalised, so the fingers' 55 N went through light objects.
- Joint drives are IsaacLab's implicit PD: an affine actuator with the damping and the spring's `kp·dt` as implicit bias
  damping (`RobotDrives`), as PhysX integrates its drives; the joint speed limit ramps the position target.

Not supported: garment, fluid and granular objects (PhysX particles and FEM cloth), robots other than the G1-D
(`robots.ROBOT_SPECS`), primitive objects without a USD asset, and annotators other than rgb and
distance_to_image_plane. These raise `NotImplementedError` naming the backend to use. A room's appearance and `table_lift`
overrides are not applied (a warning says so); MDL materials are drawn with their base-colour image. Results are not comparable with `isaacsim` numerically: contact, friction
and the solver differ.

## isaaclab_newton

IsaacLab 3 runs the same Isaac code path as `isaacsim` (USD stage, Isaac Sim core prims, RTX cameras) with
`SimulationCfg.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(...))` (`env/environment/base_env.py`; the sim config's
`newton:` overrides NewtonCfg fields, `newton.solver` the solver's and `newton.shape` the default contact). The same
code also runs IsaacLab 3 with PhysX (`isaacsim` in an Isaac Sim 6 environment); `env/environment/isaaclab_compat.py`
holds the version differences the task code sees (IsaacLab 3 returns quaternions in xyzw).

What Newton needs that PhysX does not (`env/backends/newton/physics.py`):

- Newton simulates a model imported from the stage, and IsaacLab 3 imports only the scene's asset cfgs; our tasks
  author the stage themselves. `IsaacRLEnv._setup_scene` imports env 0 for every env (`physics.replicate`), and the
  scene manager re-imports it after every scene reload (`physics.rebuild`, a hard reset; IsaacLab's articulations and
  sensors rebind on `PHYSICS_READY`). Rigid objects are therefore respawned each episode and retired ones deleted.
- Isaac Sim core's rigid prims read and write bodies through a PhysX tensor view; `physics.RigidBodyView` provides
  that view on Newton's state (a free body is a one-body articulation; its pose and twist are the free joint's
  coordinates).
- `physics.link_jacobians` gives PhysX's `get_jacobians()` layout (link origins); Newton's refer to the centres of mass.
- The default contact is critically damped with a time constant of two solver steps (as the mujoco backend), so 2 mm
  cards rest on the table instead of sinking through it. `physics.replicate` also authors the mujoco backend's stiff
  `mjc:solimp` on every collider and torsional friction (`mjc:condim` 4) on the objects, which Newton reads from USD.
- Newton's MuJoCo has no joint speed limit; `physics.limit_joint_speeds` ramps the submitted position target by the
  actuators' `velocity_limit_sim`, as PhysX's drives move (a gripper closing at full speed knocked blocks away).
- The URDF importer of IsaacLab 3 references visuals for links without one, and puts the collision APIs on the Xform
  above each collision mesh, which Newton (it collides Gprims only) ignores: `g1d._repair_imported_usd` removes the
  dangling references (Newton refuses a stage with composition errors) and moves the APIs onto the meshes.
- `SimulationContext.render()` of IsaacLab 3 no longer runs Kit's app loop; the tiled capture pumps the RTX renderer
  before reading its annotators.

Tensors stay on the sim config's device (`cpu` by default): the task and reward code reads them as CPU arrays.
`newton.device` (e.g. `cuda:0`) moves only Newton onto the GPU; the core prims' view (`physics.RigidBodyView`) still
hands the scene and reward code CPU tensors. On the CPU, Warp runs the solver kernels serially, which is hopeless for
a scene of thousands of bodies (rigid beans, `env_cfg/sim/g1d_beans_rigid_sim.yml`).

### Steel

`newton.solver_type: steel` in the sim config, or `ROBODOJO_NEWTON_SOLVER=steel` for one run, selects Newton's Steel
solver (`SteelSolverCfg`, PGS, Newton 1.8) instead of MuJoCo-Warp; `newton.steel` then sets the solver's fields
(`newton.solver` is MuJoCo-Warp's). IsaacLab 3.0 has no Steel: it needs an IsaacLab after 3.0 (`isaaclab_newton`
with `physics/steel_solver_cfg.py`, e.g. main at 9cc3a9a6), still on Isaac Sim 6.2. That IsaacLab keeps the Newton
builder in its manager instance and renamed the actuator limits; `env/backends/newton/physics.py` and
`env/environment/isaaclab_compat.py` handle both. On the Slurm cluster that environment is the shared image
`/shared/images/common/robodojo-newton-v1.sif`: the 3.0 environment plus `newton[sim]==1.8.0`, `mujoco-warp` 3.17,
`ovstorage`, `newton-usd-schemas` and every `IsaacLab/source/*` package of that commit installed editable with
`--no-deps`, under `/opt/openflywheel-newton` (definition `robodojo-newton-v1.def`, pip freeze
in `/opt/openflywheel-newton/provenance/`). It sets the backend, the rendering presets and the Kit arguments below; run it with
`SIM_SIF=/shared/images/common/robodojo-newton-v1.sif ROBODOJO_NEWTON_SOLVER=steel openflywheel-sim python ...` (cluster docs,
`scripts/cluster/SKILL.md`). The original environment on <gpu-server> was deleted after the image was verified; the image backup is OBS
`Employee/contributor/images/robodojo-newton-v1.sif`.

Rigid beans (`g1d_pour_beans_plus`, 170–460 grains, scripted expert, 11-14; `RESULTS.md` in
`<gpu-server>:/data/robodojo-runs/newton-beans-20261114` and OBS `Employee/contributor/raw/newton-beans-20261114/`): PhysX
accepted 9 of 13 episodes, MuJoCo-Warp and Steel none. On MuJoCo-Warp (CG, pyramidal cones) grains pass through the
cup walls once the gripper squeezes the cup (1634 wall leaks in 13 episodes). Steel leaks none, but the fingers knock
or tip the cup and the grains spill from its mouth, and a control step takes 1.0–1.7 s against PhysX's 0.46 s. Steel's
free-body constraint rows are capped at 16384: a 462-grain cup asks for up to 16410, 24576 do not load (the PGS CUDA
module fails) and 8192 overflow and break the contact. On both Newton solvers the step time is the GPU solve, waited
for at the first joint readback of the step. Use `isaacsim` for granular tasks.

Not supported: garment, fluid and particle granular objects (PhysX particles / FEM cloth; the rigid-bean
`granular` objects of `g1d_beans_rigid_sim` run), and Isaac Sim core articulation
objects (`env/scene_manager/objects/articulation.py`), whose PhysX articulation view has no stand-in yet.

Grasps are less reliable than on PhysX. MuJoCo-Warp has no noslip solver (`use_mujoco_cpu`, which has one, is not
supported by IsaacLab's Newton manager), and its contacts are soft: a pinch across the curved side of the sorting
task's 1.5 cm half-cylinders can squeeze the block out of the pads. The defaults (`impratio` 1000, `enable_multiccd`)
are what held best on <gpu-server>, measured with the scripted sorting expert (`g1d_sort_blocks_by_color`, seed 0). Layout 0
succeeded twice with every block within 4 mm, as on PhysX. On layouts 1 and 2, 3 of 4 blocks were placed. In the
layout 2 run the solver then diverged: a link pose came back as a zero quaternion, and the expert raised. The
`isaacsim` backend succeeds on all three. Lower `impratio` (10, 100), torsional friction 0.05, extra drive damping and
Newton's own collision pipeline (`use_mujoco_contacts: false`) were each worse. On every backend but `isaacsim` (which
has its PhysX monitor), the eval client ends an episode whose end-effector pose turns non-finite or zero-norm and counts
it as unstable (`EvalEnv._end_diverged_envs`), not as a fail.

### Environment

An Isaac Sim 6.2 / IsaacLab 3.0 environment (Python 3.12): `pip install "isaacsim[all,extscache]==6.2.0.0"`, IsaacLab
3.0 from source with `source/isaaclab`, `isaaclab_physx`, `isaaclab_newton`, `isaaclab_assets`, `isaaclab_tasks`
(Newton 1.6.1, mujoco-warp 3.12, warp 1.17, torch 2.11). Our core prims still use Isaac Sim's deprecated extensions:
pass `--kit_args "--ext-folder <site-packages>/isaacsim/extsDeprecated"`, and point `ROBODOJO_RENDERING_MODES_DIR` at
IsaacLab 2.3's `apps/rendering_modes` (IsaacLab 3 dropped the presets the render config names). A working environment
is `<runs-dir>/isaacsim6-spike-20261102/env`; `<runs-dir>/sim-backends-20261110/run.sh`
runs the sorting expert with it.
