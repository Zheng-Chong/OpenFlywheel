# Four complete G1-D variants: simulation validation, 2026-11-03

The selected simulation evidence and known failures are summarized in [evidence/20261103](evidence/20261103/final_validation.json). The immutable complete delivery contains all four models, source resources, recordings, HDF5, review media and audits. Its SHA256 and 25-volume layout are in [release_manifest.json](evidence/20261103/release_manifest.json); [complete delivery](https://github.com/Zheng-Chong/OpenFlywheel/releases/tag/g1d-four-hands-validation-20261103) is published and all 35 GitHub attachment sizes and SHA256 digests match local files. The local playback is at http://127.0.0.1:8790/. No hardware, DDS, or physical camera was used.

The native recordings were produced on `2cf5b6d177a82ce54c18dc26cfe8e61d06f47788`. This integration is prepared on current main `96466f8d2e90b20493a13c7a25101661db395aef`, preserving its initial-joint/reset handling. CPU integration checks are reported separately; old recordings must not be presented as a native rerun on the newer base.

| Complete model | Active hand axes, both sides | Direction cases | Contact shape × channel cases | Free-object lift / hold / release |
|---|---:|---:|---:|---:|
| `g1d_sharpa_wave` | 44 | 68 | 30/30 | 6/6 |
| `g1d_wuji_hand2_1` | 40 | 80 | 30/30 | 4/6; both spheres fail |
| `g1d_wuji_hand2_2` | 40 | 80 | 30/30 | 6/6 |
| `g1d_inspire_rh56dftp` | 12 | 12 | 36/36 | 6/6 |

These are finite qualifications, not proof of all poses, full range, generalized grasping, or accurate placement. The 80% combined targets for Sharpa and both Wuji variants have about 0.162–0.173 rad tracking error. Inspire's active-axis combined error is below 0.0081 rad; passive joints are checked against the native mimic equation, not an unused drive target. Its maximum coupling error in the selected grasp run is 0.000367 rad. Several accepted lift/hold/release trials drop the object after release; final position offsets are shown separately. Failed attempts are retained, never joined to create a successful physical episode.

## What the data means

- Motion increments cover all 136 active hand axes and 240 legal direction cases, including return and FK checks. CPU checks reject malformed, nonfinite or out-of-range hand commands.
- Free objects have gravity, mass 25 g, and no attachment. Sphere diameter is 50 mm, cube edge 44 mm, cylinder diameter 46 mm and height 48 mm. A retractable three-point fixture supports initialization; holding is checked after its withdrawal. Criteria are frozen in `plan.json`.
- Native motion/grasp recordings contain five same-frame RGB/depth views, measured joint state, requested/applied targets, force components in N, link/world transforms, and object states. Use the recorded timestamps: nominal preview FPS does not remove nonuniform phase-boundary sampling.
- Contact fixtures test sphere, flat and cylinder loading, shear, unloading and recovery. Collision instances in the private fixture scene are made editable before binding a test material; native raycasts verify the actual collider and material. The 280 N/m, 20 N·s/m, 0.6/0.5 friction values are simulation presets, not measured gel parameters. Wuji 2.1's quasistatic fixture is referenced to the measured pad pose at each phase boundary, which is recorded explicitly.
- Sharpa runs the official TacMap code and stores ten geometric depth maps in m and vendor uint8 encodings. The selected run's independent sphere error is about 14.6 µm and encoding difference is 0. These maps are not optical gel RGB or FEM, and depth is not converted to N.
- Wuji 2.1 has no manufacturer tip-sensor model; its distal forces are virtual aggregate contacts. Wuji 2.2 uses the real `*_tip_sensor_frame` collider names, not the shell. Inspire exposes 12 aggregate finger/palm force channels, not invented hardware patches or calibrated taxels.
- `RobotManager.get_tactile_contact` reports aggregate **normal** force, including self-contact. Per-object fixture/grasp data additionally saves filtered normal and friction components; these different quantities must not be conflated.
- HDF5 exports are independently read back. Native RGB/depth is lossless; contact JPEGs preserve their original recorded bytes. Preview MP4s are lossy. This is a dedicated validation schema, not an automatically compatible 16-dimensional G1-D training dataset.

## Reproducing or auditing

Run Isaac workers sequentially in fresh output directories; never import their launcher modules into a CPU process. The current workstation uses Isaac Sim 5.1 / IsaacLab 2.3.2, CPU PhysX (480 Hz native and Inspire, 240 Hz other contact fixtures), and rendering on GPU 1. Existing source URDFs, motor caps, mimic constraints and exclusions are preserved. Execution inputs and hashes are frozen per run.

1. `comprehensive_validate.py`: `--model`, `--mode motion|grasp`, `--plan`, `--grasps` for grasp, `--output`; plus Isaac launcher options. Candidate grasp settings are experiment inputs, not hardware-calibrated control defaults.
2. `audit_comprehensive.py`: read-only raw trace/HDF5, force frames and action/reaction, cameras, source hashes, movement and task gates; reports passive coupling separately.
3. `tactile_validate.py` / `../inspire/contact_validate.py`: controlled native fixtures; then `export_episode.py` and `audit_contact_episode.py` for independent export checks.
4. `framework_validate.py` / `../inspire/framework_validate.py`: registered action queue/order, effort caps and descriptors. `framework_contact_validate.py` tests actual loading through the public observation getter, including an independent filtered force comparison.
5. `export_comprehensive_review.py --root ... --selection ... --output ...`: explicit recordings only. Serve the generated directory on loopback with `python -m http.server 8790 --bind 127.0.0.1 --directory ...`.
6. Run `bash scripts/check.sh` with the CPU dependency environment and lint the new scripts. No simulation worker should be run against another user's active process or shared asset checkout.

Installation/TCP measurements, inertia and servo identification, real friction/gel calibration, per-serial tactile metadata and hardware transfer remain outside this simulation-only qualification. All known failures, source choices and material-binding corrections are in the local report and experiment history.

Publication: 2026-11-04, GitHub account `Joker-of-Gotham`, commit email `1211095767@qq.com`. The GitHub CPU CI passed; local private-submodule access remains restricted. Publication does not change the frozen native results or their known failures.
