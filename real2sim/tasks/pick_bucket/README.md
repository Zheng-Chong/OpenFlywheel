# pick_bucket recorded-scene alignment review - 2026-11-03

Current status: NOT aligned / NOT accepted. The earlier probe15 video uses a different initial layout and must not be treated as reconstruction of the recording, or as a diagnosis of recorded-scene load bearing.

The first frame of episode_0001 has the basket at lower right, nearer than the horizontally oriented mango. The screenshot at 14 s is an intermediate state: the mango is already inside the basket. `recorded_alignment.json` archives the source config, first-frame joint states, head camera image-fit candidate, coordinate conventions and input SHA256 values.

The new `g1d_pick_bucket` config uses the source torso frame, including the 53.371 mm torso/chassis forward offset; recorded initial arm/gripper joints; source head pose and separate fx/fy/principal point; table position and the silver closed cable hatch. The candidate preserves native asset dimensions and masses.

The native basket 00014 is circular (240 mm mouth, 263.837 x 256.9 x 252 mm complete bbox), whereas the old task proxy is elliptical (approximately 212 x 157 mm). Their identity has not been established. The projected native silhouette still exceeds the recording. Do not resize a measured asset or copy its measured mass onto a different prop to conceal this discrepancy.

Latest normal native run: `pick_bucket_recorded_alignment_gate`, Isaac Sim 5.0, seed/layout 0, g1d_pick_bucket. Basket startup stability failed; five-stage execution never began. No successful lift, return or release is claimed. Preserve the failure. Yaw hypotheses and the post-initialization collision-disable diagnostic did not establish the contact root cause.

Review media: `preview/alignment_comparison.jpg` and `preview/alignment_report.json`. The comparison labels source frame 0, projected requested silhouette, actual failed startup, and source frame 420. Neither static projections nor diagnostic renders prove task success. Camera values remain conditional image fits, not qualified real calibration.

Reproduce after resolving the basket model identity:

```bash
PYTHONPATH=. python -c 'from pathlib import Path; from scripts.g1d.build_asset import write_pattern_layouts; write_pattern_layouts(Path("Assets"), "g1d_pick_bucket", "pick_bucket", Path("Assets/Eval_Layout/RoboDojo/g1d_pick_bucket/0"), 0, 1)'
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=.:XPolicyLab EVAL_NUM=1 python scripts/g1d/scripted_demo.py --env_cfg_type g1d_pick_bucket --task_name pick_bucket --seed 0 --device cuda:0 --device_id 0
```

No shared asset geometry was changed. PR remains draft. Old probe15 evidence is retained in `validation_unaligned_probe15.json`; no batch data collection was performed.
