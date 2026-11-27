#!/usr/bin/env bash
# CPU checks for OpenFlywheel: bash scripts/check.sh [BASE_REF]
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "${1:-}" ] && git rev-parse -q --verify "${1}^{commit}" >/dev/null 2>&1; then
    base="$1"
    files=$(git diff --name-only --diff-filter=ACMR "$base" -- '*.py' | grep -vE '^(third_party|XPolicyLab|real2sim|scripts/g1d/calib_toolkit)/' || true)
    if [ -n "$files" ]; then
        ruff check $files
        ruff format --check $files
    fi
    git diff --check "$base"
fi

python -m pytest -q --import-mode=importlib \
    scripts/datagen \
    scripts/scenes/test_appearance.py \
    scripts/objects/test_manitwin.py \
    scripts/objects/test_gso.py \
    scripts/objects/test_geniesim.py \
    scripts/objects/test_articulated.py \
    scripts/objects/test_robocasa.py \
    scripts/objects/test_sand3d.py \
    scripts/objects/test_fmb.py \
    scripts/objects/test_object_names.py \
    scripts/objects/test_block_patterns.py \
    scripts/objects/test_block_sorting.py \
    scripts/objects/test_block_atomic.py \
    scripts/objects/test_deformable.py \
    scripts/objects/test_cup_profile.py \
    scripts/objects/test_cup_handle.py \
    scripts/objects/test_simready.py \
    scripts/objects/test_place_supports.py \
    scripts/objects/test_affordance.py \
    scripts/objects/test_affordance_2d.py \
    scripts/objects/test_affordance_dexhand.py \
    scripts/objects/test_affordance_vlm.py \
    scripts/objects/test_verify_affordance_dexhand_sim.py \
    scripts/objects/test_polyhaven_models.py \
    scripts/objects/test_molmospaces_thor.py \
    scripts/objects/test_molmospaces_objaverse.py \
    scripts/objects/test_palatial.py \
    scripts/objects/test_usdcraft.py \
    scripts/objects/test_projectsim.py \
    scripts/objects/test_video2data.py \
    scripts/prototypes/test_genie_bridge.py \
    scripts/g1d/test_pour_half.py \
    scripts/g1d/test_g1d_body.py \
    scripts/g1d/test_randomize.py \
    scripts/g1d/test_pick_bucket.py \
    scripts/g1d/test_affordance_grasp.py \
    scripts/g1d/test_affordance_dexhand.py \
    tests/test_g1d_wholebody.py \
    env/backends/mujoco \
    scripts/g1d/xense_gripper \
    scripts/dexhands/test_pour_beans_sharpa.py \
    scripts/dexhands/test_x_anything_sharpa.py
echo "checks passed"
