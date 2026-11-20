import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.datagen.needed_assets import ALWAYS, category_paths, clutter_paths, from_layout, needed, object_paths
from src.real2sim_platform.benchmarks import DEFAULTS, ROBODOJO_TASKS


def test_needed_assets_from_layouts_and_configs(tmp_path):
    layout = {
        "Room": {
            "default": "real_scene1",
            "appearance": [{"material": {"mdl": "/x/Assets/Material/material_0200/a.mdl"}}],
        },
        "Table": {"default": "material_0122"},
        "Background": {"category_name": "studio.hdr"},
        "Rigid": {"mug": [{"category_idx": 5}], "tray": [{"category_idx": 1, "type": "cluttered"}]},
        "Fluid": {
            "wuliangye": [
                {"category_idx": 0, "physics": {"pack": "$Robodojo_ASSETS/Object/Granular/red_bean/pack.npz"}}
            ]
        },
    }
    (tmp_path / "g1d_pour_0.json").write_text(json.dumps(layout))
    (tmp_path / "benchmark.json").write_text(json.dumps({"task": "x", "layouts": ["g1d_pour_0.json"]}))
    got = set(needed("g1d", "stack_bowls", [tmp_path]))
    assert set(ALWAYS) <= got and "?Assets/Traj/RoboDojo/stack_bowls" in got
    assert {
        "Assets/Room/real_scene1",
        "Assets/Material/material_0200",
        "Assets/Material/material_0122",
        "Assets/Object/Granular/red_bean/pack.npz",
        object_paths("Rigid", "mug", 5),
        object_paths("Clutter", "tray", 1),
        object_paths("Fluid", "wuliangye", 0),
    } <= got
    assert "Assets/Room/Simple_Room_nolight" in got and object_paths("Geometry", "camera_stand", 0) in got  # scene
    assert object_paths("Rigid", "mug", 5).split("|")[0] == "Assets/Object/Real2Sim/Rigid/mug/00005"
    assert "Assets/Material" not in got and "Assets/Material" in needed(all_materials=True)
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"objects": [{"cat": "airpods", "idx": 0}, {"cat": "block", "idx": 3}]}))
    assert object_paths("Rigid", "block", 3) in needed(pools=[pool])


def test_needed_assets_garment_articulation_geometry_and_edge_layouts():
    layout = {
        "Room": {"default": "real_scene5"},
        "Garment": {
            "Top_Long": [{"category_idx": 4}],
            "plane_cloth": [{"category_idx": None}, {}],  # Plane primitive without USD must not crash
        },
        "Articulation": {"toaster": [{"category_idx": 1}], "empty": None},
        "Geometry": {
            "bread_shelf": [
                {
                    "category_idx": 0,
                    "visual": {"visual_mdl_path": "/data/robodojo-assets/Assets/Material/material_0333/wood.mdl"},
                }
            ]
        },
        "Clutter": {"cup": [{"category_idx": 2}]},
    }
    got = from_layout(layout)
    assert {
        "Assets/Room/real_scene5",
        "Assets/Material/material_0333",
        object_paths("Garment", "Top_Long", 4),
        object_paths("Articulation", "toaster", 1),
        object_paths("Geometry", "bread_shelf", 0),
        object_paths("Clutter", "cup", 2),
    } <= got


def test_needed_assets_blocks_v2_and_robodojo_v1_all_tasks():
    blocks_jobs = next(d["jobs"] for d in DEFAULTS if d["slug"] == "blocks")
    assert len(blocks_jobs) == 5 and len(ROBODOJO_TASKS) == 54
    for job in blocks_jobs:
        got = set(needed(job["env_cfg"], job["task"], task_objects=True))
        assert set(ALWAYS) <= got and f"?Assets/Traj/RoboDojo/{job['task']}" in got
    # g1d_general_pickup declares Clutter yaml_path; g1d_build_block_pattern declares Blocks wood_block_*
    pickup = set(needed("g1d_shenzhen_lab", "g1d_general_pickup", task_objects=True))
    assert clutter_paths() in pickup and object_paths("Rigid", "pepper", 0) in pickup
    pattern = set(needed("g1d", "g1d_build_block_pattern", task_objects=True))
    assert object_paths("Rigid", "block_pattern_card", 0) in pattern

    for task in ROBODOJO_TASKS:
        got = set(needed("arx_x5", task, task_objects=True))
        assert set(ALWAYS) <= got
    # Garment task
    fold = set(needed("arx_x5", "fold_clothes_random", task_objects=True))
    assert object_paths("Garment", "Top_Long", 6) in fold and clutter_paths() in fold
    # Unindexed whole-category Articulation / Geometry / Rigid task
    egg = set(needed("arx_x5", "fill_egg_holder", task_objects=True))
    assert {
        category_paths("Articulation", "egg_holder"),
        category_paths("Geometry", "egg_basket"),
        category_paths("Rigid", "egg"),
    } <= egg
    # _task.yml scene_config: conveyor override -> Simple_Room
    conveyor = set(needed("arx_x5", "match_and_pick_from_conveyor", task_objects=True))
    assert "Assets/Room/Simple_Room" in conveyor and "Assets/Room/Simple_Room_nolight" in conveyor
