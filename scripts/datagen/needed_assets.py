"""The asset paths a simulation job needs, one a line, for `asset_client.py pull --paths-from` (pull only those from the
long-term library on <asset-server> instead of copying the whole shared library):

    the layouts' objects, rooms, materials (rsi_assets.layout_assets, plus the Dynamic and Fluid sections), every
    $RoboDojo_ASSETS/... a layout or config names (bean packs), the scene config's defaults, and what every job loads
    (robots, sensors, HDRs, fluid materials); with --task its trajectories, with --task-objects its config's objects,
    with --pool every object of a pool that layouts drawn on the node pick from (pick_anything_pool.json,
    open_anything_pool.json).

An object is "<lib path>|<lib path>|..." in OBJECT_LIBRARIES order, the order the simulation looks it up in
(env.global_configs.object_dir); a leading "?" marks a path that may be missing.

    python scripts/datagen/needed_assets.py --env-cfg g1d_lab --task g1d_pour_beans_plus --layouts <seed folder> \
        > needs.txt
    python3 OpenFlywheel/asset_client.py pull --ssh <asset-server> --cache /scratch/tmp/$USER/asset-cache \
        --root <run>/robodojo-assets --paths-from needs.txt
    ln -s robodojo-assets/Assets <run>/Assets

Eval layouts are in the library too: pull Assets/Eval_Layout/RoboDojo/<config>/<seed> first, then pass that folder.
The table and ground randomization (random_table, random_ground) draws from the Material/ listing: pass
--all-materials, or a job draws from fewer materials than with the whole library.
"""

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from env.global_configs import OBJECT_LIBRARIES  # noqa: E402

SECTIONS = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")  # LayoutManager.load_saved_layout
ALWAYS = ("Assets/Robots", "Assets/Sensor", "Assets/Background", "?Assets/Material/Fluid")
ASSET_VAR = re.compile(r"\$Robo[dD]ojo_ASSETS/([^\s\"'@]+)")


def object_paths(section, category, index):
    return "|".join(f"Assets/Object/{lib}/{section}/{category}/{int(index):05d}" for lib in OBJECT_LIBRARIES)


def category_paths(section, category):
    return "|".join(f"Assets/Object/{lib}/{section}/{category}" for lib in OBJECT_LIBRARIES)


def clutter_paths():
    return "|".join(f"Assets/Object/{lib}/Clutter" for lib in OBJECT_LIBRARIES)


def named(text):
    return {f"Assets/{m}" for m in ASSET_VAR.findall(text)}


def _material_dir(path):
    if isinstance(path, str) and "/Material/" in path:
        name = path.split("/Material/", 1)[1].split("/", 1)[0]
        if name:
            return f"Assets/Material/{name}"
    return None


def from_layout(layout):
    out = set()
    room_cfg = layout.get("Room") if isinstance(layout.get("Room"), dict) else {}
    if room := room_cfg.get("default"):
        out.add(f"Assets/Room/{room}")
    for group in room_cfg.get("appearance") or []:
        if isinstance(group, dict):
            mat = group.get("material") if isinstance(group.get("material"), dict) else {}
            if mdl := _material_dir(mat.get("mdl")):
                out.add(mdl)
    table = (layout.get("Table") if isinstance(layout.get("Table"), dict) else {}).get("default")
    ground_cfg = layout.get("Ground") if isinstance(layout.get("Ground"), dict) else {}
    ground = (ground_cfg.get("materials") if isinstance(ground_cfg.get("materials"), dict) else {}).get("default")
    for material in (table, ground):
        if material:
            out.add(f"Assets/Material/{material}")
    for section in (*SECTIONS, "Clutter"):
        sec_val = layout.get(section)
        if not isinstance(sec_val, dict):
            continue
        for category, entries in sec_val.items():
            for e in entries or []:
                if not isinstance(e, dict):
                    continue
                if e.get("category_idx") is not None:
                    sec = "Clutter" if section == "Clutter" or e.get("type") == "cluttered" else section
                    out.add(object_paths(sec, category, e["category_idx"]))
                # an articulated object's source folders (g1d_open_anything): pull does not follow a reference with
                # a space in it
                out |= set((e.get("open") or {}).get("sources") or []) if isinstance(e.get("open"), dict) else set()
                visual = e.get("visual") if isinstance(e.get("visual"), dict) else {}
                for val in visual.values():
                    if mdl := _material_dir(val):
                        out.add(mdl)
    return out | named(json.dumps(layout))


def from_config(config, clutter_pool=True):
    """A scene or task config: the defaults it falls back on, and the objects its category lists can draw."""
    out = set()
    if not isinstance(config, dict):
        return out
    room_cfg = config.get("Room") if isinstance(config.get("Room"), dict) else {}
    if room := room_cfg.get("default"):
        out.add(f"Assets/Room/{room}")
    table = (config.get("Table") if isinstance(config.get("Table"), dict) else {}).get("default")
    ground_cfg = config.get("Ground") if isinstance(config.get("Ground"), dict) else {}
    ground = (ground_cfg.get("materials") if isinstance(ground_cfg.get("materials"), dict) else {}).get("default")
    for material in (table, ground):
        if material:
            out.add(f"Assets/Material/{material}")
    for section in (*SECTIONS, "Clutter"):
        blocks = config.get(section)
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if yaml_path := block.get("yaml_path"):
                out.add(f"?Assets/{yaml_path}")
                if clutter_pool:
                    out.add(clutter_paths())
            for c in block.get("category") or []:
                if not isinstance(c, dict) or not c.get("name"):
                    continue
                if c.get("index") is not None:
                    out |= {object_paths(section, c["name"], i) for i in c["index"]}
                else:
                    out.add(category_paths(section, c["name"]))
    return out


def _env_part(cfg_dir, kind, name):
    import yaml

    path = cfg_dir / kind / f"{name}.yml"
    if not path.is_file():
        return set()
    part = path.read_text(encoding="utf-8")
    return named(part) | (from_config(yaml.safe_load(part) or {}) if kind == "scene" else set())


def _task_env_overrides(cfg_dir, task):
    """Task-level scene/robot/camera overrides in task/RoboDojo/config/_task.yml (process_config)."""
    import yaml

    index_path = ROOT / "task" / "RoboDojo" / "config" / "_task.yml"
    if not index_path.is_file():
        return set()
    info = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    task_info = (info.get("tasks") or {}).get(task) or {}
    if not task_info:
        return set()
    common_info = info.get("common") or {}
    out = set()
    for kind, key in (("scene", "scene_config"), ("robot", "robot_config"), ("camera", "camera_config")):
        if name := task_info.get(key) or common_info.get(key):
            out |= _env_part(cfg_dir, kind, name)
    return out


def needed(env_cfg=None, task=None, layouts=(), task_objects=False, all_materials=False, pools=(), clutter_pool=True):
    import yaml

    out = set(ALWAYS) | ({"Assets/Material"} if all_materials else set())
    for pool in pools:  # layouts drawn on the node (g1d_pick_anything, g1d_open_anything): every object they can draw
        doc = json.loads(Path(pool).read_text())
        out |= {object_paths("Rigid", o["cat"], o["idx"]) for o in doc.get("objects", [])}
        for e in doc.get("entries", []):
            out |= {object_paths("Articulation", e["cat"], e["idx"]), *e.get("sources", [])}
    cfg_dir = ROOT / "env_cfg"
    if env_cfg:
        text = (cfg_dir / f"{env_cfg}.yml").read_text(encoding="utf-8")
        parts = (yaml.safe_load(text) or {}).get("config") or {}
        out |= named(text)
        for kind, name in parts.items():
            out |= _env_part(cfg_dir, kind, name)
    if task:
        out.add(f"?Assets/Traj/RoboDojo/{task}")
        out |= _task_env_overrides(cfg_dir, task)
        path = ROOT / "task" / "RoboDojo" / "config" / f"{task}.yml"
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            out |= named(text)
            if task_objects:
                out |= from_config(yaml.safe_load(text) or {}, clutter_pool=clutter_pool)
    for folder in layouts:
        folder = Path(folder)
        for path in sorted(folder.rglob("*.json") if folder.is_dir() else [folder]):
            doc = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                out |= from_layout(doc)
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-cfg", help="env config name (env_cfg/<name>.yml)")
    parser.add_argument("--task", help="task name: its trajectories and the assets its config names")
    parser.add_argument("--layouts", nargs="*", default=[], help="seed folders or layout files")
    parser.add_argument("--task-objects", action="store_true", help="also every object the task config can draw")
    parser.add_argument("--all-materials", action="store_true", help="the whole Material/ (table/ground randomization)")
    parser.add_argument("--pool", nargs="*", default=[], help="object pools (scripts/objects/*_pool.json) to draw from")
    args = parser.parse_args()
    paths = needed(args.env_cfg, args.task, args.layouts, args.task_objects, args.all_materials, args.pool)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
