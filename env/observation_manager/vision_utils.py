"""Zero-torch helpers for camera selection, depth/affordance config parsing, and 3D affordance lookup."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
import json
import os
from pathlib import Path

import numpy as np

DEFAULT_AFFORDANCE_HEATMAP_HW = (48, 64)
DEFAULT_AFFORDANCE_TOP_K = 16


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def parse_camera_selection(value):
    """Parse a bool/string/sequence/set/mapping camera selection into None (off), True (all cameras), or set[str]."""
    if value is None or isinstance(value, bool):
        return True if value else None
    if isinstance(value, Mapping):
        if not value.get("enabled", True):
            return None
        return parse_camera_selection(value.get("cameras", True))
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() in ("0", "false", "no", "off"):
            return None
        if raw.lower() in ("1", "true", "yes", "on", "*", "all"):
            return True
        cams = {c.strip() for c in raw.split(",") if c.strip()}
        if any(c.lower() in ("*", "all") for c in cams):
            return True
        return cams or None
    if isinstance(value, Sequence | AbstractSet) and not isinstance(value, bytes | bytearray):
        cams = {str(c).strip() for c in value if str(c).strip()}
        if any(c.lower() in ("*", "all") for c in cams):
            return True
        return cams or None
    return True if value else None


def merge_camera_selections(*selections):
    """Combine camera selections (None | True | set[str]) into the union selection."""
    merged = set()
    for sel in selections:
        if sel is True:
            return True
        if isinstance(sel, set):
            merged.update(sel)
    return merged or None


def requested_depth_cameras(requested_from_cfg=None, env_var: str = "OPENFLYWHEEL_COLLECT_DEPTH"):
    """Return depth camera selection from config + env var (None | True | set[str])."""
    return merge_camera_selections(
        parse_camera_selection(requested_from_cfg),
        parse_camera_selection(os.environ.get(env_var, "")),
    )


def selection_matches(selection, camera_name: str) -> bool:
    return selection is True or (isinstance(selection, set) and camera_name in selection)


def parse_heatmap_hw(value, default=DEFAULT_AFFORDANCE_HEATMAP_HW) -> tuple[int, int]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        parts = value.lower().split("x")
        if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
            raise ValueError(f"Heatmap size must be WxH (e.g. 64x48), got {value!r}")
        w, h = int(parts[0]), int(parts[1])
        if w <= 0 or h <= 0:
            raise ValueError(f"Heatmap size must be positive, got {value!r}")
        return (h, w)
    if isinstance(value, Sequence) and len(value) == 2:
        h, w = int(value[0]), int(value[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"Heatmap size must be positive, got {value!r}")
        return (h, w)
    raise ValueError(f"Unsupported heatmap size spec: {value!r}")


def as_numpy(val) -> np.ndarray:
    if hasattr(val, "detach"):
        val = val.detach()
    if hasattr(val, "cpu"):
        val = val.cpu()
    return np.asarray(val)


def is_usable_intrinsics(k_arr: np.ndarray) -> bool:
    if k_arr.shape != (3, 3) or not np.isfinite(k_arr).all():
        return False
    return not (
        abs(float(k_arr[0, 0])) <= 1.0 + 1e-9 and abs(float(k_arr[1, 1])) <= 1.0 + 1e-9 and float(k_arr[0, 2]) == 0.0
    )


def iter_layout_rigid_entries(env, layout_id=None, env_idx: int = 0) -> list[dict]:
    """Yield Rigid object layout entries (dicts) for `env_idx`."""
    entries: list[dict] = []
    layout_entries_fn = getattr(env, "layout_entries", None)
    if callable(layout_entries_fn):
        try:
            by_env = layout_entries_fn("Rigid")
            rigid = by_env[env_idx] if by_env and env_idx < len(by_env) else None
            if isinstance(rigid, Mapping):
                for cat, cat_entries in rigid.items():
                    for e in cat_entries or ():
                        if isinstance(e, Mapping):
                            item = dict(e)
                            item.setdefault("category", cat)
                            entries.append(item)
        except Exception:
            entries.clear()
    if entries:
        return entries

    sm = getattr(env, "seed_manager", None)
    if sm is not None and hasattr(sm, "get_seed_scene_info"):
        lid = layout_id
        if lid is None:
            seed_map = getattr(env, "current_env_seed_map", None)
            if isinstance(seed_map, Mapping):
                lid = seed_map.get(env_idx)
        if lid is not None:
            try:
                info = sm.get_seed_scene_info(lid) or {}
                rigid = info.get("Rigid") or {}
                if isinstance(rigid, Mapping):
                    for cat, cat_entries in rigid.items():
                        for e in cat_entries or ():
                            if isinstance(e, Mapping):
                                item = dict(e)
                                item.setdefault("category", cat)
                                entries.append(item)
            except Exception:
                entries.clear()
    if entries:
        return entries

    scene_mgr = getattr(env, "scene_manager", None)
    cfg = getattr(scene_mgr, "config", None)
    if isinstance(cfg, Mapping):
        env_cfg_map = cfg.get(f"env{env_idx}")
        rigid = env_cfg_map.get("Rigid") if isinstance(env_cfg_map, Mapping) else None
        if isinstance(rigid, Mapping):
            for cat, cat_entries in rigid.items():
                for e in cat_entries or ():
                    if isinstance(e, Mapping):
                        item = dict(e)
                        item.setdefault("category", cat)
                        entries.append(item)
    return entries


def resolve_affordance_for_entry(entry, aff_dir=None):
    """Resolve an affordance.json dict for a single Rigid layout entry, or None."""
    if not isinstance(entry, Mapping):
        return None
    if isinstance(entry.get("affordance"), Mapping):
        return dict(entry["affordance"])
    if entry.get("affordance_path") and Path(entry["affordance_path"]).is_file():
        return json.loads(Path(entry["affordance_path"]).read_text(encoding="utf-8"))
    cat = entry.get("category")
    idx = entry.get("category_idx")
    if cat is not None and idx is not None and aff_dir is not None:
        override = Path(aff_dir) / str(cat) / f"{int(idx):05d}" / "affordance.json"
        if override.is_file():
            return json.loads(override.read_text(encoding="utf-8"))
    usd_path = entry.get("usd_path") or entry.get("path")
    if usd_path:
        sidecar = Path(usd_path).parent / "affordance.json"
        if sidecar.is_file():
            return json.loads(sidecar.read_text(encoding="utf-8"))
    if cat is not None and idx is not None:
        try:
            from env.global_configs import ASSETS_PATH, object_dir

            obj_dir = Path(object_dir(f"Rigid/{cat}/{int(idx):05d}", assets=ASSETS_PATH))
            sidecar = obj_dir / "affordance.json"
            if sidecar.is_file():
                return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def detect_robot_model(env) -> str | None:
    """Canonical robot model name from target robots (e.g. 'g1d_sharpa_wave', 'g1d'), or None."""
    rm = getattr(env, "robot_manager", None)
    if rm is None:
        return None
    for robot in getattr(rm, "robot_list", ()):
        if getattr(robot, "type", None) != "target":
            continue
        name = getattr(robot, "asset_name", None) or getattr(robot, "robot_name", None)
        if name:
            return str(name)
    return None


def detect_gripper_types(env, grasps_map: Mapping, fallback_model: str | None = None) -> tuple[str, ...] | None:
    """Infer matching affordance gripper_types tuple for the active target robot."""
    if not isinstance(grasps_map, Mapping) or not grasps_map:
        return None
    model = str(detect_robot_model(env) or fallback_model or "").lower()
    if "sharpa" in model and grasps_map.get("sharpa_wave"):
        return ("sharpa_wave",)
    if "wuji" in model and grasps_map.get("wuji_hand2_2"):
        return ("wuji_hand2_2",)
    if grasps_map.get("dex1_1"):
        return ("dex1_1",)
    return tuple(str(k) for k, v in grasps_map.items() if v) or None


def resolve_affordance_target(
    env,
    *,
    env_idx: int = 0,
    layout_id=None,
    preferred_label: str | None = None,
    affordance_path: str | Path | None = None,
    affordance_data: Mapping | None = None,
    robot_model: str | None = None,
) -> dict | None:
    """Resolve target object label, affordance dict, scale, fallback pose, and gripper_types for an episode."""
    aff_dir_env = os.environ.get("OPENFLYWHEEL_AFFORDANCE_DIR", "").strip()
    aff_dir = Path(aff_dir_env) if aff_dir_env else None
    entries = iter_layout_rigid_entries(env, layout_id=layout_id, env_idx=env_idx)
    by_label = {str(e["label"]): e for e in entries if e.get("label")}

    if preferred_label is None:
        preferred_label = getattr(env, "target_label", None) or getattr(env, "target_name", None)
    if preferred_label is not None:
        preferred_label = str(preferred_label)

    aff_dict = None
    if affordance_path:
        p = Path(affordance_path)
        if p.is_file():
            aff_dict = json.loads(p.read_text(encoding="utf-8"))
    elif isinstance(affordance_data, Mapping):
        aff_dict = dict(affordance_data)

    chosen_label = preferred_label
    chosen_entry = by_label.get(chosen_label) if chosen_label else None
    if aff_dict is None and chosen_entry is not None:
        aff_dict = resolve_affordance_for_entry(chosen_entry, aff_dir=aff_dir)

    if aff_dict is None:
        ordered = sorted(
            entries,
            key=lambda e: (
                0 if str(e.get("label", "")) == "target" else (1 if "target" in str(e.get("label", "")) else 2)
            ),
        )
        for e in ordered:
            cand = resolve_affordance_for_entry(e, aff_dir=aff_dir)
            if cand is not None:
                aff_dict = cand
                chosen_entry = e
                chosen_label = str(e.get("label", "")) or chosen_label
                break

    if aff_dict is None:
        return None

    if chosen_label is None and entries:
        chosen_label = str(entries[0].get("label", "target"))
        chosen_entry = entries[0]

    scale = 1.0
    fallback_pose = None
    if isinstance(chosen_entry, Mapping):
        if chosen_entry.get("scale") is not None:
            scale = chosen_entry["scale"]
        if "default_pos" in chosen_entry:
            fallback_pose = (
                np.asarray(chosen_entry["default_pos"], dtype=np.float64),
                np.asarray(chosen_entry.get("default_ori", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64),
            )

    grasps_map = aff_dict.get("grasps") or {}
    return {
        "target_label": chosen_label or "target",
        "affordance": aff_dict,
        "asset_id": str(aff_dict.get("asset_id", "")),
        "scale": scale,
        "fallback_pose": fallback_pose,
        "gripper_types": detect_gripper_types(env, grasps_map, fallback_model=robot_model),
    }
