"""Unitree G1-D + Dex1-1 physics for every simulation backend (g1d.py: IsaacLab; env/backends/mujoco: MuJoCo).

URDFs built by scripts/g1d/build_asset.py. Tabletop (robot g1d, official Standard version): chassis fixed. Mobile (robot
g1d_mobile, Flagship): the chassis on virtual planar joints.
"""

from collections.abc import Mapping
import json
import re
import sys
from typing import Any

from env.global_configs import ROBOTS_PATH
from env.robot_manager.robot_spec import ActuatorGroup, RobotSpec

# Real G1-D finish (product photo + head/wrist recordings). URDF colours are diffuse-only, so links matching these names
# get these PBR finishes; patterns are disjoint. g1d.add_black_trim adds the parts the meshes do not separate.
FINISHES = {
    "metal": {"diffuse_color": (0.62, 0.63, 0.65), "metallic": 1.0, "roughness": 0.5},  # matte
    "black": {"diffuse_color": (0.02, 0.02, 0.02), "roughness": 0.3},  # smooth plastic
    "rubber": {"diffuse_color": (0.02, 0.02, 0.02), "roughness": 0.85},
}
LINK_FINISHES = {
    r"torso_link|AGV_link|LZ_.*_Link|.*_(shoulder_.*|elbow|wrist_.*)_link": "metal",
    r".*_dex1_.*|Yaw_Link|Pitching_Link|head_link|logo_link": "black",
    r".*_Wheel_Link": "rubber",
}

# Dex1-1 pad (Link{1,2}_3) of each finger joint; RobotManager puts a ContactSensor on them for the grip force.
FINGER_PADS = {f"{s}_dex1_finger_joint_{i}": f"{s}_dex1_finger_link_{i}_tip" for s in ("left", "right") for i in (1, 2)}

# Default Dex1-1 inner gripping collider links (Link{1,2}_3 tip pads + Link{1,2}_2_col mid bars).
DEFAULT_FINGER_CONTACT_LINKS = tuple(
    f"{s}_dex1_finger_link_{i}_{part}" for s in ("left", "right") for i in (1, 2) for part in ("tip", "mid")
)

ARM_JOINT_NAMES = tuple(
    f"{s}_{j}_joint"
    for s in ("left", "right")
    for j in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
)


def _first_not_none(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def caller_robot_yml_cfg() -> tuple[Mapping[str, Any] | None, int]:
    """Inspect caller frames for `RobotManager._attach_scene_cfg` (or MuJoCo equivalent) to read per-robot YAML cfg."""
    for depth in range(1, 10):
        try:
            frame = sys._getframe(depth)
        except ValueError:
            break
        code_name = frame.f_code.co_name
        if code_name in ("_attach_scene_cfg", "_get_robot"):
            cfg = frame.f_locals.get("cfg")
            idx = frame.f_locals.get("idx", 0)
            if isinstance(cfg, Mapping):
                return cfg, int(idx) if isinstance(idx, int) else 0
    return None, 0


def resolve_finger_contact(cfg_or_fc: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalize optional `finger_contact` configuration for Dex1-1 gripper colliders; None when unset."""
    if not isinstance(cfg_or_fc, Mapping) or not cfg_or_fc:
        return None
    fc = cfg_or_fc.get("finger_contact") if "finger_contact" in cfg_or_fc else cfg_or_fc
    if not isinstance(fc, Mapping) or not fc:
        return None
    if "has_material" in fc and "links" in fc:
        return dict(fc)

    links_raw = fc.get("links")
    if links_raw is None:
        links = list(DEFAULT_FINGER_CONTACT_LINKS)
    elif isinstance(links_raw, str):
        links = [links_raw]
    else:
        links = [str(x) for x in links_raw]

    fallback_f = fc.get("friction")
    static_raw = fc.get("static_friction")
    dynamic_raw = fc.get("dynamic_friction")
    rest_raw = fc.get("restitution")
    f_comb = _first_not_none(fc, "friction_combine", "friction_combine_mode")
    r_comb = _first_not_none(fc, "restitution_combine", "restitution_combine_mode")
    c_stiff = _first_not_none(fc, "compliant_stiffness", "compliant_contact_stiffness")
    c_damp = _first_not_none(fc, "compliant_damping", "compliant_contact_damping")

    has_material = any(
        v is not None for v in (fallback_f, static_raw, dynamic_raw, rest_raw, f_comb, r_comb, c_stiff, c_damp)
    )
    static_f = float(static_raw) if static_raw is not None else (float(fallback_f) if fallback_f is not None else 0.5)
    dynamic_f = (
        float(dynamic_raw) if dynamic_raw is not None else (float(fallback_f) if fallback_f is not None else static_f)
    )
    restitution = float(rest_raw) if rest_raw is not None else 0.0

    return {
        "links": links,
        "has_material": has_material,
        "static_friction": static_f,
        "dynamic_friction": dynamic_f,
        "restitution": restitution,
        "friction_combine": str(f_comb) if f_comb is not None else None,
        "restitution_combine": str(r_comb) if r_comb is not None else None,
        "compliant_stiffness": float(c_stiff) if c_stiff is not None else None,
        "compliant_damping": float(c_damp) if c_damp is not None else None,
        "torsional_patch_radius": (
            float(fc["torsional_patch_radius"]) if fc.get("torsional_patch_radius") is not None else None
        ),
        "min_torsional_patch_radius": (
            float(fc["min_torsional_patch_radius"]) if fc.get("min_torsional_patch_radius") is not None else None
        ),
        "contact_offset": float(fc["contact_offset"]) if fc.get("contact_offset") is not None else None,
        "rest_offset": float(fc["rest_offset"]) if fc.get("rest_offset") is not None else None,
        "max_depenetration_velocity": (
            float(fc["max_depenetration_velocity"]) if fc.get("max_depenetration_velocity") is not None else None
        ),
        "linear_damping": float(fc["linear_damping"]) if fc.get("linear_damping") is not None else None,
        "angular_damping": float(fc["angular_damping"]) if fc.get("angular_damping") is not None else None,
    }


def urdf_name(mobile=False):
    return "g1d_mobile_dex1" if mobile else "g1d_dex1"


def _build_arm_actuator_groups(
    cfg: Mapping[str, Any] | None = None,
    *,
    wrist_stiffness: float | None = None,
    wrist_damping: float | None = None,
    wrist_effort_limit: float | None = None,
    wrist_velocity_limit: float | None = None,
    wrist_armature: float | None = None,
    arm_stiffness: float | None = None,
    arm_damping: float | None = None,
    arm_effort_limit: float | None = None,
    arm_stiffness_overrides: Mapping[str, float] | None = None,
    arm_damping_overrides: Mapping[str, float] | None = None,
    arm_effort_limit_overrides: Mapping[str, float] | None = None,
) -> dict[str, ActuatorGroup]:
    if isinstance(cfg, Mapping):
        wrist_sub = cfg.get("wrist") if isinstance(cfg.get("wrist"), Mapping) else {}
        if wrist_stiffness is None:
            wrist_stiffness = _first_not_none(cfg, "wrist_stiffness")
            if wrist_stiffness is None:
                wrist_stiffness = _first_not_none(wrist_sub, "stiffness")
        if wrist_damping is None:
            wrist_damping = _first_not_none(cfg, "wrist_damping")
            if wrist_damping is None:
                wrist_damping = _first_not_none(wrist_sub, "damping")
        if wrist_effort_limit is None:
            wrist_effort_limit = _first_not_none(cfg, "wrist_effort_limit", "wrist_effort_limit_sim")
            if wrist_effort_limit is None:
                wrist_effort_limit = _first_not_none(wrist_sub, "effort_limit_sim", "effort_limit")
        if wrist_velocity_limit is None:
            wrist_velocity_limit = _first_not_none(cfg, "wrist_velocity_limit", "wrist_velocity_limit_sim")
            if wrist_velocity_limit is None:
                wrist_velocity_limit = _first_not_none(wrist_sub, "velocity_limit_sim", "velocity_limit")
        if wrist_armature is None:
            wrist_armature = _first_not_none(cfg, "wrist_armature")
            if wrist_armature is None:
                wrist_armature = _first_not_none(wrist_sub, "armature")
        if arm_stiffness is None:
            arm_stiffness = _first_not_none(cfg, "arm_stiffness")
        if arm_damping is None:
            arm_damping = _first_not_none(cfg, "arm_damping")
        if arm_effort_limit is None:
            arm_effort_limit = _first_not_none(cfg, "arm_effort_limit", "arm_effort_limit_sim")
        if arm_stiffness_overrides is None and isinstance(cfg.get("arm_stiffness_overrides"), Mapping):
            arm_stiffness_overrides = cfg["arm_stiffness_overrides"]
        if arm_damping_overrides is None and isinstance(cfg.get("arm_damping_overrides"), Mapping):
            arm_damping_overrides = cfg["arm_damping_overrides"]
        if arm_effort_limit_overrides is None and isinstance(cfg.get("arm_effort_limit_overrides"), Mapping):
            arm_effort_limit_overrides = cfg["arm_effort_limit_overrides"]

    kp_base = float(arm_stiffness) if arm_stiffness is not None else 4400.0
    kd_base = float(arm_damping) if arm_damping is not None else 40.0
    eff_base = float(arm_effort_limit) if arm_effort_limit is not None else 100.0
    vel_base = 5.0
    arm_iner = 0.01

    has_wrist_scalars = any(
        v is not None
        for v in (wrist_stiffness, wrist_damping, wrist_effort_limit, wrist_velocity_limit, wrist_armature)
    )
    has_regex_maps = any(bool(m) for m in (arm_stiffness_overrides, arm_damping_overrides, arm_effort_limit_overrides))

    if not has_wrist_scalars and not has_regex_maps:
        return {
            "arm": ActuatorGroup(
                joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],
                effort_limit_sim=eff_base,
                velocity_limit_sim=vel_base,
                stiffness=kp_base,
                damping=kd_base,
                armature=arm_iner,
            )
        }

    if not has_regex_maps:
        return {
            "arm": ActuatorGroup(
                joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint"],
                effort_limit_sim=eff_base,
                velocity_limit_sim=vel_base,
                stiffness=kp_base,
                damping=kd_base,
                armature=arm_iner,
            ),
            "wrist": ActuatorGroup(
                joint_names_expr=[".*_wrist_.*"],
                effort_limit_sim=float(wrist_effort_limit) if wrist_effort_limit is not None else eff_base,
                velocity_limit_sim=float(wrist_velocity_limit) if wrist_velocity_limit is not None else vel_base,
                stiffness=float(wrist_stiffness) if wrist_stiffness is not None else kp_base,
                damping=float(wrist_damping) if wrist_damping is not None else kd_base,
                armature=float(wrist_armature) if wrist_armature is not None else arm_iner,
            ),
        }

    # Resolve per-joint parameters across the 14 arm joints and partition into strictly disjoint ActuatorGroups.
    buckets: dict[tuple[float, float, float, float, float], list[str]] = {}
    for jname in ARM_JOINT_NAMES:
        is_wrist = "_wrist_" in jname
        kp = float(wrist_stiffness) if (is_wrist and wrist_stiffness is not None) else kp_base
        kd = float(wrist_damping) if (is_wrist and wrist_damping is not None) else kd_base
        eff = float(wrist_effort_limit) if (is_wrist and wrist_effort_limit is not None) else eff_base
        vel = float(wrist_velocity_limit) if (is_wrist and wrist_velocity_limit is not None) else vel_base
        iner = float(wrist_armature) if (is_wrist and wrist_armature is not None) else arm_iner
        for pat, val in (arm_stiffness_overrides or {}).items():
            if re.fullmatch(str(pat), jname):
                kp = float(val)
        for pat, val in (arm_damping_overrides or {}).items():
            if re.fullmatch(str(pat), jname):
                kd = float(val)
        for pat, val in (arm_effort_limit_overrides or {}).items():
            if re.fullmatch(str(pat), jname):
                eff = float(val)
        buckets.setdefault((kp, kd, eff, vel, iner), []).append(jname)

    out: dict[str, ActuatorGroup] = {}
    for idx, ((kp, kd, eff, vel, iner), jlist) in enumerate(buckets.items()):
        if idx == 0:
            gname = "arm"
        elif all("_wrist_" in j for j in jlist) and "wrist" not in out:
            gname = "wrist"
        else:
            gname = f"arm_{idx}"
        out[gname] = ActuatorGroup(
            joint_names_expr=jlist,
            effort_limit_sim=eff,
            velocity_limit_sim=vel,
            stiffness=kp,
            damping=kd,
            armature=iner,
        )
    return out


def actuators(mobile=False, *, cfg: Mapping[str, Any] | None = None, **arm_overrides):
    groups = {
        # Official column speed 76.5 mm/s (AgvClient.HeightAdjust), shared by the two equal stages.
        "lift": ActuatorGroup(
            joint_names_expr=["LZ_.*_Joint"],
            effort_limit_sim=1e4,
            velocity_limit_sim=0.0765 / 2,
            stiffness=1e6,
            damping=1e4,
        ),
        "waist": ActuatorGroup(
            joint_names_expr=["Yaw_Joint", "torso_Joint"],
            effort_limit_sim=1e4,
            velocity_limit_sim=5.0,  # URDF
            stiffness=1e6,
            damping=1e4,
        ),
    }
    if mobile:
        # Official chassis limits (AgvClient.Move): 1.5 m/s, 0.6 rad/s.
        groups["base_xy"] = ActuatorGroup(
            joint_names_expr=["base_[xy]_joint"],
            effort_limit_sim=1e4,
            velocity_limit_sim=1.5,
            stiffness=1e6,
            damping=1e4,
        )
        groups["base_yaw"] = ActuatorGroup(
            joint_names_expr=["base_yaw_joint"],
            effort_limit_sim=1e4,
            velocity_limit_sim=0.6,
            stiffness=1e6,
            damping=1e4,
        )
    groups.update(_build_arm_actuator_groups(cfg, **arm_overrides))
    groups["gripper"] = ActuatorGroup(
        joint_names_expr=[".*_dex1_finger_joint_.*"],
        # Real Dex1-1: motor PD kp=5 N*m/rad, target clipped 0.18 rad ahead (teleop); at ~8.3 mm/rad per
        # finger (assumed linear, uncalibrated) that is ~3.6e4 N/m and ~55 N per finger. Not measured.
        # Full force within ~1.5 mm, so thin handles are still squeezed before the stroke ends.
        effort_limit_sim=55.0,
        stiffness=3.6e4,
        # The real controllers clip each target near the current position (teleop +-0.18 rad plus smoothing),
        # so the real G1-D closes at ~0.064 m/s per finger (peak rate in the G1D teleop recordings, 30 Hz),
        # not the URDF's 0.2 m/s (Dex1-1 fastest setting, 0.24 s full stroke).
        velocity_limit_sim=0.064,
        damping=100,
    )
    return groups


def init_joint_pos(mobile=False):
    return {
        "LZ_mt_Joint": 0.0,  # lift column stages, 0..0.21 m each; raise for taller tables
        "LZ_it_Joint": 0.0,
        "Yaw_Joint": 0.0,  # waist pitch (forward lean) despite the Unitree name
        "torso_Joint": 0.0,  # waist yaw
        ".*_shoulder_.*": 0.0,
        ".*_elbow_joint": -1.0,  # forearms raised (0 = forward at table height, through the objects)
        ".*_wrist_.*": 0.0,
        ".*_dex1_finger_joint_.*": -0.02,  # open (Dex1-1: positive closes)
        **({"base_.*_joint": 0.0} if mobile else {}),
    }


def physics_contract():
    with open(f"{ROBOTS_PATH}/g1d/physics_contract.json") as f:
        return json.load(f)


def filtered_pairs():
    """Dex1-1 rack/carrier hulls overlap inside the gripper housing, and some neighbours are not parent-child: the
    pairs listed with the v4 asset (scripts/g1d/build_asset.py) never collide."""
    contract = physics_contract()
    pairs = [(a, b) for a, others in contract["filtered_pairs"].items() for b in others]
    return pairs + [tuple(p) for p in contract["additional_internal_slider_carrier_pairs"]]


def get_robot_spec(mobile=False, cfg: Mapping[str, Any] | None = None, **arm_overrides):
    if cfg is None and not arm_overrides:
        cfg, _ = caller_robot_yml_cfg()
    return RobotSpec(
        urdf_path=f"{ROBOTS_PATH}/g1d/{urdf_name(mobile)}.urdf",
        actuators=actuators(mobile, cfg=cfg, **arm_overrides),
        init_joint_pos=init_joint_pos(mobile),
        fix_base=True,
        disable_gravity=True,
        self_collisions=True,
        filtered_pairs=filtered_pairs(),
        finger_pads=FINGER_PADS,
        # The wrist camera's lens bezel is a solid shell 10-12 mm in front of the lens (the real one has the
        # aperture): only its visual is hidden (see g1d.py), so the camera's near plane can come in to 1 cm.
        hidden_visuals=["housing_visual.*"],
        link_finishes={pattern: FINISHES[name] for pattern, name in LINK_FINISHES.items()},
    )
