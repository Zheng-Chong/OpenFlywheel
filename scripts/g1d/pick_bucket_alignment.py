"""Transfer the archived episode-0001 image-fit candidate into native Isaac coordinates.

This is conditional image alignment, not measured camera/robot calibration. Object sizes
and masses still come from the registered native assets, never the old proxy scene.
"""

import json
import math
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "real2sim/tasks/pick_bucket/recorded_alignment.json"
# Source coordinates are torso-relative, not chassis-relative. The native URDF
# torso is +53.371 mm forward and +0.168 mm left of AGV_link at zero pitch.
ROOT_XY = (0.0, -0.79)
AGV_XY = (0.000167718, -0.8433714301)
TABLE_TOP = 0.7628
NATIVE_ZERO_TORSO_Z = 0.1115 + 0.733508566


def source():
    return json.loads(SOURCE.read_text())


def xy(point):
    return [-point[1] + ROOT_XY[0], point[0] + ROOT_XY[1]]


def init_joints():
    data = source()
    lift = TABLE_TOP - data["table_source_top_m"] - NATIVE_ZERO_TORSO_Z
    joints = {"LZ_mt_Joint": lift / 2, "LZ_it_Joint": lift / 2}
    names = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    for side in ("left", "right"):
        for name, q in zip(names, data["initial_state"][side + "_arm"]["qpos"]):
            joints[side + "_" + name + "_joint"] = q
        motor = data["initial_state"][side + "_ee"]["qpos"][0]
        for finger in (1, 2):
            joints[f"{side}_dex1_finger_joint_{finger}"] = 0.0245 - motor / 5.4 * 0.0445
    return joints


def layout(assets, table_entry, room_entry):
    data = source()
    objects = data["object_initialization"]
    rigid = {}
    for role, cat, idx in (("food", "mango_prop", 0), ("basket", "basket", 14)):
        name = "mango" if role == "food" else "basket"
        yaw = math.degrees(objects[name + "_euler"][2]) + 90
        rigid[cat] = [table_entry(assets, TABLE_TOP, cat, idx, role, xy(objects[name]), yaw)]
    return {
        "Rigid": rigid,
        "Room": room_entry("real_scene4", xy(data["table_source_center_m"]) + [0.0], TABLE_TOP),
        "Robot": {"init_joint_pos": init_joints()},
    }


def configure_camera(config):
    from omegaconf import open_dict

    cam = source()["head_camera"]["mjcf_camera"]
    with open_dict(config):
        config.cam_head.camera.mount_link = "robot0/torso_link"
        config.cam_head.camera.pos = cam["pos"]
        config.cam_head.camera.ori = cam["quat_wxyz"]


def apply_lens(manager):
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdGeom

    k = source()["head_camera"]["K_pixels_640x480"]
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    for names, cameras in zip(manager.camera_names, manager.cameras):
        camera = cameras[names.index("cam_head")]
        lens = UsdGeom.Camera(get_current_stage().GetPrimAtPath(camera.prim_path))
        focal = 20.0
        lens.GetFocalLengthAttr().Set(focal)
        lens.GetHorizontalApertureAttr().Set(640 * focal / fx)
        lens.GetVerticalApertureAttr().Set(480 * focal / fy)
        lens.GetHorizontalApertureOffsetAttr().Set((320 - (cx + 0.5)) * focal / fx)
        lens.GetVerticalApertureOffsetAttr().Set(((cy + 0.5) - 240) * focal / fy)


def add_hatch(manager):
    """Closed cable hatch visible in frame 0; photo-fit dimensions, not a movable prop."""
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = get_current_stage()
    data = source()
    center = xy(data["tray"]["center_m"])
    for i in range(manager.num_envs):
        origin = manager.env_origins[i].cpu().numpy()
        root = f"/World/envs/env_{i}/pick_bucket_hatch"
        for name, size, shift, color, metal in (
            ("border", (0.298, 0.112, 0.001), (0, 0, 0.0005), (0.78, 0.79, 0.75), 0.3),
            ("lid", (0.278, 0.091, 0.001), (0, 0.003, 0.0012), (0.38, 0.47, 0.44), 0.85),
            ("slot", (0.278, 0.008, 0.001), (0, -0.047, 0.0013), (0.015, 0.02, 0.017), 0.0),
        ):
            cube = UsdGeom.Cube.Define(stage, root + "/" + name)
            cube.GetSizeAttr().Set(1)
            xf = UsdGeom.Xformable(cube)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(
                Gf.Vec3d(
                    center[0] + shift[0] + float(origin[0]),
                    center[1] + shift[1] + float(origin[1]),
                    TABLE_TOP + shift[2],
                )
            )
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            mat = UsdShade.Material.Define(stage, root + "/" + name + "_material")
            shader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
