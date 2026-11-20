"""Camera config parsing, pose sampling and intrinsics shared by every simulation backend's camera manager."""

from copy import deepcopy

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import torch

from env_cfg.camera.template import (
    D435,
    DEX1_RGB,
    G1D_HEAD,
    GEMINI_345LG,
    LARGE_D435,
    PINHOLE,
    THIRD_VIEW,
)
from utils.rotations import euler_angles_to_quat

REAL_MAP = {
    "Gemini_345Lg": GEMINI_345LG,
    "third_view": THIRD_VIEW,
    "d435": D435,
    "large_d435": LARGE_D435,
    "dex1_rgb": DEX1_RGB,
    "g1d_head": G1D_HEAD,
}

VISUAL_MAP = {
    "pinhole": PINHOLE,
    "d435": PINHOLE,
}

CAPTURE_KEYS = (
    "output_dir",
    "disable_render_product",
    "render_frame_num",
    "enable_tiled",
    "colorize_depth",
    "default_frequency",
    "annotator",
)


def split_camera_config(init_config):
    """(cameras config, random config) of a camera yml: its capture settings dropped."""
    config = deepcopy(OmegaConf.to_container(init_config, resolve=True))
    for key in CAPTURE_KEYS:
        config.pop(key, None)
    random_config = config.pop("random", {})
    return OmegaConf.create(config), OmegaConf.create(random_config)


def random_cfg_for(random_config, camera_name):
    if camera_name in random_config:
        return random_config[camera_name]
    return random_config.get("common", None)


def sample_camera_pose(camera_name, camera_config, random_config, offset=None, use_randomization=True):
    """(position, wxyz orientation) of a camera in its parent frame (mount link or env): config pose, optional domain
    randomization, then the layout's fixed extrinsic offset (pos in the parent frame, ori in the camera's own)."""
    random_cfg = random_cfg_for(random_config, camera_name)
    random_pos = torch.zeros(3)
    random_ori = torch.zeros(3)
    if use_randomization and random_cfg is not None and random_cfg.get("enabled", False):
        if hasattr(random_cfg, "pos") and random_cfg.pos is not None:
            random_pos = torch.from_numpy(
                np.random.uniform(torch.tensor(random_cfg.pos.min), torch.tensor(random_cfg.pos.max))
            )
        if hasattr(random_cfg, "ori") and random_cfg.ori is not None:
            random_ori = torch.from_numpy(
                np.random.uniform(torch.tensor(random_cfg.ori.min), torch.tensor(random_cfg.ori.max))
            )

    if hasattr(camera_config.camera, "ori") and camera_config.camera.ori is not None:
        if torch.tensor(camera_config.camera.ori).shape[0] == 3:
            orientation = euler_angles_to_quat(torch.tensor(camera_config.camera.ori) + random_ori, degrees=True)
        else:
            orientation = torch.tensor(camera_config.camera.ori)
    else:
        orientation = euler_angles_to_quat(torch.tensor([0, 0, 0]))
    if hasattr(camera_config.camera, "pos") and camera_config.camera.pos is not None:
        position = torch.tensor(camera_config.camera.pos)
    else:
        position = torch.tensor([0, 0, 1])
    final_position = position + random_pos
    if offset:
        final_position = final_position + torch.tensor(offset.get("pos", [0.0, 0.0, 0.0]))
        q = np.asarray(orientation.cpu() if hasattr(orientation, "cpu") else orientation, dtype=float)
        turned = Rotation.from_quat(np.roll(q, -1)) * Rotation.from_euler(
            "xyz", offset.get("ori", [0.0, 0.0, 0.0]), degrees=True
        )
        orientation = torch.tensor(np.roll(turned.as_quat(), 1))
    return final_position, orientation


def layout_offset(layout_offsets, env_id, camera_name):
    offsets = layout_offsets or []
    return (offsets[env_id] or {}).get(camera_name) if env_id < len(offsets) else None


def intrinsics(camera_type):
    """3x3 K from the camera template (square pixels and a centred principal point unless calibrated); identity
    without focal/aperture."""
    args_info = REAL_MAP.get(camera_type, None)
    if args_info is None or "focal_length" not in args_info or "horizontal_aperture" not in args_info:
        return np.eye(3, dtype=np.float64)
    width, height = args_info["resolution"]
    fx = (width * args_info["focal_length"]) / args_info["horizontal_aperture"]
    cx, cy = width / 2.0, height / 2.0
    fy = fx
    if args_info.get("horizontal_aperture_offset") is not None:  # calibrated types: fy and principal point
        fy = (height * args_info["focal_length"]) / args_info["vertical_aperture"]
        cx -= args_info["horizontal_aperture_offset"] * width / args_info["horizontal_aperture"]
        cy += args_info["vertical_aperture_offset"] * height / args_info["vertical_aperture"]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def pose_matrix(pos, quat):
    """4x4 from position and wxyz quaternion."""
    w, x, y, z = quat
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = pos
    return T
