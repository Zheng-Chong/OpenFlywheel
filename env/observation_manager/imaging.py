"""Image perturbation a layout asks for (its "Imaging" entry), applied to every RGB frame before it reaches the
policy or the dataset, so collection and evaluation see the same images:

    {"brightness": 1.1, "contrast": 0.9, "saturation": 1.2, "noise": 0.02, "blur": 1.5, "seed": 7,
     "cameras": ["cam_head", ...]}

brightness, contrast and saturation are factors (1 keeps the image); noise is the std of Gaussian pixel noise as a
share of full scale, drawn per frame and camera from `seed`; blur a Gaussian sigma in pixels; `cameras` limits it to
those cameras (default all). Missing keys leave that property alone.
"""

import zlib

import numpy as np
from scipy import ndimage

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def perturb(rgb, spec, camera, frame):
    """The uint8 HxWx3 frame `rgb` of `camera` (the `frame`-th of the episode) perturbed as `spec` says."""
    if not spec or (spec.get("cameras") and camera not in spec["cameras"]):
        return rgb
    img = np.asarray(rgb, dtype=np.float32) / 255.0
    if (b := spec.get("brightness")) is not None:
        img = img * b
    if (c := spec.get("contrast")) is not None:
        mean = float(img.mean())
        img = (img - mean) * c + mean
    if (s := spec.get("saturation")) is not None:
        gray = (img @ LUMA)[..., None]
        img = gray + (img - gray) * s
    if sigma := spec.get("blur"):
        img = ndimage.gaussian_filter(img, sigma=(sigma, sigma, 0))
    if noise := spec.get("noise"):
        rng = np.random.default_rng([int(spec.get("seed", 0)), int(frame), zlib.crc32(camera.encode())])
        img = img + rng.normal(0.0, noise, img.shape).astype(np.float32)
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
