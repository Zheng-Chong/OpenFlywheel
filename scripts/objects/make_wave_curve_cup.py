"""Wave curve cup: physical registration by 杨开元, 2026-10-29 21:25:39.

Usage: python scripts/objects/make_wave_curve_cup.py [RIGID_DIR]
Metres, Z up, bounding-box centre origin. Outer lip diameter, lowest support
diameter and full height are measured; local wall/groove details are photo
hypotheses. Molded grooves are geometry, not printed decoration. No visible
label in the one supplied photo; rear and underside remain undocumented.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import make_pitcher as mp
import make_plastic_cups as cups
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
ID = "Rigid/wave_curve_cup/00000"
REVISION = "4245652bc6c6f64f2d6ae929ce1efd467418b31e21bef4c2001a1550168abcbe"
PHOTO_SHA = "90e0ebefe1d1bf93438644f6c2f9fd0bb3412687a9b2490732ee23dc9014ed54"
# Measured outside rim/support diameters and total envelope height (m).
H, RB, RT = 0.09301, 0.06061 / 2, 0.07276 / 2
# Unmeasured local geometry inferred from the visible lip and wall proportions.
WALL, FLOOR = 0.0020, 0.0030
LOOKS = [
    ("yellow", "polypropylene", (1.0, 0.46, 0.006), 0.26, 0.0, None),
    ("blue", "glazed ceramic", (0.018, 0.12, 0.65), 0.15, 0.0, 2300),
    ("red", "silicone", (0.65, 0.018, 0.025), 0.78, 0.0, 1100),
    ("silver", "brushed aluminium", (0.58, 0.62, 0.68), 0.30, 1.0, 2700),
]


def verify():
    p = ROOT / "real2sim/objects" / ID
    spec = json.loads((p / "spec.json").read_text())
    text = (p / "modeling_prompt.md").read_text()
    data = json.JSONDecoder().raw_decode(text[text.index("{", text.index("## 已确认输入")) :])[0]
    assert (
        hashlib.sha256(
            json.dumps({"version": "real-object-v3", "inputs": data}, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        == REVISION
    )
    assert spec["modeling_confirmation"]["revision"] == REVISION
    assert spec["modeling_confirmation"]["version"] == "real-object-v3"
    mapping = {"dimensions": "dimensions_mm", "real_mass": "mass_g", "gen": "generalization"}
    for key in [
        "name",
        "name_en",
        "material",
        "color",
        "features",
        "requirements",
        "notes",
        "dimensions",
        "real_mass",
        "dimension_policy",
        "mass_policy",
        "reference_source",
        "physics_requirements",
    ]:
        assert data[key] == spec.get(mapping.get(key, key)), key
    gen = data["gen"]
    for key, value in spec["generalization"].items():
        assert gen[key] == value, key
    assert gen["count"] == 3 and gen["color"] and gen["material"]
    assert not any(gen[k] for k in ["size", "shape", "rt_color", "rt_scale"])
    photos = list((p / "reference").glob("*"))
    assert len(photos) == 1
    assert hashlib.sha256(photos[0].read_bytes()).hexdigest() == PHOTO_SHA
    assert spec["dimensions_mm"] == "开口直径72.76，底部直径60.61，高度93.01"
    assert spec["mass_g"] == 110.9
    return spec


def outer(z, theta):
    """Taper with five shallow molded wave grooves seen in the photograph."""
    t = z / H
    r = RB + (RT - RB - 0.0006) * t
    # Broad, rounded groove channels; spacing and depth estimated from photo.
    for index, center in enumerate([0.012, 0.024, 0.038, 0.052, 0.069]):
        center = center + 0.0025 * np.sin(theta + index * 1.3)
        r -= 0.00055 * np.exp(-(((z - center) / 0.00085) ** 2))
    if z > H - 0.003:
        r += 0.0006 * (z - (H - 0.003)) / 0.003
    return r


def inner(z):
    # The observed inside is smooth; no invented internal grooves.
    return RB - WALL + (RT - RB) * z / H


def build():
    sec = cups.unit_section(2)
    z = np.unique(
        np.r_[
            np.linspace(0, H - 0.002, 220),
            H - 0.003,
            [c + d for c in [0.012, 0.024, 0.038, 0.052, 0.069] for d in [-0.0017, -0.00085, 0, 0.00085, 0.0017]],
        ]
    )
    theta = np.arctan2(sec[:, 1], sec[:, 0])
    profile = [(0.0, 0.0)] + [(outer(h, theta)[:, None] * sec, h) for h in z]
    profile += [(RT, H - 0.001)]
    profile += [(RT - 0.001 + 0.001 * np.cos(a), H - 0.001 + 0.001 * np.sin(a)) for a in np.linspace(0, np.pi, 17)[1:]]
    profile += [(inner(FLOOR), FLOOR), (0.0, FLOOR)]
    pts, faces = cups.loft(profile, sec)
    volume = mp.signed_volume(pts, faces)
    assert volume > 0
    # Collision wall follows a conservative outer envelope. Smooth inside is
    # explicitly represented by 32 staves; no convex hull spans the mouth.
    hulls = [np.array([(*(RB * p), z) for p in sec[:: cups.N_THETA // 24] for z in [0, FLOOR]])]
    levels = [(0.0, RB * sec, inner(0) * sec), (H, RT * sec, inner(H) * sec)]
    hulls += cups.wall_staves(levels)
    c = dict(
        pts=pts,
        faces=faces,
        normals=mp.vertex_normals(pts, faces),
        hulls=hulls,
        probe=cups.cavity_probe(lambda z: inner(z) * sec, FLOOR, H),
        ml=cups.volume_ml(lambda z: inner(z) * sec, FLOOR, H),
        contact_r=RB,
        volume=volume,
    )
    cups.self_check(c)
    for hull in hulls:
        assert len(ConvexHull(hull).vertices) <= 64
    return c


def measurements(c):
    p = c["pts"]
    z = p[:, 2]
    top, bottom = p[z >= z.max() - 0.003], p[np.isclose(z, z.min(), atol=1e-8)]
    return {
        "开口直径": round(float(np.ptp(top[:, 0])) * 1000, 2),
        "底部直径": round(float(np.ptp(bottom[:, 0])) * 1000, 2),
        "高度": round(float(np.ptp(z)) * 1000, 2),
    }


def description(uid, look):
    color, material = look[:2]
    caption = dict(
        color=[color, f"{color} body", f"{color} exterior"],
        material=[material, f"{material} body", f"{material} finish"],
        size=["93 mm tall", "73 mm rim diameter", "61 mm base diameter"],
        name=["cup", "drinking cup", "wave curve cup"],
        shape=["round open mouth", "tapered hollow body", "horizontal wave grooves"],
        function=["holding small objects", "containing beans", "serving drinks"],
    )
    return dict(
        uuid=uid,
        caption=caption,
        description=[
            f"{color} {material} cup with horizontal wave grooves",
            "round open cup that widens towards the rim",
            "handleless cup with a flat circular base",
            f"{color} cup with a smooth hollow interior",
            "small grooved drinking cup about 93 mm tall",
        ],
    )


def main(output):
    spec = verify()
    c = build()
    for idx, look in enumerate(LOOKS):
        color, material, rgb, roughness, metal, density = look
        mass = spec["mass_g"] / 1000 if idx == 0 else round(density * c["volume"], 6)

        def mat(stage):
            return mp.add_material(
                stage,
                "/root/_materials/Body",
                "OmniPBR",
                {"diffuse_color_constant": rgb, "reflection_roughness_constant": roughness, "metallic_constant": metal},
                rgb,
                1.0,
                roughness,
            )

        def extra(centered):
            data = dict(
                asset_id=f"Rigid/wave_curve_cup/{idx:05d}",
                registration_revision=REVISION,
                source_photo_sha256=PHOTO_SHA,
                origin="whole visible mesh bounding-box centre",
                units="m",
                up_axis="Z",
                author="杨开元",
                generator="scripts/objects/make_wave_curve_cup.py",
                reference_source="physical photograph; shape and appearance reference",
                simulation_use="rigid container; material response uncalibrated",
                estimated_local_geometry_mm=dict(wall=2, bottom=3, groove_depth=0.55),
                estimated_capacity_ml=round(c["ml"], 1),
                missing_photos=["rear", "underside"],
                collision_approximation="base slab and convex wall staves; external grooves smoothed",
                friction_source="unmeasured; project defaults",
                real_material_calibrated=False,
                visual=dict(color_parts=["model"], material=material, color=rgb, roughness=roughness),
            )
            if idx == 0:
                data.update(measurements_mm=measurements(centered), mass_policy="measured", dimension_policy="measured")
            else:
                data.update(
                    variant_of=ID,
                    mass_policy="estimate",
                    dimension_policy="inherited geometry",
                    estimated_mass_g=round(mass * 1000, 3),
                    mass_basis=f"assumed density {density} kg/m3 times mesh shell volume",
                    inherited_dimensions_mm=measurements(centered),
                )
            return data

        d, meta = cups.save(output, "wave_curve_cup", idx, c, mat, mass, lambda uid: description(uid, look), extra)
        (d / "runtime_config.json").write_text(json.dumps({"physics": {"mass": mass}}, indent=2))
        print(d, meta["physics"], meta.get("measurements_mm"), flush=True)


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Assets/Object/Real2Sim/Rigid")
    except Exception:
        if mp.APP is None:
            raise
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    if mp.APP is not None:
        mp.APP.close()
