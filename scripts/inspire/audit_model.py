"""Read-only source fidelity, mount anatomy and contact-fixture audit."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from scripts.dexhands.kinematics import fk, origin_transform


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def semantic(e):
    return e.tag, sorted(e.attrib.items()), [semantic(c) for c in e]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = ET.parse(a.asset / "robot.urdf").getroot()
    c = json.loads((a.asset / "model_contract.json").read_text())
    result = {
        "hands": {},
        "urdf_sha256": sha(a.asset / "robot.urdf"),
        "scope": "CAD/source and simulation convention; flange holes and physical TCP unmeasured",
    }
    fixtures = []
    for side, h in c["hands"].items():
        source = a.source / h["source"].removeprefix("attachment/")
        original = ET.parse(source).getroot()
        for e in list(original):
            if (
                e.tag == "link"
                and e.get("name") == "world"
                or e.tag == "joint"
                and e.find("parent").get("link") == "world"
            ):
                original.remove(e)
        original.find("link[@name='base_link']").set("name", h["root"])
        for n in original.iter():
            if n.get("link") == "base_link":
                n.set("link", h["root"])
        unchanged, mesh_ok, inertia_ok, limit_changes = True, [], [], []
        for j in original.findall("joint"):
            dst = copy.deepcopy(root.find(f"joint[@name='{j.get('name')}']"))
            src = copy.deepcopy(j)
            if src.find("limit").attrib != dst.find("limit").attrib:
                limit_changes.append(
                    {"joint": j.get("name"), "source": src.find("limit").attrib, "effective": dst.find("limit").attrib}
                )
            src.remove(src.find("limit"))
            dst.remove(dst.find("limit"))
            unchanged &= semantic(src) == semantic(dst)
        for link in original.findall("link"):
            dst = copy.deepcopy(root.find(f"link[@name='{link.get('name')}']"))
            inertia_ok.append(semantic(link.find("inertial")) == semantic(dst.find("inertial")))
            inertia = link.find("inertial/inertia").attrib
            mat = np.array(
                [
                    [float(inertia[k]) for k in keys]
                    for keys in (("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz"))
                ]
            )
            inertia_ok.append(bool(np.linalg.eigvalsh(mat).min() > 0))
            for m, d in zip(link.findall(".//mesh"), dst.findall(".//mesh"), strict=True):
                srcmesh = source.parent.parent / "meshes" / Path(m.get("filename")).name
                mesh_ok.append(sha(srcmesh) == sha(a.asset / d.get("filename")))
                m.set("filename", d.get("filename"))
            unchanged &= semantic(link) == semantic(dst)
        errors = []
        for fraction in (0, 0.5, 1):
            q = {n: v["lower"] + fraction * (v["upper"] - v["lower"]) for n, v in h["effective_active_limits"].items()}
            srcposes, dstposes = fk(original, q), fk(root, q)
            inv = np.linalg.inv(dstposes[h["root"]])
            errors.append(max(float(abs(srcposes[n] - inv @ dstposes[n]).max()) for n in srcposes))
        q = {"left_elbow_joint": -0.6, "right_elbow_joint": -0.6, **h["initial_rad"]}
        poses = fk(root, q)
        perturbed = fk(root, {**q, side + "_index_1_joint": 0.001})
        lateral = poses[side + "_index_1"][:3, 3] - poses[side + "_little_1"][:3, 3]
        lateral /= np.linalg.norm(lateral)
        curl = perturbed[side + "_index_2"][:3, 3] - poses[side + "_index_2"][:3, 3]
        curl /= np.linalg.norm(curl)
        inward = np.array([0, -1 if side == "left" else 1, 0])
        anatomy = {
            "thumb_side_up_cosine": float(lateral[2]),
            "positive_index_flexion_inward_cosine": float(curl @ inward),
        }
        result["hands"][side] = {
            "joint_origins_axes_mimics_preserved": unchanged,
            "mesh_bytes_preserved": all(mesh_ok),
            "mesh_references": len(mesh_ok),
            "positive_source_inertias_preserved": all(inertia_ok),
            "feasible_limit_intersections": limit_changes,
            "mount_relative_fk_max_matrix_error": max(errors),
            "configured_mass_kg": h["configured_mass_kg"],
            "anatomy": anatomy,
            "passed": bool(
                unchanged
                and all(mesh_ok)
                and all(inertia_ok)
                and max(errors) < 1e-10
                and lateral[2] > 0.6
                and curl @ inward > 0.95
            ),
        }
        # Contact fixtures use the actual convex hull used by PhysX, with a
        # palm-facing outward normal; these are NOT factory taxel locations.
        thumb = side + ("_thumb_3" if side == "left" else "_thumb_4")
        for finger, name in [(f, side + "_" + f + "_2") for f in ("little", "ring", "middle", "index")] + [
            ("thumb", thumb),
            ("palm", h["root"]),
        ]:
            link = root.find(f"link[@name='{name}']")
            node = link.find("collision/geometry/mesh")
            mesh = trimesh.load(a.asset / node.get("filename"), force="mesh")
            mesh.apply_transform(origin_transform(link.find("collision/origin")))
            hull = mesh.convex_hull
            palm_normal_base = np.array([0, -1 if side == "left" else 1, 0])
            normal_link = poses[name][:3, :3].T @ poses[h["root"]][:3, :3] @ palm_normal_base
            score = hull.face_normals @ normal_link
            candidate = np.flatnonzero(score > 0.95)
            if not len(candidate):
                candidate = np.flatnonzero(score > score.max() - 0.05)
            # Largest planar triangle interior avoids seams and fingertip edges.
            index = candidate[np.argmax(hull.area_faces[candidate])]
            fixtures.append(
                {
                    "side": side,
                    "finger": finger,
                    "link": name,
                    "point_m": hull.triangles_center[index].tolist(),
                    "normal": hull.face_normals[index].tolist(),
                    "source": "native_convex_hull_face_interior",
                    "hardware_taxel_location": False,
                }
            )
    result["rejected_sources"] = [
        {"path": str(p.relative_to(a.source)), "sha256": sha(p), "reason": reason}
        for p, reason in [
            (
                next(a.source.rglob("left.urdf")),
                "negative thumb parent with positive slave limits; use coherent matching xacro",
            ),
            (
                a.source / "urdf_right_11_5.urdf",
                "different base frame, changed thumb coupling and absent matching mesh package",
            ),
        ]
    ]
    result["passed"] = all(h["passed"] for h in result["hands"].values())
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "model_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (a.output / "contact_fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
