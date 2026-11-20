"""Independent CPU convex-geometry check of Sharpa inter-finger clearance.

MuJoCo is used for collision detection only at prescribed joint states. This
does not claim that its dynamics match the native Isaac validation run.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = ET.parse(args.urdf).getroot()
    ET.SubElement(ET.SubElement(root, "mujoco"), "compiler", strippath="false")
    for mesh in root.iter("mesh"):
        mesh.set("filename", str((args.urdf.parent / mesh.get("filename")).resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]

    def body(geom):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])

    def finger(name):
        return next((f for f in ("thumb", "index", "middle", "ring", "pinky") if "_" + f + "_" in name), None)

    rows = []
    for side in ("left", "right"):
        for name in [n for n in names if n.startswith(side + "_") and n.endswith("_AA")]:
            for value in (0.0, 0.15, -0.15, 0.04):
                data = mujoco.MjData(model)
                for n in ("left_elbow_joint", "right_elbow_joint"):
                    data.qpos[model.jnt_qposadr[names.index(n)]] = -0.6
                data.qpos[model.jnt_qposadr[names.index(name)]] = value
                mujoco.mj_forward(model, data)
                contacts = []
                for c in data.contact:
                    a, b = body(c.geom1), body(c.geom2)
                    if (
                        c.dist < -0.00001
                        and a.startswith(side + "_")
                        and b.startswith(side + "_")
                        and finger(a)
                        and finger(b)
                        and finger(a) != finger(b)
                    ):
                        contacts.append({"bodies": [a, b], "separation_m": float(c.dist)})
                rows.append({"joint": name, "target_rad": value, "interfinger_contacts": contacts})
    args.output.write_text(
        json.dumps(
            {
                "urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
                "engine": "MuJoCo " + mujoco.__version__,
                "scope": "convex collision geometry only",
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
