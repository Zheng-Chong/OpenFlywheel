"""Read authored hand USD dynamics and relationships; never infer real calibration."""

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdPhysics


def inspect(path):
    stage = Usd.Stage.Open(str(path))
    result = {
        "entry": str(path),
        "joints": {},
        "filtered_pairs": {},
        "rigid_bodies": {},
        "unresolved_relationships": [],
    }
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            attrs = {}
            for a in prim.GetAttributes():
                if a.GetName().startswith(("drive:", "physxJoint:", "physics:lowerLimit", "physics:upperLimit")):
                    value = a.Get()
                    if isinstance(value, float | int | str | bool):
                        attrs[a.GetName()] = value
            result["joints"][prim.GetName()] = attrs
        if prim.HasAPI(UsdPhysics.MassAPI):
            result["rigid_bodies"][prim.GetName()] = {
                a.GetName(): str(a.Get())
                for a in prim.GetAttributes()
                if a.GetName().startswith(("physics:mass", "physics:diagonalInertia", "physxRigidBody:"))
            }
        for rel in prim.GetRelationships():
            if rel.GetName() == "physics:filteredPairs":
                result["filtered_pairs"][prim.GetName()] = [str(t) for t in rel.GetTargets()]
            for target in rel.GetTargets():
                if not stage.GetPrimAtPath(target.GetPrimPath()):
                    result["unresolved_relationships"].append([str(rel.GetPath()), str(target)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for model in ("g1d_wuji_hand2_1", "g1d_wuji_hand2_2", "g1d_sharpa_wave"):
        result[model] = {}
        for side in ("left", "right"):
            if "sharpa" in model:
                path = (
                    args.vendor
                    / "sharpa-models/wave_01"
                    / (side + "_sharpa_wave")
                    / (side + "_sharpa_wave_with_flange.usda")
                )
            else:
                rev = model.removeprefix("g1d_wuji_")
                path = args.vendor / "wuji-description/hand2" / rev / "body/usd" / side / ("wuji" + rev + ".usd")
            result[model][side] = inspect(path)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print({m: {s: len(d["joints"]) for s, d in sides.items()} for m, sides in result.items()})


if __name__ == "__main__":
    main()
