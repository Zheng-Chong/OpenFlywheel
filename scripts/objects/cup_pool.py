"""The cups of g1d_pour_beans_plus (倒豆子 Plus): every cup in the shared library that stands and holds red beans
(scripts/objects/check_cup_holds_beans.py), with its profile (scripts/objects/cup_profile.py), its family, and whether
it is a training cup or one kept for evaluation only.

    build  profiles from a cup check run (metrics.json, cavity/<n>.npz) and the cups' metadata.json:
           python scripts/objects/cup_pool.py build --run <check run dir> --assets Assets
    refine-3d  refine existing profiles in cups.json from the cups' 3D USD meshes (recovers inward cavity bellies,
               outward bulges, corner radii and recessed inner floors):
               python scripts/objects/cup_pool.py refine-3d --assets Assets
    split  the evaluation cups: whole families (unseen shapes) plus a share of every other family, ~SHARE in all, of
           the cups the expert can pour (--feasible: a JSON list of cup ids; default all):
           python scripts/objects/cup_pool.py split [--feasible feasible.json]
Writes real2sim/tasks/g1d_pour_beans_plus/cups.json: {"cups": {id: {...}}, "split": {...}}. Ids are object
directories under Assets/Object (`Real2Sim/Rigid/plastic_cup/00001`).
"""

import argparse
import json
from pathlib import Path
import random
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.objects import cup_profile  # noqa: E402

POOL = ROOT / "real2sim" / "tasks" / "g1d_pour_beans_plus" / "cups.json"
SHARE = 0.2  # of the cups kept for evaluation only
HELD_FAMILIES = ("wave_curve_cup", "simready_glass")  # shapes no training episode sees (stemmed glasses, wavy walls)
MIN_TRAIN = 2  # cups a family keeps for training when some of it is held out
TWIN = "Real2Sim/Rigid/plastic_cup/00000"  # the real cups' twin (L0, G0): always a training cup
SPLIT_SEED = 20261103
FALLBACK_MASS = 0.2  # kg, as the cup check, for SimReady cups without one
# Cavity floor over the underside: Rigid/metal_cup/00000 (0.1 mm) holds beans standing still but they stay on the desk
# when it is lifted (C probe 2026-11-04); the cup check never lifts a cup
FLOOR_MIN = 0.0005


def family(cup_id):
    """Shape family of a cup id: the category, with the imported packages grouped by kind."""
    category = cup_id.split("/")[-2]
    for pattern, name in (
        (r"geniesim_.*beer_mug", "geniesim_beer_mug"),
        (r"geniesim_.*pen_cup", "geniesim_pen_cup"),
        (r"geniesim_", "geniesim_cup"),
        (r"simready_.*glass", "simready_glass"),
        (r"simready_.*mug", "simready_mug"),
        (r"simready_", "simready_cup"),
    ):
        if re.match(pattern, category):
            return name
    return category


def entry(cup_id, record, npz, meta, names=None):
    """A pool entry: the check's record, the profile from its height map, size and mass from the metadata, its names
    (description.json "names": {"zh": [short, ..., detailed], "en": [...]}) for the instructions."""
    phys = meta.get("physics") or {}
    size = phys.get("size") or record["size_m"]
    box = (meta.get("geometry") or {}).get("aligned_bbox", {}).get("vertices")
    # SimReady packages have no bbox and need not put the root at its centre: the underside is a height below the top
    bottom = float(np.min(np.asarray(box)[:, 2])) if box else float(np.nanmax(npz["height"])) - size[2]
    prof = cup_profile.profile(npz["xs"], npz["ys"], npz["height"], npz["level"], npz["cavity"], bottom)
    return {
        "family": family(cup_id),
        "size": [round(float(v), 4) for v in size],
        "mass": float(phys.get("mass") or FALLBACK_MASS),
        "capacity": round(cup_profile.capacity(prof)),
        "volume_ml": round(cup_profile.volume(prof) * 1e6),
        # raw SimReady packages (no physics yet) need an object.usd adapter first
        "loads": (bool(meta.get("physics")) or not cup_id.split("/")[2].startswith("simready_"))
        and prof["z"][0] - prof["bottom"] >= FLOOR_MIN,
        "names": {lang: list((names or {}).get(lang) or []) for lang in ("zh", "en")},
        "cavity": prof,
    }


def _mesh_triangles(folder):
    """(points (V, 3), triangles (F, 3)) in the cup root frame from `folder/object.usd*`, or (None, None)."""
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        return None, None
    usds = [folder / n for n in ("object.usdz", "object.usd") if (folder / n).is_file()]
    if not usds:
        usds = sorted(p for p in folder.iterdir() if p.suffix in (".usd", ".usda")) if folder.is_dir() else []
    if not usds:
        return None, None
    stage = Usd.Stage.Open(str(usds[0]))
    if not stage:
        return None, None
    cache = UsdGeom.XformCache()
    vis_v, vis_t, vis_off = [], [], 0
    col_v, col_t, col_off = [], [], 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=float)
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not len(pts) or not counts or not indices:
            continue
        counts_np, idx_np = np.asarray(counts, int), np.asarray(indices, int)
        starts = np.r_[0, np.cumsum(counts_np)[:-1]]
        tris = np.asarray(
            [(idx_np[s], idx_np[s + k], idx_np[s + k + 1]) for s, c in zip(starts, counts_np) for k in range(1, c - 1)],
            dtype=int,
        ).reshape(-1, 3)
        tf = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float)
        world = pts @ tf[:3, :3] + tf[3, :3]
        if "/collision/" in str(prim.GetPath()):
            col_v.append(world)
            col_t.append(tris + col_off)
            col_off += len(world)
        else:
            vis_v.append(world)
            vis_t.append(tris + vis_off)
            vis_off += len(world)
    if vis_v:
        return np.concatenate(vis_v), np.concatenate(vis_t)
    if col_v:
        return np.concatenate(col_v), np.concatenate(col_t)
    return None, None


def build(args):
    run = Path(args.run)
    metrics = json.loads((run / "metrics.json").read_text())["cups"]
    cups = {}
    for n, (cup_id, record) in enumerate(metrics.items()):
        if not (record["pass"] and record["stands"]):
            continue
        folder = Path(args.assets) / "Object" / cup_id
        meta = json.loads((folder / "metadata.json").read_text())
        desc = folder / "description.json"
        names = json.loads(desc.read_text(encoding="utf-8")).get("names") if desc.is_file() else None
        item = entry(cup_id, record, np.load(run / "cavity" / f"{n:03d}.npz"), meta, names)
        pts, tris = _mesh_triangles(folder)
        if pts is not None:
            prof = cup_profile.refine_from_mesh(item["cavity"], pts, tris)
            item["cavity"] = prof
            item["capacity"] = round(cup_profile.capacity(prof))
            item["volume_ml"] = round(cup_profile.volume(prof) * 1e6)
            item["loads"] = item["loads"] and prof["z"][0] - prof["bottom"] >= FLOOR_MIN
        cups[cup_id] = item
    old = json.loads(POOL.read_text()) if POOL.is_file() else {}
    POOL.parent.mkdir(parents=True, exist_ok=True)
    pool = {"source": str(args.run), "cups": cups, "split": old.get("split")}
    POOL.write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(cups)} cups -> {POOL.relative_to(ROOT)}")


def refine_3d(args):
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    refined = 0
    for cup_id, item in sorted(pool["cups"].items()):
        folder = Path(args.assets) / "Object" / cup_id
        pts, tris = _mesh_triangles(folder)
        if pts is None:
            continue
        prof = cup_profile.refine_from_mesh(item["cavity"], pts, tris)
        item["cavity"] = prof
        item["capacity"] = round(cup_profile.capacity(prof))
        item["volume_ml"] = round(cup_profile.volume(prof) * 1e6)
        item["loads"] = bool(item["loads"]) and prof["z"][0] - prof["bottom"] >= FLOOR_MIN
        refined += 1
    POOL.write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"refined {refined}/{len(pool['cups'])} cups -> {POOL.relative_to(ROOT)}")


def choose_eval(cups, held_families=HELD_FAMILIES, share=SHARE, seed=SPLIT_SEED):
    """Evaluation cup ids: every cup of `held_families`, then from each other family its share of the rest (largest
    remainder) so that ~`share` of all are held out, each family keeping MIN_TRAIN for training."""
    by_family = {}
    for cup_id in sorted(cups):
        by_family.setdefault(cups[cup_id]["family"], []).append(cup_id)
    held = [c for f in held_families for c in by_family.get(f, [])]
    want = round(share * len(cups)) - len(held)
    rest = {f: ids for f, ids in by_family.items() if f not in held_families and len(ids) > MIN_TRAIN}
    pool = sum(len(ids) for ids in rest.values())
    quota = {f: want * len(ids) / pool for f, ids in rest.items()} if pool and want > 0 else {}
    take = {f: min(int(q), len(rest[f]) - MIN_TRAIN) for f, q in quota.items()}
    for f in sorted(quota, key=lambda f: quota[f] - int(quota[f]), reverse=True):
        if sum(take.values()) >= want:
            break
        if take[f] < len(rest[f]) - MIN_TRAIN:
            take[f] += 1
    rng = random.Random(seed)
    for f, k in sorted(take.items()):
        held += rng.sample([c for c in rest[f] if c != TWIN], k)
    return sorted(held)


def split(args):
    pool = json.loads(POOL.read_text())
    cups = pool["cups"]
    if args.feasible:
        ok = set(json.loads(Path(args.feasible).read_text()))
        cups = {c: v for c, v in cups.items() if c in ok}
    held = choose_eval(cups)
    pool["split"] = {
        "share": SHARE,
        "held_families": list(HELD_FAMILIES),
        "seed": SPLIT_SEED,
        "train": sorted(set(cups) - set(held)),
        "eval": held,
        "excluded": sorted(set(pool["cups"]) - set(cups)),
    }
    POOL.write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"train {len(cups) - len(held)}, eval {len(held)}, excluded {len(pool['split']['excluded'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run", required=True)
    b.add_argument("--assets", default=str(ROOT / "Assets"))
    r = sub.add_parser("refine-3d")
    r.add_argument("--assets", default=str(ROOT / "Assets"))
    s = sub.add_parser("split")
    s.add_argument("--feasible")
    args = parser.parse_args()
    {"build": build, "refine-3d": refine_3d, "split": split}[args.cmd](args)


if __name__ == "__main__":
    main()
