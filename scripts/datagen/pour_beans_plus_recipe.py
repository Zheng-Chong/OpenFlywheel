"""The next collection round's recipe for g1d_pour_beans_plus (pour_beans_plus_layouts.py recipe) from a failure
analysis (pour_beans_plus_failures.py summary.json) and the training episodes' Generalization records:

    explore_unseen_combos   the level combos (L0-L2 per dimension) the training data never had: rsi/AGENTS.md marginal
                            exploration (>= 30% of a round); over half of them at object L0, the twin pair
    same_cup_pairs          one training cup at one scale as both source and target, so only the places tell them
                            apart: plus_1000 had 69 such episodes, and its policy pours back from the full cup on 16% of
                            the benchmark's look-alike pairs
    precision_small_target  a target scaled 0.85-0.95 (the training range's lower half), the other dimensions calmer
                            than the training mix: small spills are the largest failure class
    hard_training_cups      training-split cups the policy passed under HARD_RATE of the time (as source or target, at
                            least HARD_N episodes, the twin aside) or that the training data had under FEW episodes of
    base_mix                the previous mix (TRAIN_MIX), so the round keeps the old coverage

Training content only (check_stratum): the evaluation cups, the held-out scene, templates and clutter and every L3
stay out. The candidates per stratum allow for the expert's acceptance with g1d_pour_beans_plus_v2 (0.39-0.51 per
stratum on 40-50 candidates each, 11-09; docs/pour-beans-plus-failures-20261109.md).

    python scripts/datagen/pour_beans_plus_recipe.py --summary <analysis>/summary.json \
        --train <training Generalization records json> --out scripts/datagen/recipes/<name>.json
"""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datagen.pour_beans_plus_layouts import DIMS, POOL, SCALE, TWIN, Cups, check_stratum  # noqa: E402

HARD_RATE, HARD_N, FEW = 0.35, 4, 15
CALM = [0.3, 0.4, 0.3]  # L0 / L1 / L2 of the dimensions a stratum does not target
SCALES = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)  # the training range (L2)
STRATA = {  # name -> (accepted episodes, candidates: enough at the lower 95% bound of the stratum's v2 acceptance)
    "explore_unseen_combos": (150, 560),
    "same_cup_pairs": (100, 350),
    "precision_small_target": (110, 440),
    "hard_training_cups": (80, 320),
    "base_mix": (60, 240),
}


def unseen_combos(train):
    seen = {tuple(g["levels"][d] for d in DIMS) for g in train}
    combos = set(itertools.product(range(3), repeat=len(DIMS))) - seen
    return sorted(combos, key=lambda t: (sum(v == 2 for v in t), t))


def hard_cups(factors, train, cups):
    """{cup: evidence} of the training cups the policy fails with or the data barely had."""
    episodes = Counter(g["cups"][role] for g in train for role in ("source_cup", "target_cup"))
    out = {}
    for c in cups.train:
        if c == TWIN:
            continue
        rows = [factors[role][c] for role in ("source_cup", "target_cup") if c in factors[role]]
        n, strict = sum(t["n"] for t in rows), sum(t["strict"] for t in rows)
        rate = strict / n if n else None
        if (n >= HARD_N and rate < HARD_RATE) or episodes[c] < FEW:
            out[c] = {"eval_episodes": n, "strict_rate": None if rate is None else round(rate, 2), "train": episodes[c]}
    return out


def build(summary, train, cups, name):
    hard = hard_cups(summary["factors"], train, cups)
    source = [c for c in sorted(hard) if any(cups.fits(c, s, False) for s in SCALES)]
    target = [c for c in sorted(hard) if any(cups.fits(c, s, True) for s in SCALES)]
    calm = {d: CALM for d in DIMS if d != "object"}
    spec = {
        "explore_unseen_combos": {
            "why": "level combos the training data never had (mostly one dimension or none at L2)",
            "combos": [list(t) for t in unseen_combos(train)],
        },
        "same_cup_pairs": {
            "why": "look-alike cups told apart by place only: the policy pours back from the full one",
            "same": True,
            "levels": calm,
        },
        "precision_small_target": {
            "why": "small spills are the largest failure class; a target under 0.95 scale cuts the strict odds",
            "scale": {"target": [SCALE[2][0], 0.95]},
            "levels": calm,
        },
        "hard_training_cups": {
            "why": f"training cups passed under {HARD_RATE:.0%} of the time or with under {FEW} training episodes",
            "cups": {"source": source, "target": target},
            "levels": calm,
        },
        "base_mix": {"why": "the previous mix (L0/L1/L2 = 0.1/0.3/0.6 per dimension), to keep its coverage"},
    }
    strata = [{"name": k, "target": STRATA[k][0], "candidates": STRATA[k][1], **v} for k, v in spec.items()]
    for st in strata:
        check_stratum(st, cups)
    recipe = {"name": name, "task": "g1d_pour_beans_plus", "plugin": "g1d_pour_beans_plus_v2", "strata": strata}
    return recipe, hard


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True, help="pour_beans_plus_failures.py summary.json")
    parser.add_argument("--train", type=Path, required=True, help="JSON list of the training Generalization records")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cups = Cups(json.loads(POOL.read_text(encoding="utf-8")))
    summary, train = json.loads(args.summary.read_text()), json.loads(args.train.read_text())
    recipe, hard = build(summary, train, cups, args.out.stem)
    args.out.write_text(json.dumps(recipe, indent=1) + "\n")
    explore = recipe["strata"][0]
    print(json.dumps({"unseen_combos": len(explore["combos"]), "hard_cups": hard}, indent=1))


if __name__ == "__main__":
    main()
