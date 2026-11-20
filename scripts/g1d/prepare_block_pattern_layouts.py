"""More layouts of a block task than build_asset.py's eval seeds 0-2, from the same generator
(build_asset.PATTERN_TASKS: g1d_build_block_pattern (照图搭积木) layout i shows pattern i mod len(PATTERNS), 3 until
round 2 added patterns; the sorting and atomic tasks draw their cards and blocks; each layout from an RNG keyed by
task, seed and i, so seeds never share a layout). Into
<assets>/Eval_Layout/RoboDojo/g1d/<seed>/<task>_<i>.json.

    python scripts/g1d/prepare_block_pattern_layouts.py --assets <run>/Assets --seed 900 --count 30          # benchmark
    python scripts/g1d/prepare_block_pattern_layouts.py --assets <run>/Assets --seed 600 --count 150 --train # round
    python scripts/g1d/prepare_block_pattern_layouts.py --task g1d_sort_blocks_by_color --assets ... --seed 640 ...

--train marks the folder split=train: the eval client refuses it, run_batch.py collects from it.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_asset  # noqa: E402

CONFIG = "g1d"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(build_asset.PATTERN_TASKS[CONFIG]), default="g1d_build_block_pattern")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="seed folder of the layouts")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--train", action="store_true", help="mark the folder split=train")
    args = parser.parse_args()
    if args.seed in build_asset.PATTERN_SEEDS:
        parser.error(f"seeds {build_asset.PATTERN_SEEDS} are build_asset.py's eval layouts")
    folder = args.assets / "Eval_Layout" / "RoboDojo" / CONFIG / str(args.seed)
    if list(folder.glob(f"{args.task}_*.json")):
        parser.error(f"{folder} already has {args.task} layouts")
    build_asset.write_pattern_layouts(args.assets, CONFIG, args.task, folder, args.seed, args.count)
    if args.train:
        (folder / "split.json").write_text(json.dumps({"split": "train", "task": args.task}) + "\n")
    print(f"wrote {args.count} {args.task} layouts to {folder}{' (train)' if args.train else ''}")


if __name__ == "__main__":
    main()
