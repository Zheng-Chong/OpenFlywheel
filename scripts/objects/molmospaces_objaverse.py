"""Import MolmoSpaces Objaverse objects (`allenai/molmospaces`, `isaac/objects/objaverse/20260128`) into RoboDojo.

MolmoSpaces Objaverse ships 129,710 native Z-up, meters Isaac OpenUSD `.usda` rigid-body assets
packed as `.tar.zst` archives across 76 tar shards (`shards/00000.tar` .. `00075.tar`, 150.92 GB total),
indexed by:
  - `isaac/objects/objaverse/20260128/arrow_table.json` (`path`, `shard_id`, `offset`, `size`, `inflated_size`)
  - `isaac/objects/objaverse/20260128/mjthor_resources_combined_meta.json.gz` (contained file trees / textures)
  - `mujoco/objects/objathor_metadata/20260129/objects_metadata.json.gz` (category, boundingBox, mass,
    primaryProperty, onObject, description_short/description/description_long, materials, synset)

Commercial license enforcement:
  - Cross-checks every candidate UID against the official Objaverse 1.0 Sketchfab metadata
    (`allenai/objaverse`, `object-paths.json.gz` + `metadata/000-XXX.json.gz`).
  - Strictly keeps ONLY commercially usable licenses (`cc0` -> `CC0`, `by` -> `CC-BY-4.0`).
  - Excludes all `by-nc`, `by-nc-sa`, `by-nc-nd`, `by-sa`, `by-nd`, and unknown licenses.

USD adaptation (same AI2-THOR MJCF-USD converter lineage as `molmospaces_thor.py`):
  1. Strips sibling top-level `PhysicsScene` prim.
  2. Flattens and reparents `/<Name>/Geometry/* -> /root/*` and `/<Name> -> /root`, explicitly rewriting
     `RelationshipSpec.targetPathList` and `AttributeSpec.connectionPathList` so materials/textures stay bound.
  3. Bakes negative scales (`xformOp:scale = (-1, 1, 1)` authored by the MJCF-USD converter on both visual
     and convex-hull collider meshes) directly into mesh `points` / `normals` and reverses face winding
     (including `faceVarying` primvars) so PhysX convex-hull cooking and rendering are reflection-free.
  4. Copies relative `./Textures/*` into `textures/`, centers the asset at `/root` bbox origin, and authors
     `RigidBodyAPI` + `MassAPI` on `/root` only.

Pipeline subcommands:
  filter   Fetch/load Objaverse + Objathor + Sketchfab license indices and select clean commercial candidates.
  fetch    Range-fetch only selected `obja_<uid>/` directories from the 76 HuggingFace tar shards.
  survey   Inspect downloaded `.usda` assets with `pxr` and emit a conversion manifest with deterministic IDs.
  convert  Flatten, reparent, bake negative scales, rewrite textures/paths, author physics -> `object.usd`.
  ready    Stamp `check_object.py` results into `metadata.json` (`runtime_ready`).

Needs `pxr` (`usd-core` or `isaac-sim python` with `SimulationApp` fallback) only for `survey` and `convert`.
CPU unit tests: `scripts/objects/test_molmospaces_objaverse.py`.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import threading
import urllib.parse
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import articulated  # noqa: E402
import molmospaces_thor as thor  # noqa: E402

SCRIPT = "scripts/objects/molmospaces_objaverse.py"
VERSION = 1

REPO = "allenai/molmospaces"
SUBPATH = "isaac/objects/objaverse/20260128"
OBJATHOR_SUBPATH = "mujoco/objects/objathor_metadata/20260129"
OBJAVERSE_REPO = "allenai/objaverse"
DEFAULT_REVISION = "main"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
USER_AGENT = "curl/8.0"

ARROW_TABLE_FILE = "arrow_table.json"
COMBINED_META_FILE = "mjthor_resources_combined_meta.json.gz"
OBJATHOR_META_ARCHIVE = "objathor_metadata_objects_metadata.json.gz.tar.zst"
OBJATHOR_META_FILE = "objects_metadata.json.gz"
OBJAVERSE_PATHS_FILE = "object-paths.json.gz"
LICENSE_CACHE_FILE = "objaverse_licenses.json.gz"

# Strict commercial license policy: ONLY CC0 and CC-BY (no NC, no SA, no ND, no unknown).
ALLOWED_SKETCHFAB_LICENSES = {
    "cc0": "CC0",
    "by": "CC-BY-4.0",
}

# Tabletop manipulation size and mass bounds (metres, kg).
MIN_EXTENT_M = 0.02
MAX_EXTENT_M = 0.65
MIN_THICKNESS_M = 0.005
RIGID_ESTIMATED_DENSITY = 500.0
DEFAULT_MIN_MASS_KG = 0.005
DEFAULT_MAX_MASS_KG = 2.0
RIGID_FRICTION = 0.6

# Curated first-wave tabletop manipulation whitelist (lowercase exact objathor category names).
CURATED_WHITELIST_CATEGORIES: tuple[str, ...] = (
    # Tools
    "hammer",
    "wrench",
    "screwdriver",
    "pliers",
    "flashlight",
    "scissors",
    "tape measure",
    "paintbrush",
    "paint brush",
    "hand saw",
    "mallet",
    "chisel",
    # Digital / Office
    "camera",
    "smartphone",
    "cell phone",
    "mobile phone",
    "telephone",
    "headphones",
    "earphones",
    "pen",
    "pencil",
    "marker",
    "book",
    "notebook",
    "computer mouse",
    "mouse",
    "keyboard",
    "computer keyboard",
    "remote control",
    "remote",
    "laptop",
    "tablet",
    "watch",
    "wristwatch",
    "calculator",
    "stapler",
    "usb drive",
    "usb flash drive",
    "flash drive",
    "game controller",
    "controller",
    "handheld gaming console",
    "speaker",
    "smart speaker",
    "microphone",
    "alarm clock",
    "clock",
    "desk clock",
    "magnifying glass",
    "compass",
    "binoculars",
    "binocular",
    # Kitchenware & Tableware
    "mug",
    "coffee mug",
    "cup",
    "teacup",
    "coffee cup",
    "bowl",
    "bottle",
    "water bottle",
    "wine bottle",
    "glass bottle",
    "plastic bottle",
    "pitcher",
    "jug",
    "teapot",
    "coffee pot",
    "kettle",
    "pot",
    "cooking pot",
    "pan",
    "frying pan",
    "skillet",
    "plate",
    "dinner plate",
    "dish",
    "saucer",
    "tray",
    "serving tray",
    "jar",
    "glass jar",
    "spoon",
    "fork",
    "spatula",
    "ladle",
    "whisk",
    "cutting board",
    "thermos",
    "tumbler",
    "drinking glass",
    "wine glass",
    "glass",
    "canister",
    "salt shaker",
    "pepper shaker",
    "pepper mill",
    "colander",
    "strainer",
    "rolling pin",
    # Food / Packaging / Daily
    "fruit",
    "apple",
    "banana",
    "orange",
    "lemon",
    "pear",
    "peach",
    "strawberry",
    "pineapple",
    "watermelon",
    "avocado",
    "tomato",
    "carrot",
    "potato",
    "onion",
    "pepper",
    "broccoli",
    "corn",
    "mushroom",
    "bread",
    "loaf of bread",
    "croissant",
    "donut",
    "doughnut",
    "cake",
    "cupcake",
    "muffin",
    "pizza",
    "burger",
    "hamburger",
    "sandwich",
    "cheese",
    "egg",
    "cookie",
    "ice cream",
    "beverage can",
    "soda can",
    "can",
    "tin can",
    "milk carton",
    "carton",
    "box",
    "cardboard box",
    "cereal box",
    "gift box",
    "storage box",
    "vase",
    "flower vase",
    "plant pot",
    "flowerpot",
    "candle",
    "soap",
    "soap dispenser",
    "spray bottle",
    "toothpaste",
    "toothbrush",
    "hairbrush",
    "comb",
    "sponge",
    "sunglasses",
    "eyeglasses",
    "glasses",
    "hat",
    "baseball cap",
    "cap",
    "helmet",
    "sneaker",
    "athletic shoe",
    "shoe",
    "boot",
    "slipper",
    "sandal",
    "bag",
    "handbag",
    "backpack",
    "purse",
    "wallet",
    "suitcase",
    "basket",
    "umbrella",
    "pillow",
    "cushion",
    "towel",
    "tissue box",
    "key",
    "padlock",
    "lock",
    "battery",
    "light bulb",
    "lightbulb",
    "dice",
    "die",
    "ball",
    "soccer ball",
    "basketball",
    "tennis ball",
    "football",
    "baseball",
    "globe",
    "trophy",
    "bell",
    "hourglass",
)
CURATED_WHITELIST_SET = frozenset(CURATED_WHITELIST_CATEGORIES)

# Hard noise/safety exclusions applied to both category and description_short.
NOISE_CATEGORY_RE = re.compile(
    r"\b("
    r"weapon|gun|handgun|pistol|revolver|rifle|shotgun|grenade|bomb|missile|ammo|bullet|"
    r"sword|dagger|axe|spear|bow|arrow|arrowhead|shield|armor|armour|katana|machete|"
    r"skull|skeleton|bone|fossil|corpse|anatomical|organ|brain|heart|"
    r"sculpture|statue|bust|figurine|totem|idol|artifact|relic|monument|"
    r"stone|rock|boulder|pebble|mineral|crystal|gemstone|ore|slab|driftwood|"
    r"building|house|castle|tower|bridge|ruin|arch|column|pillar|wall|doorway|"
    r"monster|demon|dragon|alien|zombie|fantasy|sci-?fi|spaceship|mech|robot|"
    r"aircraft|tank|warship|submarine|nude|nsfw"
    r")\b",
    re.I,
)
NOISE_DESC_RE = re.compile(
    r"\b("
    r"handgun|pistol|revolver|rifle|shotgun|grenade|assault rifle|firearm|weapon|sword|dagger|"
    r"human skull|animal skull|skeleton|corpse|gore|bloody|nude|nsfw|sci-?fi weapon|fantasy weapon|"
    r"warhammer|battle hammer|medieval helmet|knight helmet|viking helmet|spartan helmet|"
    r"pixel art|minecraft|low-?poly game asset|minecraft block"
    r")\b",
    re.I,
)

# Canonicalize synonymous objathor categories into clean, consistent `obja_<slug>` family directories.
CATEGORY_CANONICAL_MAP: dict[str, str] = {
    "paint brush": "paintbrush",
    "cell phone": "smartphone",
    "mobile phone": "smartphone",
    "computer mouse": "mouse",
    "computer keyboard": "keyboard",
    "remote control": "remote",
    "wristwatch": "watch",
    "usb drive": "usb_flash_drive",
    "flash drive": "usb_flash_drive",
    "controller": "game_controller",
    "smart speaker": "speaker",
    "desk clock": "clock",
    "binocular": "binoculars",
    "coffee mug": "mug",
    "coffee cup": "cup",
    "teacup": "cup",
    "water bottle": "bottle",
    "wine bottle": "bottle",
    "glass bottle": "bottle",
    "plastic bottle": "bottle",
    "cooking pot": "pot",
    "frying pan": "pan",
    "skillet": "pan",
    "dinner plate": "plate",
    "serving tray": "tray",
    "glass jar": "jar",
    "drinking glass": "glass",
    "tumbler": "glass",
    "loaf of bread": "bread",
    "doughnut": "donut",
    "hamburger": "burger",
    "soda can": "beverage_can",
    "tin can": "can",
    "cardboard box": "box",
    "gift box": "box",
    "storage box": "box",
    "flower vase": "vase",
    "flowerpot": "plant_pot",
    "eyeglasses": "glasses",
    "baseball cap": "cap",
    "athletic shoe": "sneaker",
    "lightbulb": "light_bulb",
    "die": "dice",
}

# Hand-curated bilingual (category_en, category_zh, noun_en, noun_zh) for every curated canonical slug.
# Invariants: category != noun in both languages; word/char length(category) <= length(noun); no BAD tokens.
CATEGORY_BILINGUAL_MAP: dict[str, tuple[str, str, str, str]] = {
    "alarm_clock": ("clock", "钟表", "alarm clock", "闹钟"),
    "apple": ("fruit", "水果", "apple", "苹果"),
    "avocado": ("fruit", "水果", "avocado", "牛油果"),
    "backpack": ("bag", "包袋", "backpack", "双肩背包"),
    "bag": ("container", "容器", "hand bag", "手提袋"),
    "ball": ("sport", "球类", "round ball", "圆球"),
    "banana": ("fruit", "水果", "banana", "香蕉"),
    "baseball": ("ball", "球类", "baseball", "棒球"),
    "basket": ("container", "容器", "woven basket", "收纳篮"),
    "basketball": ("ball", "球类", "basketball", "篮球"),
    "battery": ("electronics", "电子", "battery cell", "电池"),
    "bell": ("object", "物件", "hand bell", "铃铛"),
    "beverage_can": ("can", "罐子", "beverage can", "饮料罐"),
    "binoculars": ("optics", "光学", "binoculars", "双筒望远镜"),
    "book": ("stationery", "文具", "hardcover book", "书籍"),
    "boot": ("footwear", "鞋类", "ankle boot", "靴子"),
    "bottle": ("container", "容器", "drink bottle", "瓶子"),
    "bowl": ("tableware", "餐具", "round bowl", "碗具"),
    "box": ("container", "容器", "storage box", "方形盒子"),
    "bread": ("food", "食物", "bread loaf", "面包"),
    "broccoli": ("vegetable", "蔬菜", "broccoli", "西兰花"),
    "burger": ("food", "食物", "burger", "汉堡包"),
    "cake": ("dessert", "甜点", "sliced cake", "蛋糕"),
    "calculator": ("electronics", "电子", "calculator", "计算器"),
    "camera": ("electronics", "电子", "photo camera", "照相机"),
    "can": ("container", "容器", "metal can", "金属罐"),
    "candle": ("decor", "摆件", "wax candle", "蜡烛"),
    "canister": ("container", "容器", "storage canister", "密封罐"),
    "cap": ("headwear", "帽饰", "baseball cap", "棒球帽"),
    "carrot": ("vegetable", "蔬菜", "carrot", "胡萝卜"),
    "carton": ("container", "容器", "paper carton", "纸盒"),
    "cereal_box": ("box", "盒子", "cereal box", "麦片盒"),
    "cheese": ("food", "食物", "cheese wedge", "奶酪"),
    "chisel": ("tool", "工具", "hand chisel", "凿子"),
    "clock": ("timepiece", "计时", "desk clock", "台钟"),
    "coffee_pot": ("pot", "壶具", "coffee pot", "咖啡壶"),
    "colander": ("cookware", "厨具", "colander", "沥水篮"),
    "comb": ("grooming", "日用", "hair comb", "梳子"),
    "compass": ("instrument", "仪器", "pocket compass", "指南针"),
    "cookie": ("dessert", "甜点", "baked cookie", "曲奇饼干"),
    "corn": ("vegetable", "蔬菜", "corn cob", "玉米"),
    "croissant": ("pastry", "糕点", "croissant", "牛角面包"),
    "cup": ("drinkware", "杯具", "drinking cup", "杯子"),
    "cupcake": ("dessert", "甜点", "cupcake", "纸杯蛋糕"),
    "cushion": ("textile", "织物", "throw cushion", "抱枕"),
    "cutting_board": ("cookware", "厨具", "cutting board", "砧板"),
    "dice": ("game", "桌游", "game die", "骰子"),
    "dish": ("tableware", "餐具", "serving dish", "菜碟"),
    "donut": ("dessert", "甜点", "glazed donut", "甜甜圈"),
    "earphones": ("audio", "音频", "earphones", "入耳耳机"),
    "egg": ("food", "食物", "whole egg", "鸡蛋"),
    "flashlight": ("tool", "工具", "flashlight", "手电筒"),
    "football": ("ball", "球类", "football", "橄榄球"),
    "fork": ("cutlery", "餐具", "dining fork", "餐叉"),
    "fruit": ("produce", "果蔬", "fresh fruit", "水果"),
    "game_controller": ("electronics", "电子", "game controller", "游戏手柄"),
    "glass": ("drinkware", "杯具", "drinking glass", "玻璃杯"),
    "glasses": ("eyewear", "眼镜", "eyeglasses", "框架眼镜"),
    "globe": ("decor", "摆件", "desk globe", "地球仪"),
    "hairbrush": ("grooming", "日用", "hairbrush", "发刷"),
    "hammer": ("tool", "工具", "claw hammer", "锤子"),
    "hand_saw": ("tool", "工具", "hand saw", "手锯"),
    "handbag": ("bag", "包袋", "handbag", "手提包"),
    "handheld_gaming_console": ("electronics", "电子", "handheld console", "掌上游戏机"),
    "hat": ("headwear", "帽饰", "brimmed hat", "帽子"),
    "headphones": ("audio", "音频", "headphones", "头戴耳机"),
    "helmet": ("headwear", "帽饰", "safety helmet", "头盔"),
    "hourglass": ("timepiece", "计时", "sand hourglass", "沙漏"),
    "ice_cream": ("dessert", "甜点", "ice cream", "冰淇淋"),
    "jar": ("container", "容器", "glass jar", "广口罐"),
    "jug": ("container", "容器", "handle jug", "大水壶"),
    "kettle": ("cookware", "厨具", "water kettle", "水壶"),
    "key": ("hardware", "五金", "metal key", "钥匙"),
    "keyboard": ("electronics", "电子", "computer keyboard", "电脑键盘"),
    "ladle": ("utensil", "厨具", "soup ladle", "汤勺"),
    "laptop": ("electronics", "电子", "laptop computer", "笔记本电脑"),
    "lemon": ("fruit", "水果", "fresh lemon", "柠檬"),
    "light_bulb": ("lighting", "照明", "light bulb", "灯泡"),
    "lock": ("hardware", "五金", "door lock", "锁具"),
    "magnifying_glass": ("optics", "光学", "magnifying glass", "放大镜"),
    "mallet": ("tool", "工具", "hand mallet", "木槌"),
    "marker": ("stationery", "文具", "marker pen", "记号笔"),
    "microphone": ("audio", "音频", "microphone", "麦克风"),
    "milk_carton": ("carton", "纸盒", "milk carton", "牛奶盒"),
    "mouse": ("electronics", "电子", "computer mouse", "电脑鼠标"),
    "muffin": ("pastry", "糕点", "baked muffin", "玛芬蛋糕"),
    "mug": ("drinkware", "杯具", "coffee mug", "马克杯"),
    "mushroom": ("vegetable", "蔬菜", "mushroom", "蘑菇"),
    "notebook": ("stationery", "文具", "ruled notebook", "笔记本"),
    "onion": ("vegetable", "蔬菜", "whole onion", "洋葱"),
    "orange": ("fruit", "水果", "citrus orange", "橙子"),
    "padlock": ("hardware", "五金", "padlock", "挂锁"),
    "paintbrush": ("tool", "工具", "paintbrush", "油漆刷"),
    "pan": ("cookware", "厨具", "frying pan", "平底锅"),
    "peach": ("fruit", "水果", "fresh peach", "桃子"),
    "pear": ("fruit", "水果", "fresh pear", "梨子"),
    "pen": ("stationery", "文具", "ballpoint pen", "签字笔"),
    "pencil": ("stationery", "文具", "wooden pencil", "铅笔"),
    "pepper": ("vegetable", "蔬菜", "bell pepper", "甜椒"),
    "pepper_mill": ("utensil", "厨具", "pepper mill", "胡椒研磨器"),
    "pepper_shaker": ("tableware", "餐具", "pepper shaker", "胡椒瓶"),
    "pillow": ("textile", "织物", "bed pillow", "枕头"),
    "pineapple": ("fruit", "水果", "pineapple", "菠萝"),
    "pitcher": ("drinkware", "杯具", "water pitcher", "冷水壶"),
    "pizza": ("food", "食物", "baked pizza", "披萨"),
    "plant_pot": ("container", "容器", "plant pot", "花盆"),
    "plate": ("tableware", "餐具", "dinner plate", "餐盘"),
    "pliers": ("tool", "工具", "hand pliers", "钳子"),
    "pot": ("cookware", "厨具", "cooking pot", "煮锅"),
    "potato": ("vegetable", "蔬菜", "raw potato", "土豆"),
    "purse": ("bag", "包袋", "coin purse", "手包"),
    "remote": ("electronics", "电子", "remote control", "遥控器"),
    "rolling_pin": ("utensil", "厨具", "rolling pin", "擀面杖"),
    "salt_shaker": ("tableware", "餐具", "salt shaker", "盐瓶"),
    "sandal": ("footwear", "鞋类", "flat sandal", "凉鞋"),
    "sandwich": ("food", "食物", "club sandwich", "三明治"),
    "saucer": ("tableware", "餐具", "cup saucer", "茶托"),
    "scissors": ("tool", "工具", "hand scissors", "剪刀"),
    "screwdriver": ("tool", "工具", "screwdriver", "螺丝刀"),
    "shoe": ("footwear", "鞋类", "casual shoe", "鞋子"),
    "slipper": ("footwear", "鞋类", "house slipper", "拖鞋"),
    "smartphone": ("electronics", "电子", "smartphone", "智能手机"),
    "sneaker": ("footwear", "鞋类", "sport sneaker", "运动鞋"),
    "soap": ("toiletries", "洗护", "bar soap", "香皂"),
    "soap_dispenser": ("toiletries", "洗护", "soap dispenser", "洗手液瓶"),
    "soccer_ball": ("ball", "球类", "soccer ball", "足球"),
    "spatula": ("utensil", "厨具", "kitchen spatula", "锅铲"),
    "speaker": ("audio", "音频", "audio speaker", "音箱"),
    "sponge": ("cleaning", "清洁", "cleaning sponge", "海绵擦"),
    "spoon": ("cutlery", "餐具", "metal spoon", "餐勺"),
    "spray_bottle": ("container", "容器", "spray bottle", "喷雾瓶"),
    "stapler": ("stationery", "文具", "desk stapler", "订书机"),
    "strainer": ("utensil", "厨具", "mesh strainer", "滤网勺"),
    "strawberry": ("fruit", "水果", "strawberry", "草莓"),
    "suitcase": ("luggage", "箱包", "travel suitcase", "行李箱"),
    "sunglasses": ("eyewear", "眼镜", "sunglasses", "太阳镜"),
    "tablet": ("electronics", "电子", "tablet computer", "平板电脑"),
    "tape_measure": ("tool", "工具", "tape measure", "卷尺"),
    "teapot": ("drinkware", "杯具", "brew teapot", "茶壶"),
    "telephone": ("electronics", "电子", "desk telephone", "电话机"),
    "tennis_ball": ("ball", "球类", "tennis ball", "网球"),
    "thermos": ("drinkware", "杯具", "vacuum thermos", "保温杯"),
    "tissue_box": ("container", "容器", "tissue box", "纸巾盒"),
    "tomato": ("vegetable", "蔬菜", "ripe tomato", "番茄"),
    "toothbrush": ("toiletries", "洗护", "toothbrush", "牙刷"),
    "toothpaste": ("toiletries", "洗护", "toothpaste tube", "牙膏"),
    "towel": ("textile", "织物", "bath towel", "毛巾"),
    "tray": ("tableware", "餐具", "serving tray", "托盘"),
    "trophy": ("award", "奖品", "award trophy", "奖杯"),
    "umbrella": ("accessory", "配饰", "folding umbrella", "雨伞"),
    "usb_flash_drive": ("electronics", "电子", "flash drive", "优盘"),
    "vase": ("decor", "摆件", "flower vase", "花瓶"),
    "wallet": ("accessory", "配饰", "folding wallet", "钱包"),
    "watch": ("timepiece", "计时", "wristwatch", "手表"),
    "watermelon": ("fruit", "水果", "watermelon", "西瓜"),
    "whisk": ("utensil", "厨具", "kitchen whisk", "打蛋器"),
    "wine_glass": ("drinkware", "杯具", "wine glass", "高脚杯"),
    "wrench": ("tool", "工具", "metal wrench", "扳手"),
}

# Verified color / material / style modifiers (ordered for clean matching against objathor description_short).
MODIFIER_BILINGUAL_MAP: tuple[tuple[str, str, str], ...] = (
    ("red", "red", "红色"),
    ("crimson", "red", "红色"),
    ("scarlet", "red", "红色"),
    ("maroon", "maroon", "暗红"),
    ("blue", "blue", "蓝色"),
    ("navy", "navy", "深蓝"),
    ("teal", "teal", "青色"),
    ("cyan", "cyan", "青色"),
    ("turquoise", "turquoise", "青色"),
    ("green", "green", "绿色"),
    ("olive", "olive", "橄榄绿"),
    ("yellow", "yellow", "黄色"),
    ("gold", "golden", "金色"),
    ("golden", "golden", "金色"),
    ("orange", "orange", "橙色"),
    ("pink", "pink", "粉色"),
    ("purple", "purple", "紫色"),
    ("violet", "purple", "紫色"),
    ("black", "black", "黑色"),
    ("white", "white", "白色"),
    ("gray", "gray", "灰色"),
    ("grey", "gray", "灰色"),
    ("silver", "silver", "银色"),
    ("brown", "brown", "棕色"),
    ("tan", "tan", "浅棕"),
    ("beige", "beige", "米色"),
    ("cream", "cream", "米白"),
    ("creamy", "cream", "米白"),
    ("wooden", "wooden", "木制"),
    ("wood", "wooden", "木制"),
    ("brass", "brass", "黄铜"),
    ("copper", "copper", "铜制"),
    ("metallic", "metallic", "金属"),
    ("metal", "metallic", "金属"),
    ("steel", "steel", "钢制"),
    ("iron", "iron", "铁制"),
    ("glass", "glass", "玻璃"),
    ("clear", "clear", "透明"),
    ("ceramic", "ceramic", "陶瓷"),
    ("porcelain", "ceramic", "陶瓷"),
    ("plastic", "plastic", "塑料"),
    ("leather", "leather", "皮质"),
    ("woven", "woven", "编织"),
    ("vintage", "vintage", "复古"),
    ("colorful", "colorful", "彩色"),
    ("striped", "striped", "条纹"),
    ("dark", "dark", "深色"),
)


def hf_endpoint(endpoint: str | None = None) -> str:
    raw = endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    return raw.rstrip("/")


def canonical_url() -> str:
    return f"{DEFAULT_HF_ENDPOINT}/datasets/{REPO}"


def resolve_molmo_url(
    rel_path: str,
    *,
    subpath: str = SUBPATH,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
) -> str:
    quoted = urllib.parse.quote(f"{subpath}/{rel_path.lstrip('/')}", safe="/")
    return f"{hf_endpoint(endpoint)}/datasets/{REPO}/resolve/{revision}/{quoted}"


def resolve_objaverse_url(
    rel_path: str,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
) -> str:
    quoted = urllib.parse.quote(rel_path.lstrip("/"), safe="/")
    return f"{hf_endpoint(endpoint)}/datasets/{OBJAVERSE_REPO}/resolve/{revision}/{quoted}"


def canonical_category_slug(raw_category: str) -> str:
    """Normalize an objathor category string to a clean, canonical slug."""
    norm = re.sub(r"\s+", " ", (raw_category or "").strip().lower())
    mapped = CATEGORY_CANONICAL_MAP.get(norm, norm)
    return articulated.slug(mapped) or "object"


def normalize_sketchfab_license(raw_license: str | None) -> str | None:
    """Return canonical commercial license string (`CC0` / `CC-BY-4.0`) or None if non-commercial/unknown."""
    if not raw_license:
        return None
    key = str(raw_license).strip().lower()
    return ALLOWED_SKETCHFAB_LICENSES.get(key)


def uid_from_archive_path(path: str) -> str | None:
    """Extract 32-hex Objaverse UID from `objaverse_obja_<uid>.tar.zst`."""
    prefix = "objaverse_obja_"
    suffix = ".tar.zst"
    if path.startswith(prefix) and path.endswith(suffix):
        uid = path[len(prefix) : -len(suffix)]
        if uid:
            return uid
    return None


def _read_shard_bytes(
    sources_dir: Path,
    arrow_entry: dict,
    *,
    subpath: str = SUBPATH,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
) -> bytes:
    """Read a `.tar.zst` slice from local `shards/NNNNN.tar` if present, else HTTP Range GET."""
    shard_id = int(arrow_entry.get("shard_id", 0))
    offset = int(arrow_entry["offset"])
    size = int(arrow_entry["size"])
    if offset < 0 or size <= 0:
        raise ValueError(f"invalid byte range for {arrow_entry.get('path')}: offset={offset}, size={size}")

    local_shard = Path(sources_dir) / "shards" / f"{shard_id:05d}.tar"
    if local_shard.is_file() and local_shard.stat().st_size >= offset + size:
        with local_shard.open("rb") as fh:
            fh.seek(offset)
            return fh.read(size)

    shard_rel = f"shards/{shard_id:05d}.tar"
    url = resolve_molmo_url(shard_rel, subpath=subpath, revision=revision, endpoint=endpoint)
    data, _ = thor.http_get(url, headers={"Range": f"bytes={offset}-{offset + size - 1}"})
    if len(data) != size:
        if len(data) >= offset + size:
            return data[offset : offset + size]
        raise ValueError(f"short Range response for {arrow_entry.get('path')}: got {len(data)} bytes, expected {size}")
    return data


def fetch_molmo_indices(
    sources_dir: Path,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    force: bool = False,
) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    """Fetch or reuse cached arrow_table, combined_meta, and objathor metadata."""
    idx_dir = Path(sources_dir) / "_index"
    idx_dir.mkdir(parents=True, exist_ok=True)

    arrow_path = idx_dir / ARROW_TABLE_FILE
    if force or not arrow_path.is_file():
        data, _ = thor.http_get(resolve_molmo_url(ARROW_TABLE_FILE, revision=revision, endpoint=endpoint))
        thor._atomic_write_bytes(arrow_path, data)
    arrow_table = json.loads(arrow_path.read_text(encoding="utf-8"))

    cmeta_path = idx_dir / COMBINED_META_FILE
    if force or not cmeta_path.is_file():
        data, _ = thor.http_get(resolve_molmo_url(COMBINED_META_FILE, revision=revision, endpoint=endpoint))
        thor._atomic_write_bytes(cmeta_path, data)
    combined_meta = json.loads(gzip.decompress(cmeta_path.read_bytes()).decode("utf-8"))

    ometa_path = idx_dir / OBJATHOR_META_FILE
    if force or not ometa_path.is_file():
        om_arrow_data, _ = thor.http_get(
            resolve_molmo_url(ARROW_TABLE_FILE, subpath=OBJATHOR_SUBPATH, revision=revision, endpoint=endpoint)
        )
        om_arrow = json.loads(om_arrow_data.decode("utf-8"))
        entry = next((e for e in om_arrow if e.get("path") == OBJATHOR_META_ARCHIVE), None)
        if entry is None:
            raise ValueError(f"{OBJATHOR_META_ARCHIVE} not found in objathor_metadata {ARROW_TABLE_FILE}")
        raw_zst = _read_shard_bytes(
            Path(sources_dir), entry, subpath=OBJATHOR_SUBPATH, revision=revision, endpoint=endpoint
        )
        tar_bytes = thor.decompress_zst(raw_zst)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
            member = next((m for m in tf.getmembers() if m.name.endswith(OBJATHOR_META_FILE)), None)
            if member is None:
                raise ValueError(f"{OBJATHOR_META_FILE} missing inside {OBJATHOR_META_ARCHIVE}")
            extracted = tf.extractfile(member)
            assert extracted is not None
            thor._atomic_write_bytes(ometa_path, extracted.read())
    objathor_meta = json.loads(gzip.decompress(ometa_path.read_bytes()).decode("utf-8"))
    return arrow_table, combined_meta, objathor_meta


def fetch_objaverse_licenses(
    sources_dir: Path,
    candidate_uids: set[str] | None = None,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    workers: int = 8,
    force: bool = False,
) -> dict[str, dict]:
    """Fetch official Objaverse 1.0 Sketchfab license records for `candidate_uids` (or all cached UIDs).

    Uses `allenai/objaverse/object-paths.json.gz` to locate only the required `metadata/000-XXX.json.gz`
    shards, caches extracted `{uid: {license, name, uri, user}}` in `_index/objaverse_licenses.json.gz`.
    """
    idx_dir = Path(sources_dir) / "_index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    cache_path = idx_dir / LICENSE_CACHE_FILE

    cached: dict[str, dict] = {}
    if not force and cache_path.is_file():
        cached = json.loads(gzip.decompress(cache_path.read_bytes()).decode("utf-8"))

    if candidate_uids is not None:
        missing = {u for u in candidate_uids if u not in cached}
        if not missing:
            return cached
    else:
        missing = None

    paths_file = idx_dir / OBJAVERSE_PATHS_FILE
    if force or not paths_file.is_file():
        data, _ = thor.http_get(resolve_objaverse_url(OBJAVERSE_PATHS_FILE, revision=revision, endpoint=endpoint))
        thor._atomic_write_bytes(paths_file, data)
    obj_paths: dict[str, str] = json.loads(gzip.decompress(paths_file.read_bytes()).decode("utf-8"))

    shards_needed: set[str] = set()
    if missing is not None:
        for uid in missing:
            rel = obj_paths.get(uid)
            if not rel:
                cached[uid] = {"license": "", "name": "", "uri": "", "user": ""}
                continue
            parts = rel.split("/")
            if len(parts) >= 2:
                shards_needed.add(parts[1])
    else:
        for rel in obj_paths.values():
            parts = rel.split("/")
            if len(parts) >= 2:
                shards_needed.add(parts[1])

    if not shards_needed:
        thor._atomic_write_bytes(cache_path, gzip.compress(json.dumps(cached, separators=(",", ":")).encode("utf-8")))
        return cached

    def _fetch_shard(shard_name: str) -> dict[str, dict]:
        rel = f"metadata/{shard_name}.json.gz"
        raw, _ = thor.http_get(resolve_objaverse_url(rel, revision=revision, endpoint=endpoint), timeout=120.0)
        shard_data = json.loads(gzip.decompress(raw).decode("utf-8"))
        out: dict[str, dict] = {}
        for uid, rec in shard_data.items():
            if missing is not None and uid not in missing:
                continue
            if not isinstance(rec, dict):
                continue
            user = rec.get("user")
            username = user.get("username", "") if isinstance(user, dict) else ""
            out[uid] = {
                "license": str(rec.get("license") or "").strip().lower(),
                "name": str(rec.get("name") or "").strip(),
                "uri": str(rec.get("uri") or "").strip(),
                "user": str(username).strip(),
            }
        return out

    sorted_shards = sorted(shards_needed)
    print(
        f"[obja] fetching {len(sorted_shards)} Objaverse 1.0 license metadata shard(s) for "
        f"{len(missing) if missing is not None else 'all'} UID(s)...",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = {pool.submit(_fetch_shard, sh): sh for sh in sorted_shards}
        for fut in as_completed(futs):
            cached.update(fut.result())

    if missing is not None:
        for uid in missing:
            cached.setdefault(uid, {"license": "", "name": "", "uri": "", "user": ""})

    thor._atomic_write_bytes(cache_path, gzip.compress(json.dumps(cached, separators=(",", ":")).encode("utf-8")))
    return cached


def normalize_description_short(raw: object) -> dict[str, str] | str:
    """Normalize objathor `description_short` (structured dict, legacy dict-repr string, or plain string)."""
    if isinstance(raw, dict):
        return {str(k): str(v).strip() for k, v in raw.items() if v is not None and str(v).strip()}
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = ast.literal_eval(s)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                return {str(k): str(v).strip() for k, v in parsed.items() if v is not None and str(v).strip()}
        return s
    return ""


def format_description_text(desc_short: object, desc_long: str = "", fallback: str = "") -> str:
    """Return a clean human-readable English description string (never a Python dict repr)."""
    long_clean = (desc_long or "").strip() if isinstance(desc_long, str) else ""
    if long_clean:
        return long_clean
    norm = normalize_description_short(desc_short)
    if isinstance(norm, dict):
        for key in ("five_words", "four_words", "three_words", "two_words", "one_word"):
            val = norm.get(key)
            if val:
                return val
    elif isinstance(norm, str) and norm:
        return norm
    return fallback


def _description_probe_text(desc_short: object) -> str:
    norm = normalize_description_short(desc_short)
    if isinstance(norm, dict):
        return " ".join(
            norm[k] for k in ("two_words", "three_words", "four_words", "five_words", "one_word") if norm.get(k)
        )
    return norm


def build_catalog(
    arrow_table: list[dict],
    combined_meta: dict[str, dict],
    objathor_meta: dict[str, dict],
    licenses: dict[str, dict] | None = None,
) -> list[dict]:
    """Combine arrow_table, combined_meta, objathor_meta, and optional Sketchfab licenses into normalized entries."""
    rows: list[dict] = []
    for entry in arrow_table:
        if not isinstance(entry, dict):
            continue
        archive = str(entry.get("path") or "")
        uid = uid_from_archive_path(archive)
        if not uid:
            continue
        om = objathor_meta.get(uid)
        if not isinstance(om, dict):
            continue

        raw_cat = str(om.get("category") or om.get("objectType") or "").strip()
        bb = om.get("boundingBox") or {}
        dims = [float(bb.get(ax, 0.0) or 0.0) for ax in ("x", "y", "z")]
        tree = combined_meta.get(archive)
        has_tex = thor._has_textures_in_tree(tree)

        lic_rec = (licenses or {}).get(uid) or {}
        raw_lic = str(lic_rec.get("license") or "").strip().lower()
        norm_lic = normalize_sketchfab_license(raw_lic)

        rows.append(
            {
                "uid": uid,
                "asset_id": f"obja_{uid}",
                "category": raw_cat,
                "category_lower": re.sub(r"\s+", " ", raw_cat.lower()),
                "category_slug": canonical_category_slug(raw_cat),
                "primary_property": str(om.get("primaryProperty") or ""),
                "on_object": bool(om.get("onObject")),
                "bbox_m": [round(d, 4) for d in dims],
                "mass_kg": float(om.get("mass") or 0.0),
                "description_short": normalize_description_short(om.get("description_short")),
                "description": str(om.get("description") or "").strip(),
                "materials": list(om.get("materials") or []) if isinstance(om.get("materials"), list) else [],
                "synset": str(om.get("synset") or "").strip(),
                "archive": archive,
                "shard_id": int(entry.get("shard_id", 0)),
                "offset": int(entry.get("offset", -1)),
                "size": int(entry.get("size", 0)),
                "has_textures": has_tex,
                "sketchfab_license": raw_lic,
                "license": norm_lic or "",
                "sketchfab_name": str(lic_rec.get("name") or "").strip(),
                "sketchfab_uri": str(lic_rec.get("uri") or "").strip(),
                "sketchfab_user": str(lic_rec.get("user") or "").strip(),
            }
        )
    rows.sort(key=lambda r: (r["category_slug"], r["uid"]))
    return rows


def load_existing_state(existing_state_path: Path | None) -> tuple[set[str], dict[str, int]]:
    """Load already-imported `(existing_uids, next_index)` to avoid ID collisions."""
    if not existing_state_path:
        return set(), {}
    data = json.loads(Path(existing_state_path).read_text(encoding="utf-8"))
    uids = set(data.get("uids") or [])
    next_idx = {str(k): int(v) for k, v in (data.get("next_index") or {}).items()}
    return uids, next_idx


def filter_catalog(
    rows: list[dict],
    *,
    commercial_only: bool = True,
    pickup_only: bool = True,
    on_object_only: bool = True,
    textured_only: bool = True,
    min_size: float = MIN_EXTENT_M,
    max_size: float = MAX_EXTENT_M,
    min_thickness: float = MIN_THICKNESS_M,
    whitelist: set[str] | frozenset[str] | None = CURATED_WHITELIST_SET,
    categories: list[str] | None = None,
    exclude_categories: list[str] | None = None,
    exclude_noise: bool = True,
    max_per_category: int | None = None,
    limit: int | None = None,
    uids: set[str] | None = None,
    existing_uids: set[str] | None = None,
    next_index: dict[str, int] | None = None,
    revision: str = DEFAULT_REVISION,
) -> dict:
    """Filter Objaverse catalog entries and assign deterministic `Rigid/obja_<slug>/<index>` IDs."""
    chosen: list[dict] = []
    dropped: list[dict] = []
    reason_counts: dict[str, int] = {}
    per_cat_counts: dict[str, int] = {}
    excl_uids = existing_uids or set()

    def _drop(row: dict, reason: str) -> None:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        dropped.append({"uid": row["uid"], "category": row["category"], "reason": reason})

    for row in rows:
        uid = row["uid"]
        if uids is not None and uid not in uids:
            _drop(row, "uid_not_requested")
            continue
        if uid in excl_uids:
            _drop(row, "already_imported")
            continue
        if row["offset"] < 0 or row["size"] <= 0:
            _drop(row, "missing_archive")
            continue
        if pickup_only and row["primary_property"] != "CanPickup":
            _drop(row, "not_can_pickup")
            continue
        if on_object_only and not row["on_object"]:
            _drop(row, "not_on_object")
            continue
        if textured_only and not row["has_textures"]:
            _drop(row, "untextured")
            continue

        dims = row["bbox_m"]
        if len(dims) != 3 or min(dims) < min_thickness or max(dims) < min_size or max(dims) > max_size:
            _drop(row, "size_out_of_range")
            continue

        cat_lo = row["category_lower"]
        if whitelist is not None and cat_lo not in whitelist:
            _drop(row, "not_in_whitelist")
            continue
        if (
            categories
            and not thor._matches_any(cat_lo, categories)
            and not thor._matches_any(row["category_slug"], categories)
        ):
            _drop(row, "category_mismatch")
            continue
        if exclude_categories and (
            thor._matches_any(cat_lo, exclude_categories) or thor._matches_any(row["category_slug"], exclude_categories)
        ):
            _drop(row, "category_excluded")
            continue
        if exclude_noise:
            if NOISE_CATEGORY_RE.search(cat_lo):
                _drop(row, "noise_category")
                continue
            desc_probe = f"{_description_probe_text(row['description_short'])} {row['sketchfab_name']}"
            if NOISE_DESC_RE.search(desc_probe):
                _drop(row, "noise_description")
                continue
        if commercial_only and not normalize_sketchfab_license(row.get("sketchfab_license")):
            _drop(row, f"license_rejected:{row.get('sketchfab_license') or 'unknown'}")
            continue

        slug = row["category_slug"]
        if max_per_category is not None and per_cat_counts.get(slug, 0) >= max_per_category:
            _drop(row, "category_cap")
            continue
        if limit is not None and len(chosen) >= limit:
            _drop(row, "global_limit")
            continue

        per_cat_counts[slug] = per_cat_counts.get(slug, 0) + 1
        chosen.append(dict(row))

    # Deterministic ID assignment: Rigid/obja_<slug>/<00000>, respecting any existing next_index offsets.
    counters: dict[str, int] = dict(next_index or {})
    for row in chosen:
        slug = row["category_slug"]
        idx = counters.get(slug, 0)
        counters[slug] = idx + 1
        row["id"] = f"Rigid/obja_{slug}/{idx:05d}"

    license_counts: dict[str, int] = {}
    for row in chosen:
        lic = row.get("license") or "unknown"
        license_counts[lic] = license_counts.get(lic, 0) + 1

    return {
        "script": SCRIPT,
        "version": VERSION,
        "repo": REPO,
        "subpath": SUBPATH,
        "revision": revision,
        "summary": {
            "catalog_total": len(rows),
            "chosen": len(chosen),
            "dropped": len(dropped),
            "categories_chosen": len(per_cat_counts),
            "licenses": license_counts,
            "drop_reasons": dict(sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "chosen": chosen,
        "dropped": dropped[:200],  # cap dropped detail for compact JSON
    }


def _extract_asset_tar_zst(raw_zst: bytes, out_dir: Path, asset_id: str) -> Path:
    """Decompress `.tar.zst` and extract `<asset_id>/` safely under `out_dir/assets/`."""
    tar_bytes = thor.decompress_zst(raw_zst)
    assets_root = (Path(out_dir) / "assets").resolve()
    assets_root.mkdir(parents=True, exist_ok=True)
    target_dir = assets_root / asset_id

    tmp_dir = assets_root / f".{asset_id}.{os.getpid()}.{threading.get_ident()}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                norm = Path(member.name)
                if norm.is_absolute() or ".." in norm.parts:
                    continue
                parts = list(norm.parts)
                if parts and parts[0] == asset_id:
                    rel = Path(*parts[1:]) if len(parts) > 1 else Path(norm.name)
                else:
                    rel = norm
                dest = (tmp_dir / rel).resolve()
                if not str(dest).startswith(str(tmp_dir.resolve())):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                extracted = tf.extractfile(member)
                if extracted is not None:
                    dest.write_bytes(extracted.read())

        usda = tmp_dir / f"{asset_id}.usda"
        if not usda.is_file():
            usdas = sorted(tmp_dir.rglob("*.usda"))
            if not usdas:
                raise FileNotFoundError(f"no .usda found inside archive for {asset_id}")
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        os.replace(tmp_dir, target_dir)
        return target_dir
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def fetch_assets(
    entries: list[dict],
    out_dir: Path,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    workers: int = 8,
) -> dict:
    """Range-fetch and extract selected MolmoSpaces Objaverse assets."""
    out_dir = Path(out_dir)
    fetched: list[str] = []
    already: list[str] = []
    failed: list[dict] = []

    todo: list[dict] = []
    for e in entries:
        aid = e["asset_id"]
        usda = out_dir / "assets" / aid / f"{aid}.usda"
        if usda.is_file() and usda.stat().st_size > 0:
            already.append(aid)
        else:
            todo.append(e)

    def _one(entry: dict) -> tuple[str, str | None]:
        aid = entry["asset_id"]
        try:
            raw_zst = _read_shard_bytes(out_dir, entry, subpath=SUBPATH, revision=revision, endpoint=endpoint)
            _extract_asset_tar_zst(raw_zst, out_dir, aid)
            return aid, None
        except Exception as exc:
            return aid, f"{type(exc).__name__}: {exc}"

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futs = [pool.submit(_one, e) for e in todo]
            for fut in as_completed(futs):
                aid, err = fut.result()
                if err is None:
                    fetched.append(aid)
                else:
                    failed.append({"asset_id": aid, "error": err})

    return {
        "fetched": sorted(fetched),
        "already_present": sorted(already),
        "failed": sorted(failed, key=lambda x: x["asset_id"]),
    }


def _reverse_face_slices(values, face_vertex_counts):
    """Reverse per-face contiguous slices in a faceVarying array (or index list)."""
    out = list(values)
    cursor = 0
    for count in face_vertex_counts:
        end = cursor + int(count)
        if end > len(out):
            break
        out[cursor:end] = reversed(out[cursor:end])
        cursor = end
    return out


def bake_negative_scales(stage, root, *, Gf_mod=None, Usd_mod=None, UsdGeom_mod=None) -> int:
    """Bake negative `xformOp:scale` signs into mesh points/normals and flip winding when det < 0.

    Also strips stale unreferenced `xformOp:transform` attributes left behind by the MJCF-USD converter.
    Returns the number of mesh prims whose negative scale was baked.
    """
    if Gf_mod is None or Usd_mod is None or UsdGeom_mod is None:
        Gf_mod, _, Usd_mod, UsdGeom_mod, _, _, _ = thor.pxr_modules()

    baked = 0
    for prim in Usd_mod.PrimRange(root):
        if not prim.IsA(UsdGeom_mod.Mesh):
            continue
        xf = UsdGeom_mod.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        op_names = {op.GetOpName() for op in ops}

        # Remove stale unused xformOp:transform (converter writes both scale=(-1,1,1) and transform=diag(-1,1,1)).
        if "xformOp:transform" not in op_names and prim.HasAttribute("xformOp:transform"):
            prim.RemoveProperty("xformOp:transform")

        scale_op = next((op for op in ops if op.GetOpType() == UsdGeom_mod.XformOp.TypeScale), None)
        if scale_op is None:
            continue
        raw_scale = scale_op.Get()
        if raw_scale is None:
            continue
        sx, sy, sz = float(raw_scale[0]), float(raw_scale[1]), float(raw_scale[2])
        if sx >= 0.0 and sy >= 0.0 and sz >= 0.0:
            continue

        sign_x = -1.0 if sx < 0.0 else 1.0
        sign_y = -1.0 if sy < 0.0 else 1.0
        sign_z = -1.0 if sz < 0.0 else 1.0
        det_neg = (sign_x * sign_y * sign_z) < 0.0

        mesh = UsdGeom_mod.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts is not None and len(pts) > 0:
            new_pts = [Gf_mod.Vec3f(float(p[0]) * sign_x, float(p[1]) * sign_y, float(p[2]) * sign_z) for p in pts]
            mesh.GetPointsAttr().Set(new_pts)
            xs = [p[0] for p in new_pts]
            ys = [p[1] for p in new_pts]
            zs = [p[2] for p in new_pts]
            mesh.GetExtentAttr().Set([Gf_mod.Vec3f(min(xs), min(ys), min(zs)), Gf_mod.Vec3f(max(xs), max(ys), max(zs))])

        fvc = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()
        if det_neg and fvc is not None and fvi is not None and len(fvc) > 0 and len(fvi) > 0:
            mesh.GetFaceVertexIndicesAttr().Set(_reverse_face_slices(fvi, fvc))

        # Flip normals attribute if authored.
        n_attr = mesh.GetNormalsAttr()
        if n_attr and n_attr.HasAuthoredValue():
            nvals = n_attr.Get()
            if nvals is not None and len(nvals) > 0:
                new_n = [Gf_mod.Vec3f(float(n[0]) * sign_x, float(n[1]) * sign_y, float(n[2]) * sign_z) for n in nvals]
                if det_neg and fvc is not None and mesh.GetNormalsInterpolation() == UsdGeom_mod.Tokens.faceVarying:
                    new_n = _reverse_face_slices(new_n, fvc)
                n_attr.Set(new_n)

        # Handle primvars (faceVarying reversal and primvars:normals sign flip).
        pv_api = UsdGeom_mod.PrimvarsAPI(prim)
        for pv in pv_api.GetPrimvars():
            pv_name = pv.GetPrimvarName()
            interp = pv.GetInterpolation()
            if pv_name == "normals":
                vals = pv.Get()
                if vals is not None and len(vals) > 0:
                    new_vals = [
                        Gf_mod.Vec3f(float(n[0]) * sign_x, float(n[1]) * sign_y, float(n[2]) * sign_z) for n in vals
                    ]
                    if det_neg and fvc is not None and interp == UsdGeom_mod.Tokens.faceVarying and not pv.IsIndexed():
                        new_vals = _reverse_face_slices(new_vals, fvc)
                    pv.Set(new_vals)
            if det_neg and fvc is not None and interp == UsdGeom_mod.Tokens.faceVarying:
                if pv.IsIndexed():
                    idxs = pv.GetIndices()
                    if idxs is not None and len(idxs) > 0:
                        pv.SetIndices(_reverse_face_slices(idxs, fvc))
                elif pv_name != "normals":
                    vals = pv.Get()
                    if vals is not None and len(vals) > 0:
                        pv.Set(_reverse_face_slices(vals, fvc))

        # Write positive scale (preserving Vec3f vs Vec3d precision).
        abs_scale = (abs(sx), abs(sy), abs(sz))
        if scale_op.GetPrecision() == UsdGeom_mod.XformOp.PrecisionDouble:
            scale_op.Set(Gf_mod.Vec3d(*abs_scale))
        else:
            scale_op.Set(Gf_mod.Vec3f(*abs_scale))
        baked += 1

    return baked


def survey(
    sources_dir: Path,
    *,
    selection: dict | None = None,
    min_size: float = MIN_EXTENT_M,
    max_size: float = MAX_EXTENT_M,
    min_thickness: float = MIN_THICKNESS_M,
) -> dict:
    """Inspect downloaded Objaverse `.usda` assets with `pxr` and emit a conversion manifest."""
    import numpy as np

    _, _, Usd, UsdGeom, _, _, _ = thor.pxr_modules()

    sources_dir = Path(sources_dir)
    entries = list((selection or {}).get("chosen", []))
    revision = (selection or {}).get("revision", DEFAULT_REVISION)

    chosen: list[dict] = []
    dropped: list[dict] = []
    for entry in entries:
        aid = entry["asset_id"]
        src_rel = f"assets/{aid}/{aid}.usda"
        usda = sources_dir / src_rel
        if not usda.is_file():
            dropped.append({"id": entry["id"], "asset_id": aid, "reason": "missing_usda"})
            continue
        try:
            stage = Usd.Stage.Open(str(usda))
            facts = articulated.inspect(stage)
        except Exception as exc:
            dropped.append({"id": entry["id"], "asset_id": aid, "reason": f"open_failed: {exc}"})
            continue

        if facts["up_axis"] != "Z":
            dropped.append({"id": entry["id"], "asset_id": aid, "reason": f"up_axis={facts['up_axis']}"})
            continue
        if not facts["colliders"]:
            dropped.append({"id": entry["id"], "asset_id": aid, "reason": "no_colliders"})
            continue

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        root = stage.GetDefaultPrim()
        bound = articulated._bound(cache, root) if root else None
        if bound is None:
            dropped.append({"id": entry["id"], "asset_id": aid, "reason": "empty_bbox"})
            continue
        size = (bound[1] - bound[0]) * float(facts["meters_per_unit"])
        if (
            not np.all(np.isfinite(size))
            or float(np.min(size)) < min_thickness
            or float(np.max(size)) < min_size
            or float(np.max(size)) > max_size
        ):
            dropped.append(
                {
                    "id": entry["id"],
                    "asset_id": aid,
                    "reason": f"size_out_of_range:{[round(float(v), 4) for v in size]}",
                }
            )
            continue

        out_entry = dict(entry)
        out_entry["src"] = src_rel
        out_entry["survey_size_m"] = [round(float(v), 4) for v in size]
        out_entry["colliders"] = int(facts["colliders"])
        chosen.append(out_entry)

    return {
        "script": SCRIPT,
        "version": VERSION,
        "repo": REPO,
        "subpath": SUBPATH,
        "revision": revision,
        "chosen": chosen,
        "dropped": dropped,
    }


def adapt_rigid(
    source_usda: Path,
    target_dir: Path,
    *,
    preferred_mass_kg: float | None = None,
    scale: float = 1.0,
    yaw_deg: float = 0.0,
    max_mass: float = DEFAULT_MAX_MASS_KG,
    min_mass: float = DEFAULT_MIN_MASS_KG,
    usd_format: str = "usdc",
) -> dict:
    """Adapt a MolmoSpaces Objaverse `.usda` into a RoboDojo `RigidObject` (`object.usd`)."""
    import numpy as np

    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils = thor.pxr_modules()

    source_usda = Path(source_usda).resolve()
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    src_stage = Usd.Stage.Open(str(source_usda))
    src_facts = articulated.inspect(src_stage)
    if src_facts["up_axis"] != "Z":
        raise ValueError(f"up axis {src_facts['up_axis']} (only Z-up sources are adapted)")
    if not src_facts["colliders"]:
        raise ValueError("no enabled collision geometry")

    raw_mass = sum(m for m in src_facts["masses"].values() if m > thor.DUMMY_MASS_THRESHOLD_KG)
    flat = src_stage.Flatten()
    _, (rels_rewritten, conns_rewritten) = thor._reparent_geometry_to_root(flat, src_facts["root"])
    copied_textures = thor._copy_and_rewrite_textures(flat, source_usda, target_dir)

    stage = Usd.Stage.Open(flat)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = stage.GetPrimAtPath("/root")
    if root.GetTypeName() != "Xform":
        root.SetTypeName("Xform")
    stage.SetDefaultPrim(root)

    # Bake negative scale=(-1, 1, 1) out of visual and collider meshes before bounding-box / PhysX cooking.
    negative_scales_baked = bake_negative_scales(stage, root, Gf_mod=Gf, Usd_mod=Usd, UsdGeom_mod=UsdGeom)

    if root.HasAPI(UsdPhysics.ArticulationRootAPI):
        root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    root.RemoveAppliedSchema("PhysxArticulationAPI")
    if root.HasAttribute(articulated.SCRIPT_ATTR) and root.GetAttribute(articulated.SCRIPT_ATTR).Get():
        root.GetAttribute(articulated.SCRIPT_ATTR).Block()

    for prim in list(Usd.PrimRange(root)):
        if prim.GetPath() == root.GetPath():
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        prim.RemoveAppliedSchema("PhysxRigidBodyAPI")
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        prim.RemoveAppliedSchema("PhysxArticulationAPI")
        if prim.IsA(UsdPhysics.Joint) or prim.GetTypeName() in articulated.LIGHT_TYPES:
            prim.SetActive(False)
        if prim.HasAttribute(articulated.SCRIPT_ATTR) and prim.GetAttribute(articulated.SCRIPT_ATTR).Get():
            prim.GetAttribute(articulated.SCRIPT_ATTR).Block()
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding(materialPurpose="physics")

    unit = src_facts["meters_per_unit"] * scale
    root_xf = UsdGeom.Xformable(root)
    root_matrix = root_xf.GetLocalTransformation()
    root_xf.ClearXformOpOrder()

    movable_children = [c for c in root.GetChildren() if UsdGeom.Xformable(c) and not c.IsA(UsdPhysics.Joint)]
    if not movable_children:
        raise ValueError("no Xformable children under /root")
    for child in movable_children:
        xf = UsdGeom.Xformable(child)
        ops, reset = xf.GetOrderedXformOps(), xf.GetResetXformStack()
        front = []
        if unit != 1.0:
            front.append(xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "ofw_unit"))
            front[-1].Set(Gf.Vec3d(unit, unit, unit))
        if yaw_deg:
            front.append(xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble, "ofw_yaw"))
            front[-1].Set(float(yaw_deg))
        if root_matrix != Gf.Matrix4d(1):
            front.append(xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "ofw_root"))
            front[-1].Set(root_matrix)
        if front:
            xf.SetXformOpOrder(front + list(ops), reset)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bound = articulated._bound(cache, root)
    if bound is None:
        raise ValueError("empty bounding box")
    centre = (bound[0] + bound[1]) / 2
    for child in movable_children:
        xf = UsdGeom.Xformable(child)
        ops, reset = xf.GetOrderedXformOps(), xf.GetResetXformStack()
        shift = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "ofw_center")
        shift.Set(Gf.Vec3d(*map(float, -centre)))
        xf.SetXformOpOrder([shift, *ops], reset)

    cache.Clear()
    if root.HasAttribute("extentsHint"):
        root.RemoveProperty("extentsHint")
    lo, hi = articulated._bound(cache, root)
    size = hi - lo
    if np.linalg.norm((lo + hi) / 2) > 1e-4:
        raise ValueError(f"origin not at bounding-box centre: {(lo + hi) / 2}")

    # Choose mass: objathor annotated mass -> raw USD mass -> bbox volume estimate, clamped to [min_mass, max_mass].
    if preferred_mass_kg is not None and math.isfinite(preferred_mass_kg) and preferred_mass_kg > 0.0:
        mass = float(preferred_mass_kg)
        mass_source = "objathor_metadata"
    elif raw_mass > 0.0:
        mass = float(raw_mass)
        mass_source = "usd_authored"
    else:
        mass = float(np.prod(np.maximum(size, 0.01)) * RIGID_ESTIMATED_DENSITY)
        mass_source = "bbox_estimated"

    clamped_mass = round(min(max(mass, min_mass), max_mass), 4)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(clamped_mass))

    out_usd = target_dir / "object.usd"
    flat.Export(str(out_usd), args={"format": usd_format})

    reopened = Usd.Stage.Open(str(out_usd))
    bodies = [str(p.GetPath()) for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if bodies != ["/root"]:
        raise ValueError(f"rigid bodies other than /root remain: {bodies}")
    for prim in reopened.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    sv = op.Get()
                    if sv is not None and (sv[0] < 0 or sv[1] < 0 or sv[2] < 0):
                        raise ValueError(f"negative scale remains on {prim.GetPath()}: {sv}")
    _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(out_usd)))
    if unresolved:
        raise ValueError(f"unresolved asset dependencies: {list(unresolved)}")

    return {
        "size": [float(v) for v in size],
        "centre_shift": [float(v) for v in -centre],
        "unit": unit,
        "yaw_deg": yaw_deg,
        "mass_kg": clamped_mass,
        "mass_info": {
            "source": mass_source,
            "raw_kg": round(mass, 4),
            "final_kg": clamped_mass,
            "clamped": abs(clamped_mass - mass) > 1e-6,
            "bounds_kg": [min_mass, max_mass],
        },
        "negative_scales_baked": negative_scales_baked,
        "rewritten_paths": {"relationships": rels_rewritten, "connections": conns_rewritten},
        "textures": copied_textures,
    }


def convert_one(
    sources_dir: Path,
    out_root: Path,
    entry: dict,
    *,
    revision: str = DEFAULT_REVISION,
    max_mass: float = DEFAULT_MAX_MASS_KG,
    usd_format: str = "usdc",
) -> dict:
    """Convert one selected MolmoSpaces Objaverse entry into `out_root/Assets/Object/RoboDojo/<id>/`."""
    sources_dir = Path(sources_dir)
    out_root = Path(out_root)
    aid = entry["asset_id"]
    uid = entry["uid"]
    src_rel = entry.get("src") or f"assets/{aid}/{aid}.usda"
    usda_path = sources_dir / src_rel
    oid = entry["id"]
    target = out_root / "Assets/Object/RoboDojo" / oid

    norm_lic = normalize_sketchfab_license(entry.get("sketchfab_license")) or entry.get("license") or ""
    if norm_lic not in ("CC0", "CC-BY-4.0"):
        raise ValueError(f"refusing to convert non-commercial or unknown license for {uid}: {norm_lic!r}")

    user_str = f" by {entry['sketchfab_user']}" if entry.get("sketchfab_user") else ""
    attribution = (
        f"MolmoSpaces Objaverse ({uid}{user_str}), Allen Institute for AI / Sketchfab ({norm_lic}), "
        f"huggingface.co/datasets/{REPO}"
    )

    source_provenance = {
        "dataset": REPO,
        "subpath": SUBPATH,
        "url": canonical_url(),
        "object": f"{SUBPATH}/{aid}",
        "revision": revision,
        "uid": uid,
        "asset_id": aid,
        "category": entry.get("category") or entry.get("category_slug", "object"),
        "license": norm_lic,
        "sketchfab_license": entry.get("sketchfab_license", ""),
        "sketchfab_uri": entry.get("sketchfab_uri", ""),
        "sketchfab_user": entry.get("sketchfab_user", ""),
        "attribution": attribution,
        "rsi_pools": f"allowed ({norm_lic})",
        "runtime_validated": False,
    }

    facts = adapt_rigid(
        usda_path,
        target,
        preferred_mass_kg=entry.get("mass_kg") or None,
        max_mass=max_mass,
        usd_format=usd_format,
    )
    size = facts["size"]
    geom = articulated.box_geometry(size)
    meta = {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{canonical_url()}/{SUBPATH}/{aid}")),
        "physics": {
            "mass": facts["mass_kg"],
            "friction": RIGID_FRICTION,
            "size": [round(v, 4) for v in size],
        },
        "geometry": geom,
        "active": {
            "place": {
                "up": {
                    "orientation": [1, 0, 0, 0],
                    "contact_circle": {"center": [0, 0, -size[2] / 2, 1, 0, 0, 0], "radius": geom["radius"]},
                }
            }
        },
        "passive": {"functional": {}, "support": {}},
        "source": source_provenance,
        "objathor": {
            "category": entry.get("category", ""),
            "description_short": normalize_description_short(entry.get("description_short")),
            "description": entry.get("description", ""),
            "materials": entry.get("materials", []),
            "synset": entry.get("synset", ""),
        },
        "adapter": {
            "script": SCRIPT,
            "version": VERSION,
            "negative_scales_baked": facts["negative_scales_baked"],
            "rewritten_paths": facts["rewritten_paths"],
            "textures": facts["textures"],
            "mass": facts["mass_info"],
            "validation": None,
        },
        "asset_id": oid,
        "runtime_ready": False,
    }

    articulated.write_json(target / "metadata.json", meta)
    title = (entry.get("category") or entry.get("category_slug", "object")).replace("_", " ").strip()
    desc_short_norm = normalize_description_short(entry.get("description_short"))
    desc_long = entry.get("description", "")
    desc_text = format_description_text(
        desc_short_norm,
        desc_long,
        fallback=f"A {title} ({aid}) from MolmoSpaces Objaverse.",
    )
    bilingual_names = build_bilingual_entry(
        oid,
        category_slug=entry.get("category_slug") or canonical_category_slug(title),
        size_m=meta["physics"]["size"],
        desc_short=desc_short_norm,
        desc_long=desc_long,
        materials=entry.get("materials") or [],
    )
    articulated.write_json(
        target / "description.json",
        {
            "uuid": meta["uuid"],
            "caption": {"name": [title]},
            "description": [desc_text],
            "names": bilingual_names,
        },
    )
    sha = thor._sha256_file(target / "object.usd")
    return {
        "id": oid,
        "uid": uid,
        "asset_id": aid,
        "license": norm_lic,
        "src": src_rel,
        "size_m": meta["physics"]["size"],
        "mass_kg": meta["physics"]["mass"],
        "negative_scales_baked": facts["negative_scales_baked"],
        "sha256": sha,
    }


def convert(
    sources_dir: Path,
    out_root: Path,
    manifest: dict,
    *,
    skip_existing: bool = False,
    max_mass: float = DEFAULT_MAX_MASS_KG,
    usd_format: str = "usdc",
) -> dict:
    """Convert every chosen entry in `manifest`."""
    out_root = Path(out_root)
    revision = manifest.get("revision", DEFAULT_REVISION)
    done, skipped, failed = [], [], []
    for entry in manifest.get("chosen", []):
        target = out_root / "Assets/Object/RoboDojo" / entry["id"]
        if skip_existing and (target / "object.usd").is_file() and (target / "metadata.json").is_file():
            skipped.append(entry["id"])
            continue
        try:
            done.append(
                convert_one(
                    sources_dir,
                    out_root,
                    entry,
                    revision=revision,
                    max_mass=max_mass,
                    usd_format=usd_format,
                )
            )
        except Exception as exc:
            failed.append(
                {
                    "id": entry["id"],
                    "uid": entry.get("uid", ""),
                    "asset_id": entry.get("asset_id", ""),
                    "src": entry.get("src", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "script": SCRIPT,
        "version": VERSION,
        "revision": revision,
        "converted": done,
        "skipped": skipped,
        "failed": failed,
    }


def stamp_ready(
    out_root: Path,
    object_id: str,
    check_dir: Path,
    *,
    note: str | None = None,
) -> tuple[bool, list[str]]:
    """Record `check_object.py` result in `metadata.json` and set `runtime_ready`."""
    return thor.stamp_ready(out_root, object_id, check_dir, note=note)


def _fmt_cm(val_m: float) -> str:
    cm = val_m * 100.0
    if cm >= 10.0:
        return f"{round(cm)}"
    return f"{cm:.1f}".rstrip("0").rstrip(".")


_MODIFIER_LOOKUP: dict[str, tuple[str, str]] = {key: (en_mod, zh_mod) for key, en_mod, zh_mod in MODIFIER_BILINGUAL_MAP}


def _first_modifier_in_text(text: str) -> tuple[str, str] | None:
    """Return the first `(en, zh)` modifier appearing left-to-right in `text`."""
    for tok in re.findall(r"[a-z]+", text.lower()):
        hit = _MODIFIER_LOOKUP.get(tok)
        if hit is not None:
            return hit
    return None


def _extract_modifier(desc_short: object, desc_long: str, materials: list[str], size_m: list[float]) -> tuple[str, str]:
    """Pick a verified (en, zh) color/material/style modifier by prioritized left-to-right tiers.

    Tiers (prevents secondary details like 'with red stitching' or 'with green LED display' from
    overriding the primary color in `two_words`/`three_words`):
      1. `desc_short` fields in order: `two_words`, `three_words`, `four_words`, `five_words`, `one_word`.
      2. Primary subject clause of `desc_long` (before `with` / `featuring` / `,`), then `desc_long[:160]`.
      3. `materials` list.
      4. Size-tier fallback (`small` / `medium` / `tall` / `wide`).
    """
    norm_short = normalize_description_short(desc_short)
    if isinstance(norm_short, dict):
        for k in ("two_words", "three_words", "four_words", "five_words", "one_word"):
            val = norm_short.get(k)
            if val:
                hit = _first_modifier_in_text(val)
                if hit is not None:
                    return hit
    elif isinstance(norm_short, str) and norm_short:
        hit = _first_modifier_in_text(norm_short)
        if hit is not None:
            return hit

    if isinstance(desc_long, str) and desc_long.strip():
        head = re.split(r"\b(?:with|featuring|having)\b|[,.;]", desc_long[:160], maxsplit=1, flags=re.I)[0]
        hit = _first_modifier_in_text(head) or _first_modifier_in_text(desc_long[:160])
        if hit is not None:
            return hit

    if materials:
        hit = _first_modifier_in_text(" ".join(str(m) for m in materials))
        if hit is not None:
            return hit

    max_cm = max(size_m) * 100.0 if size_m else 15.0
    if max_cm < 12.0:
        return "small", "小号"
    if max_cm <= 28.0:
        return "medium", "中号"
    return "tall" if (len(size_m) == 3 and size_m[2] > max(size_m[0], size_m[1]) * 1.2) else "wide", (
        "高款" if (len(size_m) == 3 and size_m[2] > max(size_m[0], size_m[1]) * 1.2) else "宽大"
    )


def build_bilingual_entry(
    oid: str,
    *,
    category_slug: str,
    size_m: list[float],
    desc_short: object = "",
    desc_long: str = "",
    materials: list[str] | None = None,
    used_l3_en: set[str] | None = None,
    used_l3_zh: set[str] | None = None,
) -> dict:
    """Build a 4-level bilingual `object_names.json` entry that satisfies `object_names.py check`."""
    cat_en, cat_zh, noun_en, noun_zh = CATEGORY_BILINGUAL_MAP.get(
        category_slug,
        ("object", "物件", category_slug.replace("_", " "), f"{category_slug.replace('_', ' ')}物件"),
    )
    mod_en, mod_zh = _extract_modifier(desc_short, desc_long, materials or [], size_m)

    # Avoid repeating a modifier word already inside noun_en / noun_zh (e.g. "wooden wooden pencil").
    noun_en_words = set(noun_en.lower().split())
    if mod_en in noun_en_words or mod_zh in noun_zh:
        max_cm = max(size_m) * 100.0 if size_m else 15.0
        if max_cm < 12.0:
            mod_en, mod_zh = "compact", "小巧"
        elif max_cm <= 28.0:
            mod_en, mod_zh = "medium", "中号"
        else:
            mod_en, mod_zh = "solid", "大号"

    l1_en, l1_zh = noun_en, noun_zh
    l2_en, l2_zh = f"{mod_en} {noun_en}", f"{mod_zh}{noun_zh}"

    sx, sy, sz = (size_m + [0.1, 0.1, 0.1])[:3]
    cx, cy, cz = _fmt_cm(sx), _fmt_cm(sy), _fmt_cm(sz)
    l3_en = f"{cx} by {cy} by {cz} cm {mod_en} {noun_en}"
    l3_zh = f"{cx}×{cy}×{cz}厘米{mod_zh}{noun_zh}"

    # Disambiguate rare exact-size + modifier collisions deterministically using mm precision / index suffix.
    seen_en = used_l3_en if used_l3_en is not None else set()
    seen_zh = used_l3_zh if used_l3_zh is not None else set()
    if l3_en in seen_en or l3_zh in seen_zh:
        mx, my, mz = round(sx * 1000), round(sy * 1000), round(sz * 1000)
        l3_en = f"{mx} by {my} by {mz} mm {mod_en} {noun_en}"
        l3_zh = f"{mx}×{my}×{mz}毫米{mod_zh}{noun_zh}"
    if l3_en in seen_en or l3_zh in seen_zh:
        idx_num = int(oid.rsplit("/", 1)[-1]) if oid.rsplit("/", 1)[-1].isdigit() else 0
        l3_en = f"{l3_en} model {idx_num}"
        l3_zh = f"{l3_zh}款式{idx_num}"

    seen_en.add(l3_en)
    seen_zh.add(l3_zh)
    return {
        "category": {"en": cat_en, "zh": cat_zh},
        "en": [l1_en, l2_en, l3_en],
        "zh": [l1_zh, l2_zh, l3_zh],
    }


def write_compact_object_names(path: Path, raw_table: dict) -> None:
    """Write `object_names.json` in the repository's compact 5-line-per-object format."""
    lines = ["{"]
    items = list(raw_table.items())
    for i, (oid, val) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if isinstance(val, str):
            lines.append(f"    {json.dumps(oid, ensure_ascii=False)}: {json.dumps(val, ensure_ascii=False)}{comma}")
        else:
            lines.append(f"    {json.dumps(oid, ensure_ascii=False)}: {{")
            lines.append(f'        "category": {json.dumps(val["category"], ensure_ascii=False)},')
            lines.append(f'        "en": {json.dumps(val["en"], ensure_ascii=False)},')
            lines.append(f'        "zh": {json.dumps(val["zh"], ensure_ascii=False)}')
            lines.append(f"    }}{comma}")
    lines.append("}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_filter = sub.add_parser("filter", help="Fetch/load indices, verify Sketchfab commercial licenses, and filter.")
    p_filter.add_argument("--sources-dir", type=Path, required=True, help="Cache dir holding _index/ and assets/")
    p_filter.add_argument("--revision", default=DEFAULT_REVISION)
    p_filter.add_argument("--endpoint", default=None, help="Override HF_ENDPOINT")
    p_filter.add_argument("--refresh-index", action="store_true")
    p_filter.add_argument("--license-workers", type=int, default=8)
    p_filter.add_argument("--no-whitelist", action="store_true", help="Do not restrict to CURATED_WHITELIST_CATEGORIES")
    p_filter.add_argument("--categories", nargs="*", default=None)
    p_filter.add_argument("--exclude-categories", nargs="*", default=None)
    p_filter.add_argument("--allow-untextured", action="store_true")
    p_filter.add_argument("--allow-non-pickup", action="store_true")
    p_filter.add_argument("--allow-non-on-object", action="store_true")
    p_filter.add_argument("--min-size", type=float, default=MIN_EXTENT_M)
    p_filter.add_argument("--max-size", type=float, default=MAX_EXTENT_M)
    p_filter.add_argument("--max-per-category", type=int, default=None)
    p_filter.add_argument("--limit", type=int, default=None)
    p_filter.add_argument("--uids", nargs="*", default=None)
    p_filter.add_argument("--uids-file", type=Path, default=None)
    p_filter.add_argument("--existing-state", type=Path, default=None)
    p_filter.add_argument("--out", type=Path, required=True, help="Output selection JSON path")

    p_fetch = sub.add_parser("fetch", help="Selectively Range-fetch & extract Objaverse assets from selection JSON.")
    p_fetch.add_argument("--selection", type=Path, required=True)
    p_fetch.add_argument("--out", type=Path, required=True)
    p_fetch.add_argument("--revision", default=None)
    p_fetch.add_argument("--endpoint", default=None)
    p_fetch.add_argument("--workers", type=int, default=8)
    p_fetch.add_argument("--shard-index", type=int, default=0)
    p_fetch.add_argument("--shard-total", type=int, default=1)

    p_survey = sub.add_parser("survey", help="Inspect extracted Objaverse .usda assets with pxr and emit manifest.")
    p_survey.add_argument("--source", type=Path, required=True)
    p_survey.add_argument("--selection", type=Path, required=True)
    p_survey.add_argument("--min-size", type=float, default=MIN_EXTENT_M)
    p_survey.add_argument("--max-size", type=float, default=MAX_EXTENT_M)
    p_survey.add_argument("--out", type=Path, required=True)

    p_conv = sub.add_parser("convert", help="Adapt Objaverse .usda assets into object.usd + metadata.json.")
    p_conv.add_argument("--source", type=Path, required=True)
    p_conv.add_argument("--root", type=Path, required=True)
    p_conv.add_argument("--manifest", type=Path, required=True)
    p_conv.add_argument("--skip-existing", action="store_true")
    p_conv.add_argument("--max-mass", type=float, default=DEFAULT_MAX_MASS_KG)
    p_conv.add_argument("--usd-format", choices=("usdc", "usda"), default="usdc")
    p_conv.add_argument("--summary", type=Path, default=None)

    p_ready = sub.add_parser("ready", help="Stamp check_object.py results into metadata.json.")
    p_ready.add_argument("--root", type=Path, required=True)
    p_ready.add_argument("--check-dir", type=Path, required=True)
    p_ready.add_argument("--manifest", type=Path, default=None)
    p_ready.add_argument("--ids-file", type=Path, default=None)
    p_ready.add_argument("ids", nargs="*")
    p_ready.add_argument("--summary", type=Path, default=None)
    p_ready.add_argument("--note", default=None)

    args = parser.parse_args(argv)

    if args.command == "filter":
        arrow_table, cmeta, ometa = fetch_molmo_indices(
            args.sources_dir, revision=args.revision, endpoint=args.endpoint, force=args.refresh_index
        )
        req_uids = None
        if args.uids or args.uids_file:
            req_uids = set(args.uids or [])
            if args.uids_file:
                req_uids |= {
                    ln.strip()
                    for ln in args.uids_file.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")
                }
        existing_uids, next_idx = load_existing_state(args.existing_state)
        whitelist = None if args.no_whitelist else CURATED_WHITELIST_SET

        # Pass 1: pre-filter without license to narrow candidate UIDs before fetching Sketchfab metadata shards.
        pre_rows = build_catalog(arrow_table, cmeta, ometa, licenses=None)
        pre_sel = filter_catalog(
            pre_rows,
            commercial_only=False,
            pickup_only=not args.allow_non_pickup,
            on_object_only=not args.allow_non_on_object,
            textured_only=not args.allow_untextured,
            min_size=args.min_size,
            max_size=args.max_size,
            whitelist=whitelist,
            categories=args.categories,
            exclude_categories=args.exclude_categories,
            max_per_category=None,
            limit=None,
            uids=req_uids,
            existing_uids=existing_uids,
            revision=args.revision,
        )
        cand_uids = {e["uid"] for e in pre_sel["chosen"]}
        licenses = fetch_objaverse_licenses(
            args.sources_dir,
            cand_uids,
            revision=args.revision,
            endpoint=args.endpoint,
            workers=args.license_workers,
            force=args.refresh_index,
        )

        # Pass 2: final filter with strict CC0 / CC-BY license enforcement, category caps, and deterministic IDs.
        rows = build_catalog(arrow_table, cmeta, ometa, licenses=licenses)
        result = filter_catalog(
            rows,
            commercial_only=True,
            pickup_only=not args.allow_non_pickup,
            on_object_only=not args.allow_non_on_object,
            textured_only=not args.allow_untextured,
            min_size=args.min_size,
            max_size=args.max_size,
            whitelist=whitelist,
            categories=args.categories,
            exclude_categories=args.exclude_categories,
            max_per_category=args.max_per_category,
            limit=args.limit,
            uids=req_uids,
            existing_uids=existing_uids,
            next_index=next_idx,
            revision=args.revision,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(args.out, result)
        print(f"[obja] filter {json.dumps(result['summary'], ensure_ascii=False)}", flush=True)
        return 0

    if args.command == "fetch":
        sel = json.loads(args.selection.read_text(encoding="utf-8"))
        entries = sel.get("chosen", [])
        if args.shard_total > 1:
            entries = [e for i, e in enumerate(entries) if i % args.shard_total == args.shard_index]
        revision = args.revision or sel.get("revision", DEFAULT_REVISION)
        res = fetch_assets(entries, args.out, revision=revision, endpoint=args.endpoint, workers=args.workers)
        print(
            f"[obja] fetch shard {args.shard_index}/{args.shard_total}: "
            f"fetched={len(res['fetched'])}, already={len(res['already_present'])}, failed={len(res['failed'])}",
            flush=True,
        )
        return 1 if res["failed"] else 0

    if args.command == "survey":
        sel = json.loads(args.selection.read_text(encoding="utf-8"))
        res = survey(args.source, selection=sel, min_size=args.min_size, max_size=args.max_size)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(args.out, res)
        print(f"[obja] survey chosen={len(res['chosen'])}, dropped={len(res['dropped'])}", flush=True)
        return 0

    if args.command == "convert":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        res = convert(
            args.source,
            args.root,
            manifest,
            skip_existing=args.skip_existing,
            max_mass=args.max_mass,
            usd_format=args.usd_format,
        )
        summary_path = args.summary or (args.root / "molmospaces_objaverse_conversion.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(summary_path, res)
        print(
            f"[obja] convert converted={len(res['converted'])}, skipped={len(res['skipped'])}, "
            f"failed={len(res['failed'])} -> {summary_path}",
            flush=True,
        )
        return 1 if res["failed"] else 0

    # ready
    ids = list(args.ids)
    if args.manifest:
        m = json.loads(args.manifest.read_text(encoding="utf-8"))
        ids.extend(e["id"] for e in m.get("chosen", []))
    if args.ids_file:
        ids.extend(
            ln.strip()
            for ln in args.ids_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        )
    ids = list(dict.fromkeys(ids))
    if not ids:
        parser.error("ready needs ids, --manifest, or --ids-file")

    passed_ids, failed_ids, reason_counts = [], [], {}
    for oid in ids:
        ok, why = stamp_ready(args.root, oid, args.check_dir, note=args.note)
        if ok:
            passed_ids.append(oid)
        else:
            failed_ids.append({"id": oid, "why": why})
            for code in why:
                reason_counts[code] = reason_counts.get(code, 0) + 1
    summary = {
        "total": len(ids),
        "passed": len(passed_ids),
        "failed": len(failed_ids),
        "failure_reasons": reason_counts,
        "passed_ids": passed_ids,
        "not_ready": failed_ids,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(args.summary, summary)
    print(f"[obja] ready {len(passed_ids)}/{len(ids)}; reasons={json.dumps(reason_counts)}", flush=True)
    return 0 if not failed_ids else 1


if __name__ == "__main__":
    code = main()
    if "APP" in globals():
        if code:
            os._exit(code)  # Kit eats exit codes on close
        APP.close()  # noqa: F821
    raise SystemExit(code)
