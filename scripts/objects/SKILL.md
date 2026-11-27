---
name: real2sim-object-asset
description: 按 Real2Sim 数采平台的物体登记（real2sim/objects/<id>/spec.json 和实拍照片）给实物孪生建 RoboDojo 刚体资产：写参数化生成脚本，生成 object.usdz / metadata.json / description.json，在 GPU 服务器隔离目录验证，按登记的尺寸名称回报 measurements_mm，并更新登记表。用于：给平台登记过的实物建模或按实测重建、处理平台物体页给出的「建模任务」、修改 scripts/objects/ 下的生成脚本。
---

# 实物孪生物体建模

平台负责登记（实测值和照片），这里负责把登记变成仿真资产。本流程的下文针对刚体（`Rigid`）。柔性物体改用 `real2sim/prototypes/deformable_assets/README.md`：沿用登记/照片/ID，由 agent 生成几何和候选求解配置，再运行柔性专用检查；不套用刚体倾角标准。关节与流体按对应流程处理。

## 先分清改的是哪一层

| 层 | 位置 | 入库 | 这一步里怎么用 |
| :-- | :-- | :-- | :-- |
| 登记（输入） | 平台状态目录 `<asset-server>:/data/platform/state/real2sim/objects/<id>/`：`spec.json`、`reference/*.jpg` | 拷回本机后入库 | 只读；同步到本机，随脚本一起提交 |
| 生成代码 | `scripts/objects/make_*.py`；任务也要用的几何放 `scripts/objects/<类别>_shape.py`（只依赖 numpy） | 入库 | 新建或修改 |
| 生成结果 | `Assets/Object/Real2Sim/<id>/`（Real2Sim 物体库；官方物体在 `Object/RoboDojo/`，不是孪生的积木在 `Object/Blocks/`，编号各库共用不重复）（`object.usdz`、`metadata.json`、`description.json`） | 不入库 | 只由脚本写，不手改；不要往 `Assets/` 放要入库的文件 |
| 渲染预览和检查结果 | `real2sim/objects/<id>/preview/sim.png`、`check.json` | 不入库 | `check_object.py --image-dir` 生成，用 `r2s publish` 发布到平台：平台据此显示渲染图，并在资产装进共享库之前就推进「建模」「尺寸核对」「物理检验」 |
| 验证 | Slurm 作业里的隔离目录 `/data/robodojo-runs/<名称>-<时间>/`（登录机上是 `~/robodojo-runs/<名称>-<时间>/`） | 不入库 | 生成到隔离目录的覆盖目录里再检查 |
| 共享资产库 | `<asset-server>` 长期资产库；平台读 `<asset-server>:/data/robodojo-assets/Assets`，仿真机器用 `asset_client.py sync` 同步整库，或 `pull` 只拉作业要的（Slurm 节点 `robodojo-sim` 的 `SIM_ASSET_NEEDS`） | 不入库 | 只有用户同意后，按 README「改服务器共享资产库」提交、审核 |
| 登记表 | `real2sim/ASSETS.md`、README「快速开始」第 4 步、`HANDOFF.md` | 入库 | 更新 |

不在这一步改：任务配置和布局（`task/RoboDojo/`、`scripts/g1d/build_asset.py` 的 `G1D_TASKS`），除非用户要求；子模块 `XPolicyLab/`、`third_party/`。

`<id>` 是 `<类型>/<类别>/<5 位编号>`，如 `Rigid/measuring_cup/00002`。编号由平台登记时分配，资产目录、照片目录、生成脚本里的编号都用它，不要另起。

机器名、路径这类会变的信息以 `HANDOFF.md`「服务器」为准；数采平台的部署、发布和回滚见 `src/real2sim_platform/ops.sh` 开头的说明，账号、隧道这类运维细节在本机 `.agent/ops.local.md`，仓库是公开的，这些不入库，本机没有这个文件就问用户。下文写的是写这份说明时的状态。

## 流程

复制这份清单，逐项完成：

```
- [ ] 1. 读 HANDOFF.md，从 origin/main 开 worktree 短分支
- [ ] 2. 取登记：spec.json + reference/
- [ ] 3. 定几何方案，写下每个尺寸的定义
- [ ] 4. 写或改生成脚本（含自检、measurements_mm）
- [ ] 5. 本机静态检查
- [ ] 6. GPU 服务器隔离目录：生成 + check_object
- [ ] 7. 核对 measurements_mm 与实测
- [ ] 8. 渲染图对照实拍照片
- [ ] 9. 登记、提交、交付说明
- [ ] 10. （用户同意后）装共享资产库，刷新平台
```

### 1. 开工

按 `CLAUDE.md`：先读 `HANDOFF.md`（GPU 服务器、共享库、各资产的当前状态都在里面），然后从 `origin/main` 开一个自己的 `git worktree` 和短分支（`<名字>/<范围>-<简述>`，如 `contributor/objects-paper-cup`）。主工作区有好几个会话共用，共用暂存区曾把一个会话的文件提交进另一个会话的 commit。别人未提交的改动不要碰。

### 2. 取登记

登记在平台的状态目录里（`<asset-server>:/data/platform/state/`，只读；平台代码跑在 `/opt/platform/current`，不可修改，两处都不要直接写）。同步到本机：

```bash
rsync -az --exclude preview/ <asset-server>:/data/platform/state/real2sim/objects/<id>/ real2sim/objects/<id>/
```

<asset-server> 从外面只有约 0.4 MB/s：照片多的登记要等一会儿，别反复重拉。

`spec.json` 字段（空字段不出现）：

| 字段 | 含义 |
| :-- | :-- |
| `id`、`name`、`name_en` | 资产 id、中文名、英文名 |
| `tasks` | 任务配置里已经用到这件物体的任务；没有就是新物体 |
| `material`、`color` | 材质、颜色（自由文本） |
| `mass_g` | 实测质量，g |
| `static_friction` | 实测静摩擦系数（斜面法），可能没有 |
| `dimensions_mm` | 实测尺寸原文，单位 mm，每项「名称 数值」，如 `开口直径 82，底部直径 64，高 135` |
| `features` | 外观与结构（透明度、把手、刻度、纹理、壁厚等） |
| `requirements` | 建模要求（如中空能装豆子、把手可抓、刻度不用建模） |
| `notes` | 备注，常有补充的实测值 |
| `generalization` | `rt_color`（颜色随机）、`rt_scale`（整体等比缩放 ±%）：运行时按布局随机，**不建新资产**，见下方「运行时随机」；`count` 等是要另建的变体资产：`count` 数量，`color`/`material`/`size`/`shape` 为 true 的属性可以变（其余与本体一致），`requirements` 泛化要求；变体放同类别后续空号，不实测，`metadata.json` 顶层写 `"variant_of": "<本体 id>"`（平台据此把它收进本体卡片），在 `ASSETS.md` 标为泛化变体 |
| `updated_by`、`updated_at` | 最后修改人和时间，写进脚本 docstring 作为测量来源 |

- 没有 `spec.json`：物体还没登记成实物孪生。让用户先在平台登记，不要自己编测量值。
- 尺寸和质量以 `spec.json` 为准（`dimensions_mm` 和 `notes` 里的实测值都算），照片只用来定形状、比例和外观。都没给的尺寸按照片估，在脚本注释和 `ASSETS.md` 里标「估计」。
- `notes` 和 `dimensions_mm` 矛盾时，问用户。
- **运行时随机**（`rt_color`、`rt_scale`）：资产只建本体一份。换色默认覆盖整个物体；要保留标签、透明件时，在 `metadata.json` 的 `visual` 里写 `"color_parts": [主体网格名]`（如 `["model"]`）。真正随机是在任务层打开的：`scripts/g1d/build_asset.py` 的 `RANDOMIZE` 给用到它的任务和标签加 `{"scale": (lo, hi), "color": True}`，这会改那个任务的布局，先问用户；放大后仍要能被 Dex1-1 夹住（窄边 ≤ 80 mm）。

### 3. 几何方案

- 给 `dimensions_mm` 里的每个名称写清定义：外径还是内径、对边还是对角、含不含把手或翻边。把定义写在脚本常量的注释里，第 4 步按同样的定义去量模型。
- 选构造方式，从最接近的现有脚本改：
  - 回转体（圆杯、碗）：`make_water_cup.py` 的 `lathe`
  - 截面沿高度渐变（圆角方形、超椭圆）：`make_plastic_cups.py` 的 `loft`、`wall_staves`，以及 `measuring_cup_shape.py`
  - 把手、提梁：`mp.sweep`；平板、挂耳：`mp.prism`；圆角：`mp.arc`
  - 小颗粒（豆子）：`make_red_bean.py`
- 任务代码也要用这件物体的几何时（如判断豆子在不在杯里），把尺寸和截面放进只依赖 numpy 的形状模块，生成脚本和任务都从这里读，保证两边一致（参照 `measuring_cup_shape.py`）。

### 4. 生成脚本

**改哪个文件**：这个类别已有生成脚本（平台物体页「生成脚本」一栏，或 `rg -l 'Rigid/<类别>|"<类别>"' scripts/objects`）就改那个；新类别新建 `scripts/objects/make_<类别>.py`。同一族物体放一个脚本，参数表和编号一一对应（如 `measuring_cup_shape.MEASURING`）；运行时会把整族都重新生成一遍。

**约定**（新脚本照 `make_plastic_cups.py`，它是最新的写法）：

- docstring：真实对应物和测量来源（spec 的 `updated_at`）、Usage、坐标系、USD 结构。
- 命令行：`python scripts/objects/make_<类别>.py [TYPE_DIR]`，默认 `Assets/Object/Real2Sim/Rigid`，写到 `<TYPE_DIR>/<类别>/<编号>/`。不要把共享库路径写死。老脚本 `make_pitcher.py`、`make_water_cup.py` 的参数是类别目录，改它们时保持原样。
- 依赖：numpy、scipy、pxr。用 `import make_pitcher as mp` 取共享 USD 工具（`add_xform`、`add_mesh`、`add_material`、`arc`、`sweep`、`prism`、`vertex_normals`、`signed_volume`、`r4`）；直接导入不了 pxr 时它会启动 Kit。脚本结尾照 `make_plastic_cups.py` 处理 `mp.APP`：出异常时 `os._exit(1)`，否则 `APP.close()`。
- 单位：spec 用 mm、g，脚本常量用 m、kg；每个常量注释标「实测」或「估计」。
- 坐标系：Z 向上，单位米，原点在包围盒中心（生成后整体平移 `-(lo + hi) / 2`，见 `save()`）。有把手或壶嘴的物体，把手朝 +X，和官方 `mug` 一致。
- USD 结构（照 `make_plastic_cups.py` 的 `write_usdz` / `add_collision`）：
  - `/root`：defaultPrim，`RigidBodyAPI`，上轴 Z，`metersPerUnit = 1`
  - `/root/visual/model`：主体，必须是层级里第一个 Mesh（加载器的 `get_bbox`、`sample_mesh_vertices` 取第一个 Mesh），其他可见部件放 `/root/visual/<名称>`
  - `/root/_materials/<名称>`：材质
  - `/root/collision/hull_NN`：显式凸包，purpose 设 `guide`，渲染不可见，`CollisionAPI` + `MeshCollisionAPI(convexHull)`
- 碰撞体：容器要空心，用杯底一块 + 杯壁分片（`wall_staves`），开口不能被封住。每个凸包 ≤ 64 个顶点、单个面 ≤ 32 个顶点，超出 PhysX GPU 限制就不能和颗粒碰撞，豆子会穿过去；底板多边形用 24 边（`BOTTOM_SIDES`）。
- 自检（脚本里 assert，不通过不写文件）：每个可见部件 `signed_volume > 0`（绕序朝外）；凸包不侵入空腔（`cavity_probe` + `self_check`）。容器打印容积；spec 写了容量（刻度、满口毫升数）就对一下。
- 材质：不透明用 OmniPBR；半透明塑料用 OmniSurface（漫反射权重固定 1，只用透射权重调乳白程度，粗糙度 ≤ 0.35，原因见 `clear_pp` 的注释）；透明玻璃用 OmniGlass。按 spec 的材质、颜色和照片调，目测接近即可。
- **标签、贴纸、印刷图案**：实物上有标签、贴纸、logo、文字或花纹，就必须做成照片贴图，不能省略，也不能用程序画的通用文字、插图、纯色块代替（农夫山泉 00000 的程序标签、东方树叶第一版包满整瓶的标签，都因此被用户判为"很假"）。
  - 位置：先在照片上标出每块标签的起止高度和所在面或包角，只贴在实物有标签的位置；每块单独一个网格（`/root/visual/label_front`、`label_back`、`label_neck` 等），纹理打进 usdz（`textures/<名称>.png`），略高出表面 0.1–0.3 mm 免得闪烁，用不透明的 OmniPBR。
  - 贴图：平面标签从照片按四个角点做透视校正；圆柱面按柱面展开（只取正对相机那一段，边缘斜视处不可信）；再除掉照片的平滑明暗、压掉高光。分辨率 ≥ 10 px/mm，文字要可读。可复用 `bottle_label.py` 的 `photo_panel`、`photo_band`、`even_light`（东方树叶 00006 的实现，在分支 `z/objects-plastic-bottle-00006`，合并前从该分支取）。
  - 透明主体会透出标签背面：背面用纸背颜色（`two_sided`），不要镜像正面图案。
  - 照片缺某一面（背面、瓶颈、底部）：在交付说明里列出缺的角度，请登记人补拍；补拍前该面用纯色底，并在 `ASSETS.md` 写「缺照片」，不要编内容。
  - 泛化变体可以换成虚构品牌的绘制标签，但位置和覆盖范围沿用本体。
- uuid：`uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo/rigid/{类别}/{编号}")`，编号用整数。
- `description.json`：`uuid`；`caption` 下 `color`、`material`、`size`、`name`、`shape`、`function` 各 3 个英文短语；`description` 是 5 句英文描述；`names` 是指令名称（见下）。
- **指令名称**：任务指令模板里的 `<标签>` 由 `desc_manager` 按模板语言（含汉字就用中文）从大类名称和 3 个名称里随机取一个填入。同一模板里两个物体共有的名称会被跳过，比如两只量杯都是 "cup"、"measuring cup"，就只用能区分它们的那两级。2026-11-03 起共享库里所有物体（含 RoboDojo 官方、NVIDIA SimReady、AgiBot GenieSim、积木库）都有名称；没有 `names` 的物体仍取 `description` 的整句。平台物体卡片显示大类和第一级，详情的「指令名称」一节列出全族的名称。
  - 格式：`"names": {"category": {"en": "cup", "zh": "杯子"}, "en": [3 个], "zh": [3 个]}`。本体和每个泛化变体都写。`category` 是大类（泛化名称），3 个名称从简到详：类别名（measuring cup / 量杯）→ 加上在同族里区分它的一两个属性（tall clear measuring cup / 高的透明量杯）→ 在所有物体里唯一指认它的完整短语。大类不和 3 个名称重复，也不比第一个长。
  - 写名词短语，能直接填进 "pick up the <x>"、"拿起<x>"：不带冠词、数量词、句末标点；不写位置、左右、比较级、任务角色。英文小写，品牌等专名除外。
  - 颜色和标签以生成的模型为准；变体写自己的属性，大类和第一级一般和本体相同。登记开了 `rt_color` 时，不写会被换色的部件的颜色（`visual.color_parts`，没写就是整件）。
  - 写进 `scripts/objects/object_names.json`（键是资产 id；能拧盖的 Articulation 版、柔性版 `Dynamic/` 写成对应 Clutter id 的别名）。生成脚本不写 `names`，每次生成后都要跑 `python scripts/objects/object_names.py apply <物体根目录>`（如 `ovl`；没有 `description.json` 的物体会新建一个只含 `names` 的）。新物体装库前在表里补条目，`object_names.py missing <物体根目录>...` 列出表里没有的物体（全覆盖时退出码 0）；覆盖要同时查共享库三个根目录和平台登记目录 `<gpu-server>:/data/platform/state/real2sim/objects`：只登记、没进资产库的物体（建模中、GenieSim 没过验证）没有 `description.json`，平台直接显示表里的名称，所以登记时就在表里补条目。`object_names.py check` 校验有没有大类、是不是正好 3 个、顺序（按英文词数、中文字数不减）、措辞，以及最详细的名称不和其他物体重复；它也在 `check.sh` 里。旧格式（5 个以上名称、没有 `category`）运行时仍能用，但不过 `check`。

**`metadata.json`**：照 `make_plastic_cups.py` 的 `metadata()` 从网格计算（下面只是结构示意，数值保留 4 位）。

```json
{
  "uuid": "...",
  "physics": {"mass": 0.045, "size": [0.082, 0.082, 0.135]},
  "measurements_mm": {"开口直径": 82.1, "底部直径": 64.0, "高": 135.0},
  "geometry": {"faces": "...", "vertices": "...", "radius": "...", "bbox": "...", "center": "...",
               "aligned_bbox": {"vertices": "8 个角点", "extents": "..."}, "oriented_bbox": "同上"},
  "active": {"place": {"up": {"orientation": [1, 0, 0, 0], "projection_circle": "...", "contact_circle": "..."}},
             "functional": {}},
  "passive": {"functional": {}, "support": {}}
}
```

- `physics.mass` = `mass_g / 1000`。运行时会把质量截到 ≤ 0.5 kg（`env/scene_manager/objects/rigid.py`），超过的话在 `ASSETS.md` 记一笔。
- **摩擦**：运行时只读 `physics.static_friction` / `dynamic_friction`（不写时默认 0.6 / 1.5）；老脚本写的 `friction` 不生效。spec 有静摩擦才写这两个键：`static_friction` 用实测值，`dynamic_friction` 没测就取同一个值并标「估计」。没有实测就都不写，`ASSETS.md` 里标「默认」。已有任务的成功率都是在默认摩擦下调出来的，改摩擦前先告诉用户。
- **`measurements_mm`**（现有脚本的 `metadata()` 都还没写，要自己加）：
  - 键只取 `dimensions_mm` 里的名称，一字不差（平台按名称核对，对不上就核对不了）；`notes` 里的其他实测值用来建模，但不放进这个字典。
  - 值在 `save()` 平移之后，从生成的网格按第 3 步的定义量出来，不要直接抄常量；只取对应部件的顶点（比如外径只用杯身，不含把手）。单位 mm，保留 1 位小数。
  - 加载器只读 physics / visual / geometry / active / passive，这个键不影响仿真。
- `oriented_bbox`：`env/scene_manager/layout_manager.py` 的 `get_instance_bbox_vertices` 优先取它的 8 个角点（缺了才退回第一个 Mesh 的包围盒），奖励判定（在不在另一物体的包围盒里、叠放等，`env/reward_manager/func_parser.py`）都经它取包围盒。要包住把手、翻边等全部部件。
- `active.functional`：能抓的把手写抓取帧，照 `make_plastic_cups.py`：`build_mug()` 返回的 `grip` 点，在 `metadata()` 里写成 `functional.handle.frame`（和官方 `mug` 一样的 +X 竖直把手）。没有就留空。
- **铰接物体的关节作用**（`Articulation/<category>/<index>`；能动的物体只放这里，别的类型一律算不可动）：每个可动关节（revolute / prismatic，固定关节不写）在 `metadata.json` 的 `body_joints` 里一项，写 `name`、`type`（`revolute` / `prismatic`）、`role`、`parent`、`child`、`axis`，限位写 `limits_deg`（转动）或 `limits_m`（平移），和 Real2Sim 微波炉 `Articulation/microwave/00000` 的写法一致。`role` 取值：
  - 能打开：`door`（铰链门、翻门）、`drawer`（抽屉）、`lid`（有铰链、能合上的翻盖，如笔记本屏幕、蛋盒盖）；
  - 只能动：`knob`（旋钮、旋转开关）、`button`（按键、按压杆、推拉开关）、`turntable`、`handle`、`pedal`、`pointer`、`rotor`；认不出写 `other`，不要猜。
  - 不算能打开、记 `other` 并用 `part` 写明部件：螺旋瓶盖 `part: cap`、泵头 `part: pump`、烤箱烤架 `part: rack`、托盘 `part: tray`。固定的显示面板没有可动关节，不记 `lid`。
  - 物体至少有一个 `door` / `drawer` / `lid` 关节就是「能打开」（`motion: openable`），只有别的作用是「只能动」（`movable`），不在 `Articulation/` 下是「不可动」（`static`）。平台物体卡片（「可开合 · 门」「可动 · 旋钮」）、筛选和 `asset_client.py find --motion` 都由 `OpenFlywheel/asset_client.py` 的 `joint_motion()` 从这里算，扫描时不读 USD。
  - 没写 `role` 的老物体（官方 `laptop`、`toaster`、`egg_holder`、`SpringButton`、`plastic_bottle`、`cable`、`microwave`）按 id 写在同一文件的 `LEGACY_ROLES`；新物体写 `role`，不进表。两处都没有的 `Articulation/` 物体显示「可动（未分类）」。

### 5. 本机静态检查

```bash
bash scripts/check.sh     # 和 CI 一样：改动过的 Python 文件过 ruff（0.11.7，行宽 120），再跑单测和任务清单检查
python -m py_compile scripts/objects/<脚本>.py
```

本机没有 pxr，不在本机生成 USD。有 numpy 和 scipy 的话，可以把 pxr 打桩，只跑几何构建和 `self_check`，提前发现绕序和空腔问题。

### 6. GPU 服务器隔离目录：生成 + 落地检查

- 服务器：一律用 GPU Slurm 集群的作业（Isaac Sim 5.1，`robodojo-sim <命令>`，`scripts/cluster/SKILL.md`；`<gpu-server>` 11-14 起也是集群节点，不再直接登录跑）。隔离目录建在登录机 `<gpu-server>` 的 `~/robodojo-runs/<类别>-<YYYYMMDD-HHMMSS>/`，作业里就是 `/data/robodojo-runs/<类别>-<YYYYMMDD-HHMMSS>/`。连接信息看本机 `.agent/remote-validation.local.toml`，这个文件不入库，连接信息也不要写进仓库。作业里的共享资产库是只读的 `/data/robodojo-assets/Assets`，隔离目录要用的官方资产直接从它软链接或拷贝。
- 把当前工作树传进隔离目录，排除 `Assets/`、`.git/`、`output/`、`eval_result/`；子模块按 `scripts/cluster/SKILL.md` 的做法放进运行目录（位置见本机 `.agent/remote-validation.local.toml`）。生成结果写到隔离目录的 `ovl/` 里，**不要直接写共享资产库**。`check_object.py` 只需要物体目录本身；要跑任务或脚本演示，才按 README「改服务器共享资产库」第 1 步建完整的 `Assets/` 覆盖目录。
- 在作业里跑（`srun --gres=gpu:1 robodojo-sim bash -c '…'` 或 sbatch 脚本；卡由 Slurm 分配，作业里卡号从 0 起）：

```bash
python scripts/objects/<脚本>.py ovl/Rigid      # 老脚本 make_pitcher.py、make_water_cup.py 传类别目录 ovl/Rigid/<类别>
python scripts/objects/object_names.py apply ovl   # 生成会重写 description.json，名称每次都要重新写入
PYTHONPATH=. python scripts/objects/check_object.py ovl/Rigid/<类别>/<编号> --image-dir real2sim/objects
```

- 柔性物体（FEM、粒子布料、绳）不能按刚体加载，`check_object.py` 跑不了；单件模型图（平台封面）用 `scripts/objects/render_model.py <物体目录> --image-dir real2sim/objects`，同样的地面、灯光和机位，不仿真，写 `preview/00_model.png`。落地、悬垂、挤压等测试图文件名写明测试项（drop、suspend、squeeze…），平台把它们放进「形变测试」，不当封面。粒子布料（`Garment`，`GarmentObject` 加载）用 `scripts/objects/check_cloth.py <物体目录> --case drop|pinch|fold --physics-hz 300`（300 Hz 是 G1-D 的物理频率）：在物理 0 号卡上跑、不设 `CUDA_VISIBLE_DEVICES`（别的卡上布料张量视图会报非法内存访问或建不出来）；pinch、fold 各再跑一次新进程比对 `nodes.npz` 作重放证据。夹片是按速度驱动的动态刚体：瞬移或按位姿摆的运动学刚体没有速度，摩擦带不动布。fold 的夹爪像手腕一样随弧线转 180°，弧顶比绷直低 40 mm，松开时夹片直接消失（理想化释放）：从两层布之间抽出夹片会把上层整片带走 40–55 mm，这是机器人任务的释放问题，不归资产检查。
- `check_object.py` 判 OK 的标准：倾角 < 5°、落地后中心高度与半高差 < 5 mm、速度 < 0.01 m/s，退出码 0。FAIL 时先查原点是否在包围盒中心、底面碰撞体是否平。细长、球形、会滚的物体（如红豆）倾角判据可能不适用，要写明原因，不能直接当成通过。
- 容器要装颗粒时（`requirements` 写了装豆或装水，或任务里要倒），再跑 `check_cup_beans.py` 或 `check_pitcher_fluid.py`。这两个脚本只接受 `--device cpu`（默认就是），GPU 状态管线下粒子坐标不回读，结果无效。
- 预计跑十几分钟以上的用 `nohup setsid` 放后台，SSH 长连接会断，进程会被一起杀掉。

### 7. 尺寸核对

- 读隔离目录里生成的 `metadata.json`，对 `dimensions_mm` 里的每个名称：|`measurements_mm` − 实测| ≤ max(2 mm, 2% × 实测)。质量直接用实测值。
- 超差就改脚本常量后重新生成，**不要改测量值**。怀疑测量本身有问题，问用户。
- 列一张表放进交付说明：名称 / 实测 / 模型 / 误差。

### 8. 外观

- 按 `reference/` 每张照片的视角各渲染一张，和照片并排拼成对照图放进 `preview/`：逐项核对截面形状、肩颈比例、颜色、透明度、把手位置，以及每块标签的位置、大小、颜色和文字。标签缺失或与照片对不上就不算完成，回到第 4 步。
- 建了泛化变体时，再拼一张本体加全部变体的并排总览图：变体之间要一眼能区分（不能只差一点色相或几毫米），登记没允许变的属性要和本体一致。
- 把 `preview/` 同步回本机，并发布到平台：`scripts/platform/r2s publish <运行目录>/real2sim/objects/<id>/preview real2sim/objects/<id>/preview`（在哪台机器上都行，经 OBS；被替换的旧文件进 `state/backups/`，发布后自动重新扫描；写 key 和 Slurm token 见 `AGENTS.md`）。这个目录不入库。
- `preview/check.json` 由 `check_object.py` 写：落地结果（`settle`）、资产的质量、尺寸和 `measurements_mm`。其余物理检验做了哪项，就在里面加 `"checks": {"collision" | "grasp" | "liquid" | "render": "pass" | "fail" | "na"}` 和 `"notes": {项: 依据}`（如「容器口未封，check_cup_beans 保留 100%」「非容器」）；没做的不写，不要为了通过而写 pass。重跑 `check_object.py` 会覆盖这个文件，补写的项要重新加。平台上人工选的结论优先。

### 9. 登记、提交、交付说明

- **提交**：
  - 生成脚本（和形状模块）
  - `scripts/objects/object_names.json` 里本体和变体的条目
  - `real2sim/objects/<id>/spec.json` 和 `reference/` 照片（从平台状态目录拷来的原件，图片自动走 LFS）
  - 新脚本或新物体：README「快速开始」第 4 步加上生成命令
  - `real2sim/ASSETS.md` 新增或更新一行：真实对应物、生成脚本，尺寸、质量、摩擦、外观各自的匹配等级和依据（spec 里的值算「实测」，照片估的算「估计」），共享库状态
  - `HANDOFF.md`
- **不提交**：`Assets/` 下任何生成结果、`preview/`、隔离目录里的东西。
- 在第 1 步开的 worktree 短分支上提交，`git add` 逐个列出文件，提交前跑 `bash scripts/check.sh`，然后推分支、开 PR，等 CI 通过再合并进 `main`。
- 提交信息：`[scripts] feat: add <类别> twin asset from platform registration`（重建用 `update`）。
- 本机 `Assets/` 里生成后，可以按 README 跑 `format.py` 刷新 `uuid_map.json`。运行时不读这个文件，隔离目录和共享库都不需要跑。
- **交付说明**：资产 id、生成脚本、尺寸对照表、`check_object.py` 输出和渲染图路径、哪些值是估计的、装没装共享库。平台物体页的「物理检验」读 `check.json`，人工结论可以覆盖；交付说明里写清每项的依据。

### 10. 共享资产库（必须先得到用户同意）

- 按 README「改服务器共享资产库」：隔离目录验证 → 用读写 API key `asset_client.py submit` 提交**验证过的产物**（目标 id 的目录，`--note` 写清是什么、怎么检查的）→ 仿真平台管理员的 key 提交、检查没有提示时直接入库，其余在「资产 · 资产入库」审核通过 → 仿真后端约 1 分钟内自己同步（页面再等平台重扫 2–3 分钟），集群上 `r2s-asset-pull` 马上能拉 → 在 `HANDOFF.md` 记下提交编号和修订号（`submit` 打印 `published as revision #n` 或 `waiting for review`）。资产库上线前（看 `HANDOFF.md`）仍是旧的五步：征得同意、备份、拷贝并 `diff -r` 确认。
- 只拷目标 id 的目录。一个脚本会生成整族物体，同族的其他编号不要顺手覆盖。
- 新 id 不会覆盖已有目录；重建已有物体时先备份原目录。
- 装好之后：`ASSETS.md`「共享库」一栏改成「已装」；平台直接只读共享库，`scripts/platform/r2s scan`（或在页面点「同步项目数据」）后就能看到。平台会按 `measurements_mm` 自动核对尺寸，物体页的「尺寸 / 质量核对」应当变成通过。
- 没装共享库、但要让平台先看到：发布 `preview/check.json`（`check_object.py` 写的尺寸、质量），平台据此推进建模和尺寸核对。

## 从公开库批量导入（RSI 的干扰物和实例）

和上面"按平台登记建一件实物"不同，这是成批导入，给 `rsi/AGENTS.md` 的训练池和考场池补资产。不装共享库，新资产单独放一个目录，用 `scripts/datagen/rsi_assets.py link` 合并进运行目录的 `Assets/`。

- **ManiTwin-100K**（`scripts/objects/manitwin.py`，Apache-2.0）：
  1. 标注：在 GPU 服务器上用 `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://hf-mirror.com/datasets/ManiTwin/ManiTwin-100K` 一次拉下全部 json。逐个文件下载会被镜像站限流（HTTP 429）。
  2. `select`：按尺寸筛（最长边 3–30 cm、最短边 ≥ 5 mm），跳过 `_tmp` 类别，类别名改成合法的 USD prim 名；尺寸和任何 RoboDojo 物体相近的不收（RoboDojo 的杂物库就是从 ManiTwin 做的）。
  3. 只下载选中物体的 `base_rescaled.usdz`（并发 4 个）。说明文档里写的 `base_rescale` 是错的，有少数物体没有 usdz。
  4. `convert`：把模型整个引用进 `/World/visual`，关掉它自带的穹顶灯，平移到底面落在 z = 0；碰撞体用减到约 2000 面的网格做凸分解；元数据按实际网格计算。编号从 900 起。
  5. 用 `check_object.py` 做落地检查。它是按"原点在物体中心"写的，判定的 `z` 和"期望值"对不上，这里要按"原点在底面"来读：`|z| ≤ 5 mm`、倾斜 < 1.5°、速度 < 0.01 m/s 才收。
  6. 再在桌面上做一次密集筛查：每个布局放 18–22 个，跑保持姿态的策略；`layout_manager` 会打印 `Unstable actors`，出现过的物体都剔掉。
  - 10-24 的结果：4259 个里选出 765 个，转换 762 个，地面检查通过 450 个，剔掉倾斜超过 1.5° 的以后剩 390 个；桌面筛查里出问题的只有两个轻微倾斜的物体。
- **FMB**（`scripts/objects/fmb.py`，CC-BY-4.0）：在本机 `fetch` 下 STEP、`convert` 转 USD（cadquery-ocp、trimesh、shapely、usd-core，不需要 Isaac），结果在 `Assets/Object/FMB/`（单独一个库，`OBJECT_LIBRARIES` 里有它）。碰撞体按截面精确分解，孔只外扩不收紧；改了分解参数要看 `convert` 打印的体积比和凸包数，并跑 `test_fmb.py`。这个库要进共享库：落地检查通过后按第 10 步提交资产库。
- **铰接物体**（ArtVIP `artvip.py`、PartNet-Mobility `partnet_mobility.py`、RoboCasa `robocasa.py`，共用 `articulated.py`；RoboCasa 的原包是 MJCF，`robocasa.py` 先自己转成 USD 放进 `sources/robocasa/usd/`，桌面柜门 / 抽屉单元由它按固定参数表生成，见 `real2sim/ASSETS.md`「公开铰接物体」）：原包原样进资产库 `sources/<库>/`（上游 README、LICENSE、`UPSTREAM.json` 记 revision 和许可），适配层 `Object/RoboDojo/Articulation/<前缀>_*` 是用相对路径引用原包的文本 `object.usd`，按 `ArticulationObject.fix_root_link()` 的假设整理：根上放 `ArticulationRootAPI`、底座排第一个子节点、底座留 `PhysxArticulationAPI`（`fix_root_link` 从底座复制它）、停用把 link 钉在原始世界坐标的世界固定关节、去掉灯光和 BehaviorScript、根自身变换和单位缩放折进子节点、原点移到包围盒中心。被引用 prim 上方的关节体设置（PartNet 的 `/Object` 上 `enabledSelfCollisions = false`）要带过来；源质量是占位值时整件质量按 link 包围盒体积分，并显式写盒子惯量和质心（PhysX 从 PartNet 的实例化凸包算出过负惯量，导致关节扫动时数值发散）。
  - `metadata.json` 的 `body_joints[]`：`name`（DOF 名）、`type`、`parent`、`child`、`axis`、`limits_deg` / `limits_m`、`drive`，以及 `part` 和 `role`。`part` 是部件（door、drawer、lid、knob、button、tray、rack、screen、cover、pump、cap、handle、pedal、turntable、rotor、pointer、other，按子 link 名或库的部件标签推断）；`role` 取值和分类见上文「铰接物体的关节作用」（平台按它分「能打开 / 只能动」）。「能打开」只指 door、drawer、lid：有铰链的门、能拉出来的容器（空气炸锅炸篮、火柴盒内盒）、真正能合上的翻盖（笔记本电脑屏幕、纸箱翻盖、文件夹封皮）。螺旋瓶盖（`part: cap`）、泵头（`pump`）、烤箱烤架和托盘（`rack` / `tray`）、只能倾转的显示面板（`screen`）、风扇罩（`cover`）一律 `role: other`；某类物体里某个部件确实能打开时由库脚本声明（`artvip.PART_ROLES`：笔记本电脑的 screen 是 lid、文件盘的 tray 是 drawer）。名字和运动对不上时按运动改（`articulated.joint_part`）：会转的 button 是 knob，会滑的 knob 是 button；毫米行程的门 / 抽屉 / 盖是 button；转 ≥ 180° 或不限位的 tray 是 turntable（微波炉玻璃盘）；叫 knob 或认不出（other）、却是盖住物体大半面的薄板且转 ≥ 60° 的，立着是 door、躺着是 lid（ArtVIP 文件夹封皮叫 `E_knob_1`，PartNet 掀盖柜的顶板叫 countertop）；handle 和其他有名字的部件不改。PartNet 的一个 link 由几个零件组成（URDF visual 名），把手、铰链、装饰件（`partnet_mobility.FIXTURES`）跟着门或抽屉一起动，只有 link 里没有别的零件时才按它们命名；玻璃、镜面是门（`PIECE_PARTS`；微波炉门常是「glass + handle」）。`physics.fixed_base`：家电和家具 true，手持物 false（`artvip.FREE_KINDS`）。
  - 检查用 `check_articulation.py`（默认 CPU PhysX，同 `TaskEnv`）：固定底座摆在非原点带 33° 旋转、离地 5 cm，静置 3 s 不漂移不抖，每个关节用驱动斜坡推过上下限位再回位（门 / 翻盖保持原样时没走完的关节，再把同件的门 / 翻盖开到远端限位扫一遍，两次有一次合格即可：烤架、托盘会被关着的门挡住，而有的门起始就是开着的，远端反而是合上），实测范围和标称一致、没卡住、没发散；尺寸在该类真实范围内（`metadata.size_check`）；渲染初始和全开两张图。多件同进程时 RTX 偶尔挂住或返回空帧，批量用每件一个进程；作业里第一件的「全开」图常是空帧（渲染器冷启动），重跑时在前面垫一件不要的。逐件记录 `articulated.py manifest` → `real2sim/articulated/<库>.json`。
  - 入资产库走直连：网关单次上传约 10 分钟就断（11-17 实测 0.2–0.7 MB/s），大包在集群上 `cp` 到 object storage `<user>/asset-import/`，用集群网关签名下载链接（`slurm.call("POST", "/gw/v1/obs/sign-get", ...)`），在 `<asset-server>` 上 `curl` + sha256 核对后 `asset_store.py --root <asset-library-root> --import-tar <tar> --by <名字> --note ...`，仿真后端 30 s 内自动同步。小改动（如只改 `metadata.json`）可以直接 `scp` 到 `<asset-server>` 再 `--import-tar`。OBS 的 key 删不掉对象，用完的临时 tar 用空文件覆盖。见 `real2sim/ASSETS.md`「公开铰接物体」。
- **库里已有、加载器读不了的刚体**（`scripts/objects/place_supports.py`，Place Anything 的支撑物）：SimReady 原包在包目录加 `object.usd` 适配层（同 `simready.py`）；`Clutter` 物体在 `RoboDojo/Rigid/clutter_<类别>/<编号>/` 建只含 `object.usd`（相对引用 `../../../Clutter/.../object.usdz`，不复制网格）、`metadata.json`、`description.json` 的新目录，名称在 `object_names.json` 写成 Clutter id 的别名。两者都按需平放（Clutter 用自己的 `active.place.up`），`passive.support` 写 `check_cup_holds_beans.py` + `place_anything.py pool` 量出的支撑面，`ready` 记落地、支撑面和放置筛选结果。覆盖目录里要能按同样的相对位置看到 `Clutter/`。
- **Poly Haven HDR**（`scripts/scenes/polyhaven_hdris.py`，CC0）：在本机下载（服务器连不上外网），打包传到服务器。macOS 的 `tar` 会带上 `._*` 文件，解压后要删掉。

## 重建已有物体

`spec.json` 的 `tasks` 不为空，或资产库里已经有这个 id 时：

- 改现有生成脚本的常量，id 不变，不另起脚本。
- `build_asset.py` 生成布局时读物体的 `metadata.json`，尺寸变了，布局和 `scripted_demo` 的抓取参数可能都要跟着重跑。这属于任务层，先告诉用户，由用户决定要不要做。

## 常见坑

- 单位：spec 是 mm、g，`metadata.json` 是 m、kg；只有 `measurements_mm` 用 mm。
- `measurements_mm` 的名称和 `dimensions_mm` 不一致，平台就核对不了；外径量进了把手，会判成超差。
- 原点不在包围盒中心：落地检查 z 偏差、布局摆放错位。
- 凸包顶点超限：颗粒（红豆）穿过碰撞体。
- 凸包的点离某个平面不到 0.7 mm（PhysX 烹饪的平面容差）：烹饪时被合并，平底不再平，物体落地后歪着（`airpods` 曾歪 2.4°、下沉 0.7 mm）。凸包的点要么正好在平面上，要么离开 ≥ 1.5 mm（见 `make_airpods.py` 的 `hull_points`）。
- 只写了 `friction`：摩擦不生效。
- 重新生成后没跑 `object_names.py apply`：`description.json` 丢了 `names`，指令退回整句描述。装共享库前 `rg -L '"names"' <目录>/*/description.json` 应当没有输出。
- 质量超过 0.5 kg：运行时被截断。
- 把要入库的东西放进 `Assets/`：`init_assets.sh` 发现目录不完整时会把整个 `Assets/` 挪走。
