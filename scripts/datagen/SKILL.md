---
name: robodojo-datagen
description: 在 RoboDojo 里用脚本专家批量生成 G1-D 仿真示教数据并发布：布局分片、多卡采集（XPolicyLab 标准 HDF5）、官方转换器导出 LeRobot、独立审计、开环回放一致性检查、上传 ModelScope、登记 registry.json；也包括给新任务接采集插件、用本仓库评估外部训练的 checkpoint。用于：采一批数据、续跑采集、导出/发布数据集、检查数据和评估环境是否一致、给新任务写采集插件、评估外部模型。
---

# 仿真数据生成与评估

数据格式和语义以 `scripts/datagen/CONTRACT.md` 为准，本文件只讲怎么做。训练不在本仓库。跑批之前也读 `scripts/cluster/DATAGEN_SKILL.md`（`robodojo-datagen-ops`：单卡并行与显存、提前终止、假完成、版本一致、多机分工、查资产和拉资产、存储），Slurm 用法看 `scripts/cluster/SKILL.md`。

## 各层放在哪

| 层 | 位置 | 入库 |
| :-- | :-- | :-- |
| 任务定义 | `real2sim/tasks/<任务>/registration.json`：数采平台「任务定义」登记的名称、可选的一句话说明和成功判据、用户指的物体（`object_hints`）、指令转写要求（每种语言 N 条，中文 / 英文）；同目录 `videos/` 是真机数采的示例视频（只在平台状态目录，不入库）。用户没写全定义时，coding agent 看视频写 `definition.json`（`name_en`、`definition`、`success`、`roles`、`hand`、`horizon_s`、`notes`），`r2s publish` 到平台，用户确认后合进登记并 `locked: true`；之后改定义要平台管理员解锁并写原因（`lock_history`），coding agent 不改 `registration.json`。（2026-10-31 之前的登记还带候选物体、干扰物、场景和泛化档位。）任务抽屉里有给 coding agent 的实现任务文本。coding agent 从任务定义总结指令并转写，写进同目录的 `instructions.json`（`{"zh": [...], "en": [...]}`），`gen_instruction` 返回这些句子。句子里的物体写成角色标签占位符（如 `<source_cup>`），每回合按句子语言从该物体 `description.json` 的 `names`（大类名称加 3 个从简到详的名称，见 `scripts/objects/SKILL.md`「指令名称」）里取一个填入，两个物体共有的名称不用 | 是 |
| 采集轮次 | `real2sim/collections/<任务>-r<n>/`：平台「数据采集」提交的 `request.json`（本轮的候选物体、干扰物、场景、各维度档位和范围、目标合格回合），或 agent 自己跑的轮次写的 `round.json`（同样的字段）；跑完写 `result.json` | 是 |
| 流水线代码、任务插件 | `scripts/datagen/`、`scripts/datagen/plugins/<任务>.py`；专家动作在 `scripts/g1d/scripted_demo.py` | 是 |
| 数据集登记 | `scripts/datagen/registry.json` | 是 |
| 专家版本 | `scripts/datagen/experts.json`（`experts.py`） | 是 |
| 候选布局、训练分片 | GPU 服务器运行目录的 `Assets/Eval_Layout/RoboDojo/<配置>/<seed>/` | 否 |
| 原始回合 | 运行目录的 `output/datagen/<批次>/raw/seed<N>/` | 否 |
| 数据集 | 运行目录的 `output/datagen/<批次>/lerobot/<名称>/`，以及 ModelScope 私有仓库 | 否 |

服务器、环境和运行目录根路径以 `HANDOFF.md` 为准：采集在 5090 Slurm 集群的作业里跑（`robodojo-sim`，`scripts/cluster/SKILL.md`；原来的采集机 `<gpu-server>` 11-14 清盘后也并入了集群），运行目录在作业里是 `/data/robodojo-runs/<批次>/`，大输出写 OBS；仿真环境是镜像里的 RoboDojo conda 环境。原来在 `<gpu-server>` 上的导出环境 `robodojo-lerobot`（lerobot 0.4.4、h5py）已随清机删除，要用时重新建；上传用 `<asset-server>:/root/ms-venv`（modelscope 1.40.3），`<gpu-server>` 上的 `ms-venv` 归档在 OBS `Employee/contributor/raw/robodojo-runs-<gpu-server>/ms-venv.tar`。

`<gpu-server>`、`<gpu-server>`（各 8 张 96 GB H20）也能跑仿真，`<asset-server>` 没有 GPU。环境从 NAS 备份恢复（`bash /mnt/workspace/robodojo-env/current/restore_env.sh`，见 `scripts/dsw/`），代码在 `/root/RoboDojo`，资产恢复到 `/data/robodojo-assets`。g1d 积木分拣在 H20 上的实测（11-04）：每卡 1 个进程 1.7 帧/秒，8 个 6.6–8.8 帧/秒（GPU 利用率约 90–100%，显存约 35 GB），8 卡 × 8 进程整机约 70 帧/秒、每小时约 230 回合；同一张卡上逐个启动时，第 8 个进程要约 160 秒才走出第一步。

## 采集一批数据

复制这份清单，逐项完成：

```
- [ ] 1. 读 HANDOFF.md 和 CONTRACT.md；git pull --ff-only；开 worktree
- [ ] 2. 准备候选布局
- [ ] 3. 建运行目录
- [ ] 4. 单回合探针
- [ ] 5. 批量采集
- [ ] 6. 导出 + 审计
- [ ] 7. 回放一致性检查
- [ ] 8. 发布、登记、提交
```

1. **开工**：同一个工作区里常有别的会话在改，用自己的 `git worktree` 和短分支。
2. **候选布局**：用任务自己的布局脚本生成到一个新的 seed 目录。量杯任务用 `scripts/g1d/prepare_office_beans_layouts.py`（拉丁超立方采样），其他 G1-D 任务的布局来自 `scripts/g1d/build_asset.py`。候选要比目标多留余量，上一批的合格率是 92%。
3. **运行目录**：代码只传 git 跟踪的文件（`git ls-files`，排除子模块）；`XPolicyLab/`、`third_party/` 软链接到服务器上现成的副本；`Assets/` 用一份覆盖副本，只放这个任务要用的部分。**不要往服务器共享资产库写东西**，新资产走长期资产库（`OpenFlywheel/ASSET_STORE.md`）。Slurm 上资产按需从长期资产库拉，不用整库：要的物体、房间先用 `r2s-asset-pull find --ssh <asset-server> <词>`、`ls --ssh <asset-server> <路径>` 查到路径（集群节点上，不用 API key），布局写好后 `python scripts/datagen/needed_assets.py --env-cfg <env> --task <task> --layouts <布局目录> > needs.txt`，作业里 `export SIM_ASSET_NEEDS=$PWD/needs.txt` 再 `robodojo-sim ...`；缺资产直接报错，不会悄悄跳过布局（`scripts/cluster/SKILL.md`「仿真和渲染」）。
4. **单回合探针**：

   ```bash
   PYTHONPATH=.:XPolicyLab python scripts/datagen/collect.py --task <任务> --env-cfg <配置> --seed <候选 seed> \
       --layout-ids 0 --output output/datagen/probe --device cuda:<卡> --device_id <卡>
   ```

   检查 `episode_000000.json` 里的 `accepted`、`frames`、`joint_names`，再用 XPolicyLab 的 `utils.data_loader.load(path, "RoboDojo", "v1.0")` 读 HDF5，确认图像能解码、state 和 action 的维度正确。
5. **批量采集**（放后台；SSH 断开会把前台进程一起杀掉）：

   ```bash
   nohup setsid python scripts/datagen/run_batch.py --task <任务> --env-cfg <配置> --candidates <候选 seed> \
       --shards 100-107 --gpus 0-7 --target 100 --output output/datagen/<批次> --revision <commit> \
       > output/datagen/<批次>.log 2>&1 &
   ```

   上传过去的运行目录不是 git 检出，要用 `--revision` 传入代码的 commit，否则数据集卡片上写的是 unknown。原始数据每个回合约 100 MB（三路 JPEG），100 条大约 10 GB。进度看 `status.json`，每个 worker 的日志是 `worker_<seed>_<首个布局>.log`。每个 worker 跑一个分片里的 `--chunk` 个布局（默认 10），每张卡默认 `--per-gpu 4` 个进程：单进程时 GPU 大部分时间在等 CPU 上的物理和 Python，4 个进程总吞吐约是 1 个的 2.6 倍；5090 每张卡**最多 4 个**，8 个会把 32 GB 显存用满，进程卡死；96 GB 的 H20 用 `--per-gpu 8`。卡上已有别人的进程时相应减少。同一张卡上的 worker 逐个启动，前一个走出第一步后才起下一个；`--start-timeout`（默认 300 秒）内没走出第一步的 worker 会被杀掉并重排，同一块最多 3 次，次数记在 `status.json` 的 `startup_hangs`。已经不可能合格的回合当场结束（`scripts/datagen/early_abort.py`，目前只有倒豆子 Plus：两杯接触、干扰物被推动、有豆子落在桌面或地上且离两只杯子现在和原来的位置都够远并持续 15 帧），记为不合格，质检单带 `aborted`（原因和帧号），合格回合的录制和判据不变；`--no-early-abort` 关掉，`collection_config.json` 的 `early_abort` 记录开没开，续跑没有这个字段的旧批次时不开。新任务加判据前，先用 `early_abort.py scan` 在已录的回合上重放，合格回合一条都不能被截断。中断以后用 `--output ... --resume` 续跑，已经录好的布局会跳过。合格数不够时，就生成更多候选布局，换一批新的 `--shards` 另起一批，不要复制合格回合来凑数。先 `nvidia-smi` 看清楚哪些卡空着，别人正在用的卡不要占。
   **失败提前结束**（`scripts/datagen/early_abort.py`，默认开）：回合一旦不可能再合格就停录，记为 rejected，sidecar 带 `aborted: {reason, frame}`，进程接着跑下一个布局。只用插件验收永远翻不回来的条件，而且只在这一帧本身不满足验收（`stable == 0`）时才结束，所以合格回合的动作、帧数和录像与关掉时完全相同。目前只对倒豆子 Plus（插件 `g1d_pour_beans_plus`、`g1d_pour_beans_plus_v2`、`g1d_pour_beans_plus_v3`）生效，判据：
   - `touched`：两杯杯身碰过（插件的 `touched` 只会由假变真）；
   - `distractor_moved`：干扰物位移超过 `MOVED_M`（插件记的是至今最大值）；
   - `beans_lost`：某颗豆子连续 `PERSIST`（15）帧在两只杯腔外、低于开局时较低那只杯底 `FLOOR_H`（1.5 cm）以内（桌面或地上），并且离两只杯子当前和开局的中心都超过杯子外接球半径 + `REACH`（3 cm）。脚本里没有东西能把桌上的豆子捡回杯里。

   开关：`collect.py` / `run_batch.py --no-early-abort` 跑到结束；`collection_config.json` 记 `early_abort`，`--resume` 没有这个字段的旧批次时按旧行为跑到结束。给别的任务加判据前，先用 `python scripts/datagen/early_abort.py scan --sidecars '<run>/raw/seed*/episode_*.json' --out <报告>` 在已录回合上重放（读 HDF5 或 rejected 剪成的 `.npz` 的 `privileged/`），合格回合被截断数必须为 0。实测（11-10，倒豆子 Plus r1 / narrow 的已录回合）：合格 335 + 1140 条 0 条被截断，r1 的 356 条不合格里 317 条（89%）会被提前结束，总帧数少 27%；集群 A/B（同 24 个布局各跑一遍，每卡 4 路）每条合格耗时 23.8 → 20.4 分钟（−14%）。注意：被提前结束的回合 `frames` 和 `final_counts` 是停下那一帧的，统计不合格原因时不能当作跑完的结果；合格回合内容不变，但插件或 `collect.py` 的代码 commit 变了，和旧代码采的批次混在一份数据集里时 `cap_runs.py` 的 revision 检查会拒绝。

6. **导出和审计**：

   ```bash
   /root/miniconda3/envs/robodojo-lerobot/bin/python scripts/datagen/export.py --run output/datagen/<批次> --episodes 100 --name <名称>
   python scripts/datagen/audit.py --dataset output/datagen/<批次>/lerobot/<名称>    # 仿真环境的 python 即可，不启动 Isaac
   ```

   `export.py` 选出通过验收、布局互不重复的回合，交给官方转换器，然后逐回合核对长度、指令和首尾帧。`audit.py` 从原始数据重新计算验收结果、运动合理性（`motion.py`：关节速度、加速度、加加速度、离限位距离、可操作度、指垫力）和布局覆盖，最后写出 `final_validation.json` 和 `sha256.json`。任一步失败都不能发布。指令超过真机限值的回合列在 `validation_raw.json` 的 `motion_over_real_limits`；加 `--strict-motion` 时有一条就失败。限速取决于插件的 `actions()` 有没有套 `demo.Limiter`（`scripted_demo.LIMITED` 只管直接跑演示）：积木、拣物、开柜的插件套了，倒豆 Plus 到专家 v5 为止（插件 `g1d_pour_beans_plus`、`_v2`、`_v3`）和量杯倒豆没套，2026-11-18 量过的 real520 和量杯 100 条全部超限（指令速度中位数 3–5 rad/s、夹爪一步开合、离限位 0.025–0.044 rad）；这类回合上真机开环回放前要先限速。倒豆 Plus 从专家 v6（`--plugin g1d_pour_beans_plus_limited`）起限速：每条指令经 `Limiter`，手臂到位后专家才走下一步（专家内部按动作计数推进，只套 `Limiter` 时手臂落后、源杯在会合处撞上还没到位的目标杯）。只剩 LeRobot、没有原始回合的数据集用 `motion.py --lerobot <数据集> --limits <任一质检单>`（要 pyarrow）出同样的指标分布和超限计数。
7. **回放一致性**：每个分片至少挑几个回合，先在采集用的配置里重放（应该和录制结果一致），再在打算评估的配置里重放，看偏差有多大：

   ```bash
   PYTHONPATH=.:XPolicyLab python scripts/datagen/replay.py --raw output/datagen/<批次>/raw/seed100 --layout-ids 0,1,2 \
       --report output/datagen/<批次>/replay_seed100.json [--env-cfg <评估配置>] --device cuda:<卡> --device_id <卡>
   ```

   报告里记录了每个 state 键的最大偏差、`accepted`（插件判据）和 `task_success`（任务判据）。
   - 在采集配置下，state 偏差应该在 1e-3 以内（10-19 实测 ≤ 1e-4 rad）。
   - 颗粒任务的豆子计数会有一两颗的出入，所以 100% 判据偶尔不过，这是正常的。倒的时候豆堆整块滑出的任务（如 `g1d_pour_beans_half_cup`）分叉得多：目标杯里差几十到几百颗，6 回合里 1 回合不再合格（2026-11-01），手臂状态仍在 1e-3 量级。
   - 偏差大，或者计数差很多，说明仿真不可复现，先查原因再发布。
   - 在别的配置下重放失败，说明不能用那个配置评估这份数据训练出的策略。
8. **发布和登记**：用 `upload.py check`、`upload`、`verify` 三步（在上传环境里，先 `modelscope login`；令牌不要出现在命令行或聊天里）。在 `registry.json` 加一条记录（状态、格式、验收、代码、`expert`（`<任务>@v<N>`，取自 `collection_config.json` 的 `expert`）、布局、位置、发布地址、验证结果），更新 `HANDOFF.md`，一起提交。
9. **放进平台「数据质检」**：在有原始回合的机器上把导出的数据集转成平台扫描的回合目录（三路相机 H.264，每 100 回合约 150 MB），再用 `scripts/platform/r2s publish` 发布到平台的回合目录 `eval_result`（在哪台机器上都行：文件经 OBS 进平台状态目录，发布后重新扫描；要仿真平台管理员的写 key 和 Slurm token，见 `AGENTS.md`）；页面左侧先按数据集（平台「数据」→「数据集」，由多个任务的采集数据组成，草稿 → 已发布，改动新建版本），再按采集数据（`_result.json` 的 `dataset`、`round`；同一任务的多轮，如 RSI 各轮，各是一份采集数据）：

   ```bash
   python scripts/datagen/qc_publish.py --dataset output/datagen/<批次>/lerobot/<名称> --out qc_eval_result --round <任务>-r<n>
   scripts/platform/r2s publish qc_eval_result eval_result
   ```

## 给新任务接采集

在 `scripts/datagen/plugins/<任务名>.py` 写一个模块（文件名必须和任务名一致，`collect.py` 按任务名加载），参照 `g1d_pour_beans_measuring_cups.py`：

- 常量：`TASK`；`ACCEPTANCE`（验收版本号，判据一改就升版本）；`HOLD`（验收要连续保持的帧数）；`STEP_LIM`（单回合动作上限）；`TAIL`（专家动作放完之后，最后一个动作再保持多少步）。
- `Episode(env)`：
  - `reject`：开局检查不通过时写原因，否则为 None；
  - `actions()`：专家动作的迭代器，动作字典的键和 `obs["state"]` 相同；
  - `step()`：每执行一个动作后调用一次，记录审计用的数据，返回这一帧是否满足验收；
  - `privileged()`：写进 HDF5 的 `privileged/` 组的数组；
  - `summary()`：写进质检单的字段。
- `audit(privileged, quality)`：只用录下来的数据重新计算验收，不通过时抛出 `AssertionError`，返回审计指标。
- 判据要比任务的 `run_reward` 更严格，并且只用实际几何（参照 `scripts/objects/measuring_cup_shape.py`，任务代码和插件读同一个形状模块）。
- 在 `scripts/datagen/test_datagen.py` 加一个用合成数据跑 `audit()` 的用例。
- 插件和专家动作提交以后，登记成这个任务的专家版本（见下一节）。

## 专家版本

一个任务的专家会按实际不断优化，每次优化是一个新版本，旧版本一直能选来采集。版本登记在 `scripts/datagen/experts.json`，只追加：每一版是一个插件（`plugins/<插件>.py`）加上它经 import 用到的全部 `scripts/` 模块（`scripted_demo.py`、`scripts/objects/...`），按某个 commit 的内容记下每个文件的 sha256，再写一句改了什么。

- 登记：专家代码提交后 `python scripts/datagen/experts.py add --task <任务> [--plugin <插件>] --summary "改了什么"`（默认 HEAD，工作区要干净；`--commit` 登记更早的提交），把 `experts.json` 一起提交。同一插件的代码没变时会拒绝。`list` 列出版本，`status` 看当前代码和每个任务最新版本差在哪些文件。
- 采集：`collect.py` / `run_batch.py` 加 `--expert v<N>`（它决定插件，不用再给 `--plugin`），代码和这一版不一致时直接退出；不加时按插件的最新版本记录，并把不一致的文件写进 `drift`。版本记在每个质检单和 `collection_config.json` 的 `expert`（`task`、`version`、`plugin`、`commit`、`drift`），导出时同一份数据集里必须一致，数据集卡片写出版本。
- 用旧版本采集：在本机检出里 `python scripts/datagen/experts.py materialize --task <任务> --version v<N> --root <运行目录的本地副本>`，把那一版的源文件（从它的 commit 取）写进运行目录，再传到服务器；流水线其余部分（`collect.py`、env、任务）仍是运行目录自己的。共享文件（如 `scripted_demo.py`）被别的任务改过以后，各任务的旧版本也要这样写入。版本的 commit 不在 `main` 历史里（分支被 squash）时打 tag `experts/<任务>-v<N>` 推到 origin，别的 clone 先 `git fetch origin 'refs/tags/experts/*:refs/tags/experts/*'`。
- **`--expert` 只校验、不还原**（11-20 核实）：代码和登记时不一致（哪怕只是 `scripted_demo.py` 里别的任务的改动）就拒绝运行，提示先 `materialize`。11-20 起 `main` 上的 `scripted_demo.py`、`open_anything.py` 被 Open Anything（#355 / #417）改过，Pick v4、Place v5、Put v5 从 `main` 直接 `--expert` 都会被拒；`materialize` 写回登记源码后能跑，sidecar `expert.version` 记 v4 / v5（11-20 集群实测，见 `docs/plan-idle-gpu-filler.md`「按条目指定专家版本」）。
- **filler 条目指定专家版本**（11-20）：`scripts/cluster/deploy/conf/filler-queue.json` 的条目可写 `"expert": "v5"`（`run_batch --expert`；插件由 `experts.json` 定）或 `"plugin": "<插件名>"`（`--plugin`，不校验版本），两者都写时插件必须是那一版的；都不写就是任务的默认插件，和以前一样（正在采的 r2 不受影响）。只对新单元生效，续跑的单元沿用它 `collection_config.json` 里的。**条目的代码包必须正好含那一版的源码**：在本机检出（通常是 `main`）的副本里先 `python scripts/datagen/experts.py materialize --task <任务> --version vN --root <副本>`，再打 tar 传 OBS（或直接从登记的提交打包），否则每个单元都会因 `--expert` 拒绝而失败。打完先在副本里 `python -c "from scripts.datagen import experts; print(experts.record('<任务>', 'vN'))"` 确认 `drift` 为空。新一轮的例子：Pick r7 `"expert": "v4"`，Place r3 / Put r3 `"expert": "v5"`；改完 `deploy.sh tools` 部署队列。
- 平台：任务详情「专家版本」列出每一版和用它采的数据（registry 的 `expert`）、轮次；「提交一轮采集」选版本，写进 `request.json` 的 `expert`。采数据不再人工质检：合格按仿真的严格判据（插件验收），平台「数据集」的看板只看分布和视频，判据之外发现问题时标记，被标记的回合不算合格。

### 共用抓取：Pick 专家 v4（`GraspRobot`）

Pick / Place / Put / Sort / Handover Anything 共用 `scripted_demo.TopGripRobot.pick` 的顶抓（Pick 专家 v3）。v4 是可选的新版本 `GraspRobot.pick`，同签名、同返回值：第一次尝试就是 v3 的抓法（原样），之后按离线 URDF IK 可达性挑翻腕、换夹持边、换俯角、细长物体深握、靠中线时换手；每次尝试检查悬停/抓取位到位、目标没被推动、合爪没合空、试抬 3 cm 目标跟着起来，否则回悬停位按目标新位置重来，最多 3 次；Pick 保持段掉了再抓一次。设计、v3 失败分析和对照数据在 `docs/pick-expert-v4.md`，离线分析工具 `scripts/datagen/pick_failures.py`（`scan` / `report` / `frames` / `compare`）。

- 状态（11-20）：**Pick v4 已通过（Pick r7 起 `--expert v4`）；Place / Put 放置专家 v4.3 已通过，从下一轮起用 `--expert v5`**（插件 `g1d_place_anything_v43` / `g1d_put_anything_in_v43`，`scripts/g1d/place_v43.py`：第一次抓取前在 URDF 上回放仿真的 IK，选整条路线到得了抓取点和下放点的抓法，Put 的高细物体先握中部；放置段 `place_v42`：IK 按 `Limiter` 截断线停关节、按 X 实际位置下放、掉出手重抓）；Place v4 / Put v4（v4.1 抓取）仍可用；`*_v4`（第五轮）和 `*_v42` 插件不要用。sidecar 多 `place_route`、`place_log`（`hand_short_mm`）。Sort、Handover 要用时先在自己的任务上做同布局 A/B。放置段失败分析：`scripts/datagen/place_failures.py`。
- 不要替已经在采的集合换专家：Pick r6、Place/Put r2、Open Anything 继续用各自登记的版本；v3 的代码路径没变（`DEMOS` 仍指向 v3），但 `scripted_demo.py` 变了，用 `--expert v3` 采时按上面「用旧版本采集」写入。

## 上游榜单任务（dual X5）的随机布局

上游引擎只回放已保存的布局，没有生成器。`scripts/datagen/x5_layouts.py` 以任务的官方评测布局（`arx_x5/0`）为结构模板，按随机方案重抽每个布局：

- 随机方案在数采平台任务抽屉的「泛化随机方案」里填，写 `real2sim/benchmark_tasks/<任务>/randomization.json`（格式和校验见 `scripts/datagen/rsi_profile.py`），入库。方案要选池子：`train`（数据方）或 `exam`（评测方）。
- 资产分池（`rsi/AGENTS.md` 规则 1）：`python scripts/datagen/rsi_assets.py scan --assets Assets` 写 `rsi/assets/pools.json`（仿真环境的 python，要 PyYAML）。官方评测用到的资产，加上没有官方数据的任务配置里能抽到的编号，都算榜单保留资产，谁都不能用；官方训练见过的两方都能用；其余按 7:3 分给训练池和考场池。官方资产变了、或加了新资产以后要重新扫描并入库。生成器和平台只列出所选池子的资产，写布局前会再检查一遍；裁判用 `rsi_assets.py audit --assets Assets --layouts <目录> --pool <池>` 复查。
- 干扰物来自池子：没用过的杂物，加上不属于任何任务类别的、没用过的物体（写成不带标签的普通 `Rigid` 条目）。新实例（任务类别里没用过的编号）会加进换实例的候选。严格口径下这两样现成的都只剩几个，要靠新资产补。
- 换色、等比缩放只对 `x5_layouts.RESTYLE` 白名单里的任务和物体生效：纯色会盖掉物体原来的贴图，缩放会改变套碗、按大小排序、插槽类任务的答案。缩放后物体仍落在桌面上，质量按体积换算。
- 可选项：位置和朝向（只在任务配置自己的范围里重采样，从不照抄评测布局；配置里写死位置的物体不动）、实例（只抽基础任务的训练侧编号，`_random` 独有的编号是榜单留出实例，永远不抽）、干扰物（杂物库，类别不重复）、房间池、桌面和地面 MDL 材质池、背景 HDR 和亮度（整个舞台共用一个穹顶，取第一个并行环境的布局）、额外灯光（类型、数量、亮度、色温）、头部相机角度扰动。
- 相机只扰动角度：平移 2 cm 时 10 个并行环境里有 3 个出图全黑（10-23 在 <gpu-server> 上对照过：只去掉平移就恢复正常）。开了相机扰动会生成 git 忽略的 `env_cfg/arx_x5_rsi_<任务>_cam<角度>.yml` 和 `env_cfg/camera/rsi_<任务>_cam<角度>.yml`（每个角度一份，同一任务的不同级别不会互相覆盖），布局放在这个 env 配置下。
- 命令（隔离目录、仿真环境的 python）：`python scripts/datagen/x5_layouts.py --task <任务> --seed <N> --count 100 [--split val]`。训练池的方案只能写 seed 1000–4999，考场池只能写 5000–9999，seed 0 是榜单评测集。输出目录带 `split.json`（默认 train，评测客户端会拒绝）和 `report_<任务>.json`（每个布局抽到了什么）。
- RSI 一轮的批量出卷 / 出数据：`scripts/datagen/rsi_round.py`。`plan` 按裁判的 `levels_<r>.json` 和预算分配考卷（每个维度前沿 60%、上一级 25%、每个任务的 L0 锚点 15%，各任务族、族内各任务平均；能力缺口任务固定 L0 一组），`generate` 按任意 plan 合成方案、调用 `x5_layouts.py` 生成每个科目一个 seed 目录（写在 plan 目录的 `layouts/` 下，链接进 `Assets/` 覆盖目录）并写 `manifest.json`（每个布局一条），`hold-folder` 把全部布局按任务链接进一个检查目录，用 `--policy_module scripts.datagen.rsi_hold` 跑保持姿态，`check` 汇总资产审计、回合配比和保持姿态结果写 `g1.json`。不稳定的布局用 `x5_layouts.py --redraw <编号>` 原地重抽，再跑一遍 `generate` 更新 manifest。本方的 `Assets/` 覆盖目录用 `rsi_assets.py link --base <完整资产> --out <运行目录>/Assets --pool <train|exam>`：只链接官方训练见过的资产和本池，官方布局只留模板 `arx_x5/0`。
- 生成后要在 Isaac 里看一遍画面再用：用一个只保持姿态的策略模块（`--policy_module`）跑 10 个布局，检查头部相机最后一帧没有大面积黑块、亮度不过曝。
- RSI 每轮数据的发布（`rsi/AGENTS.md` 五之二）：`scripts/datagen/rsi_release.py --run <rsi 运行根目录> --round <N>`，在带 `XPolicyLab/` 的代码目录里按步骤跑。
  - 转换器必须用 XPolicyLab fork 分支 `contributor/openwam-robodojo-lerobot`（[PR #217](https://github.com/XPolicyLab/XPolicyLab/pull/217)）的 `scripts/transform_lerobot_v30_format.py`：LeRobot 0.4.4 的流式编码器队列满 0.1 秒就丢帧、只打警告，之后的帧全部错位。发布目录里放这个分支的 `scripts/`、`utils/`、`policy/OpenWAM/OpenWAM`，不要改共享的 XPolicyLab。
  - `export`（导出环境的 python，`--shards 32` 并行转换后用 LeRobot 的 `aggregate_datasets` 合并，校验也按分片并行）：先核对 `submission.json` 和裁判 `audit.json`（`--audit` 可指定）的批次、回合数、帧数，再把审计过的回合和 `rejected/` 里有录像的回合各转成一个 LeRobot v3；审计判为超长的回合（`audit.json` 的 `too_long`，v0.12）放进 `_rejected`，在 `rsi_episodes.json` 里带 `referee_reject: too_long` 和 `reject_reason`。逐回合核对：长度、指令、state / action 逐位相等、三路相机视频帧数 = 行数 = HDF5 长度、第 0、n/4、n/2、3n/4、n-1 帧和 HDF5 的 JPEG 比。写 `rsi_episodes.json`（每回合的批次、组合、布局 seed / 编号、HDF5 sha256、质检单）、`final_validation.json`（含转换器 sha256）、`sha256.json`。临时视频写在 `<output>/.tmp`（根盘满了会静默丢回合，10-31 遇到过）。转换器对单个回合出错只打印、继续，回合数对不上时 export 会失败，看日志里的 `Failed files`。
  - 在 HDF5 删除前跑 `scripts/datagen/rsi_lerobot_check.py --dataset <数据集> --stats <M_0>/normalization_stats.npy`（`PYTHONPATH=.`），核对 OpenWAM 两种读取器读出的一致。
  - `upload`（装了 modelscope 和 numpy 的 python；私有仓、逐文件核对、未登录访问必须 404，写 `<名称>.release.json`）；10-31 `<gpu-server>` 上保存的 `Vincia` 令牌已不能建仓，用 `<gpu-server>` 上的 `contributor`。`registry`（本机仓库里把 release.json 合进 `registry.json`）；`delete`（不带 `--delete-hdf5 --confirm <通过仓名>` 只列出要删的 HDF5；裁判下令前不删）。
  - 拷到训练用的 OSS：在 DSW 上跑 `scripts/datagen/rsi_oss_copy.sh <5090 的 SSH 端口> <数据集目录> /mnt/data/contributor/rsi-dlc/<运行名>/lerobot/round<NN>[_rejected]`。公网入口对每个连接限速，脚本用多路 tar 拉到本地盘、再单流写进 ossfs（不支持并发写和 truncate，rsync `--inplace` 会失败），最后按 `sha256.json` 核对。入口总带宽 10-31 只有 15–30 MB/s，ModelScope 上传和它抢出口，先拷 OSS。
  - 规模和耗时（10-31）：AV1 约每帧 75 KB（第 1 轮通过 65 GB），32 分片转换第 1 轮约 30 分钟、第 2 轮约 50 分钟。

## 数据采集轮次

一个任务的数据按轮次采，每轮一个编号 `<任务>-r<n>`（n 从 1 递增）。任务定义不随轮次变；每轮自己定候选物体、干扰物、场景和泛化档位（训练只到 L2）。

- **平台提交的轮次**：输入是 `real2sim/collections/<编号>/request.json`，抽屉里有给 coding agent 的采集任务文本。按上面「采集一批数据」做，seed 段不要和评测布局、测试基准用到的 seed 重叠；过程中在平台改状态（生成布局、采集中、导出与审计、已完成 / 失败）。
- **agent 自己跑的轮次**（RSI 的轮次由裁判在审计后同步，而且只发轮次、来源、目标和合格数，见 `rsi/AGENTS.md` 5.3）：开跑时在运行目录写 `real2sim/collections/<编号>/round.json`（`task`、`source`: `rsi` / `expert` / `teleop` / `other`、`target`、`status`、`roles`、`distractors`、`scenes`、`generalization`、`notes`，字段同 `request.json`），用 `r2s publish` 发布，平台就会列出这一轮；跑完再发布 `result.json`。这种轮次在平台上只读，改动以文件为准，删除也在仓库里删。
- `result.json`：`{"collected_at", "summary", "attempts", "accepted", "expert_success_rate", "dataset"（registry.json 里的名称）, "location", "layouts": {"<env_cfg>": [seed, ...]}, "issues"}`。平台按 `accepted` 累计任务的合格回合。两个文件都要提交入库。

## 用本仓库评估外部训练的 checkpoint

- 模型评测只能按「测试基准」跑和打分：基准（平台数据库，`<slug>@v<n>`）是一组任务 × env 配置 × 布局 seed，加上创建时定下的泛化（`generalization`：空间关系、场景视觉、物体属性、视角成像、语言对齐五个维度各 L0–L3，和必填的说明；启用后和条目一样不改，改动新建版本；之前建的基准由管理员补写一次）。提交评测时只选基准，不另选任务或泛化，`request.json` 的 `benchmark_jobs` 就是要跑的条目，`benchmark_generalization` 是布局已有的泛化（评测时不另加随机），`result.json` 的 `runs` 每个条目一条（task、env_cfg 和条目一致，带 seed），平台按回合数加权和按条目平均两种算法出成绩。
- 模型先在数采平台「评测 · 模型管理」登记（平台数据库，kind `model`，`models.py`）：登记只填显示名称、模型 ID 和模型介绍，版本登记后在模型详情里发。每个模型有只追加的版本列表：内部序号 `version`（1、2、…，评测的 `model_version` 指它），提交人自己定的版本号 `name`（如 `1.1`、`1.1.0`、`1.1.0.post`，同一模型内不重复；早先的版本没有，显示为 `v<序号>`），可选的版本标题 `label`（如 `step3000 · 夹爪 0.96`，成功率折线图和版本下拉跟在版本号后面），较上一版本的差异（第一个版本除外）、推理代码（代码仓 + 分支或 commit，或 DSW 上的路径）、权重和训练数据。权重填 DSW 上的路径：`/mnt/data/<路径>` 是 PAI OSS 的 ossfs 挂载（就是 `<oss-bucket>/<路径>`），`/mnt/workspace/<路径>` 是 NAS；不再收 `oss://`（早先的版本有）。`<gpu-server>` 磁盘有限，也没有 OSS 工具和凭据，评测时从 `<gpu-server>` 读、经本机中转只拷进这次评测的运行目录，评测完删掉。训练数据从「数据集」里选一个（`train_dataset`），平台上没有的写明「平台外的数据」（`train_external`），二选一必填；提交评测时按它预选。模型登记之前的评测，仿真平台管理员用 `POST /api/model-version` 带 `imported_from: <评测编号>` 导入成版本（权重必须等于那条评测的 `checkpoint`，可以是服务器路径），再把评测的 `reg` 改成 `{"model", "model_version"}` 挂上；挂上时申请的其他内容不变。
- 评测申请走数采平台「模型评测」：提交人选已登记的模型和版本（服务器把这个版本的 `model_name`、`version`、`changes`、`code`、`checkpoint`（即权重路径）抄进申请；`version` 是版本号，之后改模型不影响这条评测；脚本专家基线 `framework: scripted_demo` 不用登记模型），再从「数据集」里选训练数据（服务器把数据集的组成抄进 `train_dataset_parts`；本体、动作、相机、指令按这些数据对齐，不另填接口和推理环境，推理环境写在版本的推理代码里；页面按版本登记的训练数据预选（早先只写了采集数据名的版本，选含这份采集数据、份数最少的数据集）；真机或 OSS 上平台没有的数据选「平台外的数据」，写明在哪、什么格式，存为 `train_external`），选一个已启用的测试基准和每个布局的重复次数，写到 `real2sim/evals/<编号>/request.json`。没有分派和评测负责人。版本填了推理规格（`serve_command` 含 `{ckpt}` `{port}`，按 `scripts/g1d/remote_policy.py` 的 openpi WebSocket 协议服务；可选 `serve_image`、`serve_parallel`、`client_args`，见 `src/real2sim_platform/models.py`）且基准任务都有数据插件时，提交人在评测详情点「在集群上运行」：登录机上的 `robodojo-evald` 以提交人的集群账号搬权重、提交 `scripts/platform/eval_run.py`、回报进度并把回合和 `result.json` 发布回平台（`scripts/cluster/OPS_SKILL.md`「平台评测」）；先勾「冒烟」跑每个条目 1 个布局。否则评测抽屉里的「测试 Prompt」由提交人交给自己电脑上的 coding agent 运行：在集群登录机 `<gpu-server>` 上用 `robodojo-sim` 跑仿真，DSW 上的权重用 `rjob push` 搬到 `/obs/Employee/<你>/`；过程中提交人在「进度」里改状态。策略名用评测编号；跑完用 `scripts/platform/r2s publish <运行目录>/eval_result eval_result` 发布到平台扫描的回合目录，并写 `real2sim/evals/<编号>/result.json`，同样用 `scripts/platform/r2s publish` 发布到平台（`summary`、`runs`：每个任务 × env 配置 × 等级的回合数、成功率、95% 区间、专家对照，`issues`），平台据此显示结果。凭据不要写进这两个文件。
- 策略适配代码属于 XPolicyLab（子模块，不在本仓库里改）。LeRobot 格式的策略（ACT、SmolVLA、Pi0 等）读的是官方转换器的键名，只有按本契约导出的数据集能直接用。
- G1-D 目前只注册在本仓库的 `env_cfg/robot/_robot_info.json` 里。XPolicyLab 的 `utils/robot/_robot_info.json` 还没有 `g1d*`，走 XPolicyLab 的 `train.sh` 会拿不到动作维度，需要给 XPolicyLab 提 PR。
- 命令：`bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/<策略> --task <任务> --ckpt <名称> --policy-env <环境> --env-cfg <数据集的配置> --action-type joint --eval-num <N>`。
  - 只用没有 train 标记的 seed 目录（评估客户端会拒绝带 train 标记的目录）。
  - 结果里同时报两种成功率：`success_rate` 是任务判据；严格判据可以对评估视频按插件重算，或者在同一组布局上和专家的回放结果对比。
  - 仿真每次运行有随机性，同一个布局至少重复跑 3 次，成功率要附上区间。
  - 换仿真后端加 `--sim-backend mujoco|isaaclab_newton`（默认 `isaacsim`；也可设 `ROBODOJO_SIM_BACKEND`）。`mujoco` 不需要 Isaac，CPU 上就能跑 G1-D 刚体任务；`isaaclab_newton` 要 Isaac Sim 6 / IsaacLab 3 环境。环境和限制见 `env/backends/README.md`；成绩只和同一后端比较。

## 调专家

专家改动要能证明有效，不能只看一批回合的成功率。倒豆子 Plus 的调优记录、已验证的瓶颈和下一步见 `scripts/datagen/summary.md`，其中可推广的做法：

- 先量手的加速度和关节指令的单步变化，再改逻辑：失败多发生在指令跳变处（交接重算、IK 在限位或奇异附近、动作起止、反馈一步到位），改成限加速度、平滑跟随。
- 比较两版只用同布局配对 A/B，并用旧版重跑一部分布局，量出随机翻转率；布局里要有失败的，也要有好的对照。同布局重跑也可能在改动生效之前就分叉（抓点差几毫米、开始倾倒差几十帧）：单个布局翻转时，先比两版日志里改动前的打印，结论看几十个布局的合计。
- 先把失败按原因分类（例如丢豆的出口：杯口、穿壁、剩在源杯），只调专家改得动的那部分；属于物理或资产的瓶颈记下来，不在专家上反复试。
- 按物体选抓法时，只用和抓法有关的结果打分；每格只考几次的清单偏乐观，要用每格 5 次以上的确认考试证实。
- 泛化要单独考：`pour_beans_plus_layouts.py gen` 随机配对杯子（两种角色、缩放、豆量都随机），`--min-ratio` 出“大口倒小口”的困难集；结果按难度维度（口径比、豆子数）分组看，平均数会把难的组淹掉。
- 用录下的回合离线标定触发器，只能看到开环的误报。触发器一旦改变流量，就可能自己制造下一次触发（V4：杯沿上不动的豆子被预测成洒落，每次回转让豆子渗得更慢、倒不完），所以离线标定之后还要做同布局 A/B 看闭环。
- 提前反应总会有误报，每次误报都要花时间。给谨慎设一个时间预算：按剩余步数画一条进度线，落后时退回只对已经发生的事件反应。

## 常见坑

- 同一台机器上多路仿真并行时，每个进程只让它看见自己的卡：`CUDA_VISIBLE_DEVICES=N` 加 `--device cuda:0 --device_id 0`（`run_batch.py` 已经这样起 worker）。`src/eval_client/main.py` 总是关掉多卡渲染，只看见一张卡时再补上 `--/renderer/activeGpu=N --/physics/cudaDevice=0`：Vulkan 不受 `CUDA_VISIBLE_DEVICES` 限制，不指定时渲染默认在 0 号卡上（10-30 在 <gpu-server> 上 6 路同时跑，0 号卡被占到 32 GB，报 `vkAllocateMemory failed`，进程卡死不退出）。所有卡都可见、用 `--device cuda:N` 也能在 N 号卡上渲染，但每个进程会在其余每张卡上各留约 250 MB 的 CUDA 上下文，8 卡各跑 4 路时每张卡要多背约 7 GB（11-02 实测）。
- H20 上同一张卡同时启动多个仿真进程，约一半会永久卡在第一次取相机画面（`ObsManager.get_obs` → `CameraView.get_data` 里的 warp launch）：GPU 利用率 0、进程 CPU 100%、日志刷 `zenity: not found`，不响应 SIGTERM，只能 `kill -9`（11-04 在 `<gpu-server>` 上 8 卡 × 8 路每秒起一个，64 个卡死 35 个）。`run_batch.py` 已经逐个启动并超时重启；自己写脚本并行起 `collect.py` 或评估客户端时，也要等前一个日志出现 `env0 step:` 再起同一张卡上的下一个。根因在 Isaac Sim 内部，没有查清。
- 在 ssh 命令里 `pkill -f <模式>` 会匹配到这条 ssh 命令自己的 shell，把连接一起杀掉；模式要锚定，如 `pkill -9 -f "^python -u scripts/datagen/collect.py"`。
- G1-D 的 env 配置（`g1d_aurora_lite_ws` 除外）用 `observation.vision.cameras` 只渲染数据集和策略用的 3 路相机，第三人称 `cam_third` 和双目右眼 `cam_head_right` 不创建，单进程每步快约 7%。要看第三人称视频时，在运行目录里临时去掉这一行。
- 每个 worker 的 `ROBODOJO_RUN_ID` 要唯一（`run_batch.py` 已经处理了）。否则同一秒启动的评测会写进同一个 `eval_result` 时间戳目录，互相删掉对方的临时视频。
- 一个采集进程最多跑任务的 `eval_nums` 个布局（默认 50），剩下的由 `run_batch.py` 自动续跑。
- 官方转换器会把输入目录里的所有 `.hdf5` 都转换进去，并且会先删掉同名的输出目录。所以只能通过 `export.py` 调用它：它只暂存选中的回合，输出目录已存在时直接拒绝。
- 官方转换器按自己 `__file__` 解析后的路径来定 `data/` 和 `env_cfg/` 的根目录。服务器运行目录里的 `XPolicyLab` 通常是软链接，解析后会指到另一个检出，结果去那边找数据（报 No matching targets）、读那边的 `env_cfg`。`export.py` 已经把这两个根目录固定为本仓库。
- 服务器访问 PyPI 要走镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...`。DSW 上用阿里云内网镜像 `https://mirrors.cloud.aliyuncs.com/pypi/simple/`（约 20 MB/s）；Isaac Sim 的 wheel 只在 `pypi.nvidia.cn`，单连接 1–2 MB/s，NAS 备份里已有 `wheels/`，别再重下。
- 颗粒任务用 `g1d_beans_collect`（1/600 s）。
- 向服务器同步代码时，`rsync` 要加 `--checksum`：只改了数字时，文件大小和修改时间可能都没变，会被漏传。
- 服务器上的 `ffmpeg` 如果不带 libx264，录评估视频时会报 BrokenPipe；仿真环境的 `bin` 里要链接系统的 `ffmpeg`。
