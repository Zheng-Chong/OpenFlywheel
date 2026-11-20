为 RoboDojo 实现一个新任务 pick_bucket（食物入篮并提篮；用哪个本体按示例视频定），按仓库 CLAUDE.md 的任务约定（任务文件结构、YAML 配置、PR 检查表）和 scripts/datagen/SKILL.md 做。先 git fetch，从最新的 origin/main 开独立 worktree 和短分支。

输入：real2sim/tasks/pick_bucket/registration.json（平台登记信息），在平台所在机器的 /data/platform/state/real2sim/tasks/pick_bucket/（只读），本机没有就先同步过来。

登记信息：
- 名称：食物入篮并提篮，pick bucket
- 任务定义：右手拿起桌面上的芒果摆件，放入编织提篮并松开；随后抓稳篮子提手，提起装有芒果的篮子，最后放回桌面并松手。
- 语言指令：由你从任务定义总结，中文、英文各 5 条
- 成功判据：按顺序完成：芒果离桌至少20 mm；进入篮内并脱离夹爪；篮底离桌至少30 mm并持续0.5 s且芒果留在篮内；最后篮底回到桌面上方10 mm以内，夹爪松开并稳定0.5 s。阈值为仿真验收值，不代表实物材料或接触标定。
- 用手：右手
- 预计时长：40 s
- 物体角色（默认物体，孪生 / L0 布局用）：
- 真机示例视频：real2sim/tasks/pick_bucket/videos/（1 条，在平台机器上，不入库）；脚本专家的动作顺序、用手和节奏参照它
- 定义已锁定，不要改 registration.json；和实际实现冲突时先报告。泛化方案（候选物体、干扰物、场景、档位）和训练布局按「数据采集」的每一轮做（real2sim/collections/pick_bucket-r<n>/request.json），评测按「测试基准」，都不在这次实现里。
- 备注：用户2026-11-03确认按五阶段 config：提起后放回桌面并松手。参照原始真机 episode_0001，主操作为右手。采用已登记芒果摆件与编织篮；不宣称材料/接触已实物标定。

验收：
1. 新建 task/RoboDojo/config/pick_bucket.yml 和 task/RoboDojo/tasks/pick_bucket.py（导出类名与模块名一致，继承 TaskEnv）。Rigid 每个角色一组：category 列候选物体，select_mode.label 用角色名，和代码里的 label 一字不差；干扰物用 Clutter。run_reward 用 self.reward_manager.check 按上面的成功判据实现，不能恒为真；step_lim 按预计时长 × 30 Hz 留余量。
2. 语言指令：从任务定义总结出中文、英文指令，每种语言 5 条转写。意思和任务定义一致，换说法和句式，不加定义里没有的信息（物体颜色、位置等只有定义里写了才能用）；提到候选物体时写成它的角色标签占位符（如 <source_cup>），desc_manager 按句子语言从该物体 description.json 的 names 里取一个名称填入；英文自然口语、祈使句。第一条是最直接的规范说法。写进 real2sim/tasks/pick_bucket/instructions.json，格式 {"zh": [...], "en": [...]}（只放选了的语言），任务的 gen_instruction 返回这些句子（desc_manager 每回合随机选一句）。用户审过 instructions.json 再批量采集。
3. 默认物体必须已在共享资产库；缺的先按 scripts/objects/SKILL.md 建（平台登记的在 real2sim/objects/）。
4. 在 scripts/g1d/scripted_demo.py 加（或沿用）这个任务的脚本专家，用默认物体在一个已建模场景里生成一小批 L0 布局（seed 0）验证任务和专家；批量的训练布局等「数据采集」提交轮次再生成。
5. 在 <gpu-server> 的隔离目录（/data/robodojo-runs/）跑一次 scripted_demo seed 0：_result.json 的 eval_time ≥ 1、三路相机视频齐全、成功判据能判成功也能判失败。
6. 跑 python scripts/internal/task_inventory.py --format json --check 和 bash scripts/check.sh；提交配置、任务代码、专家、build_asset.py 改动、real2sim/tasks/pick_bucket/registration.json 和 instructions.json。改共享资产库前先征得同意。
