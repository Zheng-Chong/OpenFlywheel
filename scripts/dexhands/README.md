# G1-D · Sharpa Wave / Wuji Hand2 仿真模型

本说明保留最初建模阶段的证据；2026-11-03 四套完整机器人的最新有限验证及失败项统一见 [综合核验](../dexhands/FOUR_HAND_VALIDATION.md)。

本目录构建并验证三个**完整 G1-D**：`g1d_sharpa_wave`、`g1d_wuji_hand2_1`、`g1d_wuji_hand2_2`。只运行仿真，不加载设备 SDK、不发现设备、不连接 DDS 或真实相机。提交状态以 Git 记录为准。

## 模型与来源

| 模型 | 每手自由度 | 双臂双手动作维数 | 指尖观测 |
|---|---:|---:|---|
| Wuji Hand2.1 | 20 | 54 | 厂家模型无传感器垫；十路虚拟原生接触力 |
| Wuji Hand2.2 | 20 | 54 | 十个刚体传感器垫；原生接触力，真实逐点标定待设备元数据 |
| Sharpa Wave | 22 | 58 | 十路 240×240 TacMap 深度与厂家编码；接触力独立保存 |

动作顺序固定为 `[左臂7, 左手20/22, 右臂7, 右手20/22]`，单位 rad。腰部 `[yaw,pitch]` rad、升降高度 m 保留独立接口，轮子固定。没有复用平行夹爪的单开度映射或指间耦合。

固定厂家提交：

- [wuji-description](https://github.com/wuji-technology/wuji-description/tree/5c4dfa16917355d073e2e1d7e3d03897b6cf9508)：`5c4dfa1`，MIT。
- [Sharpa 模型](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/0d19cac602f46456b819e4b6a2c09a74982c9a3e)：`0d19cac`，Apache-2.0，含 NOTICE。
- [Sharpa TacMap](https://github.com/sharpa-robotics/sharpa-tacmap/tree/e4f8827407e15625368ef0b9cf5a72c44304e16b)：`e4f8827`，Apache-2.0，含 NOTICE。

交付含三套完整厂家快照及 945 个源文件 SHA256。手部关节轴、原点、上下限、质量、惯量和网格字节保留。Sharpa 选 `with_flange`，不重复加入腕关节或浮动基座。Wuji 标准集成裸手；两版罩壳、手套与安装 CAD 原件仍在 vendor 中，未把手套外形当作柔性物理。

Wuji 2.1 每手 28 links，2.2 每手 33 links，均 20 DoF、配置质量 687.8 g。2.2 新增传感器垫并重新分配远端质量，垫质量为厂家临时配置，不是实测。Sharpa 每手 35 links、22 DoF，左右配置质量约 1.311765/1.309765 kg。完整机器人分别 81/91/95 links。

腕端安装为**仿真候选**：平移 `[0.0415, ±0.003, 0]` m；旋转见各 `model_contract.json`。未证明当前 G1-D 孔位、转接板与实测 TCP。头部双目沿用另一台设备的参考外参；没有沿用被替换的 Dex1 腕相机。

## 2026-10-30 拇指复核与修正

旧版 `20261029.sim.v1` 的 Wuji 两版腕端旋转只对齐了手指伸展轴，漏查掌面和拇指侧，导致默认姿态拇指朝下、正向屈曲朝外。修正为 RPY `[0, -pi/2, 0]`；仅改整手与 G1-D 腕端之间的固定变换，厂家手内部的关节、网格、惯量不改。平移和孔位/TCP 待实测状态保持。旧录像的有限运动通过不能证明腕端方向正确，旧 Wuji 记录只作为历史证据。

`audit_hand_placement.py` 分开检查厂家子树原样保留、根节点相对 FK，以及整机“拇指侧朝上、正向食指屈曲朝内”的仿真安装约定。回归测试含旧错误变换，必须能拒绝它。该约定不等于实物安装标定。

Sharpa 的竖拇指是原始 URDF 全关节零位，不能称为自然张手。默认控制零位保持，另用 `native_validate.py --thumb-review` 逐拇指轴、CMC 屈曲演示、回零录制。CMC 屈曲演示仅给两侧 `thumb_CMC_FE=1.0 rad`，其余手指为零；这是显式仿真目标，不是厂家推荐休息姿态或当前设备零点。预览与 GLB 标明取自哪段录像、哪一帧，不能用换图代替模型修正。

## 控制与碰撞

保留 G1-D 控制、腰部、升降及完整碰撞结构，开启重力和自碰撞。手部有限排除对来源于厂家 USD/MJCF，没有整指禁碰撞。碰撞采用明确的 convex hull。

保留厂家 USD 增益/摩擦/armature；手部与臂部的模拟力矩、速度上限按各 URDF 电机设置。Wuji 展示 USD 对全部电机设置 2 Nm，部分高于 URDF 的 0.2/0.3/0.6 Nm，故不直接沿用该上限。Sharpa 使用厂家建议的 IdealPD 模式；没有声称验证 MIT 控制。无几何查询帧使用厂家 USD 的极小数值质量，避免导入器用 1 kg 回退值；这些值不是实体质量。

## 已执行的验收

Isaac Sim 5.1.0-rc.19、Isaac Lab 包元数据 0.54.3（私有 Kit 配置 2.3.2）、Warp 1.11.0，在 <gpu-server> 的独立目录运行；GPU 0，未改共享库或其他任务。物理步长 1/240 s，记录 10 Hz。

1. 六份厂家 URDF 的树、轴、网格引用、正定惯量与惯量三角关系核对。
2. 完整机器人逐关节有限动作与回零，0.4 rad 屈曲/卸载、独立 FK 和复位；三视角原生录像。Wuji 为 430 帧/43 s，Sharpa 为 470 帧/47 s。
3. 平头、D4 球头、D4 圆柱依次对十个位置压入 0.2/0.5/0.8 mm，切向 ±0.8 mm，卸载恢复；每套 30 帧/3 s。固定夹具不会追随手指位移。
4. 原生接触力及其反力、world↔link 变换、同帧机器人状态、Sharpa 深度与编码核对。
5. 实际 RobotManager 注册/动作队列、54/54/58 维顺序、同一整机绑定、实际电机上限与头部相机配置检查。
6. 六个独立 HDF5 回合读回、源文件摘要、视频全帧解码与帧数检查。

2026-10-30 修正后 Wuji 两版已按相同流程重跑通过，最大稳态误差分别为 0.326° / 0.269°，FK 最大误差约 0.0302 mm；另有 130 帧 Sharpa 拇指复验。精确结果见对应版本交付 `validation/summary.json`。Sharpa 球头深度最大误差约 10.36 μm（阈值 100 μm），厂家编码最大差异 0（阈值 1）。Sharpa 0.15 rad 侧摆出现真实指间碰撞；采用几何核对无穿插的 0.04 rad 窗口，失败记录保留，不能据此声称全工作空间安全。

接触测试不是拾取、搬运和放置任务；登记中的 `full_task_simulation_verified=false`。没有实物成功率或硬件标定结论。

## 触觉标定边界

Sharpa TacMap 是几何穿透深度的射线采样，非胶体 FEM 或光学 RGB。毫米采样文件经 `correction_scale=1e-3` 转米；以实际碰撞凸壳投影并保留掩码。4F 有效采样 37607/57600，TH 为 34470/57600。原始采样图未经覆盖。厂家滤波、分段量化与 9×9 Gaussian blur 不改：前 0.5 mm 为 5 μm/code，之后 30 μm/code，5.15 mm 饱和为 255。

兼容层处理当前 Warp 对小尺寸几何的数值问题：内部毫米查询、边界返回米，并从最小距离重建显示用命中位置。原始球头在米尺度生成时两极三角形被删；测试件改为单位网格生成后缩放，检查闭合性。失败报告和修复前后对照保留，未放宽阈值。

夹具接触材料为演示参数：k=280 N/m、c=20 Ns/m、静/动摩擦 0.6/0.5；不是实测胶体参数，也不是通用抓取接触配置。原生力 N 与深度 m 独立保存，未用未经校准的线性关系把深度换成力。

Wuji 离线解码器 `env/robot_manager/wuji_tactile_contract.py` 不导入 SDK。根据明确的逐模块元数据解析布局、单位、点坐标及局部轴；固件新版的 normalized 数据不冒充 N。只有已知合力应用点时才能形成合力矩。缺少真实模块信息、序列号、零载基线及实测响应，因此不制造触点位置或热图。2.1 的虚拟接触不冒充厂家传感器。

真实后续标定：当前设备孔位与 TCP、关节零位、手眼/相机、质量惯量、伺服、摩擦与滑移、胶体载荷曲线/迟滞、逐 SN 元数据/零点/温度补偿。公开模型无法替代这些实测，本次按用户要求不连接真机。

## 复现与数据

先将交付 `repo_overlay/` 的文件与基线 `2cf5b6d177a82ce54c18dc26cfe8e61d06f47788` 对照合并；`changes.patch` 是已有文件的差异，新文件在 overlay 中。使用私有 Assets 目录，勿覆盖共享库。`install_assets.py --repo /path/to/OpenFlywheel --assets /private/Assets` 只安装三套资产，遇到不同内容的现有目录会拒绝。

平台扫描只发现配置/预览，登记文字需显式导入私有数据库：`python scripts/dexhands/register_local.py --repo /path/to/OpenFlywheel --db /private/platform.sqlite3`。只处理这三项，已有不同登记会拒绝覆盖。私有库仅装机器人，平台通用提示“资产库未找到”指未接入物体库，不影响本次三套机器人。

预览服务：`python scripts/dexhands/review_server.py --directory /delivery --port 8785`，仅绑定本机，支持视频拖动。

在 CPU Python 中需要 numpy/scipy/trimesh/rtree/shapely/h5py/Pillow/matplotlib/pytest；测试工具 ruff 0.11.7。USD 审计使用 Isaac 的 pxr 或 usd-core。GPU 命令在已配置 Isaac Lab 的私有运行环境执行：

```bash
python scripts/dexhands/audit_sources.py --vendor /delivery/vendor --output source_audit.json
python scripts/dexhands/build_models.py --vendor /delivery/vendor --source-manifest /delivery/source_manifest.json --usd-audit /delivery/vendor_usd_audit.json --output /private/Assets/Robots
python scripts/dexhands/native_validate.py --model g1d_wuji_hand2_2 --output /new/native_run --device cuda:0
python scripts/dexhands/tactile_validate.py --model g1d_sharpa_wave --vendor /delivery/vendor --maps /delivery/tactile_cad_maps02 --probes /delivery/contact_probes02.json --output /new/contact_run --device cuda:0
python scripts/dexhands/framework_validate.py --model g1d_sharpa_wave --output framework.json --device cuda:0
```

每次使用新输出目录；不覆盖既有回合。需要时用 `--experience /private/runtime/apps/isaaclab.python.kit` 指定已验证 Kit 配置。当前 5.1 私有运行源对 URDF 导入扩展采用无固定版本名称，并对缺失的 `set_merge_fixed_ignore_inertia` 做兼容保护；勿修改共享安装。

```bash
python scripts/dexhands/export_episode.py --run /accepted/run --asset /private/Assets/Robots/g1d_sharpa_wave --output /new/episode.h5 --maps /delivery/tactile_cad_maps02
python scripts/dexhands/audit_delivery.py /delivery --output audit.json
```

HDF5 `observation/state` 与 `action/position_rad` 采用上述固定关节顺序。`robot/joint_position_si` 另存全部原生关节，`robot/joint_units` 区分 rad 与升降 m；原始 JSON 的历史字段 `joint_position_rad` 包含升降 m，不能全按 rad 解读。触觉包含 N 力、link pose、探头位置/姿态；Sharpa 另存米深度和厂家 uint8。每帧有 frame/time_s，文件带单回合来源与 SHA256。该格式是建模核验回合格式，尚未声称接通训练策略或完成 LeRobot 转换。

GLB 与各套预览图片取自同一段原生录像的明确帧：Wuji 两版为 corrected02 首帧机械零位；Sharpa 为 thumb_review02 第 119 帧 CMC 屈曲示例。默认控制零位仍是全零，不是自然张手标定。物理以 URDF 为准。视频来自真实运行的原生渲染，非生成图像。`evidence/` 的执行快照保留当时文件名；最后 helper 以 `_` 开头避免被平台误列为第四种机器人，控制逻辑保持不变。build04 与旧录像作为历史保留；当前 build05 只修正 Wuji 整手固定安装旋转并补充姿态说明。六个主回合的 URDF 摘要逐一与当前资产核对；额外 Sharpa 拇指回合独立保存。
