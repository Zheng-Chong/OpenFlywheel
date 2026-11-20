# G1-D + Inspire RH56DFTP

本说明保留最初建模阶段的证据；2026-11-03 四套完整机器人的最新有限验证及失败项统一见 [综合核验](../dexhands/FOUR_HAND_VALIDATION.md)。

新增完整桌面 G1-D 本体 `g1d_inspire_rh56dftp`，双侧 RH56DFTP-0L / -0R。保留双臂、腰部、升降、固定底盘、头部双目配置与对外碰撞。全程只用仿真和离线文件，没有连接真机、DDS、真实相机，提交状态以 Git 记录为准。

## 来源与装配

用户附件 `inspire_RH56DFTP.rar` 的 SHA256 为 `fd4eb08c419f5754df87e28d160044670a07bb0338f4dc6f18415fdec5a671ab`。交付保留左右手匹配的 xacro/网格、替代 URDF、两份 STEP、尺寸图和三份触觉手册原件。未执行附件中的 ROS 构建产物或设备程序。附件未附开放许可证，未替厂家补写许可声明。

选配套 `left.xacro` 与 `urdf_right.xacro`，移除独立展示用 world 固定关节，重命名左右手根节点以避免重名，保留手内部关节原点、轴、联动比例、网格字节与惯量。左侧另一个 URDF 的拇指符号/从动范围冲突；`urdf_right_11_5.urdf` 改变了基准和联动且缺匹配网格，故没有混用。

每手 6 个主动、6 个从动关节，13 个实体 links；整机 51 links。左右手模型配置质量分别约 0.403415 / 0.445471 kg，均不是实测质量。四指从动比例 1.05，因此主动上限取 `1.6 / 1.05 = 1.5238095 rad`；原始上下限保存在 `model_contract.json`。右拇指弯曲初始下限为 0.035 rad，不能一律清零。

以 CAD 底部平面圆形包围盒中心对齐 G1-D 腕参考 `[0.0415, ±0.003, 0] m`，手坐标旋转 RPY `[0, pi/2, 0]`：手指伸展朝腕端 +X，拇指侧朝上，正向食指弯曲朝内。每侧实际平移、CAD 锚点、源数据摘要在模型合同内。**G1-D 安装孔位、转接板、实际安装变换和实测 TCP 仍待确认；源 CAD 根节点不冒充实测 TCP。**

## 控制与碰撞

动作顺序 `[左臂7, 左手6, 右臂7, 右手6]`，共 26 维，手部单位 rad。每手顺序为 `[小指, 无名指, 中指, 食指, 拇指弯曲, 拇指旋转]`。从动关节没有独立动作或驱动增益，由十二条原生 PhysX mimic 约束联动。保留腰部与升降的独立接口。

`angle_set_to_rad()` 提供离线 0–1000 仿真端点映射与 -1 保持语义。前五路 1000 为张开；拇指旋转方向参数显式可选，尚未作实际设备零位/端点校准。**这不是厂家角度标定表，不能与方向相反的 POS_SET 0–2000 混用。**

开启重力和自碰撞。只排除两处有几何证据的“掌壳—拇指近节”内部重叠，保留手指之间及与物体的接触。碰撞采用原网格 convex hull；在私有实例取消碰撞网格实例化之后，才应用 0.5 mm contact offset。

保留 G1-D 臂部控制并按其 URDF 限力矩/速度。手部是明确未辨识的仿真预设：主动刚度 3 Nm/rad、阻尼 0.15 Nms/rad、额定配置上限 1 Nm、主动速度上限 2 rad/s，内部目标领先测量角最多 0.05 rad。该限制保护堵转，不代表测得的伺服响应。

已验证默认使用 CPU PhysX、GPU 渲染，480 Hz 物理步、64/8 求解迭代；公开观测频率 30 Hz，核验录像 10 Hz。不要把不同物理后端的未通过中间记录当作验证结果。Isaac Lab 2.3.x 的 mimic 配置参数名称与导入器标志方向不一致，代码做了兼容探测并强制核对十二条约束。导入的 USD 还带 25/.005 的弱软约束预设；必须在强层显式覆盖为零以恢复刚性联动，删除局部属性会重新显露弱层。运行记录验证了实际从动关系。

## 触觉与数据

- 压阻协议：每手 17 分区、1062 点、2124 字节，覆盖寄存器 3000–5123；16 位小端，0–4096 原始 ADC。掌心按列、从下向上传输，解码时转为标准二维矩阵。手指布局的手册排版重复项按配套旧手册的行列说明核对。没有将 ADC 换算成 N。
- 电容协议：每个指尖模块 58 字节，八路原始值、四个派生 float、方向、接近值、内部校验和与保留字段。手册没有给出该内部校验算法与明确派生力单位，因此输出 `checksum_verified=False`、`force_N=None`；不伪造校验通过或物理单位。
- 仿真：每侧五个远端指节与掌心，共十二路原生接触。注册环境的辅助 `obs['tactile_contact']` 输出同一仿真时钟、link 名称/位姿与聚合**法向**力；独立接触实验另外保存过滤接触对的法向力、摩擦力及总力。二者不混为厂家触点阵列。
- 缺少厂家逐点坐标、SN 标定、胶体参数与 FEM 资源，未制造逐点力热图、真实光学图像或 FEM 响应。原始包解码接口不加载设备 SDK。

运动、接触分别导出独立 HDF5。`action/position_rad` 是请求的主动目标，`robot/target_si` 是实际施加的伺服目标；被动关节仍保存在全关节状态中。腰部转动为 rad，两段升降为 m，`robot/joint_units` 是权威单位。视频、原始帧、状态、接触和 link 变换逐帧对应；GLB 用录制的所有关节角回放，包括测得的从动角。

## 已完成与边界

最终采用 `native07`：400 帧 / 40 s，逐主动轴有限动作和回位、半范围、同步机械端点目标和最终回位；最大有限运动/回位稳态差 0.003911 rad，联动差 0.000242 rad，FK 差小于 0.001 mm。`contact08` 为 36 帧 / 3.6 s，十二个位置的球头/平头压入 0.2/0.5/0.8 mm、切向 ±1 mm与卸载，全部接触，恢复后记录力为零，反力核对通过。实际 RobotManager 26 维队列、顺序、限力矩、非法目标拒绝、辅助接触观测和参考相机描述检查通过。

**同步机械端点发生自接触，最大目标差约 0.658 rad，没有计作全范围自由跟踪通过。完整抓取任务、真机安装/关节/相机/动力学标定、真实触觉材料响应均未验证。** 这些状态在平台登记中保持待办，不能用夹具接触或有限运动替代。

整套 CPU 检查：115 passed、1 skipped、82 subtests passed；其中包括本次九项协议/时序边界测试和原三套灵巧手的回归。精确报告、同帧导出/视频解码核对和摘要在交付 `validation/`；中间尝试与错误原因保留，不改写为通过。

## 使用与复现

本包是此前 `G1D_Sharpa_Wuji_20261030_corrected` 的新增本体覆盖目录，完整包含这一套 G1-D 的物理网格；不替换原三套模型。代码基线与已有文件差异见 `BASELINE.json`、`changes.patch`。先在自己的 OpenFlywheel 副本中对照合并 `repo_overlay/`，然后将新本体安装到私有资产目录：

```bash
python scripts/inspire/install_asset.py --repo /path/to/OpenFlywheel --assets /private/Assets
python scripts/inspire/register_local.py --repo /path/to/OpenFlywheel --db /private/platform.sqlite3
```

仓库的 `Assets` 应指向选定私有库。安装器验证每个文件摘要，遇到不同的已有本体会拒绝覆盖。登记脚本只更新本条记录的自动扫描信息，不重启服务、不改其他本体或账号。

在已安装 Isaac Sim 5.1、Python 3.11 的环境中使用包内私有 `runtime/isaaclab_source`（Isaac Lab 元数据 0.54.3）与 Kit 配置；它们不包含 Isaac Sim 二进制。私有兼容改动只涉及缺失导入器选项、可选 TacSL 导入和已安装 URDF 扩展版本，未改共享安装。`PYTHONPATH` 同时包含该目录和 OpenFlywheel 根目录。每次使用新的输出目录，物理配置与实际命令见录制输入及日志：

```bash
python scripts/inspire/audit_model.py --asset Assets/Robots/g1d_inspire_rh56dftp --source /package/vendor/inspire --output /new/audit
python scripts/inspire/native_validate.py --output /new/native --device cpu --experience /package/runtime/apps/isaaclab.python.kit --kit_args '--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false'
python scripts/inspire/contact_validate.py --fixtures /package/validation/contact_fixtures.json --output /new/contact --device cpu --experience /package/runtime/apps/isaaclab.python.kit --kit_args '--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false'
python scripts/inspire/framework_validate.py --output /new/framework.json --device cpu --experience /package/runtime/apps/isaaclab.python.kit --kit_args '--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false'
python scripts/inspire/offline_tactile.py --format piezoresistive --side left --payload /saved/frame.bin --timestamp-s 1.0 --output /new/decoded.json
```

运行时只读取本地包与模拟相机；不设置任何真机地址。480 Hz 内部控制逐步刷新有限伺服目标，外部策略/观测保持 30 Hz。评估时记录请求目标与施加目标，不能丢失限幅语义。CPU 工具依赖 numpy/scipy/trimesh/PyYAML/h5py/Pillow/matplotlib/pytest，几何审计还需其常用依赖；版本清单随包提供。

打开交付 `index.html` 可离线看录像；也可启动本机范围读取服务：

```bash
python scripts/dexhands/review_server.py --directory /package --port 8789
python scripts/inspire/audit_delivery.py /package --output /new/delivery_audit.json
```

本机核验页为 <http://127.0.0.1:8789/>，本体管理为 <http://127.0.0.1:8786/#robots>。端口 8787 属于其他项目，未终止或改动。包里不包含账号数据库、会话或密钥。

当前私有库只安装机器人；平台通用的“资产库未找到”提示指物体库未接入。本体详情中的任务/采集回合为零，不将本次诊断录像冒充任务采集成功。
