# 木质拱形积木 · 实测重建报告

资产：`Rigid/wooden_arch_block/00000`。登记/平台署名：杨开元；执行：Codex。日期：2026-10-29（Asia/Shanghai）。
状态：研究候选；刚体数值复用通过，GPU夹持接口失败/机器人抓取与完整外观未验；未替换共享库。

## 当前输入与确认

最初版本 `da62c0f282627330eabfb1bc813285c9ca66ea15edd71083e36ce7a96c26c040` 与线上记录及照片一致，
但尺寸字段“实测”与旧备注“尚未实测”矛盾。本次用户明确答复：
“按这些实测尺寸重建；9.58 mm 是拱顶厚度”。据此定向更新当前说明，旧说明和旧估计保留历史，
以该用户确认重新生成正式 v3 任务，当前版本为 `ff33fb65e47df98cac53a62e1c55494a3c78ab537595ae5c8748367a95bc6e75`。
生成器重验正式prompt完整输入摘要、spec字段、照片SHA；没有继续用旧prompt生成。
旧尺寸60×30×40mm、估计25g不再作本次目标，旧模型、参数和历史原生报告没有重写。
尺寸确认回执在 `registration_reconfirmation_20261029.json`，各参数依据在 `parameters_measured_20261029.json`。

| 同名测量项 | 实测 mm | 网格提取 mm | 误差 mm | 定义 |
| :-- | --: | --: | --: | :-- |
| 长度 | 60.10 | 60.10 | 0.00 | X方向完整外包围盒 |
| 宽度 | 29.72 | 29.72 | 0.00 | Y方向完整外包围盒/拱洞深度 |
| 高度 | 30.04 | 30.04 | 0.00 | Z方向完整外包围盒 |
| 最薄处 | 9.58 | 9.58 | 0.00 | 中截面拱顶到平顶面的厚度 |

本体质量实测24.6g，原生工厂读回 24.599997g。
单位m/kg，Z向上，原点为全部显示网格包围盒中心。保留既有UUID，未改ID。
惯量与质心按均匀密度的完整拱形实体计算并缩放到实测总质量，均经原生读回，正定；
惯量、密度分布与木材种类没有实测。摩擦未知，不在metadata伪填实测值，使用项目默认0.6/1.5。

## 几何、贴图与未覆盖区域

矩形外轮廓减去贯通Y方向的半圆拱洞，半径由高度减冠厚推得20.46mm；
半圆与贯通结构依据照片和登记特征，未将额外孔径冒充直接实测。
外缘细小倒圆没有尺寸依据，未精细重建。显示网格单独置于 `/root/visual/model`。
碰撞为两个支墩与32片拱顶条，每片凸包8顶点，拱洞开放；最大弦逼近误差约0.025mm，
不是用整体凸包封洞。底面接触区为两只支墩，未把空洞中心当作支撑圆。

木纹来自实际照片 `reference/original_20261024.jpg`（632×494px），不是程序木纹。
先在源图标注正面、顶面、左端及可见内拱局部范围，再分别透视校正、去除宽尺度明暗，
背景边界按邻近实拍纹理修补；4张PNG纹理打包进USDZ，用材质子集覆盖各自的实体面。
文件输出按约10px/mm采样，但原照片有效细节较低，放大不增加真实纹理分辨率。
没有可见印刷文字/贴纸/logo，秤上的字不属于木块。未拍到背面、右端、底面和部分内拱，
这些区域用纯色，登记“缺照片”，没有给它们编造木纹。
颜色、粗糙度0.64和几何小细节为目视拟合/假设，不代表木材辨识或色彩标定。
源图SHA和裁剪顶点见 `preview/texture_provenance.json`，标注见 `preview/photo_regions.png`。

## 实际运行及边界

实际项目 `SceneManager.instantiate_object_wrapper` 加载，运行时为生产env隔离快照，
未把其他会话的env改动提交进本分支，SHA见 `preview/runtime_source_sha256.json`。
物理步长1/240s。最终工厂原生报告为 `preview/native_report.json`（validation/v5-cpu）；
GPU标准落地报告另为 `preview/check.json`。二者对应最终同一USDZ和参数文件SHA。

| 检查 | 实际后端/结果 |
| :-- | :-- |
| 初始无外部穿插 | CPU工厂初始位姿读回、保守碰撞包围盒间距为正；几何自检 |
| 落地静置 | CPU工厂及GPU `check_object.py` 均通过 |
| 质量/惯量 | CPU原生读回一致、惯量正定；质量实测、惯量模型计算 |
| 拱洞通行/实体阻挡 | 直径4mm同形球穿过拱洞；支墩与拱顶阻挡；运动实心推块推动木块 |
| 第二次加载 | 独立第二项目工厂实例、质量读回通过；分离通行测试时以运动学方式持定 |
| 夹持/抬升代理 | CPU运动学目标接口，夹块命令50mm，木块抬升49.976mm，相对滑移0.024mm，释放回落通过 |
| 自由滑移 | 初速度0.25m/s，滑行2.038mm后减速至静置；没有反推出实测摩擦 |
| 复位 | 注入位移、旋转及非零六维速度，位置/姿态恢复、速度归零，恢复后再次落地 |
| 机器人抓取/实际木材标定 | 未做；夹块运动与0.25mm命令重叠是诊断条件，非测得夹持力或机器人驱动 |
| GPU运动学夹块目标 | 失败：当前GPU张量接口未实现该setter，不得据CPU结果宣称GPU夹持通过 |

保留失败：v1报告序列化遇到numpy bool错误后退出崩溃，没有有效原生验收；
v2直接改夹块位姿的GPU夹持未抬起木块，报告passed=false（不能以其进程退出0当通过）；
v3正确目标API在GPU报未实现。失败日志/报告均留在交付包validation与logs中。
v4在CPU正确目标API下通过；v5只隐藏预览阶段夹块以生成干净多视角图，再次完整CPU检查通过。
没有用修改摩擦来凑通过，未宣称实际机器人、实物滑移或木材强度已标定。

最终模型SHA256：`da4d03d9040db62ceb24b1f4fecdeb830c2a504ab6ec6c6d6f4658054d8ee6c0`。
最终USDZ、metadata、runtime_config、description四文件SHA绑定在原生报告中。
完整来源照片、称量记录、历史溯源文件保留；生成脚本为 `scripts/objects/make_wooden_arch_block.py`。
原multi-object生成器是其他会话未提交的生产代码，此次用专用可复现入口隔离目标资产，
没有把它的其他物体、旧程序木纹或未提交修改带进本分支。

## 复现

```bash
python scripts/objects/make_wooden_arch_block.py ovl/Rigid
python scripts/objects/object_names.py apply ovl
python scripts/objects/object_names.py check
PYTHONPATH=runtime:. python scripts/objects/check_wooden_arch_block.py --device cpu --asset ovl/Rigid/wooden_arch_block/00000 --output validation/new-cpu
PYTHONPATH=runtime:. python scripts/objects/check_object.py ovl/Rigid/wooden_arch_block/00000 --image-dir real2sim/objects
```

全仓最终检查91 passed、1 skipped、82 subtests passed，日志为 `cpu-check-final.log`；
ruff/py_compile通过。共享当前60×30×40mm/25g旧资产仍然保留，未自动接纳或覆盖人工审核；
待补缺面与真实夹持/材质验证后再考虑基础库接纳和授权部署。

## 全部指令名称

en: block → arch block → wooden arch block → light brown wooden arch block → light brown rectangular wooden arch block with semicircular opening

zh: 积木 → 拱形积木 → 木质拱形积木 → 浅棕色木纹拱形积木 → 底部半圆洞的浅棕色矩形木纹拱形积木


## 平台与共享库状态

新实测候选发布在平台研究区 `wooden_arch_block_measured`，原资产关联本候选交付，署名杨开元。
原资产的既有preview/check、历史原生报告、共享库和镜像中的60×30×40mm/25g旧模型全部保留，
没有用候选通过结果给旧模型背书，也没有自动批准人工审核。报告内preview路径以交付包或
研究区候选根目录为准。原登记新确认版本保持有效，非目标平台记录不变，操作前有备份。
GitHub HTTPS认证在本会话缺失；源码保存独立本地分支，未推送/PR/合并。
