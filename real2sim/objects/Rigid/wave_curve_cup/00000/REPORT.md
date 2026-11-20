# 波浪纹杯子 · 实物建模交付

登记人/平台署名：杨开元；建模执行：Codex。日期：2026-10-29（Asia/Shanghai）。
资产：`Rigid/wave_curve_cup/00000`；泛化：同类别 `00001–00003`。

## 输入与版本

开工时从平台只读回读当前记录，使用平台 `object_registration.revision` 重新计算，
real-object-v3 确认有效，revision 为
`4245652bc6c6f64f2d6ae929ce1efd467418b31e21bef4c2001a1550168abcbe`。
本体 spec 中 `asset_origin=physical_object` 是导出归一化值；确认摘要中的该字段为 null，
平台原始记录与摘要一致，不能用导出 spec 的归一化字段替代原始确认输入计算。
原照片 2899792 bytes，SHA256
`90e0ebefe1d1bf93438644f6c2f9fd0bb3412687a9b2490732ee23dc9014ed54`。
生成器重验登记字段、正式 prompt 输入摘要及本地照片 SHA，不自动重确认，也不修改输入。

## 尺寸、几何和材料

单位 m/kg，Z 向上，原点为完整可见网格包围盒中心。开口直径解释为杯口卷边最大外径，
底部直径是最低支撑圆外径；高度为外底最低点到卷边最高点。内口径没有实测。

| 同名字段 | 实测 mm | 模型 mm | 误差 mm |
| :-- | --: | --: | --: |
| 开口直径 | 72.76 | 72.76 | 0.00 |
| 底部直径 | 60.61 | 60.61 | 0.00 |
| 高度 | 93.01 | 93.01 | 0.00 |

本体质量实测 110.9 g，未用密度反推或更改。静/动摩擦未知，metadata 不写实测摩擦，
使用项目默认值（0.6/1.5），未做真实接触材料标定。
壁厚2 mm、底厚3 mm、卷边半径1 mm、五道凹槽的位置/深度0.55 mm均为单张照片推断，
不属于实测；照片未覆盖背面、底面，缺照片。几何为圆形渐扩杯体、开放杯口、光滑内壁、
横向模压波浪凹槽（槽线沿周向起伏2.5mm，曲率与背面延拓为单图推断）；可见面未见标签、logo或印字，照片中的秤面文字不属于杯子。
颜色和粗糙度从实拍目视拟合，未作色卡或灯光标定；没有已有模型参数可沿用。

显示网格与碰撞分开：单一完整外壳 `/root/visual/model`，凸碰撞为杯底与杯壁分片，
开口不封闭；每凸包<=64顶点，杯底24边，杯壁内侧弦误差<=0.3 mm。
碰撞外壁使用光滑包络，外侧微凹槽不逐槽参与接触。
三个变体外尺寸、形状、壁厚完全相同，仅换颜色/材质；未开运行时换色/缩放。
00001蓝色釉面陶瓷、00002红色硅胶外观、00003银色拉丝铝；均为刚体后端。
变体质量由同一外壳体积和假定密度2300/1100/2700 kg/m³估计，不能视为实测。
硅胶变体只表达外观与假定密度，不模拟柔性材料响应。

## 实际验证

在已核实的 RTX 5090 服务器 GPU1 隔离目录运行，实际项目 SceneManager 工厂加载，
PhysX GPU 刚体，物理步长1/240s。每件执行：自然落地静置、质量读回、第二独立实例加载、
8 mm动态球穿过开口并被杯底留住、侧壁/杯底实体阻挡、自由杯子留物、运动学实心推块推动杯子，
注入非零平移/旋转和六维速度后复位、速度归零和再次静置。另跑 `check_object.py`，4/4通过。
杯腔分离阻挡测试用浮空运动学杯排除地面混淆，自由杯留物另外验证；未做机器人抓取、
流体、水密性、真实材料响应或真实摩擦校准。报告中的 pass 只覆盖已运行的数值项目。

| 编号 | 质量 g | 工厂落地倾角 deg | 线速度 m/s | 工厂全部检查 | object.usdz SHA256 |
| :-- | --: | --: | --: | :-- | :-- |
| 00000 | 110.900 | 0.000671 | 0.001190 | pass | `bd574802aacd6d3979a95b370391855195883dd33c65eda8c031bef06047638c` |
| 00001 | 87.703 | 0.000775 | 0.001330 | pass | `2703e894566b151d452648f95c7ebd7a73ed427b4608df0d6fb83c0f2dc2505a` |
| 00002 | 41.945 | 0.000819 | 0.001343 | pass | `7db78e91d081d11fdfb131c4d812e0a25d004c26f68f7e4580d9f97f5f0dc275` |
| 00003 | 102.956 | 0.000795 | 0.001335 | pass | `d1be0aa2affc23d6136cf7193f9baae5e08433cb85138f178716c50786ee5038` |

原生报告分别为 `preview/native_report.json`（本体）和各变体同名文件；
`preview/check.json` 为标准项目落地报告并附分项依据，均绑定最终四个资产文件SHA。
第一次验证保留在隔离目录 validation/00000-v1，v2调整卷边与颜色，v3补周向起伏槽线并重新生成全族，
最终使用 validation/*-v3，旧模型保留 ovl-v1/ovl-v2，不与最终报告混用。
统一中性光照多视角图、变体拼图、单张源照片视角对照均在本体 preview/。
相机由单图估计；对照图为可见轮廓检查，不代表相机或外观已标定。

运行时使用生产项目 env 的隔离快照；文件SHA记录在 `preview/runtime_source_sha256.json`。
生产树含其他会话的质量默认值/速度复位等更新，没有将那些改动提交进本次分支。
复现时需匹配该运行时快照，或已合入等价功能的项目版本。
全仓 `scripts/check.sh`：90 passed，1 skipped，82 subtests passed；ruff及py_compile通过。
初次75个机器人网格哈希子检查失败源于LFS指针未展开，日志保留；执行git lfs checkout后全仓通过。

## 状态、交付与复现

本体和三个变体为通过数值复用检查的仿真候选，外观待人工复核；缺背面/底面照片，
不宣称完整实物标定。尚未安装共享资产库，未自动通过人工审核。
已生成与核验的模型位于隔离目录 ovl/Rigid/wave_curve_cup/00000–00003；
交付包按 Assets/Object/Real2Sim/Rigid/wave_curve_cup/ 同名目录组织。
源代码位于独立分支 `yangkaiyuan/objects-wave-curve-cup`；没有修改任务布局和子模块源码。

```bash
python scripts/objects/make_wave_curve_cup.py ovl/Rigid
python scripts/objects/object_names.py apply ovl
python scripts/objects/object_names.py check
python scripts/objects/check_wave_curve_cup.py --asset ovl/Rigid/wave_curve_cup/00000 --output validation/new
python scripts/objects/check_object.py ovl/Rigid/wave_curve_cup/0000{0,1,2,3} --image-dir real2sim/objects
```

## 全部指令名称

### Rigid/wave_curve_cup/00000

en: cup → drinking cup → yellow drinking cup → yellow grooved polypropylene drinking cup → yellow tapered open polypropylene drinking cup with horizontal wave grooves

zh: 杯子 → 饮水杯 → 黄色饮水杯 → 黄色波浪纹塑料饮水杯 → 黄色圆口无把手锥形波浪纹塑料饮水杯

### Rigid/wave_curve_cup/00001

en: cup → drinking cup → blue drinking cup → blue grooved glazed ceramic drinking cup → blue tapered open glazed ceramic drinking cup with horizontal wave grooves

zh: 杯子 → 饮水杯 → 蓝色饮水杯 → 蓝色波浪纹陶瓷饮水杯 → 蓝色圆口无把手锥形波浪纹陶瓷饮水杯

### Rigid/wave_curve_cup/00002

en: cup → drinking cup → red drinking cup → red grooved silicone drinking cup → red tapered open silicone drinking cup with horizontal wave grooves

zh: 杯子 → 饮水杯 → 红色饮水杯 → 红色波浪纹硅胶饮水杯 → 红色圆口无把手锥形波浪纹硅胶饮水杯

### Rigid/wave_curve_cup/00003

en: cup → drinking cup → silver drinking cup → silver grooved brushed aluminium drinking cup → silver tapered open brushed aluminium drinking cup with horizontal wave grooves

zh: 杯子 → 饮水杯 → 银色饮水杯 → 银色波浪纹铝制饮水杯 → 银色圆口无把手锥形波浪纹铝制饮水杯


## 平台同步回执

最终预览和原生报告已定向同步本体与三变体，署名杨开元，状态为研究候选/待人工复核，
没有替换人工审核结论，也没有安装共享资产库。输入spec、正式prompt、原照片SHA保持，
平台确认摘要仍有效；仅更新建模交付、check/渲染/核对数据，没有覆盖版本化输入和旧模型默认值。
平台将来全量扫描新模型默认值时，旧prompt会按项目规则失效，后续重建需重新确认。
定向事务中所有非目标记录保持不变，有数据库及目标文件备份，私有交付含同步回执。
Git远程HTTPS推送dry-run因缺少认证失败（terminal prompts disabled）；源码已保存独立本地分支，
未推送、未创建PR、未合并main。
