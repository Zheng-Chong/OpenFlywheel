# 自建资产登记表

更新时间：2026-10-20（共享资产库状态于 2026-10-19 只读核对）

这里列出所有自建资产：对应哪件实物、怎么生成、每个属性和真实的匹配程度、服务器共享资产库里装了没有。官方 RoboDojo 资产（`block`、`car` 等）不在表里。

## 匹配等级

| 等级 | 含义 |
| :-- | :-- |
| 实测 | 对在用的那件实物直接量：卷尺、卡尺、秤、棋盘格标定 |
| 官方 | 厂商给的模型或规格，且对应的就是在用的那件实物 |
| 拟合 | 从照片、视频或真机画面反求，有残差可查 |
| 估计 | 常见规格、同款产品参数或目测 |
| 未做 | 用的是代码默认值或随手定的值 |

## 物体

路径 `Assets/Object/Real2Sim/<类型>/<类别>/<编号>/`（Real2Sim 物体库，和官方的 `Object/RoboDojo/` 分开；加载器先找 Real2Sim），由生成脚本写入，不入库。共享库里从真实场景资产包来的孪生（`metadata.json` 有 `source_asset`）、它们的泛化变体（`variant_of`）和平台登记的孪生也都在 Real2Sim 库，2026-10-28 从 `Object/RoboDojo/` 搬来，原路径留了相对软链接给旧代码用。

| 资产 | 真实对应物 | 生成脚本 | 尺寸 | 质量 | 摩擦 | 外观 | 共享库 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `Rigid/wave_curve_cup/00000` 波浪纹杯子 | 杨开元登记的黄色PP波浪杯 | `make_wave_curve_cup.py` | 实测外径72.76/60.61、高93.01mm，网格同值；局部壁厚/槽深估计 | 实测110.9g | 默认，未知 | 单张实拍拟合；背面/底面缺照片，外观待复核 | 未装；4项族资产工厂/留物/阻挡/复位/复载通过 |
| `Rigid/wave_curve_cup/00001–00003` 泛化 | 无实物：蓝陶瓷、红硅胶外观、银铝 | `make_wave_curve_cup.py` | 同本体几何，不冒充实测 | 密度×外壳体积估计87.703/41.945/102.956g | 默认 | 不同颜色与粗糙度，均为刚体假设 | 未装，数值检查通过，外观待复核 |
| `pitcher/00000–00002` 冷水壶 | Luminarc Quadro 1.1 L 同款，未确认有实物 | `scripts/objects/make_pitcher.py` | 估计：按产品规格高 24.5 cm、宽 8.9 cm | 未做：0.5 kg 是加载器上限 | 默认 | 估计：烟灰/琥珀/透明三色；00001、00002 是 00000 的颜色变体（`variant_of`） | 已装 |
| `water_cup/00000` 搪瓷杯 | `scene3_pour_pack` 里的接水杯 | `make_water_cup.py` | 估计：取自 `scene_spec.json`（该文件写明物体尺寸是估计的），高 118 mm、口径 73 mm | 估计 0.165 kg | 默认 | 估计 | 已装 |
| `plastic_cup/00000` 透明杯 | 真机片段 G1D-PourBeansPlus 里的一次性杯 | `make_water_cup.py --plastic` | 估计：对着视频里的夹爪目测，高 95 mm、口径 72 mm | 估计 30 g | 默认 | 估计 | 已装 |
| `measuring_cup/00000、00001` 量杯 | 2026-10-17 桌面照片中的 400 / 500 ml 量杯 | `make_plastic_cups.py` | 实测：高 140 / 170 mm，壁厚 2 mm，杯口（翻边）75 mm、杯底 63 mm（对边）；杯底倒角方形、杯口圆角方形按俯视照片 | 实测 40 / 60 g | 默认 | 估计：对照照片调材质 | **未装** |
| `measuring_cup/00002–00005` 泛化变体（`variant_of` 00001）：透明 170 mm、烟灰半透明 200 mm、不透明橙色 120 mm、浅蓝半透明 155 mm | 无实物 | `make_plastic_cups.py`（`MEASURING_VARIANTS`） | 估计：截面与实测量杯相同（`measuring_cup_shape.CUP`，任务杯腔判据按高度仍精确），只改高度；满口 439–733 ml，刻度取满口约 80%（350–600 ml） | 估计：按两只实测量杯随高度线性插值，27–80 g | 默认 | 估计：00002/00003/00005 着色半透明 OmniSurface，00004 OmniPBR | 已装 `Object/Real2Sim`（2026-10-28，本体 00000/00001 仍未装），无任务使用 |
| `plastic_mug/00000` 白色马克杯 | 同一组照片中的塑料马克杯 | `make_plastic_cups.py` | 实测：直筒高 100 mm、内径 80 mm、壁厚 2 mm；把手高 70 mm、外伸 35 mm、厚 5 mm（底厚、把手条宽 13 mm 为估计） ；模型 `measurements_mm` 总长 119.0、外径 84.0、高 100.0，误差 0 | 实测 89 g | 默认（不再写不生效的 `friction`） | 估计 | **未装**；`g1d_pour_beans_cup_or_mug` 用 |
| `plastic_mug/00001–00005` 泛化变体（`variant_of` 00000）：天蓝儿童杯、粉色高身锥形杯、藏青陶瓷咖啡杯、烟棕半透明杯、薄荷绿圆角方杯 | 无实物 | `make_plastic_cups.py`（`MUG_VARIANTS`） | 估计：同一构造（直筒/锥形/超椭圆截面 + 扁条把手），高 78–120 mm、杯身宽 72–84 mm、276–490 ml | 估计：网格体积 × 密度（PP 0.9、陶瓷 2.3 g/cm³），49–289 g | 默认 | 估计：各色 OmniPBR，00004 为着色半透明 OmniSurface | **未装**，无任务使用 |
| `red_bean/00000` 单颗红豆 | 红小豆 | `make_red_bean.py` | 实测：7 × 5 × 5 mm | 实测约 0.1 g | — | 估计 | 已装（无任务使用，只用于渲染） |
| `paper_cup/00000` 白纸杯、`00001` 黑纸杯 | 学校实验室绿幕桌上的纸杯，每只是两只套在一起的一次性纸杯 | `scripts/objects/make_school_tableware.py` | 拟合：同事 MuJoCo 孪生 `g1d_food_sorting` 用真机画面拟合的轮廓（`real2sim/scenes/real_scene4_school/bundle/cup_shapes.json`），白 85.6 × 155.4 mm、黑 86.6 × 153.8 mm；壁厚 2 mm、杯底 3 mm 为估计 | 估计：孪生的密度假设，19.3 / 20.7 g | 默认 | 估计：孪生的纯色 | **未装**，无任务使用；未在平台登记 |
| `paper_plate/00000–00002` 纸盘（盛橘子、红色食物、花材的三个） | 同一张桌上的三个一次性纸盘 | `make_school_tableware.py` | 拟合：孪生的纸盘网格，口径 160 / 137 / 151 mm、高 16 / 14 / 15 mm；纸厚 1.5 mm 为估计 | 估计：孪生的密度假设，4.3 / 3.1 / 3.8 g | 默认 | 估计：孪生的纯色 | **未装**，无任务使用；未在平台登记 |
| `airpods/00000` 耳机充电盒 | 数采平台登记的白色耳机充电盒（AirPods 款式），平放 | `scripts/objects/make_airpods.py` | 实测：高 × 宽 × 厚 46.40 × 54.40 × 21.38 mm（模型 46.4 / 54.4 / 21.4）；俯视圆角 R15、边缘圆角 R8、盖深 16 mm 为照片估计；扬声器孔、指示灯、挂绳孔未建 | 实测 38 g | 默认 | 估计：对照 3 张实拍，白色亮面、盖缝、USB-C 口 | 已装（2026-10-20），无任务使用 |
| `basket/00000` 奶白镂空收纳篮（灰色把手） | 数采平台登记的 PP 收纳篮（徐鹤鸣 10-29） | `scripts/objects/make_basket_cream.py` | 实测：长 358、宽 253、高 96 mm（模型误差 0；包围盒含把手 359.2 × 253 × 97.4）；壁孔 311 个（约 6 mm，交错 7 排，只在四角附近成片、向长边中部渐疏）、底孔 253 个（12 mm 方格，椭圆形孔区、中间留实心块和 3 个孤孔），孔径、外扩、圆角为照片估计 | 实测 266 g | 默认 | 估计：奶白色 PP、灰色把手嵌件（OmniPBR），对照三张实拍 | 已装（2026-10-29） |
| `basket/00002–00007` 泛化变体（`variant_of` 00000）：粉蓝、深筐浅灰黑把手、粉色浅托盘、薄荷绿方篮、哑光炭灰无把手嵌件、亮白小篮满壁孔 | 无实物 | `make_basket_cream.py` | 估计：长 270–358 mm、高 60–130 mm | 估计：按壳面积从 266 g 换算，156–321 g | 默认 | 估计 | 已装（2026-10-29） |
| `basket/00001` 浅蓝镂空收纳篮 | 数采平台登记的 PP 收纳篮（徐鹤鸣 10-29） | `scripts/objects/make_basket_blue.py` | 实测：长 338、宽 258、高 84 mm（含口沿，模型误差 0）；长边每面 4 × 20 = 80 孔、短边每面 4 × 15 去掉把手槽占的 10 个 = 50 孔（按照片数过，脚本校验孔数），实心底；孔径约 9 mm、孔距约 14 mm、篮壁外倾（短边 10°、长边 4°）、口沿外伸和四角垂片为照片估计 | 实测 178 g | 默认 | 估计：浅蓝亮面 PP（OmniPBR，不透明），对照三张实拍 | 已装（2026-10-29） |
| `basket/00008–00013` 泛化变体（`variant_of` 00001）：薄荷绿深篮 5 排孔、粉色浅篮 3 排孔、灰色方篮、黄色长圆孔、藏青哑光带底孔、磨砂透明方孔小篮 | 无实物 | `make_basket_blue.py` | 估计：长 250–338 mm、高 60–110 mm | 估计：按面积从 178 g 换算，101–199 g | 默认 | 估计 | 已装（2026-10-29） |
| `Clutter/paper_cup/00000` 黑色单层纸杯 | 数采平台登记的纸杯（徐鹤鸣 10-29） | `scripts/objects/make_paper_cup.py`、`paper_cup_shape.py` | 实测：开口外径 87.9 mm（含卷边）、底部直径 60.5 mm、高度 136.0 mm（模型误差 0）；直锥杯壁、3.4 mm 白色卷边、杯底内凹 10 mm（下面是向内折边的裙边）、一条竖向接缝，壁厚 0.7 mm 为照片估计；容积约 494 ml | 实测 22 g | 默认 | 估计：外壁哑光黑纸、内壁卷边和杯底米白纸（OmniPBR），对照三张实拍 | 已装（2026-10-29） |
| `Clutter/paper_cup/00001–00006` 泛化变体（`variant_of` 00000）：白色、牛皮纸 112 mm、红色浓缩杯 58 mm、藏青宽口矮杯、鼠尾草绿双层杯、芥末黄高杯 158 mm | 无实物 | `make_paper_cup.py` | 估计：高 58–158 mm、101–599 ml | 估计：按纸面积从 22 g 换算，7.8–38.4 g | 默认 | 估计 | 已装（2026-10-29） |
| `plastic_box/00000` 灰白塑料收纳盒 | 数采平台登记的 PP 方盒（徐鹤鸣 10-28） | `scripts/objects/make_plastic_box.py` | 实测：外边长 151.5 mm、内边长 140.2 mm（模型误差 0）；高约 100 mm、灰色外沿高 27 mm、白色盒身每边内收 8.5 mm 并向底部收 2.5 mm、圆角为照片估计 | 实测 151 g | 默认 | 估计：白色盒身、深灰外沿（OmniPBR），对照两张实拍 | 已装（2026-10-29） |
| `plastic_box/00001–00006` 泛化变体（`variant_of` 00000）：粉色盒身白外沿、米棕高长方盒、藏青浅托盘、黄灰大圆角盒、半透明盒身黑外沿、薄荷绿小盒 | 无实物 | `make_plastic_box.py` | 估计：边长 112–200 mm、高 60–140 mm | 估计：按壳面积从 151 g 换算，82–195 g | 默认 | 估计 | 已装（2026-10-29） |
| `lounge_chair/00000` 蓝色扶手椅、`00001` 白色扶手椅 | 吧台左侧客区圆桌边的软包扶手椅（`real_scene2_counter/reference/lounge*.jpg`） | `scripts/objects/make_lounge_chair.py` | 估计：按客区拟合（桌高 0.75 m 假设）对照片目测，宽 0.58、深 0.56、高 0.80 m，坐面 0.48 m；结构按照片：平面前直后圆的 U 形，一圈软包底座托着内嵌坐垫，一片弧形壳包住背和两侧、顶边后高前低（0.80 → 0.68 m），两侧前段落到底座上，后面和侧后方在底座与壳之间留一道圆角镂空，四条外撇木腿；白色椅只换了颜色，真实是另一款更圆的椅子 | 估计 7 kg，布局用 `mass_cap` 放开 0.5 kg 上限 | 估计：静 0.5 / 动 0.4（木脚对水磨石，推得动） | 估计 | **未装**；任务 `g1d_deliver_cup_to_lounge` 用；未在平台登记 |
| `Clutter/plastic_bottle/00000` 农夫山泉 550 ml 塑料瓶（带盖，单刚体） | 数采平台登记的透明塑料瓶，红白标签 | `scripts/objects/make_plastic_bottle.py`（标签 `bottle_label.py`） | 实测：开口直径（外径）29.8 mm、底部直径 64.7 mm（平台登记），总高 227 mm（不含盖，用户 2026-10-23 补测；带盖 228.5 mm）；瓶肩、瓶口螺纹和支撑环、标签位置、握持凹槽、瓶底内凹、瓶盖（外径 33 mm、高 16.4 mm）为照片估计；瓶壁 0.4 mm（估计），碰撞壁 2 mm | 实测：空瓶 18 g（登记的 328 g 含水，按空瓶建模，算作含盖） | 实测：静摩擦 0.35；动摩擦未测，取同值 | 估计：透明 OmniGlass；标签是程序绘制的贴图（品牌字、山水图、成分表、条码、二维码，照实拍的红白版式，文案除品牌外是通用文字）；红色竖纹盖 | 已装（2026-10-23），无任务使用 |
| `Clutter/plastic_bottle/00001–00005` 泛化变体（`variant_of` 00000）：380 ml 小瓶、1 L 淡蓝大瓶、磨砂圆角方瓶、绿色束腰瓶、琥珀宽口果汁瓶 | 无实物 | `make_plastic_bottle.py` | 估计：瓶口、瓶底结构同本体，高 150–262 mm、宽 58–75 mm | 估计：按容积从本体 18 g 换算，10–28 g | 同本体 | 估计：虚构品牌的贴图标签，各色瓶盖 | 已装（2026-10-23），无任务使用 |
| `Articulation/plastic_bottle/00000–00005` 上面 6 个瓶子的能拧盖版本 | 同上 | `make_plastic_bottle.py`（同一次运行写出） | 同 Clutter 版；关节 `cap_twist`（转 0–720°，拧两圈松开）+ `cap_lift`（mimic 跟随，每圈升 2.7 mm 螺距，估计） | 同上；瓶盖 2 g、中间连杆 0.1 g（估计） | 同上，摩擦写进 USD 物理材质（`ArticulationObject` 不读 metadata） | 同上 | 已装（2026-10-23），无任务使用；盖子拧到头也不会脱离瓶身 |
| `Clutter/metal_cup/00000` 金属杯（无把手，卷边杯口） | 数采平台登记的铝杯，外壁和杯底白色、内壁反光金属 | `scripts/objects/make_metal_cup.py` | 实测：开口外径 80.2 mm（含卷边）、杯底外径 57.3 mm、高 105.4 mm（平台登记，模型三项误差 0）；壁厚 0.6 mm、卷边半径 1.2 mm、杯底内凹、内底凸台为照片估计，杯身小红标未建 | 实测 65 g | 默认 | 估计：白色涂层 OmniPBR，内壁、卷边拉丝铝 | 已装（2026-10-29）；未做物理验证（用户要求），已渲染 |
| `Clutter/metal_cup/00001–00006` 泛化变体（`variant_of` 00000）：黑色浓缩咖啡杯、淡蓝高筒杯、不锈钢直筒杯、绿色圆角方杯（内壁铜色）、红色宽口矮杯（内壁金色）、奶白鼓腹杯 | 无实物 | `make_metal_cup.py` | 估计：高 70–140 mm、宽 64–80 mm | 估计：按外表面积从本体 65 g 换算，不锈钢 ×2.9，37–198 g | 默认 | 估计 | 已装（2026-10-29）；未做物理验证，已渲染 |
| `ceramic_cup/00000` 黑色陶瓷杯（无把手，外壁阶梯圈层） | 数采平台登记的黑色釉面陶瓷杯 | `scripts/objects/make_ceramic_cup.py` | 实测：杯口外径 73.1 mm、杯底 58.4 mm、杯高 92 mm（平台登记，模型三项误差 0）；外壁 7 圈 6 级台阶按照片数，壁厚 4 mm、圈足、内底圆角为照片估计 | 实测 112 g | 默认 | 估计：黑色釉 OmniPBR（粗糙度 0.08） | 已装（2026-10-29） |
| `ceramic_cup/00001–00006` 泛化变体（`variant_of` 00000）：白瓷光面直筒、青瓷 4 级台阶、陶土哑光（内白釉）、钴蓝 12 级细纹（内白）、灰色宽口矮杯、红色鼓腹杯（内黑） | 无实物 | `make_ceramic_cup.py` | 估计：高 65–100 mm、宽 64–80 mm | 估计：按陶瓷体积从本体 112 g 换算，85–169 g | 默认 | 估计 | 已装（2026-10-29） |
| `iphone_17_pro_max/00000` iPhone 17 Pro Max 宇宙橙（黑屏） | 未登记实物，按苹果官网参数建（用户 10-27 同意） | `scripts/objects/make_iphone_17_pro_max.py` | 官方：机身 77.98 × 163.43 × 8.75 mm（模型高 163.4、宽 78.0、厚 8.75，误差 0）、平面圆角、凸台 2.55 mm、镜头再高 1.88 mm、镜头 / 闪光灯 / 传感器 / 麦克风位置、按键位置（苹果开发者网站公开的尺寸图 2025-09-09，PDF 不入库）；估计：边缘圆角、凸台下沿、背板内缩；未建：logo、扬声器孔、天线条 | 官方：国行 231 g | 默认 | 估计：颜色取样 Apple Newsroom 产品图 | 已装（2026-10-29） |
| `iphone_17_pro_max/00001、00002` 深蓝、银色（`variant_of` 00000，只换颜色） | 同上 | `make_iphone_17_pro_max.py` | 同本体 | 官方 231 g | 默认 | 估计 | 已装（2026-10-29） |
| `plastic_cup/00001` 半透明蓝色塑料杯 | 数采平台登记的 PP 塑料杯（种政 10-28） | `scripts/objects/make_plastic_cup.py` | 实测：开口外径 80.2 mm、底部直径 54.8 mm、高度 132.6 mm（模型三项误差 0）；圆唇、内底平圆盘和台阶、外底圈足按照片，壁厚 0.9 mm 按质量反推 | 实测 24 g | 默认 | 估计：OmniSurface 半透明蓝（80% 透射、按 1 mm 壁厚体积吸收），对照照片调过 | 已装（2026-10-28） |
| `plastic_cup/00002–00007` 泛化变体（`variant_of` 00001）：乳白磨砂、透明小杯、红色宽口矮杯、绿色圆角方杯、琥珀鼓腹杯、紫色高杯 | 无实物 | `make_plastic_cup.py` | 估计：高 90–150 mm、宽 68–80 mm | 估计：按外表面积从 24 g 换算，18–24 g | 默认 | 估计 | 已装（2026-10-28） |
| `soda_can/00006` 零度可乐罐（已开罐） | 数采平台登记的细长铝罐（种政 10-28） | `scripts/objects/make_soda_can.py`、`soda_can_label.py` | 实测：直径 57.7 mm、高度 146.5 mm（到卷边顶，不含拉环；模型误差 0）；泪滴形倒饮口（约 18 × 14 mm）和压入罐内的盖板、双孔拉环（13 × 21.8 × 1 mm，立起 80°）、罐盖沉槽、罐底拱顶为照片估计 | 实测 13 g | 默认 | 估计：涂装为从三张参考照片展开的贴图（标志、FIFA 字样、无糖、黑黄红色块、奖杯徽标、背面成分表和条码；照片没拍到的约 50° 侧面为纯红），罐盖和卷边拉丝铝 | 已装（2026-10-28） |
| `soda_can/00007–00012` 泛化变体（`variant_of` 00006）：330 ml 红罐、绿色细长罐、500 ml 蓝色高罐、250 ml 哑光黑细罐、拉丝铝矮罐、金色鼓腹罐 | 无实物 | `make_soda_can.py` | 估计：直径 53–66 mm、高 92–168 mm | 估计：按罐身面积从 13 g 换算，8–17 g | 默认 | 估计 | 已装（2026-10-28） |
| `Clutter/plastic_bottle/00006` 东方树叶 500 ml 茶饮料瓶（带盖） | 数采平台登记的东方树叶红茶瓶（徐鹤鸣 10-28） | `scripts/objects/make_plastic_bottle.py`（`TeaBottle`）、`bottle_label.py` | 实测：瓶口外径 39.6 mm、瓶身边长 62.2 mm、高度 205 mm（含瓶盖；登记写 20.5 mm，用户 10-28 确认 205；模型三项误差 0）；大倒角八边形瓶身（斜面竖纹）、短瓶肩、Ø41.6 mm 长瓶颈、白顶透明裙边瓶盖、圆形内凹瓶底，按照片估计 | 实测 33 g（含盖） | 默认 | 估计：标签只在正面、背面和瓶颈前半圈（侧面无标签），00006 的贴图由脚本从参考照片裁剪、校正透视 | 已装（2026-10-28） |
| `Clutter/plastic_bottle/00007–00012` 泛化变体（`variant_of` 00006）：600 ml 茉莉清茶、400 ml 琥珀乌龙、圆瓶荔枝红茶、磨砂软方瓶大红袍、收腰柠檬茶、300 ml 白桃乌龙（虚构品牌，瓶口同 00006） | 无实物 | `make_plastic_bottle.py` | 估计：高 150–227 mm、宽 54–64 mm | 估计：按满口容积从 33 g 换算，18–39 g | 默认 | 估计 | 已装（2026-10-28） |
| `Articulation/plastic_bottle/00006–00012` 上面 7 个瓶子的能拧盖版本 | 同上 | `make_plastic_bottle.py` | 同上 | 同上 | 盖与瓶身之间的物理材质沿用脚本的 0.35 | 同上 | 已装（2026-10-28） |
| `Clutter/plastic_bottle/00013` 透明扁圆 PP 瓶（带盖，无标签） | 数采平台登记的透明塑料瓶（徐鹤鸣 10-28） | `scripts/objects/make_plastic_bottle.py`（`OvalBottle`） | 实测：开口外径 40.1 mm、瓶身短边 70 mm、长边 104.1 mm（模型三项误差 0）；类椭圆截面向底部收到 0.85、宽瓶肩、短圆瓶颈、五角星内凹瓶底，总高约 235 mm（含盖）、壁厚 0.9 mm 按照片估计 | 实测 65 g | 默认 | 估计：透明 PP 加细颗粒法线贴图，半透明白色螺纹盖（Ø44.5 × 24 mm） | 已装（2026-10-29） |
| `Clutter/plastic_bottle/00014–00019` 泛化变体（`variant_of` 00013）：淡蓝小瓶、光面圆角方瓶、磨砂椭圆瓶、琥珀高瓶、绿色矮宽瓶、粉色细瓶（瓶口同 00013） | 无实物 | `make_plastic_bottle.py` | 估计 | 估计 | 默认 | 估计 | 已装（2026-10-29） |
| `Articulation/plastic_bottle/00013–00019` 上面 7 个瓶子的能拧盖版本 | 同上 | `make_plastic_bottle.py` | 同上 | 同上 | 盖与瓶身之间的物理材质沿用脚本的 0.35 | 同上 | 已装（2026-10-29） |
| `test_object/00000` 测试用物体 | 数采平台登记的测试用物体（徐鹤鸣 2026-11-05），无照片 | `scripts/objects/make_test_object.py` | 实测（登记）：立方体边长 10 mm；模型 `measurements_mm` 边长 10.0，误差 0 | 估计 0.9 g：PP 密度 0.9 g/cm³ × 1 cm³（登记的 100 g 与尺寸矛盾，经登记人同意改为估计） | 默认 | 估计：乳白半透明 PP（`clear_pp`，乳白占比 0.5），无照片可对照 | 已提交资产库（待审核）；落地检查通过 |

- **摩擦“默认”**：运行时 `env/scene_manager/objects/rigid.py` 只读 `physics.static_friction` / `dynamic_friction`，不写就是 0.6 / 1.5。生成脚本写进 `metadata.json` 的 `friction`（0.25 / 0.5）不生效，USD 里也没有绑定物理材质。要让实测摩擦生效，生成脚本要改写这两个键。
- **任务里的红豆**是 5.6 mm PBD 球形颗粒（与实测单粒等体积）（`scripts/g1d/build_asset.py` 的 `RED_BEAN_PHYSICS`，单粒 0.1 g（同实测）），不是上面的网格。摩擦 11-01 起是 0.3（原 0.2，是 4 mm 时为了能从 13 mm 壶嘴倒出来调的），按真机 G1D-PourBeansEps380 的倒豆标定（`scripts/g1d/bean_tilt_probe.py`）：高杯装到真机的 131 mm（约 3100 颗，生成方式 `spawn: {region: box, packing: hex}`），按真机的约 2°/s 倾转，88° 出 5%、98° 出 25%；真机目标杯 86–90° 开始接到豆子、93–99° 峰值时约半杯（高杯的四分之一）。0.2 时同样的杯子 82° / 92° 就出，倒在桌上的豆堆只有 13°（约 atan 0.2），0.3 时 18°，真实红豆约 25–30°，仍偏低。粒径影响很小（4.5 mm 起流只差 1°）。5.6 mm 颗粒会堵住壶嘴，壶倒杯任务需另行处理。
- **质量上限**：`rigid.py` 把刚体质量截到 ≤ 0.5 kg；布局条目的 `physics.mass_cap` 可以放开（只有客区椅子用）。

## 积木库（Blocks）

不是实物孪生的积木组：按商品图建，路径 `Assets/Object/Blocks/<类型>/<类别>/<编号>/`（加载器在 Real2Sim 之后、RoboDojo 之前找，编号和其他库不重复）；数采平台上单独一个资产库「积木库」（`blocks`）。

| 资产 | 真实对应物 | 生成脚本 | 尺寸 | 质量 | 摩擦 | 外观 | 共享库 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `wood_block_*` 彩色实木积木 11 种形状 × 8 色 = 88 件：类别 `plank`（90 mm 长条）、`short_plank`（60 mm 长条）、`tile`（方片）、`cuboid`、`cube`、`cylinder`（高）、`short_cylinder`（矮）、`arch`、`prism`（大三棱柱）、`small_prism`、`half_cylinder`；每类 `00000` 原木色（榉木纹贴图），`00001–00007` 是它的颜色变体（`variant_of` 00000）：蓝、红、橙红、绿、黄、橙、粉 | 实验室那套彩色积木（商品图 `real2sim/objects/wood_blocks_product_20261031.jpg`）；正方体、大三棱柱、拱形各有一块在平台登记（`cube_building_blocks`、`triangular_prism_building_block`、`wooden_arch_block` 的 00000），这里另起一族，不动那三个登记和杨开元的资产（用户 2026-10-31） | `scripts/objects/make_wood_blocks.py` | 实测：正方体 30 mm，大三棱柱直角边 40.24、厚 29.27 mm，拱形 60.10 × 29.72 × 30.04 mm、拱顶厚 9.58 mm（模型 60.1 / 29.7 / 30.0 / 9.6）；其余按商品图标称（长条 90/60 × 30 × 15、方片 30 × 30 × 15、长方体 60 × 30 × 30、圆柱 Ø30 × 60/30、小三棱柱底 40 × 厚 30、半圆柱 Ø30 × 30），模型误差都 ≤ 0.2 mm；估计：小三棱柱按直角等腰（高 21.5）、半圆柱高 15、棱边圆角 R1.5 / 端面圆角 R1.2、拱洞为落在底面的半圆；同一形状各色几何相同 | 实测：拱形 24.6 g、大三棱柱 20.1 g；其余估计：网格体积 × 两块实测积木的平均密度 749 kg/m³，7.9–40.3 g；各色相同 | 默认 | 估计：原木色是程序生成的可平铺榉木纹（底色、细纹、短斑点，按实验室原木拱形的照片调色），按面法向投影 UV、圆柱按柱面展开；漆色是商品图配色的哑光 OmniPBR。任务可以再按布局随机换色（`build_asset.py` 的 `RANDOMIZE` `color`），换色后名称里的颜色就不对了，只有大类和第一级名称还准 | 已装 `Object/Blocks/`（2026-10-31），无任务使用 |
| `block_pattern_card/00000–00002` 图案卡：小房子、牌坊、拱桥 | 无实物：任务 `g1d_build_block_pattern` 的视觉参考，画的是 `scripts/objects/block_patterns.py` 里的图案正视图 | `scripts/objects/make_pattern_cards.py` | 估计：A6 横版 148 × 105 mm、2 mm 卡纸 | 估计 20 g | 默认 | 程序绘制：米白卡面，积木用原木纹贴图（提亮）加深色描边、地面线；图案上方朝 +Y（平放在机器人前方时正对机器人） | 已装 `Object/Blocks/`（2026-10-31），任务 `g1d_build_block_pattern` 用 |
| `block_sort_card/00000–00009` 分拣卡：`00000–00004` 蓝、红、绿、黄、粉色块，`00005–00009` 正方体、圆柱、半圆柱、三棱柱、方片的灰色立体图标 | 无实物：分拣任务 `g1d_sort_blocks_by_color` / `_by_shape` 里平放在每只托盘后方，说明这只托盘装哪一类，见 `scripts/objects/block_sorting.py` | `scripts/objects/make_sort_cards.py` | 估计：A7 74 × 105 mm、2 mm 卡纸，图案卡的卡面 | 估计 20 g | 默认 | 程序绘制：米白卡面；色块用积木漆色同一个 sRGB 值；形状图标从积木网格按 30°/28° 视角投影，剔除背面、由远及近填充、按法向打光 | 已装（2026-11-01，`check_object.py` 10/10 OK） |
| `block_tray/00000` 窄木托盘 | 无实物：分拣任务里并排放两只，装分好的积木 | `scripts/objects/make_block_tray.py` | 估计：外形 70 × 130 × 25 mm，壁厚 5 mm、底厚 4 mm（两块积木沿长边各占一格；壁矮，积木从壁顶上方投进去） | 按山毛榉 749 kg/m³ × 体积 ≈ 57 g | 默认 | 程序木纹，同原木积木 | 已装（2026-11-01，`check_object.py` OK） |

## 场景

源文件在 `real2sim/scenes/`（入库，LFS），`build_asset.py` 安装到 `Assets/Room/real_scene{1,...,6}/`。每个场景目录的 `PLAN.md` 有逐项尺寸和置信度。

| 场景 | env 配置 | 真实对应 | 重建依据 | 尺度 | 桌面 | 机器人站位 | 灯光 | 共享库 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `real_scene1` 办公室 | `g1d` | 办公室白色升降桌；10-24 起南墙玻璃展示柜和木门高柜、白色活动柜、两款办公椅（网背 4 把、高背皮椅 1 把）、电视墙腰线白板墙板、窗台高度按照片返工；10-25 西墙（隔断、门）改成磨砂玻璃，门带白框、执手、刷卡器 | 5 张照片 + 视频，SfM；家具和墙面按 SfM 相机反投影、三角化 | 估计：桌长 1.2 m、桌高 0.75 m、地毯 0.5 m 三个假设互相吻合 | 照片拟合 1.186 × 0.590 m，高 0.715 m（10-30，`fit_table.py`；之前是 1.20 × 0.62 × 0.68 的先验） | 未做：根 y=-0.73 按仿真能放下选定；升降柱各 0.0175 | 估计；10-30 RTX 按照片对色（调暗调暖） | 已装（10-30 对色版） |
| `real_scene2` 吧台 | `g1d_counter`、`g1d_counter_mobile` | 吧台落地柜；左侧客区（墙纸凹龛、两张圆桌、垃圾桶；椅子是布局里的刚体）；10-28 起过道里的高吧台和绿色岛台（静态，含 6 把吧凳、2 盏线形吊灯）、垂直于墙的天花板灯槽、按正面照片重排的右侧木柜墙 | 吧台 2 张照片联合拟合；客区 3 张照片另行拟合（`fit_lounge.py`）；岛台 4 张照片拟合，相机锚在吧台上（`fit_island.py`） | 估计：吧台和岛台只靠“微波炉宽 0.485 m、高 0.29 m”，客区只靠“圆桌高 0.75 m”；左木柱宽 1.0 m 由两组照片的下限估计 | 拟合，高 0.833 m | 未做：根 y=-0.80（底盘伸不进柜下） | 估计；10-30 RTX 按照片对色（顶灯 3 倍、5500 K） | 已装（10-30，带客区、岛台；`g1d_counter_mobile` 布局和 `lounge_chair` 未装） |
| `real_scene3` 实验室 | `g1d_lab` | 实验室白色折叠培训桌；房间里的工位桌、网椅、活动柜、文件柜、边桌、圆桌（10-22 起是立体模型，此前压平在墙面照片上）；墙面 10-23 起是纯色饰面加门、白板、电视，不再贴照片；角落停着一台静态 G1-D（URDF 烘焙） | 4 张照片联合拟合；家具按拟合相机反投影落地点摆放，尺寸取常见规格 | 估计：只靠“桌高 0.75 m”；长宽与标准 1.4 × 0.7 m 差约 2% | 拟合 1.369 × 0.686 m，高 0.749 m | 估计：假设站在无挡板一侧 | 拟合：按手机照片调色温和亮度比，未对真机相机 | 已装（10-22 换成带立体家具的版本） |
| `real_scene4` 学校实验室 | `g1d_school` | 学校实验室绿幕前的木色桌板 | 同事 MuJoCo 孪生：用真机腕部、头部画面拟合桌面四角；`build_scene.py` 换到底盘坐标系后拟合矩形 | 拟合：机器人相机标定 + URDF；无卷尺 | 拟合 1.065 × 0.596 m，离地 0.763 m（躯干下 0.082 m），四角残差 15 mm；桌腿未观测 | 拟合：桌沿在根节点前 0.19 m，桌心在右 0.083 m，底盘左偏 1.63°（按记录的躯干偏航） | 未做：照搬 real_scene3 | **未装** |
| `real_scene5` 深圳实验室 | `g1d_shenzhen_lab` | 白色升降办公桌，后沿贴墙，右侧金属储物柜；10-27 起整层房间：两根角柱、石膏吊顶 + 穿孔铝板天花和灯条、东墙 5 工位、升降桌、货架、5 把椅子、电梯厅栏杆和闸机 | 平台登记 + 5 张照片联合拟合（桌面尺寸固定为卷尺值）；房间按 8 张全景照片拟合（`fit_room.py`，柜子和插座定尺度） | **卷尺**：桌面 1.20 × 0.60 m，高 0.75 m，厚 2 cm；柜 75 × 120 × 45 cm（2026-10-21） | 卷尺值，`acceptance.py` 三项误差 0 | 未实测：根 y=-0.73（桌沿在前 0.13 m）、桌子中线，按物体落在 y≈[-0.55, -0.25] 选定；升降柱各 0.035 | 已记录（顶条面光源）：10-27 起按照片位置建 3 条灯条 RectLight + 5 个后吊顶筒灯 DiskLight，亮度按旧标定估算的照度换算，RTX 未重新对色；未对头部相机 | 已装（2026-10-27，`<gpu-server>` 共享库：`Room/real_scene5` 整层房间版 + `g1d_shenzhen_lab` 574 个布局，无量杯任务） |

- `real_scene1` 里的 5 个桌面物体（`usd/objects/`）没有被 Room 入口 `real_scene1.usd` 引用，评测时不加载。2026-09-21 真机运行时的桌子、沙发摆放与这个场景不同。
- `real_scene2` 自身有 6–7 个不兼容 GPU 的凸包，红豆落到这些表面会穿过去；`cam_third` 被场景里一块板挡住。10-28 起去掉了微波炉（现场已撤，尺度仍按它的假设尺寸），客区白色圆桌改为 4 把椅子。
- `real_scene4` 的升降柱保持 0：桌面相对躯干的高差按拟合值保留（0.082 m），不像其他三个场景那样统一成办公室的 0.165 m。
- `real_scene5` 是第一个按平台登记、用卷尺定尺度的场景；登记写“四条腿”“条纹瓷砖”，照片是两条 T 形升降腿和方块地毯，按照片建（见该目录 `PLAN.md`）。
- 摩擦系数在六个场景里都是估计值。
- 评测布局 `Assets/Eval_Layout/RoboDojo/g1d*/` 由 `build_asset.py` 生成。共享库三个配置都有拾取、叠块、壶倒杯、杯对杯四个任务的布局，缺 `g1d_pour_beans_measuring_cups`（要先装 `measuring_cup`）。

`real2sim/scene3_pour_pack/` 是另一套管线的倒水场景包（桌子实测 1.2 × 0.6 × 0.75 m，物体尺寸估计），`build_asset.py` 不安装它，只有 `water_cup` 的尺寸取自其中。

## 机器人 G1-D + Dex1-1

源文件 `real2sim/robot/g1d_v4/`（入库），`build_asset.py` 生成 `Assets/Robots/g1d/`。共享库已装 2026-10-17 带相机夹爪版。

| 项 | 等级 | 依据 |
| :-- | :-- | :-- |
| 运动学（关节原点和轴） | 官方 | 24 个公共关节与真机上的官方 `g1_d.urdf` 一致；v4 把腰部两个关节名互换了 |
| 头部相机内参 | 实测 | 棋盘格 fx ≈ 279.5，重投影 0.44–0.51 px；未建模畸变（< 2%） |
| 腕部相机内参 | 实测 | 棋盘格 fx ≈ 314；主点偏移（最多 17 px）未建模 |
| 头部、腕部相机外参 | 官方 + 拟合核对 | 宇树合同值；真机第 5/11 帧投影，夹爪轮廓误差 0.4–3.6 px，头部画面胸口上沿 −3 到 +2 px |
| 带相机夹爪外形 | 估计 | 项目成员在官方 CAD 上补的后部和安装脚（工程估计），惯量沿用原值 |
| 外观颜色 | 估计 | 按产品图和真机画面逐段目测核对；仿真胸口偏浅灰，真机偏深灰 |
| 手臂关节驱动 | 未做 | `env/robot_manager/robot_config/g1d.py`：刚度 4400、阻尼 40、力矩上限 100 |
| 夹爪力 | 官方额定值 | 每指 20 N；刚度 1e4 是为夹住细把手定的，开合时间和实际夹持力未测 |

## 待实测清单

按对 sim2real 的影响排序。量完把数值、日期和工具写进上面的“依据”，等级改成“实测”。

1. **三张桌子**（卷尺）：桌面离地高度、长、宽；机器人底盘前沿到桌沿的距离和左右偏移。写回：尺度用 `build_scene.py --scale-correction`，单个尺寸在 `fit_photo.py` 里把对应先验固定成实测值；站位改 `env_cfg/robot/g1d*.yml` 和 `build_asset.py` 的 `SCENES`。
2. **在用的杯子**（卡尺 + 厨房秤）：两只量杯、透明塑料杯、白色马克杯的高度、杯口外宽、底宽、壁厚、空杯质量。写回 `make_plastic_cups.py` / `make_water_cup.py` 顶部常量，再跑 `check_object.py`。
3. **摩擦**（斜面滑动角，μ ≈ tan θ）：杯子在桌面上、红豆在杯壁上。需要先让生成脚本写 `static_friction` / `dynamic_friction`，否则量了也不生效。
4. **红豆**：随机 20 粒的长宽厚、100 粒的质量。
5. **冷水壶、搪瓷杯**：先确认实验室有没有实物；没有就把“真实对应物”标为无实物。
6. **灯光**：每个场景用真机头部相机拍一张，和仿真同视角画面对比。
7. **机器人动力学**：关节阶跃和正弦响应、夹爪开合时间和夹持力（见 README TODO）。

## 维护规则

- 新增资产、改生成参数、量到实测值，都更新本表对应行，和代码放在同一个提交里。
- 实物照片放 `real2sim/objects/<类型>/<类别>/<编号>/reference/`（数采平台物体页也能上传到这里），仿真渲染由 `check_object.py --image-dir real2sim/objects` 生成到同一 id 的 `preview/`（不入库）。
- 装进或更新服务器共享资产库后改“共享库”一栏（流程见 README「改服务器共享资产库」）。
- 核对共享库：`ssh <gpu-server> 'ls /data/robodojo-assets/Assets/Object/Real2Sim/Rigid/<类别>'`。

### 裁纸刀 paper_knife（2026-10-30）

| ID | 对应物 / 生成脚本 | 尺寸与质量依据 | 外观 / 摩擦 / 验证 | 共享库 |
| :-- | :-- | :-- | :-- | :-- |
| Rigid/paper_knife/00000 | 实物裁纸刀；`scripts/objects/make_paper_knife.py` | 长152.88、最大宽39.44、中间宽32.83、厚17.14 mm；58.9 g 均实测 | 黄色 PP 外壳、开放钢导轨、分段刀片、黑色滑块、贯通挂孔；局部几何按照片拟合；背面/底面缺照片、纯色；摩擦默认，未标定；项目工厂落地/通道/阻挡/夹持代理/复位通过，未验机器人抓取 | 未装；研究资产待审核 |
| Rigid/paper_knife/00001 | 00000 颜色泛化；同脚本 | 几何和质量继承本体设计值，非新实测 | 青绿外壳、象牙色挂孔边框；rt_color 只换 model；独立工厂加载落地通过 | 未装；研究资产 |
| Rigid/paper_knife/00002 | 00000 颜色泛化；同脚本 | 同上 | 深蓝外壳、橙色挂孔边框；独立工厂加载落地通过 | 未装；研究资产 |
| Rigid/paper_knife/00003 | 00000 颜色泛化；同脚本 | 同上 | 紫色外壳、浅灰挂孔边框；独立工厂加载落地通过 | 未装；研究资产 |

补测来源：用户在 2026-10-30 对话提供厚度17.14 mm 并指示继续；确认人注明“用户对话补测；Codex记录”，不冒用原登记人的重新签名。新 revision `e249245a730f5980ca6d5b91285c3dad45f476e8a4ded6581e561aa9bbf0fdd4`。运行时换色保留 tail_rim / metal / slider，指令名称不引用外壳颜色。刀片圆孔旁浅压印已用独立照片贴图保留（label_blade_stamp）；金属磨损从实拍裁剪矫正，未猜写标识。
完整复现、名称和证据见 `objects/Rigid/paper_knife/00000/REPORT.md`。预览是中性灯光；照片机位为同侧近似匹配，未标定相机。夹持代理不是机器人场景成功或实物摩擦标定。共享库部署仍需用户确认。

## 湿巾袋实测研究候选 · 2026-10-30

| ID | 来源/尺寸/质量 | 外观/物理 | 共享库 |
| --- | --- | --- | --- |
| `Rigid/wipe_bag/00000` | 实拍实物；实测 190×96.02×38 mm、240.2 g | 三块照片贴图、独立盖外形；整包等效 FEM；缺底/背/开盖照片，端部与贴图外观待审核；材料/摩擦未标定 | 未装，研究资产 |
| `Rigid/wipe_bag/00001–00003` | 泛化；尺寸、形状、质量继承本体 | 绿色、红色、蓝色固定翻盖，膜颜色/粗糙度变化；bulk 可运行时换色，名称不写 bulk 颜色 | 未装，研究资产 |

生成 `scripts/objects/make_wipe_bag.py`，检查 `check_wipe_bag.py`，渲染 `render_wipe_bag.py`。原点为完整显示包围盒中心，米制 Z 上。完整来源、实跑证据、全部名称、SHA 与限制见 `prototypes/wipe_bag_measured/REPORT.md`。

## 蓝色超细纤维家政毛巾 · 2026-11-02

| ID | 来源/尺寸/质量 | 外观/物理/验证 | 共享库 |
| --- | --- | --- | --- |
| `Garment/microfiber_towel/00000` | 平台登记（3 张商品图，不是这件实物的实拍）；尺寸估计：商品标称 300×300 mm，未用卷尺量；质量实测 25 g（用户称重） | 商品图 3 的毛圈纹理（只取高频颗粒，随机拼贴）+ 5 mm 锁边带；角上布标未建模；PhysX 粒子布料（`GarmentObject`），961 节点；几乎不可拉伸（用户说明）：拉伸刚度 1e8、粒子摩擦 1.0，调出来的候选值，未标定 | 已装 `Object/Real2Sim/Garment`，候选 |
| `Garment/microfiber_towel/00001` | 本体的尺寸变体（`variant_of`），没有单独登记；商品标称 700×300 mm；质量 58.3 g（本体面密度 277.8 g/m² × 面积） | 同上，2201 节点 | 已装（同上），候选 |

生成 `scripts/objects/make_microfiber_towel.py`，检查 `check_cloth.py --physics-hz 300`（G1-D 的物理频率）：落桌 + 复位、摩擦夹片捏住一边提起 15 cm、对折（夹爪像手腕一样随弧线翻转，把 -X 两角放到 +X 两角上，然后夹片消失）。两件 drop / pinch / fold 和 pinch、fold 的新进程重放全部通过（2026-11-02，<gpu-server>）：质量读回 25.000 / 58.333 g；复位误差 0.1 mm；捏住一边悬垂时经纬线最大拉伸 6.4% / 6.1%（在夹口，p99 1.6% / 1.7%）、滑移 0；对折后两角离对面角 2–8 mm（重放 8 mm / 35 mm）、折后长度 0.98 倍半长、上层没有穿到下层、回弹 ≤ 1.5 mm。重放不逐位一致（夹片按速度驱动，GPU 接触有随机性）：悬垂摆动最大差 77 / 161 mm，对折 18 / 60 mm。平台柔性验收（本体）6 项通过，材料标定待验收。对折用的是**理想化释放**：从两层布之间抽出夹片会把上层整片带走 40–55 mm，抬起上夹片会把节点拉到 66% 应变（PhysX 粒子摩擦是基于位置的，夹片每步位移小时会粘住粒子），这是机器人任务的释放问题，接任务时要处理。没做：揉皱、擦拭、三角面自穿插审计、机器人抓取。

## GenieSim 公开物体库（2026-10-29）

公开源模型，不是实测孪生。来源 `agibot_world/GenieSimAssets`（ModelScope），revision `f65609e1c2bb153947ca9eee2ae1dc7392cdf3b7`，许可 CC BY-NC-SA 4.0；物体前缀 `Rigid/geniesim_`，导入脚本 `scripts/objects/geniesim.py`，复现命令见 README。

| 阶段 | 数量 | 依据 |
| :-- | --: | :-- |
| 源物体登记 | 1245 | objects/**/object_parameters.json |
| 无关节桌面候选 | 963 | 标注、米制尺寸及 3–50 cm 范围 |
| USD 转换通过 | 866 | 依赖、单刚体、碰撞体、真实 USD 包围盒、源质量/摩擦 |
| 基础落地与渲染通过、已装共享库 | 684 | 逐件三秒落地、正放/高度/速度判据、白底渲染 |
| 运行验证待复核、未装共享库 | 182 | 保留具体检查结果；自然倾斜也可能不满足正放判据 |
| 静态排除 | 97 | 质量超范围、多质量、缺依赖或尺寸问题 |

倒豆装豆检查（2026-11-03，`scripts/objects/check_cup_holds_beans.py`，结果 `real2sim/objects/cup_beans_check.json`）：共享库 138 件杯子固定住倒入任务红豆，128 件 ≥ 95% 装住；Real2Sim 60/60、RoboDojo 官方 26/27、GenieSim 12/18（`benchmark_beer_mug` 都建模了杯里的啤酒，杯腔只剩上面 1–7 cm）、SimReady 30/33（`dishware_mug_a01` 碰撞体封口，`kitchenware_cup_d01` 碰撞体没有杯壁）。SimReady 的杯子没有 `object.usd(z)`，任务加载器还读不到。

866 条已显示于平台「AgiBot GenieSim」分组。尺寸/质量/摩擦为源模型参数而非实测；抓取、容器、任务与真机一致性未验收。原始 checkout 必须保留，适配层引用其模型和材质；SDK 内置 MDL 在 Isaac Sim 运行时解析，不打包不完整的着色器副本。详细文件与验证入口见 HANDOFF 的 GenieSim 条目。

## Sand3D 公开刚体（2026-11-08）

公开泛化候选，不是实测孪生，也不是积木库。导入脚本 `scripts/objects/sand3d.py`。来源是 Hugging Face `Sandflow/sand3d-simready`（2 334 件）和 `Sandflow/sand3d-articraft`（9 209 件）里 `n_joints = 0` 的行；数据集文件 CC-BY-4.0，每件另保留 `source_license` 与 `attribution`。库代码 Apache-2.0（https://github.com/Sandflow-ai/sand3d）。`select` 不读取、不执行 `program` 列。

上游 USDA 为 Z 向上、米制，带刚体；官方导出不含质量、惯量、摩擦和碰撞体。`convert` 只收最长边 3–50 cm 的无关节物体，写成 `Assets/Object/RoboDojo/Rigid/sand3d_<family>/<编号>/object.usdz`：凸包碰撞（多于 64 顶点时收到 64），质量为凸包体积 × 250 kg/m³ 的估计值，`metadata.json` 的 `source` 写明估计方法和两级许可，`runtime_ready` 为 false。摩擦用加载器默认 0.6 / 1.5。有关节的行在 `select` 里排除。不写入 `Object/Blocks/`，不替换 88 件 `wood_block_*`。

2026-11-08 全量导入：两库共 1970 件无关节候选，转换 1389 件（580 件尺寸不在 3–50 cm，1 件无多边形网格）。`check_object.py` 3 s 落地 + 白底渲染：967 件通过，422 件待复核（412 件落地未静止，多为水果、蛋、球、扁平工具、锅；10 件白色细小物体在白底上渲染判不出）。1389 件都登记在平台「Sand3D」分组（`sand3d`，`twin` false，`asset_origin` public_dataset），带渲染和 `check.json`；只有 967 件通过的提交了资产库，待复核的不提交。中英三级名称在 `scripts/objects/object_names.json`：与 SimReady 同源的 830 件引用 `Rigid/simready_*` 的名称，其余 559 件按渲染图起名（上游只有漫反射色，名称按渲染外观，如上游 brown 纸箱渲染为白色）。运行目录 `<gpu-server>` / `<runs-dir>/sand3d-20261108/`。仓库不包含 USDA 或生成结果。CPU 单测：`scripts/objects/test_sand3d.py`。

## FMB 插块与装配（2026-11-09）

公开 CAD 转换，不是实测孪生，也不是积木库。来源 [Functional Manipulation Benchmark](https://functional-manipulation-benchmark.github.io/)（Luo et al., IJRR 2024）发布的 STEP 文件，CC-BY-4.0，每件 `metadata.json` 的 `source` 写数据集、许可和署名。转换脚本 `scripts/objects/fmb.py`（`fetch` 下 7 个 STEP，`convert` 写 `Assets/Object/FMB/Rigid/`，`names` 写名称表），CPU 单测 `scripts/objects/test_fmb.py`。加载器在 Blocks 之后、RoboDojo 之前找 `Object/FMB/`；平台上单独一个资产库「FMB」（`fmb`，[OpenFlywheel#17](https://github.com/Zheng-Chong/OpenFlywheel/pull/17) 的 `DIR_LIBS`）。

| 类别 | 内容 | 几何 | 质量 | 颜色 |
| :-- | :-- | :-- | :-- | :-- |
| `fmb_peg_<截面>`（9 类 × 48 件） | 截面 rectangle、round、oval、hexagon、arch、square_circle、double_square、three_prong、star；编号 `(尺寸 × 2 + 长度) × 8 + 颜色`，尺寸 large / medium / small，长度 100 / 150 mm；颜色 0 深红色（配色表写 brown，`#660000` 渲染出来是暗红）是基件，1–7 是它的颜色变体（`variant_of`） | 官方 CAD | 估计 16–175 g | 官方配色表：深红、红、浅红、黄、绿、浅蓝、深蓝、紫 |
| `fmb_peg_board/00000–00002` | 230 × 230 × 50 mm 插孔板，按孔大小排；`fmb.holes` 逐孔写对应插块、孔心（m，板坐标）、插入端、CAD 间隙和碰撞体间隙 | 官方 CAD | 估计 1134–1380 g | 灰 |
| `fmb_peg_fixture`、`fmb_board_fixture` | V 形插块翻转工装、L 形板翻转工装 | 官方 CAD | 估计 88 g、80 g | 灰 |
| `fmb_assembly_board/00000–00002` + `fmb_assembly_part/<a × 4 + k>` | 三套互锁装配：底板 + 4 个零件（板、条、销），按装配好的位姿给出；`fmb.assembly` 写每个零件和它在底板坐标里的中心 | 官方 CAD | 估计 | 底板灰，零件红 / 黄 / 绿 / 浅蓝 |

- 质量都是 CAD 体积 × 600 kg/m³ 的估计（FMB 没给材料），摩擦用加载器默认。插孔板和装配底板超过运行时的 0.5 kg 上限，加载时按 0.5 kg；要真实质量就在布局里设 `mass_cap`。
- 碰撞体按截面精确分解：沿棱柱轴切成截面不变的段，每段截面切成凸片再拉伸成凸包（每个 ≤ 64 点、每面 ≤ 32 点），体积为 CAD 的 0.989–1.004。孔的轮廓只往外放（0.3 mm 简化后外扩同样距离），碰撞体不会把孔变紧：CAD 每边间隙约 1.5 mm（拱形 1.17 mm），碰撞体间隙 1.22–1.65 mm。插孔板 408–441 个凸包，V 形工装 182 个。
- 多叶插块（square_circle、double_square、three_prong）立在实心握把上，叶片朝上；要绕 X 翻半圈、叶片一端先插（`inserted_end` 为 `top`）。
- 名称（大类 插块 / 板子 / 工装 / 零件 + 中英各 3 级）在 `scripts/objects/object_names.json`。

## 公开铰接物体（2026-11-17）

公开库的铰接物体，不是实测孪生。共用适配层 `scripts/objects/articulated.py`：原包原样放资产库 `sources/<库>/`（带上游 README、`UPSTREAM.json` 记 revision 和许可），每件一个文本层 `Assets/Object/RoboDojo/Articulation/<前缀>_<类别>/<编号>/object.usd` 用相对路径引用原包，在引用之上：去掉灯光、相机、BehaviorScript；停用把 link 钉在原始世界坐标上的世界固定关节；`ArticulationRootAPI` 移到被引用的根上、底座排成第一个子节点（`ArticulationObject.fix_root_link()` 的假设）；根自身的变换折进子节点，原点移到包围盒中心；没有质量的 link 补估计质量（整件质量按 link 包围盒体积分；源质量是占位值时连盒子惯量和质心一起显式写）；上下限写反的关节交换；被引用 prim 上方的关节体设置（如 `enabledSelfCollisions`）带过来。`metadata.json` 的 `body_joints` 写关节清单（名称、类型、`part`、`role`、父子 link、轴、限位、原驱动），写法同 `Articulation/microwave/00000`。`role` 只有 door / drawer / lid / knob / button / other，「能打开」只指门、抽屉（含空气炸锅炸篮、火柴盒内盒这类能拉出的容器）、真正能合上的翻盖（笔记本电脑屏幕、纸箱翻盖）；泵头、瓶盖、烤架和托盘、只能倾转的显示面板、风扇罩都是 `role: other`，部件写在 `part`（pump / cap / rack / tray / screen / cover …），约定见 `scripts/objects/SKILL.md`；`physics.fixed_base` 家电家具为 true、手持物品为 false（`ArticulationObject` 从 USD 读质量，不读 metadata）。检查 `scripts/objects/check_articulation.py`：走 `ArticulationObject`，摆在 (0.37, −0.21) 转 33°、离地 5 cm 固定底座，底座 link 与摆放位姿差 < 2 mm / 0.5°、静置 3 s 漂移 < 2 mm 且速度 < 5 mm/s；静置期间门、抽屉、翻盖离起始位置不超过 2° / 5 mm（`rest`，不会自己开，11-18 起计入通过）；门、抽屉、翻盖放到半开、按 USD 自己的驱动放 2 s 后不超过 2° / 5 mm（`hold`，停在哪就在哪；只报告不计入通过，ArtVIP 的回位弹簧本来就会把门关上）；每个关节驱动目标按斜坡推过上限再推过下限再回到起点（11-18 起：某个关节在门和翻盖保持原样时没走完，就把同件的门和翻盖开到离起点远的限位再扫一遍，两次有一次合格即合格；烤架、托盘因此不会被关着的门挡住，而 ArtVIP `oven/00001` 这种门在 0° 是放下打开、转到远端反而合上的，仍按第一遍判定），到达位置与标称限位差 ≤ 2° / 2 mm（或量程 3%），走完量程 ≥ 97%，回位误差同阈值，关节速度 < 50；没有限位的关节各转 90°；`metadata.json` 的 `size_check`（最长边在该类真实产品范围内）；初始姿态和各关节到远端限位两张渲染。没检查穿模和机器人抓取。逐件记录在 `real2sim/articulated/<库>.json`。

| 库 | 许可 | 候选 → 转换 → 通过 | 路径 / 平台资产库 | 资产库 |
| :-- | :-- | :-- | :-- | :-- |
| [ArtVIP](https://huggingface.co/datasets/X-Humanoid/ArtVIP) `d22f2095bc9db29fdd4b60f4a3c4f8177e1f5e9e`（`scripts/objects/artvip.py`） | Apache-2.0（数据集卡；仓库没有 LICENSE 文件，`sources/artvip/LICENSE` 是 Apache-2.0 原文）；可进 RSI 池 | 小家电 / 小家具 / 日用品 229 → 122 → **96** | `Object/RoboDojo/Articulation/artvip_*`；平台「ArtVIP 铰接物体」（`artvip`） | 已入库：修订 25（直连导入）；修订 26、28 只改 `metadata.json` 的 `role`/`part`（26：14 件按「能打开只指门、抽屉、翻盖」重标；28：14 件把只能动的 handle / turntable / rotor / pedal 写回各自 role） |
| [PartNet-Mobility USD](https://huggingface.co/datasets/AlbertS666/PartNet-Mobility-USD) `faf11263510b3d41264b0d7644ea9e7282226890`（社区转换，`scripts/objects/partnet_mobility.py`） | SAPIEN / ShapeNet 条款：**仅限非商用研究和教育**，只能给先同意同一条款的同事用；**不进 RSI 池** | 微波炉 16 → 16 → **16**；烤箱 29 → 3 → **3**；储物柜 346 → 135 → **134**（共 391 → 154 → 153） | `Object/RoboDojo/Articulation/partnet_{microwave,oven,storagefurniture}`；平台「PartNet-Mobility（非商用研究）」（`partnet_mobility`） | 已入库：修订 27（直连导入，`--note` 标「非商用研究，不进 RSI 池」，`sources/partnet_mobility/LICENSE` 写明条款）；修订 28 改 7 件 `metadata.json`（毫米行程的「门」记 button、会转的「按键」记 knob）；修订 33 补入 11-18 复查通过的 5 件 |
| [RoboCasa](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF) `420a04af939c34873e6839a586b70844baf28aab`（NVIDIA 发布的 MJCF，`scripts/objects/robocasa.py`） | CC BY 4.0（数据集卡；仓库没有 LICENSE 文件，`sources/robocasa/LICENSE` 是 CC BY 4.0 原文，署名在 `UPSTREAM.json`）；可进 RSI 池（用户 11-17 定） | 11-18：微波炉 50 → 50 → **50**；烤箱 25 → 21 → **17**；台式烤箱 50 → 47 → **47**；生成的桌面柜门 / 抽屉单元 48 → 48 → **48**（共 173 → 166 → 162）。11-19 其余 fixture（转换 → 通过）：搅拌机 25 → **25**、咖啡机 50 → **49**、洗碗机 25 → **15**、电水壶 25 → **23**、冰箱 50 → **50**、抽油烟机 25 → **22**、水槽 50 → **50**、厨师机 25 → **22**、炉灶 26 → **26**、灶台 24 → **23**、烤面包机 50 → **48**（共 375 → 353；窗户 50 件没有关节，不导）；可抓取物体 69 类 968 → 967 → **801**（64 类有通过件） | `Object/RoboDojo/Articulation/robocasa_*`（铰接 fixture，含生成的柜子单元）、`Object/RoboDojo/Rigid/robocasa_<类别>`（刚体物体，自包含 `object.usdz`）；平台「RoboCasa」（`robocasa`） | 11-18 部分已入库：修订 30（直连导入）；11-19 部分已入库：修订 40（直连导入运行时包，新增 4168 个文件）和修订 41（上游原包 `sources/robocasa/objects_lightwheel/`、`fixtures_lightwheel/` 的 11 类 fixture，新增 21350 个文件，更新 `UPSTREAM.json`）；平台 `robocasa` 共 1316 件（11-19 新登记 1154 件并发布预览和 `check.json`） |

- **ArtVIP 取舍**（`real2sim/articulated/artvip.json` 逐件写理由）：排除整类 39 件（桌子 28、座椅、落地灯、门、水龙头、书柜、拖把、花洒、饮水机、画架），大件超出桌边范围（最长边 > 0.9 m）29 件，模型尺度错误（订书机 1.75 m、纸箱 3.77 m 等，超出该类真实范围 1.5 倍）、关节挂在世界上（`oven_1` 的门）、只有钟表指针能动、关节重名各若干。保留的：最长边 ≤ 0.65 m 算桌面，≤ 0.9 m 算桌边。
- **ArtVIP 未通过 26 件**：关节到不了限位（被自身部件挡住：纸箱翻盖互相挡、钟表指针、夹子两臂相碰、垃圾桶踏板、推拉门差 6–8 mm 等）20 件，回位被挡 8 件，尺寸超出真实范围 4 件（咖啡机 0.64 m、电饭煲 0.53 m、两只 0.38 m 的按压瓶），可重叠计数。11-18 用新检查脚本（含 #344 的作用规则、需要时开门重扫）把尺寸合理的 22 件重跑，仍全部不通过，原因都与门无关：纸箱翻盖互相挡（7 件，到 70–85° / 120°，部分回位差 1–4°）、钟表指针被挡或回不到位（2）、夹子两臂相碰（2，到 7.6° / 10°、17.6° / 20°）、垃圾桶踏板到不了下限或回位差 3–6°（4）、提手回位差 3–7°（篮子、工具箱、垃圾桶）、床头柜两个抽屉关不回 9 cm 以内、推拉门差 6–8 mm、抽屉柜回位差 2 cm、工具箱门到 85.7° / 90°、提手到 87° / 90°；逐关节数值在 `real2sim/articulated/artvip.json` 的 `failing_joints`。
- **资产库修订**：25 ArtVIP 导入，26 ArtVIP 14 件只改 `role`/`part`，27 PartNet 导入，28 两库 21 件 `role`/`part` 对齐「铰接物体的关节作用」，29 是同一改动的重复导入（0 个文件变化，空修订，资产库修订删不掉）；35 RoboCasa 162 件补关节摩擦和电枢（只改 `metadata.json` 和 `sources/robocasa/usd/*/model.usdc`），36 ArtVIP 41 件门、抽屉、翻盖不再自己打开（只改 `metadata.json` 和 `object.usd`，见上）。
- **`asset_client.py pull` 漏拉带空格的引用**（11-17 Open Anything 采集会话发现）：`usd_refs()` 的正则 `@([^@\s]+)@` 不收含空格的路径，而 ArtVIP 原包目录名里有空格（`rice cooker`、`table fan`、`coffee machine` 等，7 件），按物体路径自动跟踪引用时会漏掉原包；直接点名 `sources/artvip/...` 拉取正常。没有改名：原包按上游路径原样保存，改名要重导 1 GB 原包、改写引用，还会让 `sources/` 和上游 revision 对不上；改为在网页仓库修正则（[OpenFlywheel#65](https://github.com/Zheng-Chong/OpenFlywheel/issues/65)），修好前集群作业对这 7 件把原包目录也写进清单。
- **ArtVIP 数值**：尺寸是源模型的（ArtVIP 本身米制、真实尺度）；质量为同类产品典型重量的估计（`artvip.py` 的 `MASS_KG`，如微波炉 12 kg、剪刀 0.08 kg）；碰撞体保留原包的 SDF 网格；关节驱动保留原包（多为 acceleration 型弹簧回位），只有一处例外：原包把不少门、抽屉、翻盖的驱动目标放在打开的限位上（笔记本屏幕 ±90°、文件夹 ±80°、纸箱翻盖 ±120°、空气炸锅炸篮 −0.3 m 等，刚度 10），静置时自己弹开（11-18 静置检查 34 件不合格）。适配层 `articulated.still_joints`（`artvip.py convert` 打开）对目标不在关闭位（0）的门、抽屉、翻盖把弹簧刚度置 0、保留阻尼，并给这些关节加 `PhysxJointAxisAPI` 库仑摩擦：水平铰取 1.5 × 部件重力对铰链的力矩（m·g·半个最长边），任意角度都停得住；竖直铰 0.3 N·m、滑轨 2 N，比手的力小得多。都是估计值，写进 `metadata.json` 关节的 `friction`（`source: estimated (still_joints)`）和 `drive.stiffness_source`（原刚度）。驱动目标是 0 的（微波炉门等）保留回位弹簧，`hold` 不合格是因为门会自己关上，不是打开。修订 36：41 件，复查 96 件静置全部合格，`airfryer/00000` 渲染失败、`filing_cabinet_drawers/00003` 抽屉加摩擦后扫不到限位，不进池。原包的 BehaviorScript（开门联动等）去掉了，垃圾桶踏板不会带动桶盖。
- **PartNet-Mobility**：社区 USD 在 `/Object/<类别>` 上统一缩放 0.5，模型是归一化尺寸；移动关节限位没跟着缩放；link 质量都是 0.3 kg 占位；浮动基座；`/Object` 上写着 `enabledSelfCollisions = false`。脚本按类别估计真实尺寸（微波炉、台式烤箱宽 0.50 m，矮柜高 0.70 m），移动关节行程同比例换算，绕 Z 转 90° 让正面朝 −Y，质量按类别估计（微波炉 12 kg、台式烤箱 8 kg、储物柜 15 kg）并显式写盒子惯量和质心（让 PhysX 从实例化凸包算时得到负惯量，关节一扫就数值发散），关节作用从 URDF 部件名读（door_frame、drawer_front、knob …）。取舍：烤箱里 24 件是落地灶台（高 ≥ 宽，或渲染目测带灶面：101908、101946、101947、102001、102018、102044），门在桌面以下够不到，排除；2 件还原后只有 6 / 14 cm 高，是灶面板，排除；储物柜按高 0.70 m（高宽比 ≥ 1.5 的衣柜类按 1.8 m）还原后最长边 > 0.9 m 的 211 件排除。未通过 6 件里，「全开」渲染空帧的 5 件（微波炉 00004，储物柜 00060、00065、00075、00110；物理检查本来就通过）11-18 用新检查脚本每作业先预热渲染重跑，全部通过，修订 33 入库；还剩柜门被挡到 93°（`partnet_storagefurniture/00063`，四扇门都是）1 件。名称按渲染主色、门抽屉数和厘米尺寸自动生成（`partnet_mobility.py names`），近重复模型最后用 PartNet 编号区分；颜色是近似。
- **RoboCasa 来源和许可**：RoboCasa365 的 Lightwheel fixture，由 NVIDIA 以 CC BY 4.0 发布在 [nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF)（按类别分包，commit `420a04af939c34873e6839a586b70844baf28aab`）。数据集卡没写作者、没提 Lightwheel，`rsi/AGENTS.md` 把 Lightwheel 自有资产列为 CC BY-NC；用户 11-17 定以 NVIDIA 这份发布为授权依据，可进 RSI 池（依据和区别在 `rsi/AGENTS.md` 第八节）。只用这份 MJCF，不用 Lightwheel SDK 的 USD 版和来源不明的 v0.2 旧 fixture。资产库 `sources/robocasa/`：通过件的上游目录原样（`fixtures_lightwheel/<类别>/<型号>/`，加柜子用的 `cabinets/cabinet_panels`、`handles`）、上游 README、`LICENSE`、`UPSTREAM.json`（revision、许可、署名），以及转换出的 `usd/`（可由前者重新生成）。
- **RoboCasa 转换**：没用 Isaac Sim 5.1 的 MJCF importer（丢 `springref`，把关节刚度阻尼按度写进驱动），`robocasa.py convert` 用 pxr 直接写：每个有关节的 body 一个刚体（无关节的 body 并进父 body），碰撞体是原 MJCF 手放的 box / cylinder 原语（guide，不显示），外观是 OBJ + 贴图；关节轴按 MJCF（有斜轴），有刚度或阻尼的关节写力驱动、目标为 `springref`（刚度阻尼从 MJCF 的每弧度换算成 USD 的每度）；`frictionloss`、`armature` 写成 PhysX 5.6 的逐轴关节参数 `PhysxJointAxisAPI:<angular|linear>`：`staticFrictionEffort` = `dynamicFrictionEffort` = `frictionloss`（两边都是库仑摩擦力矩 N·m / 力 N，单位相同，原值照搬），`armature` 原值（kg·m² / kg）；旧的 `physxJoint:jointFriction` 是乘在关节传递力上的无量纲系数，MuJoCo 没有对应量，不用。生成的柜门 / 抽屉单元没有 MJCF 关节，取同一发布里 Lightwheel fixture 最常见的一组（`frictionloss` 1、`armature` 0.01）。`metadata.json` 的 `body_joints[].friction` 记 `static` / `dynamic` / `armature`，`source.joint_friction` 写换算。11-18 修订 34 之前的版本没写这两项：台式烤箱下翻门（0.5–0.8 kg，重心约在铰链上方 10 cm，开 40° 时自重约 0.5 N·m，MJCF 摩擦 1 N·m 本可以托住）在 PhysX 里靠自重自己掉开，双抽屉柜的上层抽屉开局滑出 2–4 cm（Open Anything 试点发现）。`region` / `spawn` 辅助几何和电机执行器丢掉（都记在 `metadata.json` `source.dropped_from_mjcf`），再走 `articulated.py` 适配层。尺寸：台式烤箱按 RoboCasa 自己的注册表（robocasa @ `425684d3` `fixture_registry`，MIT）逐型号缩放到注册深度；微波炉注册的默认宽 0.8 m 是灶台上方槽位用的，不用，保持原尺寸；烤箱注册表没有尺寸，保持原尺寸。注册表自己剔掉的 7 件（Oven035/048/049/055，ToasterOven005/006/037）不导。质量按类估计（微波炉 12、烤箱 30、台式烤箱 6 kg）。关节作用：门 door、转盘 turntable、按键 button、旋钮 knob，烤架和托盘 other（`part: rack / tray`）。
- **RoboCasa 其余部分（11-19）**：同一 revision 的其余 12 类 fixture 和 `objects_lightwheel/` 的 69 类物体，授权依据不变（论文里来自 Objaverse 的物体不在这份发布里，没导）。fixture 走同一条 MJCF → USD → `articulated.py` 路线；新类别按各自真实尺寸范围判（`SIZE_M`，冰箱 1.2–2.2 m 等），不受「桌边 ≤ 0.9 m」限制（那条只管原来三类，保证已入库编号不变：重新 `select` 后 162 件编号和来源逐件一致）；质量按类估计（`MASS_KG`）。烤面包机的碰撞体类名是 `collision_1` … `collision_4`、碰撞体有网格，解析器和 `write_usd` 已支持（网格碰撞体取凸包，≤ 64 顶点）。物体是单刚体：`write_rigid` 写自包含 `object.usdz`（视觉网格 + 贴图打进包、原点在视觉包围盒中心、发布件自带的凸分解碰撞体逐个取凸包、`/World` 是刚体），质量 = 碰撞体凸包体积 × MJCF 密度 1000 kg/m³，截到 5 g–0.5 kg（估计值）；部分物体（奶锅）的子 body 带位姿，没有关节的 body 位姿并进几何。检查：铰接件 `check_articulation.py`（同上）；物体 `check_object.py` 落地（倾斜 < 5°、高度差 < 5 mm、速度 < 0.01 m/s）加单件渲染，通过的写 `runtime_ready: true`（`robocasa.py ready`）。未通过：洗碗机 10 件下层碗篮被开着的门挡住拉不满（门本身正常）；尺寸超出真实范围 12 件（抽油烟机 3、厨师机 3、烤面包机 2、电水壶 2、咖啡机 1、灶台 1）；物体 142 件按建模朝向放不稳会倒（餐具架、滤网、柠檬角、鸡腿、热狗面包、打蛋器、木勺、吸管等，其中滤网、打蛋器、洗碗刷、烤串、餐具架整类没有通过件），22 件碰撞体和外观高度对不上（烤盘悬空约 1 cm、搅拌杯外观陷进地面），2 件多次渲染空帧，1 件番茄片上游尺寸 5 m（比例错）。`check_object.py` 顺带改为渲染器冷启动返回空帧时多渲染几帧再取图（原来整批崩掉）。名称（`robocasa.names`）：fixture 同前；物体按 `OBJECT_NAMES` 的类别名 + 渲染主色（`OBJECT_COLOURS`，比家电多红橙黄绿蓝紫粉）+ 厘米尺寸，透明玻璃会叫成浅灰色。
- **RoboCasa 柜门和抽屉单元是生成的**：RoboCasa 的柜子是运行时按厨房布局程序化拼的，没有独立的柜子资产。`robocasa.py` 用同一数据集里的门板（`cabinets/cabinet_panels`，51 种）和把手（`handles`，51 种）生成 4 种桌面单元各 12 件：单门 36 × 30 × 42 cm、双门 56 × 30 × 42 cm、单抽屉 44 × 38 × 18 cm、双抽屉 44 × 38 × 32 cm（宽 × 深 × 高）。参数怎么定的：高 ≤ 42 cm、最长边 ≤ 0.65 m，是本库「桌面」的尺寸上限，放在桌上 G1-D 能够到门把和抽屉；板厚 16 mm、门缝 3 mm 取常见板式家具；门绕外侧竖直铰链开 0–105°，双门对开；抽屉行程为深度的 0.75（常见 3/4 拉出滑轨）；质量按板材体积 × 650 kg/m³（刨花板 / 中纤板）估计；门板和把手按单元编号确定性轮换（`(i·4 + k·13) mod 51`、`(i·4 + k·7) mod 51`），没有随机数，重跑得到同样的 48 件（`test_robocasa.py` 覆盖）。门板 OBJ 缩放到开口大小；把手竖装在门的开启边内 4 cm（双门在中缝两侧），抽屉上横装在面板中线以上 1/4 高处，过长时等比缩到面板边长的 80%。
- **RoboCasa 检查结果**：转换 166 件，通过 162；能打开（有门或抽屉）的就是全部 162 件：门关节 150 个（微波炉 50、烤箱 17、台式烤箱 47、柜子 36），抽屉 36 个；只能动的有旋钮 161、按键 48、转盘 50、烤架 80、托盘 8。未通过 4 件烤箱（Oven041/044/047/053）只差尺寸：最长边 0.38–0.46 m，小于真实嵌入式烤箱（0.5–0.85 m）；RoboCasa 运行时把烤箱缩放进柜槽，原尺寸不代表实物，没有可靠的缩放依据，不入库。第一轮检查里烤架、托盘只拉出 3 cm 左右，是被关着的门挡住了，`check_articulation.py` 因此改为先开门再扫；4 件柜子「全开」渲染空帧都是作业里的第一件（渲染器冷启动），单独重跑通过。名称按渲染主色、门 / 抽屉 / 旋钮 / 烤架数和厘米尺寸自动生成（`robocasa.py names`），撞名时依次加高、深、上游型号。

## Place Anything 支撑物适配（2026-11-18）

给 `g1d_place_anything`（把 X 放到 Y 上）补支撑物 Y：共享库里可商用、能当桌面支撑面、但任务加载器读不了的物体，做运行时适配层，原包和原物体不动。脚本 `scripts/objects/place_supports.py`（CPU 单测 `test_place_supports.py`），沿用 `simready.py` 的写法：`object.usd` 的 `/root` 放刚体和质量，`normalize` 把物体平放（Clutter 用它自己的 `active.place.up` 朝向，SimReady 把包围盒最薄的轴转到 +Z：砧板原包是立着的）并把包围盒中心移到原点，`asset` 引用源文件，在这一层去掉源里的刚体、质量和物理材质绑定。

- **NVIDIA SimReady**（CC-BY-4.0）：`RoboDojo/Rigid/simready_kitchenware_cuttingBoard_{a..l}01/00000`（砧板 12）、`simready_misc_notebook_a0{1..4}/00000`（笔记本 4）。包目录里加 `object.usd`、替换 `metadata.json`，`description.json`（名称）不变。质量：闭合网格体积 × 包内物理材质密度；网格不闭合或有几种材质（笔记本）时用包围盒体积 × 0.8 × 平均密度；都截到材质范围（木 0.15–0.5、纸 0.05–0.5 kg，上限是运行时的 0.5 kg，所以大砧板偏轻），全部是估计。碰撞体是原包的 SDF。
- **RoboDojo Clutter**（随仓库 MIT；杂物库源自 ManiTwin-100K，Apache-2.0）：新目录 `RoboDojo/Rigid/clutter_<类别>/<编号>/`，只有 `object.usd`（引用 `../../../Clutter/<类别>/<编号>/object.usdz`，不复制网格）、`metadata.json`、`description.json`（照抄 Clutter 的，换 uuid；名称在 `object_names.json` 里写成 Clutter id 的别名）。质量按同类实物手估（`place_supports.MASS`）。候选：书 `book/00000–00004`、桌游盒 `box/00001`、盘子 `plate/00000–00001`。
- `metadata.json`：`physics.size`、`mass`、`geometry`（含 `oriented_bbox`）、`active.place.up`，`passive.support` 写测到的支撑面（`{"top/0": {"center": [[x, y, z, 1, 0, 0, 0]], "radius": [r], "half": [[hx, hy]]}}`，物体坐标系，沿用已有的圆形写法再加最大内接矩形），`adapter` 记引用、平放旋转、质量依据和验证结果，三项检查都过才 `runtime_ready: true`。

验证（Slurm，运行目录 `<gpu-server>:~/openflywheel-runs/pl-sup/`，资产用集群旧整库副本 + 覆盖目录，因为资产隧道拉不动；结果在 `out/<批>/`）：① `check_object.py` 5 s 落地；② `check_cup_holds_beans.py` 高度图 + `place_anything.py pool` 量支撑面（Place Anything 作业 5962 的方法）；③ 只用这些支撑物当 Y、按 r1 的 L2 配方生成办公室布局，跑 Place Anything 专家 v2（`run_batch --no-trajectory`），按支撑物统计合格率和失败原因，规则同 r1 筛选（≥ 3 条、合格率 < 20% 剔除）。

| 来源 | 候选 | 落地 | 支撑面 | 放置筛选（合格 / 录制） | 入资产库 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Clutter 书 5、桌游盒 1 | 6 | 6 | 6 | 43/87（49%）：书 00000 7/15、00001 10/25、00002 3/6、00003 10/19、00004 2/7，桌游盒 11/15 | 6，修订 37 |
| Clutter 盘子 | 2 | 2 | 0：盘心只比盘沿低约 1 cm，`check_cup_holds_beans` 的腔体门槛是溢出高度下 1 cm，只剩几个格子 | — | 否 |
| SimReady 砧板 | 12 | 12 | 10：c01 两次结果不一致（一次腔体测不出平面，一次单独站立翻倒 119°），g01 把手槽区域测不出平面 | 长边 ≤ 32 cm 的 4 件：e01 4/10、j01 6/10、k01 4/16，i01 1/11 剔除；a01 b01 d01 f01 h01 l01 长 33–41 cm，超出桌面规则没筛 | e01 j01 k01，修订 38 |
| SimReady 笔记本 | 4 | 4 | 4 | a01 6/12、a02 6/11、a04 2/6，a03 2/11 剔除（失败 7/11 是抓空 X，和 a01/a02/a04 同一模型） | a01 a02 a04，修订 38 |

两轮筛选（作业 6912、6973，各 100 个布局）合格 49% / 36%，r1 支撑物筛选是 43%；失败是专家抓空（`holding`）和放下后判据不过（`not_placed`），**没有一次是 Y 被推走或碰翻**（只有 1 次碰动干扰物）。入库的 12 件 `runtime_ready: true`，作为 Place Anything 第 2 轮的 Y 候选（`docs/place-anything-r2-supports.md`）；没入库的（大砧板、两只盘子、c01、g01、i01、a03）适配层可由脚本重新生成，库里仍是原样。没做：摩擦实测（用加载器默认）、抓取 Y 本身。

## 2026-10-30 纳米胶实测研究候选

| ID | 对应实物/脚本 | 尺寸/质量 | 摩擦 | 外观/共享库 |
| --- | --- | --- | --- | --- |
| `Rigid/nano_glue/00000` | 纳米胶；`make_nano_glue.py` | 实测72.50/38.77/29.85mm、91.8g，网格误差≤0.04mm | 未实测，项目默认 | 实拍印字+卷绕端面；缺完整内壁及底面照片，完整外观未通过；研究候选，未装 |

| `Rigid/wooden_arch_block/00000` 木质拱形积木 | 杨开元实物与称量/尺寸确认 | `make_wooden_arch_block.py` | 实测60.10×29.72×30.04、冠厚9.58mm，网格同值 | 实测24.6g；惯量计算 | 默认，未知 | 照片木纹；背/右/底/内拱缺照片，外缘细节未验 | 新研究候选未装；共享旧60×30×40mm/25g模型保留；CPU夹持代理通过、GPU setter失败 |

<!-- paper_knife 2026-10-30: 当前登记62175d66重新核对，实测与照片不变；研究资产，未装共享库。署名：刘小凡（登记），Codex（建模复核）。 -->
