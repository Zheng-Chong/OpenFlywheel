"""Compose a local review from accepted recordings only; never generate motion or tactile values."""
# ruff: noqa: E501 -- Embedded HTML/CSS is intentionally kept together.

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delivery", type=Path, required=True)
    a = p.parse_args()
    root = a.delivery
    motion, contact = root / "evidence/native07", root / "evidence/contact08"
    native = json.loads((motion / "report.json").read_text())
    tactile = json.loads((contact / "report.json").read_text())
    assert native["passed"] and tactile["passed"]
    rows = [json.loads(v) for v in (contact / "frames.jsonl").read_text().splitlines()]
    timeline = [json.loads(v) for v in (motion / "frames.jsonl").read_text().splitlines()]
    media = root / "media"
    media.mkdir(exist_ok=True)
    jobs = []
    for run in (motion, contact):
        for view in ("whole", "left", "right"):
            out = run / (view + ".mp4")
            jobs.append(
                subprocess.Popen(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-framerate",
                        "10",
                        "-i",
                        str(run / view / "%05d.jpg"),
                        "-c:v",
                        "libx264",
                        "-threads",
                        "2",
                        "-crf",
                        "20",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(out),
                    ]
                )
            )
    for job in jobs:
        if job.wait():
            raise RuntimeError("Video encoding failed")
    normal = np.linalg.norm([r["normal_force_world_N"] for r in rows], axis=-1)
    friction = np.linalg.norm([r["friction_force_world_N"] for r in rows], axis=-1)
    time = np.array([r["time_s"] for r in rows])
    names = ("little", "ring", "middle", "index", "thumb", "palm")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for side in range(2):
        for j, name in enumerate(names):
            axes[0, side].plot(time, normal[:, 6 * side + j], label=name)
            axes[1, side].plot(time, friction[:, 6 * side + j], label=name)
        axes[0, side].set_title(("Left" if side == 0 else "Right") + " hand: native filtered contacts")
        for i in range(2):
            axes[i, side].grid(alpha=0.2)
            axes[i, side].axvline(1.8, c="#777", ls="--", lw=0.8)
        axes[1, side].set_xlabel("Recording time (s): sphere / flat")
    axes[0, 0].set_ylabel("Normal force magnitude (N)")
    axes[1, 0].set_ylabel("Friction force magnitude (N)")
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(media / "contact_forces.png", dpi=160)
    plt.close(fig)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    montage = Image.new("RGB", (1280, 4 * 690), "#edf0f4")
    draw = ImageDraw.Draw(montage)
    for i, (frame, label) in enumerate(
        [
            (0, "Open reference"),
            (349, "Half-range command"),
            (374, "Endpoint command: self-contact stop"),
            (399, "Return and settle"),
        ]
    ):
        draw.text((20, i * 690 + 12), f"{label} | native07 frame {frame}", font=font, fill="#172033")
        for j, side in enumerate(("left", "right")):
            montage.paste(Image.open(motion / side / f"{frame:05d}.jpg"), (j * 640, i * 690 + 50))
    montage.save(media / "pose_review.jpg", quality=94)
    Image.open(motion / "whole/00000.jpg").save(media / "robot.png")
    short = [{k: r[k] for k in ("frame", "time_s", "phase")} for r in timeline]
    videos = "".join(
        f'<video id="v{i}" controls playsinline preload="metadata" src="evidence/native07/{v}.mp4"></video>'
        for i, v in enumerate(("whole", "left", "right"))
    )
    touch_videos = "".join(
        f'<video controls playsinline preload="metadata" src="evidence/contact08/{v}.mp4"></video>'
        for v in ("whole", "left", "right")
    )
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1-D · Inspire RH56DFTP 仿真核验</title><style>
body{{font:16px/1.7 system-ui,sans-serif;color:#172033;background:#f3f5f7;margin:0}}main{{max-width:1250px;margin:auto;padding:36px 24px}}h1{{font-size:38px;line-height:1.2;margin:14px 0}}h2{{font-size:24px;margin:32px 0 12px}}a{{color:#245cb6}}.muted{{color:#626d7b}}.card{{background:white;border:1px solid #dce1e7;border-radius:16px;padding:22px;margin:18px 0}}.hero{{display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:28px}}.hero img{{width:100%;border-radius:12px}}.stats,.trio{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.stats b{{font-size:32px;display:block}}.warn{{background:#fff5db;border-left:4px solid #c98c20;padding:15px 20px}}video{{width:100%;background:#e1e5eb;border-radius:10px}}input[type=range]{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #e3e7ec;text-align:left}}button{{border:1px solid #c1cad5;background:white;padding:10px 18px;border-radius:10px;cursor:pointer}}.wide{{width:100%;border-radius:10px}}details{{margin:18px 0}}@media(max-width:850px){{.hero,.trio{{grid-template-columns:1fr}}h1{{font-size:30px}}main{{padding:24px 16px}}}}
</style><main><div class="muted">仿真模型核验 · 2026.11.02 · 全程未连接真机</div><h1>G1-D + Inspire RH56DFTP</h1>
<section class="card hero"><div><p>完整 G1-D 本体，双侧厂家灵巧手。保留腰部、升降、双臂、碰撞与头部双目配置。</p><div class="stats"><div><b>26</b>动作维度</div><div><b>12</b>原生联动约束</div><div><b>51</b>整机 links</div></div><p>已完成来源核对、腕端方向核查、逐轴有限运动、接触测试、控制接口与数据导出。</p><p><a href="http://127.0.0.1:8786/#robots">打开本地本体管理</a> · <a href="repo_overlay/real2sim/embodiments/g1d_inspire_rh56dftp/preview/model.glb">下载整机 GLB 预览</a> · <a href="README.md">查看复现说明</a></p></div><img src="media/robot.png" alt="native07 第0帧的完整机器人"></section>
<p class="warn">验收边界：有限运动与回位通过；同步机械端点目标出现自接触，最大目标差 {native["endpoint_errors_rad"]["closed_range"]:.3f} rad，未标为全范围自由运动通过。G1-D 孔位、转接板、实测 TCP、当前设备零位和触觉标定仍待实测。</p>
<h2>同一录制的三视角运动</h2><p class="muted">400 帧 / 40 秒，CPU PhysX + GPU 渲染，物理步长 1/480 秒。两个手部视角与整机视角取自同一步状态。</p><div class="trio">{videos}</div><section class="card"><button id="play">同步播放 / 暂停</button> <span id="phase">第0帧</span><input id="seek" type="range" min="0" max="399" value="0"><small class="muted">拖动按帧核对。机械零位下拇指姿态来自厂家模型，并非实测自然张手。</small></section>
<h2>实际观测到的姿态</h2><details><summary>展开左右手：初始、半程、闭合受限、回位</summary><img class="wide" src="media/pose_review.jpg" alt="同一native07录制的四个明确标注帧"></details>
<h2>接触、摩擦与卸载</h2><p>球头和平头对左右各五个远端指节与掌心逐步压入，并作切向移动。12/12 位置两种形状均有原生接触，卸载后记录力为 0 N，接触与反力核对误差为 {tactile["force_opposition_max_N"]:.1g} N。红色夹具为测试工具。</p><div class="trio">{touch_videos}</div><img class="wide" src="media/contact_forces.png" alt="同帧原生法向力与摩擦力曲线"><p class="warn">这里是原生刚体接触力，不是厂家 1062 点触觉阵列。压阻原始 ADC 与电容模块已分别实现离线解码；厂家未提供的触点坐标、压强/力标定、胶体 FEM 响应均未伪造。</p>
<h2>核验结果</h2><section class="card"><table><tr><th>项目</th><th>结果</th></tr><tr><td>整机 FK / 原生物理位置</td><td>最大差 {native["fk_max_error_m"] * 1000:.6f} mm</td></tr><tr><td>联动关系</td><td>最大差 {native["mimic_max_error_rad"]:.6f} rad</td></tr><tr><td>有限逐轴与回位</td><td>最大稳态差 {native["bounded_axis_error_rad"]:.6f} rad</td></tr><tr><td>控制接口</td><td>26 维实际 RobotManager 队列、单位/顺序、边界拒绝与观测检查通过</td></tr><tr><td>实体安装 / 实测标定 / 完整抓取任务</td><td>未验证；本包不作通过声明</td></tr></table></section>
<h2>文件与证据</h2><section class="card"><p><a href="datasets/native_motion.h5">运动 HDF5</a> · <a href="datasets/native_contacts.h5">接触 HDF5</a> · <a href="evidence/native07/trajectory.npz">完整关节轨迹</a> · <a href="evidence/contact08/forces.npz">接触力</a></p><p><a href="evidence/native07/report.json">运动报告</a> · <a href="evidence/contact08/report.json">接触报告</a> · <a href="validation/framework05.json">控制链路</a> · <a href="validation/model_audit.json">来源与坐标核对</a> · <a href="validation/delivery_audit.json">只读交付核对</a></p><p>厂家原件与摘要在 <a href="vendor/source_manifest.json">vendor/source_manifest.json</a>；失败尝试保留在 failures/。源文件与录制输入分别冻结，运动与接触是独立实验，没有拼接成一次任务成功。</p></section>
</main><script>const timeline={json.dumps(short)};const vs=[0,1,2].map(i=>document.getElementById('v'+i)),seek=document.getElementById('seek'),phase=document.getElementById('phase');const labels={{open:'初始参考',half_range:'半程目标',closed_range:'机械端点目标／自接触',return_open:'回位稳定'}};function show(f){{f=Math.max(0,Math.min(399,f));seek.value=f;phase.textContent='第 '+f+' 帧 · '+(labels[timeline[f].phase]||timeline[f].phase);}}seek.oninput=()=>{{vs.forEach(v=>v.currentTime=seek.value/10);show(+seek.value)}};document.getElementById('play').onclick=()=>{{if(vs[0].paused)vs.forEach(v=>v.play());else vs.forEach(v=>v.pause())}};vs[0].ontimeupdate=()=>{{let t=vs[0].currentTime;vs.slice(1).forEach(v=>{{if(Math.abs(v.currentTime-t)>.15)v.currentTime=t}});show(Math.floor(t*10+1e-6))}};show(0);</script></html>"""
    (root / "index.html").write_text(html)


if __name__ == "__main__":
    main()
