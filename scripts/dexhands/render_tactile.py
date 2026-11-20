"""Render recorded geometric depth maps and native solver force traces; no generated contact data."""

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
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(r) for r in (a.run / "frames.jsonl").read_text().splitlines()]
    forces = np.linalg.norm([r["force_world_N"] for r in rows], axis=-1)
    t = np.array([r["time_s"] for r in rows])
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for side, ax in enumerate(axs):
        for j, finger in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
            ax.plot(t, forces[:, side * 5 + j], label=finger)
        ax.set_ylabel(("Left" if side == 0 else "Right") + " force (N)")
        ax.grid(alpha=0.2)
        for x in [t[10], t[20]]:
            ax.axvline(x, color="#666", lw=0.8, ls="--")
    axs[0].legend(ncol=5)
    axs[1].set_xlabel("Recorded simulation time (s): sphere / flat / cylinder")
    fig.tight_layout()
    fig.savefig(a.output / "force_trace.png", dpi=160)
    plt.close(fig)
    if not (a.run / "tactile.npz").exists():
        return
    depth = np.load(a.run / "tactile.npz")["depth_m"]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    cmap = matplotlib.colormaps["magma"]
    directory = a.output / "depth_frames"
    directory.mkdir(exist_ok=True)
    for i, frame in enumerate(depth):
        canvas = Image.new("RGB", (1200, 560), "#f3f4f6")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (18, 12),
            f"Sharpa geometric depth | frame {i} | {rows[i]['shape']} / {rows[i]['phase']}",
            font=font,
            fill="#111827",
        )
        for k, d in enumerate(frame):
            rgb = (cmap(np.clip(d / 0.0008, 0, 1))[..., :3] * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb), (k % 5 * 240, 50 + k // 5 * 254))
            draw.text(
                (k % 5 * 240 + 8, 51 + k // 5 * 254),
                ("L" if k < 5 else "R") + " " + ["thumb", "index", "middle", "ring", "pinky"][k % 5],
                font=font,
                fill="white",
            )
        draw.rectangle((0, 540, 1200, 560), fill="#f3f4f6")
        draw.text(
            (18, 538),
            "Fixed scale: black = 0 mm; bright = 0.8 mm. Geometric depth, not force or optical RGB.",
            font=font,
            fill="#111827",
        )
        canvas.save(directory / f"{i:05d}.png")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-framerate",
            "10",
            "-i",
            str(directory / "%05d.png"),
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(a.output / "depth.mp4"),
        ],
        check=True,
    )
    import shutil

    shutil.copyfile(directory / "00005.png", a.output / "depth_preview.png")
    shutil.rmtree(directory)


if __name__ == "__main__":
    main()
