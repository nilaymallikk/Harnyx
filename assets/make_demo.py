#!/usr/bin/env python3
"""Record the Harnyx CLI demo and render it as a terminal GIF and MP4.

The command is executed for real in a pseudo-terminal on this machine; its
captured output drives the animation. Output: ``assets/demo.gif`` and
``assets/demo.mp4``.

Run:  python assets/make_demo.py
"""

from __future__ import annotations

import os
import pty
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
W, H = 900, 480
BAR = 34
BG = "#1e1e2e"
BAR_BG = "#181825"
FG = "#e5e5e5"
MUTED = "#a0a0a0"
GREEN = "#22c55e"
CYAN = "#06b6d4"
YELLOW = "#f59e0b"
FONT_CANDIDATES = [
    "/usr/share/fonts/jetbrains-mono-fonts/JetBrainsMono-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if Path(p).exists()), FONT_CANDIDATES[-1])
FS = 16
LH = 26
PAD = 22


def capture(cmd: list[str], cwd: Path) -> list[str]:
    """Run ``cmd`` in a pty and return its output split into lines."""
    master, slave = pty.openpty()
    proc = subprocess.Popen(cmd, cwd=cwd, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    chunks = []
    while True:
        try:
            data = os.read(master, 4096)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    proc.wait()
    os.close(master)
    text = b"".join(chunks).decode("utf-8", errors="replace")
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def _line_color(line: str) -> str:
    if "engineer reward" in line or "patched success" in line:
        return GREEN
    if "baseline success" in line:
        return YELLOW
    if "accepted versions" in line:
        return CYAN
    if "research run directory" in line:
        return MUTED
    return FG


def render(lines: list[str], typed: str = "", show_prompt: bool = True) -> Image.Image:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, FS)

    draw.rectangle([0, 0, W, BAR], fill=BAR_BG)
    for i, color in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        cx = 20 + i * 20
        draw.ellipse([cx - 6, BAR // 2 - 6, cx + 6, BAR // 2 + 6], fill=color)
    draw.text((W / 2 - 55, BAR / 2 - 9), "harnyx - zsh", font=font, fill=MUTED)

    y = BAR + PAD
    prompt = "~ "
    if show_prompt:
        draw.text((PAD, y), prompt, font=font, fill=CYAN)
        draw.text((PAD + draw.textlength(prompt, font=font), y), "harnyx " + typed, font=font, fill=FG)
        if not typed:
            cx = PAD + draw.textlength(prompt + "harnyx ", font=font) + 4
            draw.rectangle([cx, y, cx + 9, y + FS + 3], fill=GREEN)
    for line in lines:
        y += LH
        draw.text((PAD, y), line, font=font, fill=_line_color(line))
    return image


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        output = capture([sys.executable, "-m", "harnyx", "run", "--run-dir", "runs"], Path(tmp))
    if not output:
        print("no output captured", file=sys.stderr)
        return 1

    command = "run"
    frames: list[Image.Image] = []
    durations: list[int] = []

    for i in range(len(command) + 1):
        frames.append(render([], command[:i]))
        durations.append(70)
    durations[-1] = 350

    typed_line = "~ harnyx run"
    lines = [typed_line]
    frames.append(render(lines))
    durations.append(300)

    for line in output:
        lines.append(line)
        frames.append(render(lines))
        durations.append(160)
    durations[-1] = 1600

    gif = HERE / "demo.gif"
    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {gif} ({len(frames)} frames)")

    if shutil.which("ffmpeg"):
        mp4 = HERE / "demo.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(gif),
                "-movflags",
                "+faststart",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                str(mp4),
            ],
            check=True,
        )
        print(f"wrote {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
