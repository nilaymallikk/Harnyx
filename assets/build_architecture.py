#!/usr/bin/env python3
"""Render the Harnyx architecture diagram.

Single source of truth for three outputs, so they never drift:

* ``architecture.excalidraw`` — editable Excalidraw scene (dark background)
* ``architecture.svg``        — vector image (crisp on GitHub)
* ``architecture.png``        — raster image at 2x (required for PyPI, which
                                does not render SVG)

Run:  python assets/build_architecture.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SCALE = 2
W, H = 880, 560
BG = "#1e1e2e"
INK = "#e5e5e5"
MUTED = "#a0a0a0"
GREEN = "#22c55e"
LABEL = 14
FONT = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"

# id, x, y, w, h, label, fill
NODES: list[tuple[str, int, int, int, int, str, str]] = [
    ("task", 20, 95, 120, 60, "Task batch", "#1e3a5f"),
    ("agent", 180, 82, 200, 76, "HarnessedAgent", "#1a4d4d"),
    ("harness", 420, 82, 190, 76, "ExecutableHarness\n4 hooks", "#2d1b69"),
    ("env", 650, 95, 140, 60, "Environment", "#5c3d1a"),
    ("traj", 650, 232, 140, 60, "Trajectory", "#1e3a5f"),
    ("packet", 450, 232, 150, 60, "FailurePacket", "#1a4d4d"),
    ("eng", 170, 222, 200, 76, "HarnessEngineer\n(LLM)", "#2d1b69"),
    ("validate", 170, 392, 200, 76, "PatchValidator\n+ Sandbox", "#5c3d1a"),
    ("eval", 410, 392, 180, 76, "rerun same tasks", "#1e3a5f"),
    ("reward", 630, 392, 160, 76, "OutcomeReward\n+ select", "#1a4d2e"),
]

# id, x, y, text, size, color
TEXTS: list[tuple[str, int, int, str, int, str]] = [
    ("title", 20, 8, "Harnyx - Architecture", 24, INK),
    ("sub", 20, 46, "learn to edit executable harnesses from failure trajectories", 13, MUTED),
    ("frozen", 180, 168, "frozen policy - weights never edited", 12, MUTED),
    ("sandbox", 170, 480, "AST policy - no imports / IO / network", 12, MUTED),
]

# id, points, color, dashed, label, label_xy
EDGES: list[tuple[str, list[tuple[int, int]], str, bool, str, tuple[int, int] | None]] = [
    ("a1", [(140, 125), (180, 125)], MUTED, False, "", None),
    ("a2", [(380, 120), (420, 120)], MUTED, False, "", None),
    ("a3", [(610, 120), (650, 120)], MUTED, False, "", None),
    ("a4", [(700, 155), (700, 232)], MUTED, False, "", None),
    ("a5", [(650, 262), (600, 262)], MUTED, False, "", None),
    ("a6", [(450, 262), (370, 262)], MUTED, False, "", None),
    ("a7", [(270, 298), (270, 392)], MUTED, False, "", None),
    ("a8", [(370, 430), (410, 430)], MUTED, False, "", None),
    ("a9", [(590, 430), (630, 430)], MUTED, False, "", None),
    # Feedback: around the right-hand margin, over the top, into the harness.
    (
        "a10",
        [(790, 430), (838, 430), (838, 40), (515, 40), (515, 82)],
        GREEN,
        True,
        "accepted patch (harness-vN)",
        (588, 16),
    ),
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size * SCALE)


def _segments(points: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def _draw_label_lines(draw: ImageDraw.ImageDraw, cx: float, cy: float, label: str, size: int, color: str) -> None:
    font = _font(size)
    lines = label.split("\n")
    heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    line_h = (max(heights) + 4) * SCALE
    total = line_h * len(lines)
    y = cy * SCALE - total / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text((cx * SCALE - (bbox[2] - bbox[0]) / 2, y - bbox[1]), line, font=font, fill=color)
        y += line_h


def render_png() -> None:
    image = Image.new("RGB", (W * SCALE, H * SCALE), BG)
    draw = ImageDraw.Draw(image)

    for _, x, y, w, h, label, fill in NODES:
        draw.rounded_rectangle(
            [x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE],
            radius=10 * SCALE,
            fill=fill,
            outline=INK,
            width=2 * SCALE,
        )
        _draw_label_lines(draw, x + w / 2, y + h / 2, label, LABEL, INK)

    for _, x, y, text, size, color in TEXTS:
        draw.text((x * SCALE, y * SCALE), text, font=_font(size), fill=color)

    for _, points, color, dashed, label, label_xy in EDGES:
        for (x1, y1), (x2, y2) in _segments(points):
            if dashed:
                _dashed_line(draw, (x1 * SCALE, y1 * SCALE), (x2 * SCALE, y2 * SCALE), color)
            else:
                draw.line((x1 * SCALE, y1 * SCALE, x2 * SCALE, y2 * SCALE), fill=color, width=2 * SCALE)
        _arrowhead(draw, points[-2], points[-1], color)
        if label and label_xy:
            draw.text((label_xy[0] * SCALE, label_xy[1] * SCALE), label, font=_font(13), fill=color)

    image.save(HERE / "architecture.png")


def _dashed_line(draw: ImageDraw.ImageDraw, start, end, color, dash=12 * SCALE, gap=8 * SCALE) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        seg = min(dash, length - pos)
        draw.line(
            (start[0] + ux * pos, start[1] + uy * pos, start[0] + ux * (pos + seg), start[1] + uy * (pos + seg)),
            fill=color,
            width=2 * SCALE,
        )
        pos += dash + gap


def _arrowhead(draw: ImageDraw.ImageDraw, prev, tip, color) -> None:
    dx, dy = tip[0] - prev[0], tip[1] - prev[1]
    length = math.hypot(dx, dy) or 1
    ux, uy = dx / length, dy / length
    size = 9
    left = ((tip[0] - ux * size - uy * size / 1.6) * SCALE, (tip[1] - uy * size + ux * size / 1.6) * SCALE)
    right = ((tip[0] - ux * size + uy * size / 1.6) * SCALE, (tip[1] - uy * size - ux * size / 1.6) * SCALE)
    draw.polygon([(tip[0] * SCALE, tip[1] * SCALE), left, right], fill=color)


def _svg() -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    out.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    out.append(
        '<defs><marker id="ah" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">'
        f'<path d="M0,0 L10,4 L0,8 z" fill="{MUTED}"/></marker>'
        '<marker id="ahg" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">'
        f'<path d="M0,0 L10,4 L0,8 z" fill="{GREEN}"/></marker></defs>'
    )
    for _, x, y, w, h, label, fill in NODES:
        out.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{INK}" stroke-width="2"/>'
        )
        lines = label.split("\n")
        start = y + h / 2 - (len(lines) - 1) * (LABEL * 0.62)
        for i, line in enumerate(lines):
            out.append(
                f'<text x="{x + w / 2}" y="{start + i * LABEL * 1.25 + LABEL * 0.35}" fill="{INK}" '
                f'font-family="DejaVu Sans, sans-serif" font-size="{LABEL}" text-anchor="middle">{_esc(line)}</text>'
            )
    for _, x, y, text, size, color in TEXTS:
        out.append(
            f'<text x="{x}" y="{y + size}" fill="{color}" font-family="DejaVu Sans, sans-serif" '
            f'font-size="{size}">{_esc(text)}</text>'
        )
    for _, points, color, dashed, label, label_xy in EDGES:
        d = " ".join(("M" if i == 0 else "L") + f"{px},{py}" for i, (px, py) in enumerate(points))
        marker = "ahg" if color == GREEN else "ah"
        dash = ' stroke-dasharray="8 5"' if dashed else ""
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{dash} marker-end="url(#{marker})"/>')
        if label and label_xy:
            out.append(
                f'<text x="{label_xy[0]}" y="{label_xy[1] + 12}" fill="{color}" '
                f'font-family="DejaVu Sans, sans-serif" font-size="13">{_esc(label)}</text>'
            )
    out.append("</svg>")
    return "\n".join(out)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _base_element(eid: str, etype: str, x: int, y: int, w: int, h: int, rng: random.Random) -> dict:
    return {
        "id": eid,
        "type": etype,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": INK,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "roundness": None,
        "seed": rng.randint(1, 10**6),
        "version": 1,
        "versionNonce": rng.randint(1, 10**6),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
    }


def _excalidraw() -> dict:
    rng = random.Random(7)
    elements: list[dict] = []
    bg = _base_element("darkbg", "rectangle", -4000, -3000, 10000, 7500, rng)
    bg.update({"strokeColor": "transparent", "backgroundColor": BG, "strokeWidth": 0, "roughness": 0})
    elements.append(bg)

    for nid, x, y, w, h, label, fill in NODES:
        rect = _base_element(nid, "rectangle", x, y, w, h, rng)
        rect.update({"backgroundColor": fill, "roundness": {"type": 3}, "label": {"text": label, "fontSize": LABEL}})
        elements.append(rect)
    for tid, x, y, text, size, color in TEXTS:
        el = _base_element(tid, "text", x, y, int(len(text) * size * 0.5), int(size * 1.25), rng)
        el.update(
            {
                "strokeColor": color,
                "fontSize": size,
                "fontFamily": 1,
                "text": text,
                "textAlign": "left",
                "verticalAlign": "top",
                "containerId": None,
                "originalText": text,
                "lineHeight": 1.25,
            }
        )
        elements.append(el)
    for eid, points, color, dashed, label, _ in EDGES:
        ox, oy = points[0]
        el = _base_element(eid, "arrow", ox, oy, points[-1][0] - ox, points[-1][1] - oy, rng)
        el.update(
            {
                "strokeColor": color,
                "strokeStyle": "dashed" if dashed else "solid",
                "roundness": {"type": 2},
                "points": [[px - ox, py - oy] for px, py in points],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": "arrow",
                "label": {"text": label, "fontSize": 13} if label else None,
            }
        )
        elements.append(el)
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "harnyx/build_architecture.py",
        "elements": elements,
        "appState": {"viewBackgroundColor": BG, "gridSize": None},
        "files": {},
    }


def main() -> None:
    render_png()
    (HERE / "architecture.svg").write_text(_svg(), encoding="utf-8")
    (HERE / "architecture.excalidraw").write_text(json.dumps(_excalidraw(), indent=2), encoding="utf-8")
    print("wrote architecture.png / architecture.svg / architecture.excalidraw")


if __name__ == "__main__":
    main()
