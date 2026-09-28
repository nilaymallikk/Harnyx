#!/usr/bin/env python3
"""Render the Harnyx logo (monochrome: white ink on black).

Outputs ``assets/logo.png`` (for PyPI, which does not render SVG) and
``assets/logo.svg`` (crisp on GitHub), from one spec.

Run:  python assets/build_logo.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 1000, 340
BG = "#0b0b0f"
INK = "#ffffff"
MUTED = "#9aa0a6"
BOLD = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
REG = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"

CX, CY, R = 150, 170, 104
RING = 16
DOT = 11


def _mark(draw: ImageDraw.ImageDraw) -> None:
    box = [CX - R, CY - R, CX + R, CY + R]
    draw.ellipse(box, outline=INK, width=RING)
    for dx, dy in ((0, -R), (R, 0), (0, R), (-R, 0)):  # the four lifecycle hooks
        x, y = CX + dx, CY + dy
        draw.ellipse([x - DOT, y - DOT, x + DOT, y + DOT], fill=INK)
    # Bold "H": two bars plus a crossbar through the core.
    draw.rounded_rectangle([CX - 40, CY - 46, CX - 22, CY + 46], radius=4, fill=INK)
    draw.rounded_rectangle([CX + 22, CY - 46, CX + 40, CY + 46], radius=4, fill=INK)
    draw.rounded_rectangle([CX - 40, CY - 9, CX + 40, CY + 9], radius=4, fill=INK)


def _spaced_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, spacing: int, fill: str) -> None:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + spacing


def render_png() -> None:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    _mark(draw)
    _spaced_text(draw, (300, 82), "HARNYX", ImageFont.truetype(BOLD, 120), 16, INK)
    draw.text((304, 214), "harness optimization loop", font=ImageFont.truetype(REG, 30), fill=MUTED)
    image.save(HERE / "logo.png")


def render_svg() -> str:
    dots = "".join(
        f'<circle cx="{CX + dx}" cy="{CY + dy}" r="{DOT}" fill="{INK}"/>'
        for dx, dy in ((0, -R), (R, 0), (0, R), (-R, 0))
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect width="{W}" height="{H}" fill="{BG}"/>
  <circle cx="{CX}" cy="{CY}" r="{R}" fill="none" stroke="{INK}" stroke-width="{RING}"/>
  {dots}
  <rect x="{CX - 40}" y="{CY - 46}" width="18" height="92" rx="4" fill="{INK}"/>
  <rect x="{CX + 22}" y="{CY - 46}" width="18" height="92" rx="4" fill="{INK}"/>
  <rect x="{CX - 40}" y="{CY - 9}" width="80" height="18" rx="4" fill="{INK}"/>
  <text x="300" y="180" fill="{INK}" font-family="DejaVu Sans, Arial, sans-serif" font-weight="bold"
        font-size="120" letter-spacing="16">HARNYX</text>
  <text x="304" y="240" fill="{MUTED}" font-family="DejaVu Sans, Arial, sans-serif"
        font-size="30" letter-spacing="1">harness optimization loop</text>
</svg>
"""


def main() -> None:
    render_png()
    (HERE / "logo.svg").write_text(render_svg(), encoding="utf-8")
    print("wrote logo.png / logo.svg")


if __name__ == "__main__":
    main()
