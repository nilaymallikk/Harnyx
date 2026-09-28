#!/usr/bin/env python3
"""Render the Harnyx logo (transparent background, monochrome).

Two variants so it reads on either theme:

* ``logo.png`` / ``logo.svg``        white ink  (dark backgrounds)
* ``logo-light.png`` / ``logo-light.svg``  black ink (light backgrounds)

Run:  python assets/build_logo.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 1000, 340
BOLD = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
REG = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
MUTED_LIGHT = "#6b7280"
MUTED_DARK = "#9aa0a6"

CX, CY, R = 150, 170, 104
RING = 16
DOT = 11


def _mark(draw: ImageDraw.ImageDraw, ink: str) -> None:
    draw.ellipse([CX - R, CY - R, CX + R, CY + R], outline=ink, width=RING)
    for dx, dy in ((0, -R), (R, 0), (0, R), (-R, 0)):  # the four lifecycle hooks
        x, y = CX + dx, CY + dy
        draw.ellipse([x - DOT, y - DOT, x + DOT, y + DOT], fill=ink)
    draw.rounded_rectangle([CX - 40, CY - 46, CX - 22, CY + 46], radius=4, fill=ink)
    draw.rounded_rectangle([CX + 22, CY - 46, CX + 40, CY + 46], radius=4, fill=ink)
    draw.rounded_rectangle([CX - 40, CY - 9, CX + 40, CY + 9], radius=4, fill=ink)


def _spaced_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, spacing: int, fill: str) -> None:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + spacing


def render_png(name: str, ink: str, muted: str) -> None:
    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    _mark(draw, ink)
    _spaced_text(draw, (300, 82), "HARNYX", ImageFont.truetype(BOLD, 120), 16, ink)
    draw.text((304, 214), "harness optimization loop", font=ImageFont.truetype(REG, 30), fill=muted)
    image.save(HERE / name)


def render_svg(name: str, ink: str, muted: str) -> None:
    dots = "".join(
        f'<circle cx="{CX + dx}" cy="{CY + dy}" r="{DOT}" fill="{ink}"/>'
        for dx, dy in ((0, -R), (R, 0), (0, R), (-R, 0))
    )
    (HERE / name).write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <circle cx="{CX}" cy="{CY}" r="{R}" fill="none" stroke="{ink}" stroke-width="{RING}"/>
  {dots}
  <rect x="{CX - 40}" y="{CY - 46}" width="18" height="92" rx="4" fill="{ink}"/>
  <rect x="{CX + 22}" y="{CY - 46}" width="18" height="92" rx="4" fill="{ink}"/>
  <rect x="{CX - 40}" y="{CY - 9}" width="80" height="18" rx="4" fill="{ink}"/>
  <text x="300" y="180" fill="{ink}" font-family="DejaVu Sans, Arial, sans-serif" font-weight="bold"
        font-size="120" letter-spacing="16">HARNYX</text>
  <text x="304" y="240" fill="{muted}" font-family="DejaVu Sans, Arial, sans-serif"
        font-size="30" letter-spacing="1">harness optimization loop</text>
</svg>
""",
        encoding="utf-8",
    )


def main() -> None:
    render_png("logo.png", "#ffffff", MUTED_DARK)
    render_png("logo-light.png", "#111111", MUTED_LIGHT)
    render_svg("logo.svg", "#ffffff", MUTED_DARK)
    render_svg("logo-light.svg", "#111111", MUTED_LIGHT)
    print("wrote logo.{png,svg} and logo-light.{png,svg} (transparent)")


if __name__ == "__main__":
    main()
