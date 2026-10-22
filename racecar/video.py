"""
Video helpers for Steps 4 and 7: high-quality MP4 writing, caption bars, heat-map overlays and side-by-side layouts.
Every clip is made by replaying a recorded race exactly (racecar.attention_frames.race_moments) and checking the
replay against the recorded result.
"""

from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from racecar import plot_style

FPS = 50            # every game frame, real time (the game runs at 50 frames per second)
CRF = 16            # H.264 quality (lower = better; 16-18 is visually lossless)
HEADER = 64         # caption bar height in pixels
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
HEAT_COLORS = ["#000000", "#7a1e00", "#eb6834", "#ffd166", "#ffffff"]
_HEAT_LUT = (matplotlib.colors.LinearSegmentedColormap.from_list("heat", HEAT_COLORS)(np.linspace(0, 1, 256))[:, :3]
             * 255).astype(np.float32)


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def writer(path, fps=FPS):
    """H.264 MP4, visually lossless, plays everywhere (yuv420p)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                              ffmpeg_params=["-crf", str(CRF), "-preset", "slow", "-movflags", "+faststart"])


def caption(width, title, subtitle="", color=plot_style.INK, height=HEADER):
    """A light caption bar: bold title (in the agent's color) and a grey subtitle."""
    img = Image.new("RGB", (width, height), plot_style.SURFACE)
    d = ImageDraw.Draw(img)
    d.text((18, 8), title, font=font(24, bold=True), fill=color)
    if subtitle:
        d.text((18, 38), subtitle, font=font(17), fill=plot_style.INK_SECONDARY)
    return np.asarray(img)


def label(picture, text, xy=(18, 14), size=26):
    """Text with a dark backing box, drawn onto a picture (e.g. "race over")."""
    img = Image.fromarray(picture)
    d = ImageDraw.Draw(img)
    f = font(size, bold=True)
    box = d.textbbox(xy, text, font=f)
    d.rectangle((box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6), fill=(0, 0, 0))
    d.text(xy, text, font=f, fill=(255, 255, 255))
    return np.asarray(img)


def heat_overlay(picture, heat, top):
    """Blend a 64x64 heat map onto a picture of any size (the AI's view covers the whole picture)."""
    h, w = picture.shape[:2]
    m = np.clip(cv2.resize(heat.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) / top, 0, 1)
    colors = _HEAT_LUT[(m * 255).astype(np.uint8)]
    alpha = (0.85 * m ** 0.9)[..., None]          # no heat = picture unchanged
    return ((1 - alpha) * picture + alpha * colors).astype(np.uint8)


def ai_view_big(view64, size):
    """The AI's 64x64 view, enlarged with hard pixel edges so its real resolution stays visible."""
    return cv2.resize(view64, (size, size), interpolation=cv2.INTER_NEAREST)


def side_by_side(*panels, gap=8):
    """Panels of equal height next to each other, with a thin gap in the page color."""
    h = max(p.shape[0] for p in panels)
    spacer = np.full((h, gap, 3), int(plot_style.SURFACE[1:3], 16), np.uint8)
    out = []
    for i, p in enumerate(panels):
        if p.shape[0] < h:
            p = np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 252, np.uint8)])
        out += [p] + ([spacer] if i < len(panels) - 1 else [])
    return np.hstack(out)


def even(frame):
    """Crop to even width and height (H.264 needs it)."""
    return frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
