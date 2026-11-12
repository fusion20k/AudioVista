"""Regenerate the logo assets in assets/ from the vector logo in ui/theme.py.

    python tools/make_icons.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from audiovista.ui import theme as T  # noqa: E402
from audiovista.ui.winshell import write_ico  # noqa: E402


def main():
    pygame.init()
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    pygame.image.save(T.render_logo(512), str(out / "logo.png"))
    pygame.image.save(T.render_logo(256, tile=False), str(out / "logo-mark.png"))
    write_ico(out / "audiovista.ico", [T.render_logo(s) for s in (16, 24, 32, 48, 64, 128, 256)])
    print("wrote", ", ".join(p.name for p in sorted(out.iterdir())))


if __name__ == "__main__":
    main()
