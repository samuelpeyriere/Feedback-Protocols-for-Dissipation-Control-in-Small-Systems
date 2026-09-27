"""Reusable matplotlib style for Wiley figures.

- STIX Two Text loaded from TrueType (.ttf) files. The .otf versions embed CFF
  outlines, which makes pdffonts report a "Mismatch" warning.
- STIX math, fonts embedded as Type 42.
- Colorblind-safe Okabe-Ito palette. Curves differ in color, line style and marker.
- Consistent font sizes and line widths, set through rcParams. You rarely need
  to pass `fontsize=` or `labelsize=` by hand.

Typical use::

    import matplotlib.pyplot as plt
    import wiley_style as ws

    ws.use()                                   # once per session / script
    fig, ax = plt.subplots()
    for k, y in enumerate(curves):
        ax.plot(x, y, **ws.curve_style(k, marker=False))
    ws.panel_label(ax, "a)")
    fig.savefig("fig.pdf")                     # transparent + tight bbox via rcParams

The STIX Two .ttf files are looked for in $STIX2_TTF_DIR if set, otherwise in
`fonts/stix2_ttf` next to this file or one directory up. If none of these holds them,
a system-wide installation of STIX Two Text is used. The fonts are free to download
from https://github.com/stipub/stixfonts.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager

# --------------------------------------------------------------------------- sizes
FS_LABEL = 20      # axis labels, titles, legend, annotations
FS_TICK = 16       # tick labels
LW_CURVE = 3       # data curves
LW_THICK = 4.0     # emphasized lines (e.g. protocol traces)
LW_THIN = 2.0      # guides, slope markers
MARKER_SIZE = 9

# --------------------------------------------------------------------------- colors
# Okabe-Ito palette (+ a neutral gray that pairs well with dark blue in grayscale)
OKABE_ITO = dict(
    blue='#0072B2', vermillion='#D55E00', green='#009E73', purple='#CC79A7',
    orange='#E69F00', skyblue='#56B4E9', black='#000000', yellow='#F0E442',
    gray='#999999',
)

CURVE_COLORS = [OKABE_ITO[c] for c in
                ('blue', 'vermillion', 'green', 'purple', 'orange', 'skyblue', 'black')]
CURVE_LINESTYLES = ['-', '--', '-.', ':']
CURVE_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']


def curve_style(k: int, marker: bool = True) -> dict:
    """Style for the k-th curve: color, line style and (optionally) marker all vary,
    so curves stay distinguishable without relying on color alone."""
    st = dict(color=CURVE_COLORS[k % len(CURVE_COLORS)],
              linestyle=CURVE_LINESTYLES[k % len(CURVE_LINESTYLES)])
    if marker:
        st['marker'] = CURVE_MARKERS[k % len(CURVE_MARKERS)]
    return st


# --------------------------------------------------------------------------- fonts
FONT_FAMILY = "STIX Two Text"
_HERE = Path(__file__).resolve().parent
FONT_DIRS = ([Path(os.environ["STIX2_TTF_DIR"])] if "STIX2_TTF_DIR" in os.environ else
             [_HERE / "fonts/stix2_ttf", _HERE.parent / "fonts/stix2_ttf"])

_registered_dirs: set[Path] = set()


def register_fonts(font_dir: str | Path | None = None, verbose: bool = False) -> str:
    """Register the STIX Two .ttf files with matplotlib (idempotent).

    `font_dir` overrides the default FONT_DIRS. Raises if the font can't be found,
    which stops matplotlib from silently falling back to DejaVu. Returns the path
    of the resolved font file.
    """
    for d in ([Path(font_dir)] if font_dir is not None else FONT_DIRS):
        if d in _registered_dirs:
            break
        ttf_files = sorted(d.glob("STIXTwo*.ttf"))
        if ttf_files:
            for path in ttf_files:
                font_manager.fontManager.addfont(str(path))
            _registered_dirs.add(d)
            break

    try:
        path = font_manager.findfont(FONT_FAMILY, fallback_to_default=False)
    except ValueError:
        raise FileNotFoundError(
            f"{FONT_FAMILY} not found: put the STIX Two .ttf files in one of "
            f"{[str(d) for d in FONT_DIRS]} or set $STIX2_TTF_DIR") from None
    if verbose:
        print(f"{FONT_FAMILY}: {path}")
    return path


# --------------------------------------------------------------------------- rcParams
def rc_params() -> dict:
    """rcParams of the Wiley style."""
    return {
        # fonts
        "font.family": FONT_FAMILY,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # sizes
        "font.size": FS_LABEL,          # default for ax.text / fig.text
        "axes.titlesize": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LABEL,
        "lines.linewidth": LW_CURVE,
        "lines.markersize": MARKER_SIZE,
        # colors
        "axes.prop_cycle": cycler(color=CURVE_COLORS),
        # saving
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": "figure",
    }


def use(font_dir: str | Path | None = None, verbose: bool = False) -> None:
    """Register fonts and apply the style globally."""
    register_fonts(font_dir, verbose=verbose)
    mpl.rcParams.update(rc_params())


# --------------------------------------------------------------------------- small helpers
def panel_label(ax, text: str, x: float = -0.1, y: float = 1.1, **kw):
    """Panel label such as 'a)' placed in axes coordinates of `ax`."""
    kw = {'va': 'top', 'ha': 'right', **kw}
    return ax.text(x, y, text, transform=ax.transAxes, **kw)