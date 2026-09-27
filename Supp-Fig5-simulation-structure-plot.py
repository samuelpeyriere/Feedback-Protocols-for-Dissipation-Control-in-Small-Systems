"""Supp. Fig. 5: structure of one simulation step.

Table of the quantities stored at steps p-1, p and p+1 (time, force, state, rate),
with numbered arrows showing the order of the updates within step p:
  1) t_p = t_{p-1} + dt
  2) sigma_p from sigma_{p-1} and f_{p-1} (jump rates at the previous force)
  3) r_p from f_p (and sigma_p)
  4) f_{p+1} from f_p and r_p

Pure schematic (no simulation data needed).
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path

import wiley_style as ws

# --------------------------------------------------------------------------- figure settings
COLOR_HEADER = '#404040'                 # header row fill (neutral, so arrows stand out)
COLOR_ROWS = ('#E3E3E3', '#F1F1F1')      # alternating row fills
COLOR_GRID = 'white'
COLOR_TEXT = ws.OKABE_ITO['black']
COLOR_STEP = {1: ws.OKABE_ITO['black'], 2: ws.OKABE_ITO['green'],
              3: ws.OKABE_ITO['blue'], 4: ws.OKABE_ITO['vermillion']}
FIGSIZE = (9.0, 5.2)
LW_GRID = 3
LW_ARROW = ws.LW_THIN + 0.5
ARROW_STYLE = '-|>,head_length=0.5,head_width=0.25'

# table layout (data units): column x-edges and row y-intervals
COL_EDGES = [0.0, 2.0, 4.0, 6.0, 8.0]
TEXT_DX = 0.25                           # left padding of the cell text
ROWS = {                                 # name: (y_bottom, y_top, y_text)
    'header': (3.8, 4.6, 4.2),
    'time':   (3.0, 3.8, 3.4),
    'force':  (1.6, 3.0, 2.6),           # taller row: room for arrows 2 and 3
    'state':  (0.8, 1.6, 1.2),
    'rate':   (0.0, 0.8, 0.4),
}
HEADER = ['step', r'$p-1$', r'$p$', r'$p+1$']
ROW_LABELS = {'time': 'time', 'force': 'force', 'state': 'state', 'rate': 'rate'}
ENTRIES = {
    'time':  [r'$t_{p-1}$', r'$t_p$', r'$t_{p+1}$'],
    'force': [r'$f_{p-1}$', r'$f_p$', r'$f_{p+1}$'],
    'state': [r'$\sigma_{p-1}$', r'$\sigma_p$', r'$\sigma_{p+1}$'],
    'rate':  [r'$r_{p-1}$', r'$r_p$', r'$r_{p+1}$'],
}


# --------------------------------------------------------------------------- helpers
def bezier(*points):
    """Path through a start point followed by groups of 3 (ctrl1, ctrl2, end) points."""
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(points) - 1)
    return Path(points, codes)


def arrow(ax, path, step, head=True):
    """Draw `path` in the color of `step`, with an arrow head at its end if `head`."""
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle=ARROW_STYLE if head else '-',
                                 color=COLOR_STEP[step], lw=LW_ARROW, mutation_scale=20,
                                 capstyle='round', joinstyle='round', zorder=4))


def step_label(ax, x, y, step):
    """Circled step number."""
    ax.text(x, y, str(step), ha='center', va='center', color=COLOR_STEP[step],
            fontsize=ws.FS_TICK, fontweight='bold', zorder=5,
            bbox=dict(boxstyle='circle,pad=0.2', fc='none', ec=COLOR_STEP[step], lw=ws.LW_THIN))


def straight(p0, p1):
    return Path([p0, p1], [Path.MOVETO, Path.LINETO])


# --------------------------------------------------------------------------- figure
def draw_table(ax):
    for k, (name, (y0, y1, yt)) in enumerate(ROWS.items()):
        is_header = name == 'header'
        fill = COLOR_HEADER if is_header else COLOR_ROWS[k % 2]
        ax.add_patch(Rectangle((COL_EDGES[0], y0), COL_EDGES[-1] - COL_EDGES[0], y1 - y0,
                               fc=fill, ec='none', zorder=0))
        texts = HEADER if is_header else [ROW_LABELS[name], *ENTRIES[name]]
        for x0, s in zip(COL_EDGES[:-1], texts):
            ax.text(x0 + TEXT_DX, yt, s, ha='left', va='center', zorder=3,
                    color='white' if is_header else COLOR_TEXT,
                    fontweight='bold' if (is_header or x0 == COL_EDGES[0]) else 'normal')

    # white grid lines between cells
    y_min, y_max = ROWS['rate'][0], ROWS['header'][1]
    for x in COL_EDGES[1:-1]:
        ax.plot([x, x], [y_min, y_max], color=COLOR_GRID, lw=LW_GRID, zorder=1)
    for _, (y0, _, _) in list(ROWS.items())[:-1]:
        ax.plot([COL_EDGES[0], COL_EDGES[-1]], [y0, y0], color=COLOR_GRID, lw=LW_GRID, zorder=1)


def draw_arrows(ax):
    y_t, y_f, y_s, y_r = (ROWS[n][2] for n in ('time', 'force', 'state', 'rate'))

    # 1) time increment
    for x0 in (COL_EDGES[1], COL_EDGES[2]):
        arrow(ax, straight((x0 + 1.0, y_t), (x0 + 2.1, y_t)), 1)
        ax.text(x0 + 1.45, y_t + 0.05, r'$\Delta t$', ha='center', va='bottom', zorder=3)
    step_label(ax, 3.8, 3.6, 1)

    # 2) sigma_{p-1} and f_{p-1} -> sigma_p (the simulator uses the jump rates at f_{p-1})
    arrow(ax, straight((2.95, y_s), (4.1, y_s)), 2)
    arrow(ax, bezier((2.95, y_f - 0.1),
                     (3.35, y_f - 0.35), (3.45, y_s + 0.2), (3.75, y_s)), 2, head=False)
    step_label(ax, 3.65, 2.2, 2)

    # 3) f_p -> r_p
    arrow(ax, bezier((4.45, y_f - 0.3),
                     (4.9, y_f - 0.75), (5.05, y_f - 1.4), (4.5, y_r + 0.25)), 3)
    step_label(ax, 4.35, 1.85, 3)

    # 4) f_p and r_p -> f_{p+1}
    arrow(ax, straight((4.75, y_f), (6.1, y_f)), 4)
    arrow(ax, bezier((4.8, y_r),
                     (5.3, y_r - 0.3), (5.85, y_r - 0.2), (5.65, y_r + 0.5),
                     (5.4, y_r + 1.3), (5.05, y_f - 0.5), (5.2, y_f - 0.1),
                     (5.3, y_f), (5.45, y_f), (5.55, y_f)), 4, head=False)
    step_label(ax, 5.75, 2.2, 4)


def make_figure(outfile=None):
    """Build the full figure."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_table(ax)
    draw_arrows(ax)

    ax.set_xlim(COL_EDGES[0], COL_EDGES[-1])
    ax.set_ylim(ROWS['rate'][0], ROWS['header'][1])
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ws.use(verbose=True)
    make_figure(outfile=os.path.join(here, 'Supp-Fig5-simulation-structure.pdf'))
