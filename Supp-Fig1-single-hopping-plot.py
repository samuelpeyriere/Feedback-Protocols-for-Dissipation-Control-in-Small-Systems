"""Supp. Fig. 1: observed state sigma(f) vs force in the zero-hopping, single-hopping and
general cases.

Each panel is a step curve between the folded (F) and unfolded (U) states, with the
unfolded intervals shaded. Force ranges of arbitrary length are drawn dashed, and the
x-axis is dotted over them.

Pure schematic (no simulation data needed).
"""
import os

import matplotlib.pyplot as plt

import wiley_style as ws

# --------------------------------------------------------------------------- figure settings
COLOR_STATE = ws.OKABE_ITO['blue']       # step curve and shading of the U intervals
COLOR_AXIS = ws.OKABE_ITO['black']
ALPHA_FILL = 0.27
LS_GAP = (0, (4, 3))                     # curve over an arbitrary-length force range
LS_AXIS_GAP = (0, (1, 1.5))              # x-axis under an arbitrary-length force range
FIGSIZE = (20, 5.3)

Y_F, Y_U = 0.0, 1.0                      # y-levels of the two states
X_MARGIN, Y_MARGIN = 0.05, 0.1

# Each case: title, positions of the F<->U transitions (the curve starts in F at
# f_min = 0 and ends at f_max = 1), their tick labels, and the force ranges drawn
# as gaps (dashed curve, dotted axis).
CASES = [
    dict(title='Zero-hopping case',
         jumps=[0.5],
         labels=[r'$f_1$'],
         gaps=[(0.1, 0.5)]),
    dict(title='Single-hopping case',
         jumps=[0.25, 0.5, 0.7],
         labels=[r'$f_1$', r'$f_1^\prime$', r'$f_2$'],
         gaps=[(0.06, 0.25)]),
    dict(title='General case',
         jumps=[0.125, 0.25, 0.35, 0.44, 0.625, 0.75, 0.875],
         labels=[r'$f_1$', r'$f_1^\prime$', r'$f_2$', r'$f_2^\prime$', r'$f_k$', r'$f_k^\prime$',
                 r'$f_{k+1}$'],
         gaps=[(0.03, 0.125), (0.47, 0.625), (0.9, 0.97)]),
]


# --------------------------------------------------------------------------- helpers
def split_by_gaps(x0, x1, gaps):
    """Split [x0, x1] into (a, b, in_gap) pieces."""
    cuts = sorted({x0, x1, *(g for gap in gaps for g in gap if x0 < g < x1)})
    return [(a, b, any(g0 <= a and b <= g1 for g0, g1 in gaps))
            for a, b in zip(cuts[:-1], cuts[1:])]


def draw_case(ax, jumps, labels, gaps, title):
    edges = [0.0, *jumps, 1.0]
    for k, (x0, x1) in enumerate(zip(edges[:-1], edges[1:])):
        y = Y_U if k % 2 else Y_F
        # horizontal level, dashed where the force range has arbitrary length
        for a, b, in_gap in split_by_gaps(x0, x1, gaps):
            ax.plot([a, b], [y, y], color=COLOR_STATE, lw=ws.LW_THIN,
                    ls=LS_GAP if in_gap else '-', solid_capstyle='butt', zorder=3)
        if y == Y_U:
            ax.fill_between([x0, x1], Y_F, Y_U, color=COLOR_STATE, alpha=ALPHA_FILL,
                            lw=0, zorder=1)
    for x in jumps:
        ax.plot([x, x], [Y_F, Y_U], color=COLOR_STATE, lw=ws.LW_THIN, zorder=3)

    # bottom axis drawn by hand: solid, but dotted over the gaps
    ax.spines['bottom'].set_visible(False)
    tr = ax.get_xaxis_transform()
    for a, b, in_gap in split_by_gaps(-X_MARGIN, 1 + X_MARGIN, gaps):
        ax.plot([a, b], [0, 0], transform=tr, color=COLOR_AXIS, clip_on=False,
                lw=0.8 if in_gap else ax.spines['left'].get_linewidth(),
                ls=LS_AXIS_GAP if in_gap else '-', zorder=4)

    ax.set_xlim(-X_MARGIN, 1 + X_MARGIN)
    ax.set_ylim(Y_F - Y_MARGIN, Y_U + Y_MARGIN)
    ax.set_xticks([0.0, *jumps, 1.0])
    ax.set_xticklabels([r'$f_{\rm min}$', *labels, r'$f_{\rm max}$'], fontsize=ws.FS_LABEL)
    ax.set_yticks([Y_F, Y_U])
    ax.set_yticklabels(['F', 'U'], fontsize=ws.FS_LABEL)
    ax.set_xlabel(r'Force $f$ [pN]', labelpad=0)
    ax.set_title(title)


# --------------------------------------------------------------------------- figure
def make_figure(outfile=None):
    """Build the full figure."""
    fig, axes = plt.subplots(1, len(CASES), figsize=FIGSIZE, sharey=True)
    for ax, case in zip(axes, CASES):
        draw_case(ax, **case)
    axes[0].set_ylabel(r'State $\sigma(f)$')

    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ws.use(verbose=True)
    make_figure(outfile=os.path.join(here, 'Supp-Fig1-single-hopping.pdf'))
