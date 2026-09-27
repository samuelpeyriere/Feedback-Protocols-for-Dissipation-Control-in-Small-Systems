"""Fig. 3: observed state (F/U) vs force for a single hopping trajectory.

Pure schematic (no simulation data needed).
"""
import os

import matplotlib.pyplot as plt

import wiley_style as ws

# --------------------------------------------------------------------------- figure settings
COLOR_STATE = ws.OKABE_ITO['vermillion']   # step curve of the observed state
COLOR_GUIDE = ws.OKABE_ITO['black']        # F/U levels and dotted force markers

# positions (arbitrary units)
F_MIN, F1, FP, FPP, F_MAX = 0.75, 3.65, 4.55, 5.65, 9.4
X_START, X_END = 0.0, 9.65      # extent of the step curve / guide lines
JUMP_1 = 3.1                     # first F->U transition (drawn just before f1)
Y_F, Y_U = 0.24, 0.65            # y-levels of the two states
Y_TOP = 0.75                     # top of the dotted guide lines


# --------------------------------------------------------------------------- figure
def make_figure(outfile=None):
    """Build the full figure."""
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # thin guide lines at the F and U levels
    for y in (Y_F, Y_U):
        ax.plot([X_START, X_END], [y, y], color=COLOR_GUIDE, lw=0.8, zorder=1)

    # dotted vertical markers at the labelled forces
    for x in (F_MIN, F1, FP, FPP, F_MAX):
        ax.plot([x, x], [0, Y_TOP], color=COLOR_GUIDE, ls=(0, (1, 2)), lw=ws.LW_THIN, zorder=2)

    # step curve: F -> U -> F -> U
    xs = [X_START, JUMP_1, JUMP_1, FP,  FP,  FPP, FPP, X_END]
    ys = [Y_F,     Y_F,    Y_U,    Y_U, Y_F, Y_F, Y_U, Y_U]
    ax.plot(xs, ys, color=COLOR_STATE, lw=ws.LW_CURVE, solid_joinstyle='miter', zorder=3)

    # only left and bottom spines, no arrows
    ax.set_xlim(X_START, 10.0)
    ax.set_ylim(0, 1.0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(1.5)

    ax.set_xticks([F_MIN, F1, FP, FPP, F_MAX])
    ax.set_xticklabels([r'$f_{\rm min}$', r'$f_1$', r"$f'$", r"$f''$", r'$f_{\rm max}$'])
    ax.tick_params(axis='x', direction='in', length=6, width=1.5, pad=6)

    ax.set_yticks([Y_F, Y_U])
    ax.set_yticklabels(['F', 'U'])
    ax.tick_params(axis='y', length=0, pad=8)

    ax.set_xlabel(r'Force $f$ [pN]', labelpad=0)
    ax.set_ylabel(r'State $\sigma(f)$')

    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ws.use(verbose=True)
    make_figure(outfile=os.path.join(here, 'Fig3-single-hopping.pdf'))
