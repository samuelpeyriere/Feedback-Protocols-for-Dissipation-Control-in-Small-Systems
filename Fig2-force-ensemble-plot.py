"""Fig. 2: the force ensemble.

a) Optical-trap cartoon of the hairpin in the folded and unfolded states. The trap
   holds the force f fixed; unfolding releases the extension x_m.
b) Free-energy landscape along the reaction coordinate, without bias and tilted
   by the bias -f x. The barriers become dG0^dagger - f x^dagger (from F) and
   dG0^star + f x^star (from U), and the free-energy difference becomes dG0 - f x_m.

Pure schematic (no simulation data needed).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
from scipy.interpolate import PchipInterpolator

import wiley_style as ws

# --------------------------------------------------------------------------- figure settings
COLOR_CURVE = ws.OKABE_ITO['blue']          # free-energy landscape
COLOR_FWD = ws.OKABE_ITO['green']           # F -> barrier (dagger), force arrow
COLOR_BWD = ws.OKABE_ITO['vermillion']      # U -> barrier (star)
COLOR_DG = ws.OKABE_ITO['purple']           # dG0, x_m, state names
COLOR_GUIDE = '0.35'                        # dash-dot guides and axes
COLOR_BEAD = ws.OKABE_ITO['skyblue']
COLOR_TRAP = ws.OKABE_ITO['vermillion']
COLOR_DX = ws.OKABE_ITO['blue']             # Delta x / x_m arrows in panel a
COLOR_STAND = '0.6'

LS_GUIDE = (0, (6, 3, 1, 3))                # dash-dot
LW_GUIDE = 1.2
LW_ARROW = 2.5
LW_THIN_DOT = 2.0                           # dotted level lines in panel a
FIGSIZE = (16, 5.2)

# landscape: F well at x=0, barrier at x=X_DAG, U well at x=X_M (reduced units)
X_DAG, X_M = 1.0, 1.8
DG_DAG, DG0 = 1.0, 0.45                     # barrier and F->U free-energy difference
F_BIAS = 0.42                               # tilt: dG0 - f x_m < 0, U becomes stable
X_RANGE = (-0.45, 2.45)
LABEL_ROOM = 0.9                            # x-room left of the forward-barrier arrow for its label


# --------------------------------------------------------------------------- helpers
def arrow(ax, xy_from, xy_to, color, both=False, lw=LW_ARROW, **kw):
    """Straight arrow with a solid head (double-headed if `both`)."""
    style = '<|-|>' if both else '-|>'
    ax.annotate('', xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=20,
                                shrinkA=0, shrinkB=0), **kw)


def guide(ax, xs, ys, **kw):
    ax.plot(xs, ys, color=COLOR_GUIDE, ls=LS_GUIDE, lw=LW_GUIDE, **kw)


def dashdot_axes(ax, xlabel=None, ylabel=None):
    """Dash-dot x/y axes with solid arrowheads (axes-fraction coordinates)."""
    ax.set_axis_off()
    tr = ax.transAxes
    ax.plot([0, 0.97], [0, 0], transform=tr, color=COLOR_GUIDE, ls=LS_GUIDE, lw=1.5,
            clip_on=False)
    ax.plot([0, 0], [0, 0.97], transform=tr, color=COLOR_GUIDE, ls=LS_GUIDE, lw=1.5,
            clip_on=False)
    for xy_from, xy_to in (((0.95, 0), (1.0, 0)), ((0, 0.95), (0, 1.0))):
        ax.annotate('', xy=xy_to, xytext=xy_from, xycoords=tr, textcoords=tr,
                    arrowprops=dict(arrowstyle='-|>', color=COLOR_GUIDE, lw=1.5,
                                    mutation_scale=18, shrinkA=0, shrinkB=0))
    if xlabel:
        ax.text(0.5, -0.04, xlabel, transform=tr, ha='center', va='top')
    if ylabel:
        ax.text(0.0, 1.03, ylabel, transform=tr, ha='left', va='bottom', fontsize=ws.FS_TICK)


def landscape(bias=0.0):
    """Free energy G(x) - bias * x on a dense grid, plus its F, barrier and U extrema."""
    knots_x = [X_RANGE[0], 0.0, X_DAG, X_M, X_RANGE[1]]
    knots_g = [1.35, 0.0, DG_DAG, DG0, 1.2]
    x = np.linspace(*X_RANGE, 2000)
    g = PchipInterpolator(knots_x, knots_g)(x) - bias * x
    i_f = np.argmin(np.where(x < X_DAG, g, np.inf))
    i_u = np.argmin(np.where(x > X_DAG, g, np.inf))
    i_b = i_f + np.argmax(g[i_f:i_u])
    return x, g, (x[i_f], g[i_f]), (x[i_b], g[i_b]), (x[i_u], g[i_u])


# --------------------------------------------------------------------------- panel a
def draw_stand(ax, xc, y0=0.0):
    """Gray pedestal with a fixed bead on top. Returns the top of the bead."""
    ax.add_patch(Polygon([[xc - 0.42, y0], [xc + 0.42, y0], [xc + 0.2, y0 + 0.6],
                          [xc - 0.2, y0 + 0.6]], closed=True, fc='0.75', ec=COLOR_STAND,
                         lw=3, zorder=1))
    r = 0.33
    ax.add_patch(Circle((xc, y0 + 0.6 + r), r, fc=COLOR_BEAD, ec='none', zorder=2))
    return y0 + 0.6 + 2 * r


def draw_trapped_bead(ax, xc, y_bottom, dx, label_side='left'):
    """Trapped bead sitting on `y_bottom`, trap displaced upward by dx, force arrow.

    Returns the y of the bead center.
    """
    r_bead, r_trap = 0.33, 0.62
    yb = y_bottom + r_bead
    yt = yb + dx
    ax.add_patch(Circle((xc, yt), r_trap, fc=COLOR_TRAP, alpha=0.35, ec='none', zorder=3))
    ax.add_patch(Circle((xc, yb), r_bead, fc=COLOR_BEAD, ec='none', zorder=2))
    arrow(ax, (xc, yb - 0.05), (xc, yb + 0.95), COLOR_FWD, lw=5, zorder=4)
    ax.text(xc - 0.14, yb + 0.63, r'$\vec{f}$', color=COLOR_FWD, ha='right', va='center',
            zorder=5)

    # Delta x between bead center and trap center
    x0, x1 = xc - 0.75, xc + 0.75
    for y in (yb, yt):
        ax.plot([x0, x1], [y, y], color=COLOR_GUIDE, ls=':', lw=LW_THIN_DOT, zorder=6)
    if label_side == 'left':
        xa, ha, xt = x0 + 0.08, 'right', x0 + 0.02
    else:
        xa, ha, xt = x1 - 0.08, 'left', x1 - 0.02
    arrow(ax, (xa, yb), (xa, yt), COLOR_DX, lw=1.5, zorder=6)
    ax.text(xt if label_side == 'left' else xt + 0.1, (yb + yt) / 2, r'$\Delta x$',
            ha=ha, va='center', zorder=6)
    return yb


def draw_panel_a(ax):
    ax.set_aspect('equal')
    ax.set_xlim(0, 3.9)
    ax.set_ylim(0, 5.0)
    ax.set_axis_off()

    # folded: hairpin drawn as a short stem ending in a loop, beads touching
    xf = 1.0
    y_top = draw_stand(ax, xf)
    ax.plot([xf - 0.8, xf], [y_top, y_top], color='k', lw=4, zorder=4,
            solid_capstyle='butt')
    ax.add_patch(Circle((xf - 0.85, y_top), 0.07, fc='white', ec='k', lw=2.5, zorder=4))
    draw_trapped_bead(ax, xf, y_top, dx=0.3, label_side='right')
    ax.text(xf, 3.25, 'FOLDED', color=COLOR_DG, ha='center', va='bottom',
            fontweight='bold')

    # unfolded: released extension x_m between the two beads
    xu = 2.9
    y_top = draw_stand(ax, xu)
    y_bead = y_top + 1.55                      # bottom of the trapped bead
    t = np.linspace(0, 1, 400)
    zig = 0.14 * np.sin(2 * np.pi * 3.5 * t) * np.sin(np.pi * t) ** 0.5
    ax.plot(xu + zig, y_top + (y_bead - y_top) * t, color='k', lw=3.5, zorder=1)
    draw_trapped_bead(ax, xu, y_bead, dx=0.45, label_side='right')

    for y in (y_top, y_bead):
        ax.plot([xu - 0.3, xu + 0.75], [y, y], color=COLOR_GUIDE, ls=':', lw=LW_THIN_DOT)
    xa = xu + 0.62
    arrow(ax, (xa, y_top), (xa, y_bead), COLOR_DX, lw=1.5)
    ax.text(xa + 0.1, (y_top + y_bead) / 2, r'$x_m$', ha='left', va='center')
    ax.text(xu, 4.55, 'UNFOLDED', color=COLOR_DG, ha='center', va='bottom',
            fontweight='bold')


# --------------------------------------------------------------------------- panel b
def draw_landscape(ax, bias, labels, show_x=True, ylabel=None):
    x, g, (xf, gf), (xb, gb), (xu, gu) = landscape(bias)
    ax.plot(x, g, color=COLOR_CURVE, lw=ws.LW_CURVE, zorder=3)

    y_lo = min(gf, gu) - (0.62 if show_x else 0.3)
    # when U is below F the dG arrow sits past the end of the curve (the U well is too
    # narrow for it), so the backward-barrier arrow moves further right
    tilted = gu < gf
    x_fwd = X_RANGE[0] - 0.12
    x_bwd = X_RANGE[1] + (0.4 if tilted else 0.1)
    ax.set_xlim(x_fwd - LABEL_ROOM, x_bwd + 0.2)
    ax.set_ylim(y_lo, gb + 0.22)
    dashdot_axes(ax, xlabel='Reaction coordinate', ylabel=ylabel)

    # state levels
    for xs, y, name in ((xf, gf, 'F'), (xu, gu, 'U')):
        ax.plot([xs - 0.2, xs + 0.2], [y, y], color='0.2', lw=4, zorder=4,
                solid_capstyle='butt')
        ax.text(xs, y + 0.02, name, ha='center', va='bottom', fontweight='bold', zorder=5)
    ax.text(xb, gb + 0.02, r'$\ast$', ha='center', va='bottom')

    # barrier heights: forward (from F) on the left, backward (from U) on the right
    guide(ax, [x_fwd, x_bwd], [gb, gb])
    guide(ax, [xb, xb], [min(gf, gu) - 0.1, gb])
    arrow(ax, (x_fwd, gf), (x_fwd, gb), COLOR_FWD)
    arrow(ax, (x_bwd, gu), (x_bwd, gb), COLOR_BWD)
    ax.text(x_fwd - 0.12, (gf + gb) / 2, labels['fwd'], color=COLOR_FWD, ha='right',
            va='center', multialignment='left')
    ax.text(x_bwd + 0.12, (gu + gb) / 2, labels['bwd'], color=COLOR_BWD, ha='left',
            va='center')

    # free-energy difference F -> U
    x_dg = X_RANGE[1] + 0.12 if tilted else xu + 0.2
    guide(ax, [xf + 0.2, x_dg + 0.05], [gf, gf])
    if tilted:
        guide(ax, [xu + 0.2, x_dg + 0.05], [gu, gu])
    arrow(ax, (x_dg, gf), (x_dg, gu), COLOR_DG)
    if not tilted:  # label beside the arrow
        ax.text(x_dg + 0.08, (gf + gu) / 2, labels['dg'], color=COLOR_DG, ha='left',
                va='center')
    else:           # U below F: label under the arrow tip, clear of the curve
        ax.text(x_dg, gu - 0.04, labels['dg'], color=COLOR_DG, ha='center', va='top')

    # distances along the reaction coordinate
    if show_x:
        y1 = min(gf, gu) - 0.1
        y2 = y1 - 0.3
        arrow(ax, (xf, y1), (xb, y1), COLOR_FWD, both=True, lw=2)
        arrow(ax, (xb, y1), (xu, y1), COLOR_BWD, both=True, lw=2)
        arrow(ax, (xf, y2), (xu, y2), COLOR_DG, both=True, lw=2)
        ax.text((xf + xb) / 2, y1 - 0.02, r'$x^\dagger$', color=COLOR_FWD, ha='center',
                va='top')
        ax.text((xb + xu) / 2, y1 - 0.02, r'$x^\star$', color=COLOR_BWD, ha='center',
                va='top')
        ax.text((xf + xu) / 2, y2 - 0.03, r'$x_m$', color=COLOR_DG, ha='center',
                va='top')


def draw_bias(ax):
    """Small inset showing the linear bias potential -f x."""
    dashdot_axes(ax)
    ax.plot([0.08, 0.9], [0.8, 0.12], color=COLOR_CURVE, lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, -0.12, r'Bias $f$', transform=ax.transAxes, ha='center', va='top')


# --------------------------------------------------------------------------- figure
def make_figure(outfile=None):
    """Build the full figure."""
    fig = plt.figure(figsize=FIGSIZE)

    ax_a = fig.add_axes([0.0, 0.0, 0.24, 1.0])
    draw_panel_a(ax_a)

    ax_l = fig.add_axes([0.285, 0.1, 0.265, 0.76])
    draw_landscape(ax_l, 0.0, dict(fwd=r'$\Delta G_0^\dagger$', bwd=r'$\Delta G_0^\star$',
                                   dg=r'$\Delta G_0$'), ylabel='Free energy')

    ax_bias = fig.add_axes([0.625, 0.22, 0.05, 0.16])
    draw_bias(ax_bias)
    fig.text(0.65, 0.55, r'$\Longrightarrow$', ha='center', va='center', fontsize=34)

    ax_r = fig.add_axes([0.715, 0.1, 0.265, 0.76])
    draw_landscape(ax_r, F_BIAS, dict(fwd=r'$\Delta G_0^\dagger$' '\n' r'$- f x^\dagger$',
                                      bwd=r'$\Delta G_0^\star$' '\n' r'$+ f x^\star$',
                                      dg=r'$\Delta G_0 - f x_m$'),
                   show_x=False)

    fig.text(0.65, 0.99, 'Force ensemble', ha='center', va='top', fontweight='bold',
             fontsize=ws.FS_LABEL + 4)
    ws.panel_label(ax_a, 'a)', x=0.03, y=0.99, ha='left')
    fig.text(0.265, 0.99, 'b)', ha='left', va='top')

    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ws.use(verbose=True)
    make_figure(outfile=os.path.join(here, 'Fig2-force-ensemble.pdf'))
