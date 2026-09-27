"""Generic axis helpers that go with wiley_style (not specific to any one figure)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import wiley_style as ws

# Broken x-axis between a finite range and a separate "x = inf" point
BREAK_GAP = 0.1        # gap between main and inf axes (fraction of mean axis width)
BREAK_MARK_SIZE = 5    # size of the '//' marks
BREAK_MARK_LW = 1


def text_width_in_data(ax, s: str, fontsize=None) -> float:
    """Width of the string `s` in data units of `ax`.

    The result depends on the current layout, so call this only after
    tight_layout / subplots_adjust.
    """
    t = ax.text(0, 0, s, fontsize=fontsize)
    bbox = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    t.remove()
    return bbox.transformed(ax.transData.inverted()).width


def draw_slope_marker(ax, x0, y0, dx, dy, label, color, lw=ws.LW_THIN, label_dy=0.1,
                      label_dx=None, **text_kw):
    """Small slope 'staircase' starting at (x0, y0), with a label to its right.

    The label sits at (x0 + label_dx, y0 + label_dy); label_dx defaults to 1.2 * dx.
    Extra keyword arguments (e.g. ha, va) go to the label's ax.text.
    """
    ax.plot([x0, x0 + dx, x0 + dx], [y0, y0, y0 + dy], c=color, linewidth=lw)
    if label_dx is None:
        label_dx = 1.2 * dx
    ax.text(x0 + label_dx, y0 + label_dy, label, **{'ha': 'left', 'color': color, **text_kw})


def draw_axis_break(ax_left, ax_right, size=BREAK_MARK_SIZE, lw=BREAK_MARK_LW):
    """Draw '//' marks on the top and bottom spines where the x-axis is broken."""
    kw = dict(marker=[(-1, -1.5), (1, 1.5)], markersize=size, markeredgewidth=lw,
              linestyle='none', color='k', mec='k', clip_on=False, zorder=10)
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **kw)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **kw)


def plot_with_inf_axis(fig, spec, x, ys, labels, *, title='', unit='', xlabel=None,
                       hide_x=False, share_inf_y=False, xscale='log',
                       width_ratios=(3, 1), gap=BREAK_GAP, inf_marker_area=70):
    """Plot curves ys[k](x). If x contains inf, those points go in a narrow side axis.

    The side axis sits to the right of the main axis, joined by a broken x-axis.
    With `share_inf_y` the side axis uses the main y scale and shows no y ticks.
    Otherwise it gets its own y scale with tick labels on the right.

    `title` is placed top-left and `unit` top-right of the panel.
    Returns (ax, ax_inf or None, legend_handles, labels).
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    has_inf = not finite.all()

    if has_inf:
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=spec,
                                                 width_ratios=width_ratios, wspace=gap)
        ax = fig.add_subplot(inner[0])
        ax_inf = fig.add_subplot(inner[1], sharey=ax if share_inf_y else None)
    else:
        ax = fig.add_subplot(spec)
        ax_inf = None

    handles = []
    for k, y in enumerate(ys):
        st = ws.curve_style(k)
        y = np.asarray(y, dtype=float)
        ax.plot(x[finite], y[finite], color=st['color'], linestyle=st['linestyle'])
        if has_inf:
            ax_inf.scatter(np.ones((~finite).sum()), y[~finite], s=inf_marker_area,
                           color=st['color'], marker=st['marker'], zorder=3)
        # legend entry shows color, line style and the marker used for x = inf
        handles.append(Line2D([], [], linewidth=ws.LW_CURVE, **st))

    # main axis
    ax.set_xscale(xscale)
    ax.set_title(title, loc='left')
    if hide_x:
        ax.tick_params(labelbottom=False)
    elif xlabel is not None:
        ax.set_xlabel(xlabel)

    # unit goes to the right-most axis of the panel
    (ax_inf if has_inf else ax).set_title(unit, loc='right')

    if has_inf:
        ax_inf.set_xlim(0.5, 1.5)
        ax_inf.set_xticks([1])
        ax_inf.set_xticklabels([r'$\infty$'])
        ax_inf.tick_params(labelbottom=not hide_x)
        ax.spines['right'].set_visible(False)
        ax_inf.spines['left'].set_visible(False)
        ax.yaxis.tick_left()
        ax_inf.yaxis.tick_right()
        if share_inf_y:
            ax_inf.tick_params(right=False, labelright=False)
        draw_axis_break(ax, ax_inf)

    return ax, ax_inf, handles, list(labels)

# --------------------------------------------------------------------------- approximation comparison
# Supplementary figures comparing the full simulation, the formula + partial simulation
# and the single-hopping estimation. Methods differ in color, marker/line and fill.
APPROX_LEGEND_SPACE = 0.14   # fraction of figure height reserved for the legend at the bottom
APPROX_MARKER_AREA = 40
APPROX_MAX_POINTS = 100      # denser point sets are thinned so the markers stay readable

APPROX_STYLES = {
    'simulation': dict(color=ws.OKABE_ITO['blue'], marker='o', filled=True),
    'formula': dict(color=ws.OKABE_ITO['vermillion'], marker='s', filled=False),
    'estimation': dict(color=ws.OKABE_ITO['green'], linestyle='-'),
}
APPROX_LABELS = {
    'simulation': 'simulation',
    'formula': 'formula + partial simulation',
    'estimation': 'single hopping',
}


def _approx_legend_handle(key):
    st = APPROX_STYLES[key]
    if 'marker' in st:
        return Line2D([], [], linestyle='none', marker=st['marker'], color=st['color'],
                      markerfacecolor=st['color'] if st['filled'] else 'none',
                      markeredgewidth=1.5)
    return Line2D([], [], color=st['color'], linestyle=st['linestyle'], linewidth=ws.LW_CURVE)


def plot_approx_curves(ax, curves):
    """Plot the methods present in `curves` ({method key: (x, y)}) on `ax`.
    Points (simulation, formula) are drawn below the single-hopping line, and the
    filled simulation points above the open formula markers."""
    for key in ('formula', 'simulation'):
        if key in curves:
            x, y = curves[key]
            step = max(1, int(np.ceil(len(x) / APPROX_MAX_POINTS)))
            x, y = np.asarray(x)[::step], np.asarray(y)[::step]
            st = APPROX_STYLES[key]
            fill = dict(color=st['color']) if st['filled'] else \
                dict(facecolors='none', edgecolors=st['color'], linewidths=1.5)
            ax.scatter(x, y, s=APPROX_MARKER_AREA, marker=st['marker'], zorder=2, **fill)
    if 'estimation' in curves:
        x, y = curves['estimation']
        st = APPROX_STYLES['estimation']
        ax.plot(x, y, color=st['color'], linestyle=st['linestyle'], zorder=3)


def make_approx_figure(panels, xlabel, xscale='linear', figsize=(17, 5), outfile=None):
    """One row of panels, each comparing the approximations of one quantity.

    `panels` is a list of dicts with keys 'title', 'unit', 'curves' ({method key: (x, y)})
    and optionally 'ylim'. The legend sits in one row under the panels.
    """
    fig, axs = plt.subplots(1, len(panels), figsize=figsize)
    used = []
    for ax, label, panel in zip(axs, 'abcdefgh', panels):
        plot_approx_curves(ax, panel['curves'])
        used += [key for key in APPROX_STYLES if key in panel['curves'] and key not in used]

        ax.set_title(panel['title'], loc='left')
        ax.set_title(panel['unit'], loc='right')
        ax.set_xlabel(xlabel)
        ax.set_xscale(xscale)
        if xscale == 'linear':
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        if panel.get('ylim') is not None:
            ax.set_ylim(panel['ylim'])
        ws.panel_label(ax, label + ')', x=-0.12, y=1.12)

    used = [key for key in APPROX_STYLES if key in used]   # fixed legend order
    fig.tight_layout(rect=(0, APPROX_LEGEND_SPACE, 1, 1))
    fig.legend([_approx_legend_handle(k) for k in used], [APPROX_LABELS[k] for k in used],
               loc='lower center', ncol=len(used), bbox_to_anchor=(0.5, 0.005))

    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig
