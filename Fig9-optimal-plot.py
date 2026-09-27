"""Fig. 9: comparison of the feedback protocols (LdF ensemble).
(a) Relative dissipation reduction vs relative information cost for all protocols of
    Figs. 5-7, over lines of constant eta_I and eta_M;
    the optimal protocols used in (b) and (c) are circled. Points are colored by <tau>.
(b) Bias B_N of the free-energy estimator vs the cumulative measurement time N<tau>,
    for no feedback, the optimal DTF and constant-DS protocols and a CTF protocol of similar <tau>.
(c) Same, weighted by N<tau> / ln(N<tau>).

Run `Fig9-optimal-simulation.py` first: it writes `Fig9-optimal-data.npz`, which this
script loads. Panel (a) also loads the data of Figs. 5-7.
You can also import `make_figure` and call it directly.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import wiley_style as ws

# --------------------------------------------------------------------------- figure settings
# One color + marker per protocol, shared by all panels.
# key in Fig9-optimal-data.npz, legend label, color, marker
PROTOCOLS = [
    ('none', 'No feedback',              ws.OKABE_ITO['blue'],       's'),
    ('DTF',  r'DTF (optimal $\eta_M$)',  ws.OKABE_ITO['vermillion'], '^'),
    ('CTF',  'CTF',                      ws.OKABE_ITO['green'],      'o'),
    ('CDS',  r'constant-DS (optimal $\eta_M$)', ws.OKABE_ITO['purple'], 'X'),
]
STYLE = {key: dict(color=color, marker=marker) for key, _, color, marker in PROTOCOLS}
NAME = {'DTF': 'DTF', 'CTF': 'CTF', 'CDS': 'constant-DS'}   # protocol names in panel (a)

# panels (b, c)
MARKED_N = [1, 10, 100, 1000]   # N at which the curves get a marker
LABELED_N_CURVE = 'CTF'   # curve whose markers carry the 'N = ...' labels

# panel (a): data file, keep every n-th point along each sweep (dense f_1 sweeps)
MAP_DATA = {
    'DTF': ('Fig5-DTF-data.npz', 4),
    'CTF': ('Fig6-CTF-data.npz', 1),
    'CDS': ('Fig7-CDS-data.npz', 4),
}
# protocols of panels (b, c) (optimal DTF and constant-DS, reference CTF), located in panel (a): key, row of the data arrays,
# parameter swept along the row, its value, interpolate in log(parameter)
OPTIMAL = [
    ('DTF', 3, 'cp1_list',   16.1,   False),  # r_U/r_F = inf, f_1 = 16.1 pN
    ('CTF', 1, 'ratio_list', 17 / 4, True),   # r_F = 4, r_U = 17 pN/s
    ('CDS', 0, 'f1_list',    14.3,   False),  # r_F'/r_F = 1/4, f_1 = 14.3 pN
]
MARKER_AREA_MAP = 70
CIRCLE_AREA = 600         # circle around the protocols of (b, c), in their (b, c) color
CMAP = 'viridis'          # points of panel (a) colored by <tau>
XYMAX = 1.5               # panel (a) shows [0, XYMAX]^2

COLOR_ISO = ws.OKABE_ITO['gray']
COLOR_BOUND = ws.OKABE_ITO['black']
COLOR_FILL = ws.OKABE_ITO['yellow']
LW_ISO = 1.2
FS_ISO = ws.FS_TICK - 2   # iso-line value labels

# iso lines drawn and labeled in panel (a) (labels placed where each line leaves the plot)
ETAI_ISO = [0.1, 0.2, 0.3, 0.4, 0.5]
ETAM_ISO = [0.25, 0.5, 0.75]


# --------------------------------------------------------------------------- panel (a)
def optimal_point(d, row, param, value, log, key=None):
    """Panel (a) coordinates of one protocol, interpolated along row `row` of dataset `d`
    at `param` = value (inf entries of the swept parameter are left out).
    With `key`, returns that quantity (e.g. 'T') instead."""
    xs = np.asarray(d[param], dtype=float)
    ok = np.isfinite(xs)
    xs, x0 = (np.log(xs[ok]), np.log(value)) if log else (xs[ok], value)

    def at(name):
        return np.interp(x0, xs, np.asarray(d[name])[row][ok])

    if key is not None:
        return at(key)
    return at('UPSILON') / at('WD0'), 1 - at('WD') / at('WD0')


def draw_map(ax, datasets, norm):
    """Iso lines, bounds, the protocol points (colored by <tau> with `norm`) and the
    circled protocols of (b, c). Returns the last scatter (for the colorbar)."""
    X = np.array([-1, 3])
    for eta_I in ETAI_ISO:
        ax.plot(X, eta_I * (1 + X), color=COLOR_ISO, ls='--', linewidth=LW_ISO, zorder=1)
    for eta_M in ETAM_ISO:
        ax.plot(X, eta_M + X, color=COLOR_ISO, ls=':', linewidth=LW_ISO, zorder=1)
    ax.plot(X, 1 + X, color=COLOR_BOUND, linewidth=ws.LW_THIN, zorder=2)
    ax.plot(X, X, color=COLOR_BOUND, ls='-.', linewidth=ws.LW_THIN, zorder=2)
    ax.fill_between(X, X, 1 + X, color=COLOR_FILL, alpha=0.25, linewidth=0, zorder=0)
    ax.axhline(1, color=COLOR_ISO, linewidth=LW_ISO, zorder=1)

    for key, d in datasets.items():
        sl = (Ellipsis, slice(None, None, MAP_DATA[key][1]))
        sc = ax.scatter((d['UPSILON'] / d['WD0'])[sl], (1 - d['WD'] / d['WD0'])[sl],
                        c=d['T'][sl], cmap=CMAP, norm=norm, marker=STYLE[key]['marker'],
                        s=MARKER_AREA_MAP, zorder=3)

    for key, row, param, value, log in OPTIMAL:
        d = datasets[key]
        x, y = optimal_point(d, row, param, value, log)
        # the point itself (it may fall between the subsampled points), then the circle
        tau = optimal_point(d, row, param, value, log, key='T')
        ax.scatter(x, y, c=[tau], cmap=CMAP, norm=norm, marker=STYLE[key]['marker'],
                   s=MARKER_AREA_MAP, zorder=4, clip_on=False)
        ax.scatter(x, y, s=CIRCLE_AREA, facecolors='none', edgecolors=STYLE[key]['color'],
                   linewidths=2.5, zorder=5, clip_on=False)
    return sc


def label_iso_lines(ax):
    """Value of each iso line where it leaves the plot (right or top edge)."""
    kw = dict(color=COLOR_ISO, fontsize=FS_ISO, textcoords='offset points',
              annotation_clip=False)
    for eta_I in ETAI_ISO:   # all leave through the right edge
        ax.annotate(r'$\eta_I = %g$' % eta_I, (XYMAX, eta_I * (1 + XYMAX)), xytext=(5, 0),
                    ha='left', va='center', **kw)
    for eta_M in ETAM_ISO:   # all leave through the top edge
        ax.annotate(r'$\eta_M = %g$' % eta_M, (XYMAX - eta_M, XYMAX), xytext=(0, 5),
                    ha='center', va='bottom', **kw)


def draw_panel_a(fig, spec, datasets):
    """Delta<W_d>/<W_d>_0 vs k_B T Upsilon/<W_d>_0 with a zoom on the origin.
    Returns the axis, the scatter used for the colorbar, and legend handles/labels."""
    ax = fig.add_subplot(spec)
    norm = LogNorm(vmin=min(np.nanmin(d['T']) for d in datasets.values()),
                   vmax=max(np.nanmax(d['T']) for d in datasets.values()))
    sc = draw_map(ax, datasets, norm)
    ax.set_xlim(0, XYMAX)
    ax.set_ylim(0, XYMAX)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(r'$k_B T \Upsilon \,/\, \langle W_d \rangle_0$')
    ax.set_ylabel(r'$\Delta \langle W_d \rangle \,/\, \langle W_d \rangle_0$')
    label_iso_lines(ax)

    handles = [
        Line2D([], [], color=COLOR_ISO, ls='--', linewidth=LW_ISO),
        Line2D([], [], color=COLOR_ISO, ls=':', linewidth=LW_ISO),
        Line2D([], [], color=COLOR_BOUND, linewidth=ws.LW_THIN),
        Line2D([], [], color=COLOR_BOUND, ls='-.', linewidth=ws.LW_THIN),
        Patch(facecolor=COLOR_FILL, alpha=0.25),
        Line2D([], [], color=COLOR_ISO, linewidth=LW_ISO),
    ]
    handles += [Line2D([], [], ls='none', marker=STYLE[key]['marker'], color='k', mfc='none',
                       mew=1.5) for key in MAP_DATA]
    handles.append(Line2D([], [], ls='none', marker='o', markersize=16, mfc='none', mec='k',
                          mew=2))
    labels = [
        r'iso $\eta_I$',
        r'iso $\eta_M$',
        r'$\eta_M = \eta_I = 1 \Rightarrow B_1 = 0$',
        r'$\eta_M = 0 \Rightarrow B_1 = \langle W_d \rangle_0$',
        r'$\langle W_d \rangle_0 \geq B_1 \geq 0$',
        r'$\langle W_d \rangle = 0$ (work-extraction threshold)',
    ]
    labels += [NAME[key] for key in MAP_DATA]
    labels.append('used in (b, c)')
    return ax, sc, handles, labels


# --------------------------------------------------------------------------- panels (b, c)
def draw_bias_panel(fig, spec, data, weight, ylabel, sharex=None, label_N=False):
    """weight(T) * B_N vs N<tau> for all protocols, with markers at MARKED_N."""
    ax = fig.add_subplot(spec, sharex=sharex)
    idx = np.array(MARKED_N) - 1
    for key, *_ in PROTOCOLS:
        T, B = data[f'T_{key}'], data[f'B_{key}']
        y = weight(T) * B
        ax.plot(T, y, color=STYLE[key]['color'])
        ax.plot(T[idx], y[idx], ls='none', **STYLE[key])
        if label_N and key == LABELED_N_CURVE:
            for n, i in zip(MARKED_N, idx):
                ax.annotate(r'$N = %d$' % n if n == MARKED_N[0] else r'$%d$' % n,
                            (T[i], y[i]), xytext=(6, 6), textcoords='offset points',
                            ha='left', va='bottom', fontsize=ws.FS_TICK,
                            color=STYLE[key]['color'])

    ax.axhline(0, color=COLOR_ISO, ls=':', linewidth=1, zorder=0)
    ax.set_xscale('log')
    ax.set_xlim(right=max(data[f'T_{key}'][-1] for key, *_ in PROTOCOLS))
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]), top=1.15 * ax.get_ylim()[1])
    ax.set_ylabel(ylabel)
    ax.grid(True, which='major', color=COLOR_ISO, alpha=0.3, linewidth=0.5, zorder=0)
    return ax


# --------------------------------------------------------------------------- figure
def add_legend_colorbar(fig, leg, mappable, label, n_free=2, height=0.8, raise_=0.3,
                        pad_pt=8):
    """Horizontal colorbar with its `label` in the `n_free` empty last cells of the second
    column of a 2-column legend, from where that column's handles start to the right edge
    of the legend. `height` and `raise_` (shift up from the center of the free cells,
    leaving room for the tick labels) are in units of the row height.
    Call only after the final layout is fixed (it is placed in figure coordinates)."""
    fig.canvas.draw()   # lays out the legend
    r = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    texts = leg.get_texts()
    nrows = (len(texts) + 1) // 2
    top = texts[nrows - n_free].get_window_extent(r)   # first-column labels of the
    bottom = texts[nrows - 1].get_window_extent(r)     # rows of the free cells
    col2 = texts[nrows].get_window_extent(r)           # first entry of the second column
    frame = leg.get_frame().get_window_extent(r)
    fs = texts[0].get_fontsize() * fig.dpi / 72
    pad = pad_pt * fig.dpi / 72
    x_start = col2.x0 - (leg.handlelength + leg.handletextpad) * fs
    yc = (top.y1 + bottom.y0) / 2
    row_h = (top.y1 - bottom.y0) / n_free

    title = fig.text(*to_fig.transform((x_start, yc)), label, ha='left', va='center',
                     fontsize=texts[0].get_fontsize())
    fig.canvas.draw()
    (x0, y0), (x1, _) = to_fig.transform(
        [(title.get_window_extent(r).x1 + pad, yc + (raise_ - height / 2) * row_h),
         (frame.x1 - 2 * pad, 0)])
    cax = fig.add_axes([x0, y0, x1 - x0, height * row_h / fig.bbox.height])
    cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=FS_ISO, pad=1)
    return cbar


def make_figure(data, datasets, outfile=None):
    """Build the full figure.

    `data` holds T_<key> and B_<key> for the keys of PROTOCOLS (Fig9-optimal-data.npz).
    `datasets` maps the keys of MAP_DATA to dicts with UPSILON, WD, WD0 and T.
    """
    fig = plt.figure(figsize=(20, 12))
    # top row: map (a) | bias panels (b, c)
    # bottom row: legend of (a), with the <tau> colorbar in its last cell | protocol legend
    outer = gridspec.GridSpec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 0.34],
                              wspace=0.3, hspace=0.25)
    right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], hspace=0.08)

    ax_a, sc, handles_a, labels_a = draw_panel_a(fig, outer[0, 0], datasets)
    ax_b = draw_bias_panel(fig, right[0], data, lambda T: 1, r'$B_N$ [$k_B T$]',
                           label_N=True)
    ax_c = draw_bias_panel(fig, right[1], data, lambda T: T / np.log(T),
                           r'$B_N \, N \langle \tau \rangle / \ln(N \langle \tau \rangle)$',
                           sharex=ax_b)
    ax_b.tick_params(labelbottom=False)
    ax_c.set_xlabel(r'Cumulative measurement time $N \langle \tau \rangle$ [s]')

    ax_leg_a = fig.add_subplot(outer[1, 0])
    ax_leg_a.axis('off')
    # two empty entries leave the last two cells of the second column free for the <tau>
    # colorbar (added below, once the layout is fixed)
    blank = Line2D([], [], ls='none')
    leg_a = ax_leg_a.legend(handles_a + [blank] * 2, labels_a + [''] * 2, loc='center',
                            ncol=2, columnspacing=1.2, borderpad=0.9)

    ax_leg = fig.add_subplot(outer[1, 1])
    ax_leg.axis('off')
    ax_leg.legend([Line2D([], [], **STYLE[key]) for key, *_ in PROTOCOLS],
                  [label for _, label, *_ in PROTOCOLS], loc='center', ncol=2,
                  columnspacing=1.5)

    add_legend_colorbar(fig, leg_a, sc, r'$\langle \tau \rangle$ [s]')

    ws.panel_label(ax_a, 'a)', x=-0.12, y=1.02)
    ws.panel_label(ax_b, 'b)', x=-0.1, y=1.04)
    ws.panel_label(ax_c, 'c)', x=-0.1, y=1.04)

    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(here, 'Fig9-optimal-data.npz'))
    datasets = {key: np.load(os.path.join(here, file)) for key, (file, _) in MAP_DATA.items()}

    ws.use(verbose=True)
    make_figure(data, datasets, outfile=os.path.join(here, 'Fig9-optimal.pdf'))
