"""Fig. 8: Crooks fluctuation theorem with the DTF protocol.
(a) Forward and backward work distributions at one decision force f_1, total (left) and
    conditioned on the observation sigma_1 (right), and the corresponding Crooks plots.
(b) I_U, I_F and Upsilon_2 vs f_1, estimated with the Jarzynski, Bennett and Crooks equalities.

Run `Fig8-CFT-simulation.py` first: it writes `Fig8-CFT-data.npz`, which this script loads.
You can also import `make_figure` and call it directly.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import wiley_style as ws

# --------------------------------------------------------------------------- figure settings
N_BINS = 100
HIST_ALPHA_FW, HIST_ALPHA_BW = 0.35, 0.8
WD_LIM = (-8, 8)          # W_d range of panel (a)
LOG_LIM = (-6, 6)         # y range of the Crooks plot
LEGEND_SPACE = 0.02       # fraction of figure height reserved for the legend at the bottom

# Panel (a): total and conditional distributions
COLOR_TOT = ws.OKABE_ITO['blue']
COLOR_U = ws.OKABE_ITO['vermillion']
COLOR_F = ws.OKABE_ITO['green']

# Panel (b): estimators (key suffix in the data file, label, color, line width)
ESTIMATORS = [
    ('J',      'Jarzynski',   ws.OKABE_ITO['purple'],  ws.LW_THIN),
    ('B',      'Bennett',     ws.OKABE_ITO['skyblue'], ws.LW_THIN),
    ('C',      'Crooks',      ws.OKABE_ITO['gray'],    ws.LW_THIN),
    ('theory', 'Theoretical', ws.OKABE_ITO['black'],   ws.LW_CURVE),
]

# Panel (a) distributions:  suffix of the data keys, color, value symbol, condition label
DISTRIBUTIONS = [
    ('',   COLOR_TOT, r'\Upsilon_2', ''),
    ('_U', COLOR_U,   r'I_U',        r' | \sigma_1 = U'),
    ('_F', COLOR_F,   r'I_F',        r' | \sigma_1 = F'),
]


# --------------------------------------------------------------------------- panel (a) helpers
def common_bins(Wd_fw, Wd_bw):
    return np.histogram_bin_edges(np.concatenate((Wd_fw, -Wd_bw)), bins=N_BINS)


def crooks_points(Wd_fw, Wd_bw):
    """(W_d, ln(rho_->(W_d) / rho_<-(-W_d))) on the bins where both densities are > 0."""
    bins = common_bins(Wd_fw, Wd_bw)
    rho_fw, _ = np.histogram(Wd_fw, bins=bins, density=True)
    rho_bw, _ = np.histogram(-Wd_bw, bins=bins, density=True)
    Wd = (bins[:-1] + bins[1:]) / 2
    ok = (rho_fw > 0) & (rho_bw > 0)
    return Wd[ok], np.log(rho_fw[ok] / rho_bw[ok])


def draw_histograms(ax, Wd_fw, Wd_bw, color, cond):
    """rho_->(W_d) (light) and rho_<-(-W_d) (dark) on common bins. Returns the legend handles."""
    bins = common_bins(Wd_fw, Wd_bw)
    kw = dict(bins=bins, density=True, histtype='stepfilled', color=color)
    *_, p_fw = ax.hist(Wd_fw, alpha=HIST_ALPHA_FW, **kw)
    *_, p_bw = ax.hist(-Wd_bw, alpha=HIST_ALPHA_BW, **kw)
    return [(p_fw[0], r'$\rho_\rightarrow(W_d%s)$' % cond),
            (p_bw[0], r'$\rho_\leftarrow(-W_d%s)$' % cond)]


def format_wd_axis(ax, xlabel=True):
    ax.set_xlim(*WD_LIM)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    if xlabel:
        ax.set_xlabel(r'$W_d$ [$k_B T$]')


# --------------------------------------------------------------------------- panels
def draw_panel_a(fig, spec, ex):
    """Histograms (top) and Crooks plot (bottom), legends in the right column."""
    gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=spec, width_ratios=[1, 1, 0.95],
                                          wspace=0.1, hspace=0.35)
    ax_tot = fig.add_subplot(gs[0, 0])
    ax_cond = fig.add_subplot(gs[0, 1], sharey=ax_tot)
    ax_log = fig.add_subplot(gs[1, :2])

    hist_items, fit_items = [], []
    for suffix, color, symbol, cond in DISTRIBUTIONS:
        Wd_fw, Wd_bw = ex['ex_Wd_fw' + suffix], ex['ex_Wd_bw' + suffix]
        ax_hist = ax_tot if suffix == '' else ax_cond
        hist_items += draw_histograms(ax_hist, Wd_fw, Wd_bw, color, cond)

        # Crooks plot: log ratio, fit of slope 1, and its intercept (dashed)
        x, y = crooks_points(Wd_fw, Wd_bw)
        value = np.mean(y - x)   # least-squares fit of y = x + value
        ax_log.scatter(x, y, color=color, s=30, zorder=3)
        line, = ax_log.plot(WD_LIM, np.add(WD_LIM, value), color=color, linewidth=ws.LW_THIN)
        ax_log.plot([WD_LIM[0], 0], [value, value], color=color, linestyle='--',
                    linewidth=ws.LW_THIN * 0.75)
        fit_items.append((line, r'$W_d + %s$' % symbol))

        # rho_-> and rho_<- cross where W_d = -value
        ax_hist.axvline(-value, color=color, linewidth=ws.LW_THIN)

    ax_tot.set_ylabel('Density')
    ax_tot.set_ylim(0, 0.6)
    ax_tot.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_tot.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax_cond.tick_params(labelleft=False)
    for ax in (ax_tot, ax_cond):
        format_wd_axis(ax)

    format_wd_axis(ax_log)
    ax_log.set_ylim(*LOG_LIM)
    ax_log.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax_log.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax_log.set_ylabel(r'$\ln \left(\frac{\rho_\rightarrow(W_d)}{\rho_\leftarrow(-W_d)}\right)$')
    ax_log.grid()

    for row, items in ((0, hist_items), (1, fit_items)):
        ax_leg = fig.add_subplot(gs[row, 2])
        ax_leg.axis('off')
        ax_leg.legend(*zip(*items), loc='center')
    return ax_tot


def draw_panel_b(fig, spec, data):
    """I_U and I_F stacked on the left, Upsilon_2 on the right. Returns the legend handles."""
    gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, wspace=0.3, hspace=0.08)
    ax_IU = fig.add_subplot(gs[0, 0])
    ax_IF = fig.add_subplot(gs[1, 0], sharex=ax_IU)
    ax_ups = fig.add_subplot(gs[:, 1], sharex=ax_IU)

    cp1_list = data['cp1_list']
    for key, ax, ylabel in (('I_U', ax_IU, r'$I_U$'), ('I_F', ax_IF, r'$I_F$'),
                            ('UPSILON', ax_ups, r'$\Upsilon_2$')):
        for suffix, _, color, lw in ESTIMATORS:
            ax.plot(cp1_list, data['%s_%s' % (key, suffix)], color=color, linewidth=lw,
                    zorder=3 if suffix == 'theory' else 2)
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax_IU.set_xlim(13, 16)
    ax_IU.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_IU.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax_IU.tick_params(labelbottom=False)
    for ax in (ax_IF, ax_ups):
        ax.set_xlabel(r'$f_1$ [pN]')

    handles = [Line2D([], [], color=color, linewidth=ws.LW_CURVE)
               for _, _, color, _ in ESTIMATORS]
    return ax_IU, handles, [label for _, label, _, _ in ESTIMATORS]


# --------------------------------------------------------------------------- figure
def make_figure(data, outfile=None):
    """Build the full figure from the arrays saved by `Fig8-CFT-simulation.py`."""
    fig = plt.figure(figsize=(12, 15 / (1 - LEGEND_SPACE)))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.22)

    ax_a = draw_panel_a(fig, outer[0], data)
    ax_b, handles, labels = draw_panel_b(fig, outer[1], data)

    ws.panel_label(ax_a, 'a)', x=-0.3, y=1.15)
    ws.panel_label(ax_b, 'b)', x=-0.3, y=1.15)

    # layout: tight_layout ignores `rect` with nested gridspecs, so free the bottom strip by hand
    fig.tight_layout()
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + LEGEND_SPACE)
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.005))

    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(here, 'Fig8-CFT-data.npz'))

    ws.use(verbose=True)
    make_figure(data, outfile=os.path.join(here, 'Fig8-CFT.pdf'))
