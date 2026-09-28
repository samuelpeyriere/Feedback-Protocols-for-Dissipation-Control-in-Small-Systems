"""Supp. Fig. 7: dependence of the protocols of Fig. 9 on the loading rate r_F (LdF ensemble).
Thermodynamic quantities vs r_F (log scale) for the DTF, CTF and constant-DS protocols at the
fixed parameters of Fig. 9, with the no-feedback protocol at r_F as reference. The dotted
vertical line marks r_F = 4 pN/s, the value used in Figs. 5-7.

Run `Supp-Fig7-rF-simulation.py` first: it writes `Supp-Fig7-rF-data.npz`, which this script loads.
You can also import `make_figure` and call it directly.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import wiley_style as ws

# --------------------------------------------------------------------------- figure settings
COLOR_NONE = ws.OKABE_ITO['black']    # no-feedback reference
COLOR_RF_MAIN = ws.OKABE_ITO['gray']  # r_F of Figs. 5-7
RF_MAIN = 4                           # pN/s

LEGEND_SPACE = 0.1        # fraction of figure height reserved for the legend at the bottom

# (row, col), key in `quantities`, title, unit, log y, ylim, key of the no-feedback reference
PANELS = [
    ((0, 0), 'UPSILON', r'$k_B T \Upsilon$',        r'[$k_B T$]', False, None,       None),
    ((0, 1), 'WD',      r'$\langle W_d \rangle$',   r'[$k_B T$]', False, None,       'WD_none'),
    ((0, 2), 'WD0',     r'$\langle W_d \rangle_0$', r'[$k_B T$]', True,  None,       'WD_none'),
    ((1, 0), 'ETAI',    r'$\eta_I$',                '',           False, None,       None),
    ((1, 1), 'ETAM',    r'$\eta_M$',                '',           False, (-1.5, 1),  None),
    ((1, 2), 'T',       r'$\langle \tau \rangle$',  '[s]',        True,  None,       'T_none'),
]


def protocol_labels(data):
    return {
        'DTF': r'DTF ($r_U = \infty$, $f_1 = %g$ pN)' % data['DTF_f1'],
        'CTF': r'CTF ($r_U = %g \cdot r_F$)' % data['CTF_ratio'],
        'CDS': r"constant-DS ($r_{F'} = r_F/%g$, $r_U = \infty$, $f_1 = %g$ pN)"
               % (1 / data['CDS_ratio'], data['CDS_f1']),
    }


# --------------------------------------------------------------------------- figure
def make_figure(data, outfile=None):
    """Build the full figure from the arrays saved by the matching simulation script
    (quantities of shape (len(protocols), len(rF_list)))."""
    rF = data['rF_list']
    labels = protocol_labels(data)

    # taller figure so the subplots keep ~ their size despite the legend strip
    fig = plt.figure(figsize=(14, 7.5 / (1 - LEGEND_SPACE)))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.25)

    for (row, col), key, title, unit, logy, ylim, ref in PANELS:
        ax = fig.add_subplot(gs[row, col])
        handles = []
        for k, protocol in enumerate(data['protocols']):
            line, = ax.plot(rF, data[key][k], label=labels[str(protocol)],
                            **ws.curve_style(k, marker=False))
            handles.append(line)
        if ref is not None:
            line, = ax.plot(rF, data[ref], color=COLOR_NONE, linewidth=ws.LW_THIN,
                            label=r'no feedback at $r_F$')
            handles.append(line)
        if key in ('ETAI', 'ETAM', 'UPSILON', 'WD'):
            ax.axhline(0, color=COLOR_RF_MAIN, linestyle=':', linewidth=1, zorder=0)
        ax.axvline(RF_MAIN, color=COLOR_RF_MAIN, linestyle=':', linewidth=ws.LW_THIN, zorder=0)

        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title(title, loc='left')
        ax.set_title(unit, loc='right')
        if row == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r'$r_F$ [pN/s]')
        if ref is not None:
            legend_handles = handles   # a panel that has every curve

    # layout: free the bottom strip for the legend by hand (as in Figs. 5-7)
    fig.tight_layout()
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + LEGEND_SPACE)
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, 0.005))

    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(here, 'Supp-Fig7-rF-data.npz'))

    ws.use(verbose=True)
    make_figure(data, outfile=os.path.join(here, 'Supp-Fig7-rF.pdf'))
