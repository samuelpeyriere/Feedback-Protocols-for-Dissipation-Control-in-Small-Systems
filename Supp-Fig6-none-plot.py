"""Supp. Fig. 6: dissipated work without feedback, <W_d>_0, vs the mean protocol duration.
Simulation compared with the exact expression (Eq. 25 of the main text); the top axis gives the
corresponding loading rate r_0 = (f_max - f_min) / <tau>.

Loads `Supp-Fig6-none-data.npz`, written by `Supp-Fig6-none-simulation.py` (run automatically if the file is missing).
You can also import `make_figure` and call it directly.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import wiley_style as ws
from none_Wd0 import ensure_data


# --------------------------------------------------------------------------- figure
def make_figure(T0, Wd0, r, Wd0_est, delta_f, outfile=None):
    """Build the full figure.

    `delta_f` = f_max - f_min, which converts a duration into a loading rate.
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))

    # primary axis
    ax1.loglog(T0, Wd0, label='simulation', **ws.curve_style(0, marker=False))
    ax1.loglog(delta_f / r, Wd0_est, label='two-state\nmodel',
               **ws.curve_style(1, marker=False))

    ax1.set_xlabel(r'$\langle \tau \rangle$ [s]')
    ax1.set_ylabel(r'$\langle W_d \rangle_0$ [$k_B T$]')
    ax1.grid()

    # secondary x-axis: loading rate r_0 = delta_f / <tau> (its own inverse)
    def inv(x):
        with np.errstate(divide='ignore'):   # matplotlib probes x = 0 when setting up the axis
            return delta_f / x

    ax2 = ax1.secondary_xaxis('top', functions=(inv, inv))
    ax2.set_xlabel(r'$r_0$ [pN/s]')

    # smaller legend in the empty lower-left corner (label wrapped to fit)
    ax1.legend(loc='lower left', fontsize=ws.FS_TICK)
    fig.tight_layout()

    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = np.load(ensure_data())

    ws.use(verbose=True)
    make_figure(data['T0'], data['Wd0'], data['r'], data['Wd0_est'],
                delta_f=float(data['fmax'] - data['fmin']),
                outfile=os.path.join(here, 'Supp-Fig6-none.pdf'))
