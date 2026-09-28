"""Supp. Fig. 2: DTF approximations.
Upsilon, <W_d>_0(<tau>) - <W_d> and <tau> vs f_1: simulation, formula + partial simulation, single hopping.
The dissipation saving is taken against the no-feedback protocol with the same mean
cycle time <tau>, as in the main figures.

Run `Supp-Fig2-DTF-approx-simulation.py` first: it writes `Supp-Fig2-DTF-approx-data.npz`,
which this script loads.
"""
import os

import numpy as np

import wiley_style as ws
from plot_helpers import make_approx_figure


def make_figure(data, outfile=None):
    """Build the full figure from the arrays saved by the matching simulation script."""
    cp1_list, force_list = data['cp1_list'], data['force_list']
    panels = [
        dict(title=r'$k_B T \Upsilon_2$', unit=r'[$k_B T$]', curves={
            'simulation': (cp1_list, data['Upsilon_JZ']),
            'formula': (cp1_list, data['Upsilon_formula']),
            'estimation': (force_list, data['Upsilon_estimation'])}),
        dict(title=r'$\langle W_d \rangle_0(\langle \tau \rangle) - \langle W_d \rangle$', unit=r'[$k_B T$]', curves={
            'simulation': (cp1_list, data['DeltaWdT_JZ']),
            'formula': (cp1_list, data['DeltaWdT_formula']),
            'estimation': (force_list, data['DeltaWdT_estimation'])}),
        dict(title=r'$\langle \tau \rangle$', unit='[s]', curves={
            'simulation': (cp1_list, data['T']),
            'formula': (cp1_list, data['T_formula']),
            'estimation': (force_list, data['T_estimation'])}),
    ]
    return make_approx_figure(panels, xlabel=r'$f_1$ [pN]', outfile=outfile)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(here, 'Supp-Fig2-DTF-approx-data.npz'))

    ws.use(verbose=True)
    make_figure(data, outfile=os.path.join(here, 'Supp-Fig2-DTF-approx.pdf'))
