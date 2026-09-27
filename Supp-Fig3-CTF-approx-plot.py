"""Supp. Fig. 3: CTF approximations.
Upsilon, <W_d>_0(<tau>) - <W_d> and <tau> vs r_U/r_F: simulation, formula + partial simulation, single hopping.
The dissipation saving is taken against the no-feedback protocol with the same mean
cycle time <tau>, as in the main figures.

Run `Supp-Fig3-CTF-approx-simulation.py` first: it writes `Supp-Fig3-CTF-approx-data.npz`,
which this script loads.
"""
import os

import numpy as np

import wiley_style as ws
from plot_helpers import make_approx_figure


def make_figure(data, outfile=None):
    """Build the full figure from the arrays saved by the matching simulation script."""
    ratio_list = data['ratio_list']
    panels = [
        dict(title=r'$k_B T \Upsilon$', unit=r'[$k_B T$]', curves={
            'simulation': (ratio_list, data['Upsilon_JZ']),
            'formula': (ratio_list, data['Upsilon_formula']),
            'estimation': (ratio_list, data['Upsilon_estimation'])}),
        dict(title=r'$\langle W_d \rangle_0(\langle \tau \rangle) - \langle W_d \rangle$', unit=r'[$k_B T$]', curves={
            'simulation': (ratio_list, data['DeltaWdT_JZ']),
            'formula': (ratio_list, data['DeltaWdT_formula']),
            'estimation': (ratio_list, data['DeltaWdT_estimation'])}),
        dict(title=r'$\langle \tau \rangle$', unit='[s]', curves={
            'simulation': (ratio_list, data['T']),
            'formula': (ratio_list, data['T_formula']),
            'estimation': (ratio_list, data['T_estimation'])}),
    ]
    return make_approx_figure(panels, xlabel=r'$r_U/r_F$', xscale='log', outfile=outfile)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(here, 'Supp-Fig3-CTF-approx-data.npz'))

    ws.use(verbose=True)
    make_figure(data, outfile=os.path.join(here, 'Supp-Fig3-CTF-approx.pdf'))
