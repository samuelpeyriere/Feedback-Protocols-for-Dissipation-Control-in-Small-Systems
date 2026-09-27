"""Dissipated work without feedback, <W_d>_0, as a function of the protocol duration <tau>.

Linear interpolation, in log-log, of the mean-field estimate (Eq. 25 of the main text) stored in
`Supp-Fig6-none-data.npz` (LdF ensemble), with <tau> = (f_max - f_min) / r_0.
If that file does not exist yet, `Supp-Fig6-none-simulation.py` is run first to create it.
"""
import os
import subprocess
import sys

import numpy as np
from scipy.interpolate import make_interp_spline

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILE = os.path.join(HERE, "Supp-Fig6-none-data.npz")
SIMULATION = os.path.join(HERE, "Supp-Fig6-none-simulation.py")


def ensure_data():
    """Run the no-feedback simulation if its data file is missing; return the file path."""
    if not os.path.exists(DATAFILE):
        print(f"{os.path.basename(DATAFILE)} not found: running {os.path.basename(SIMULATION)}")
        subprocess.run([sys.executable, SIMULATION], check=True)
    return DATAFILE


_data = np.load(ensure_data())
_T0 = (_data['fmax'] - _data['fmin']) / _data['r']
_spline = make_interp_spline(-np.log(_T0), np.log(_data['Wd0_est']), k=1)


def Wd0_T(T):
    """<W_d>_0 [k_B T] of the no-feedback protocol of mean duration T [s] (scalar or array)."""
    return np.exp(_spline(-np.log(T)))
