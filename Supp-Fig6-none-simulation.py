"""Simulation data for Supp. Fig. 6 (no feedback, LdF ensemble).

Dissipated work <W_d>_0 of the no-feedback protocol (constant loading rate r from f_min
to f_max) for 200 loading rates, simulated and in the mean-field approximation (Eq. 25 of
the main text).

Saves the results to `Supp-Fig6-none-data.npz`, which `Supp-Fig6-none-plot.py` loads.
`none_Wd0.py` interpolates the mean-field values to give <W_d>_0 at any mean duration <tau>,
which is the no-feedback reference of Figs. 5-7 and Supp. Figs. 2-4.
"""
import os
import warnings

# numerical overflow / empty-slice warnings from hairpyn would bury the progress bar
# (set via env var so the joblib worker processes inherit it)
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=RuntimeWarning)

import hairpyn as hp
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(HERE, "Supp-Fig6-none-data.npz")

ensemble = 'LdF'
simulation = hp.Simulation(N=10000, ensemble=ensemble)
r = np.logspace(-2, 2, 200)   # loading rates [pN/s]


def computation(r):
    # time step scaled with 1/r, so that the force step r * Deltat = 4e-3 pN is the same at all rates
    simulation.Deltat = 1e-3*(4/r)
    simulation.protocol = hp.NOProtocol(rF=r)
    simulation.run()
    Wd0 = simulation.dissipated_work().mean()
    Wd0_est, _ = simulation.Wd0_estimation()
    T0, _ = simulation.avg_time()
    simulation.reset()
    return Wd0, T0, Wd0_est


# one progress step per loading rate simulated
results = list(tqdm(Parallel(n_jobs=-1, return_as="generator")(delayed(computation)(r_) for r_ in r),
                    total=len(r), unit="rate", desc="r"))
Wd0, T0, Wd0_est = map(np.array, zip(*results))

np.savez(OUTFILE, ensemble=ensemble, fmin=simulation.fmin, fmax=simulation.fmax,
         r=r, Wd0=Wd0, T0=T0, Wd0_est=Wd0_est)
print(f"Saved results to {OUTFILE}")
for name, arr in [("Wd0", Wd0), ("T0", T0), ("Wd0_est", Wd0_est)]:
    if np.isnan(arr).any():
        print(f"Warning: {name} has {np.isnan(arr).sum()} NaN value(s)")
