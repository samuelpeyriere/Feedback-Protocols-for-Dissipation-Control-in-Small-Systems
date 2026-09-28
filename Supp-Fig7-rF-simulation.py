"""Simulation data for Supp. Fig. 7 (dependence on the loading rate r_F, LdF ensemble).

The protocols of Fig. 9 are kept at fixed parameters and r_F is varied over 40 values:
  - DTF: r_U = inf, f_1 = 16.1 pN (hairpyn DTFProtocol formulas);
  - CTF: r_U/r_F = 17/4 (hairpyn CTFProtocol formulas);
  - constant-DS: r_F'/r_F = 1/4, r_U = inf, f_1 = 14.3 pN (Upsilon from the hairpyn
    Strategy formula, <W_d> and <tau> from 3000 simulated trajectories).
As in Supp. Fig. 6, the time step is scaled with 1/r_F, so that the force step r_F * Deltat = 4e-3 pN
is the same at all rates (Deltat = 1 ms at r_F = 4 pN/s, the value used in Figs. 5-7).

Saves the results to `Supp-Fig7-rF-data.npz`, which `Supp-Fig7-rF-plot.py` loads.
"""
import os
import warnings

# numerical overflow / empty-slice warnings from hairpyn would bury the progress bar
# (set via env var so the joblib worker processes inherit it)
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=RuntimeWarning)

import hairpyn as hp
import numpy as np
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

from none_Wd0 import Wd0_T

HERE = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(HERE, "Supp-Fig7-rF-data.npz")

rF_list = np.logspace(-1, 2, 40)   # pN/s
DTF_f1 = 16.1                      # pN
CTF_ratio = 17/4                   # r_U/r_F
CDS_ratio, CDS_f1 = 1/4, 14.3      # r_F'/r_F, f_1 [pN]
N_CDS = 3000                       # trajectories per r_F for the constant-DS protocol
N_JOBS = 12                        # a job holds up to ~1 GB of trajectories
ensemble = 'LdF'
protocols = np.array(['DTF', 'CTF', 'CDS'])


def new_simulation(rF, N=1):
    simulation = hp.Simulation(N=N, ensemble=ensemble)
    simulation.Deltat = 1e-3*(4/rF)
    return simulation


def computation(rF):
    """(Upsilon, <W_d>, <tau>) of each protocol at loading rate r_F."""
    # the hairpyn formulas run their own Parallel calls on a shared Simulation: nested in a
    # joblib worker these would be threads racing on it, so they are made sequential here
    with parallel_config(backend='sequential'):
        return _computation(rF)


def _computation(rF):
    simulation = new_simulation(rF)
    Wd0 = Wd0_T((simulation.fmax - simulation.fmin)/rF)   # no-feedback <W_d>_0 at r_F

    # DTF: DeltaWd is referenced to its value at the last f_1 (f_max), which must be in the list
    simulation.protocol = hp.DTFProtocol(rF=rF, rU=np.inf)
    cp1 = np.array([DTF_f1, simulation.fmax])
    dtf = (simulation.protocol.Upsilon(simulation, cp1)[0],
           Wd0 - simulation.protocol.DeltaWd(simulation, cp1)[0],
           simulation.protocol.avg_time(simulation, cp1)[0])

    simulation = new_simulation(rF)
    simulation.protocol = hp.CTFProtocol(rF=rF)
    ratio = np.array([CTF_ratio])
    ctf = (simulation.protocol.Upsilon(simulation, ratio)[0],
           Wd0 - simulation.protocol.DeltaWd(simulation, ratio)[0],
           simulation.protocol.avg_time(simulation, ratio)[0])

    simulation = new_simulation(rF, N=N_CDS)
    simulation.protocol = hp.Strategy(rF=rF, rFprime=CDS_ratio*rF, rU=np.inf, cp1=CDS_f1)
    Upsilon = simulation.protocol.Upsilon(simulation, np.array([CDS_f1]))[0]
    simulation.run()
    Wd = simulation.dissipated_work().mean()
    t, _ = simulation.avg_time()
    simulation.reset()
    cds = (Upsilon, Wd, t)

    return dtf, ctf, cds


# one progress step per loading rate
results = list(tqdm(Parallel(n_jobs=N_JOBS, return_as="generator")(
    delayed(computation)(rF) for rF in rF_list), total=len(rF_list), unit="rate", desc="r_F"))
# arrays of shape (len(protocols), len(rF_list))
UPSILON, WD, T = np.array(results, dtype=float).transpose(2, 1, 0)

WD0 = Wd0_T(T)   # no-feedback <W_d>_0 at the same mean duration <tau>
ETAI = (WD0 - WD)/(WD0 + UPSILON)
ETAM = ((WD0 - WD) - UPSILON)/WD0

simulation = hp.Simulation(ensemble=ensemble)
T_none = (simulation.fmax - simulation.fmin)/rF_list   # no-feedback reference at r_F
np.savez(OUTFILE, ensemble=ensemble, rF_list=rF_list, protocols=protocols,
         DTF_f1=DTF_f1, CTF_ratio=CTF_ratio, CDS_ratio=CDS_ratio, CDS_f1=CDS_f1,
         UPSILON=UPSILON, WD=WD, WD0=WD0, ETAI=ETAI, ETAM=ETAM, T=T,
         T_none=T_none, WD_none=Wd0_T(T_none))
print(f"Saved results to {OUTFILE}")
for name, arr in [("UPSILON", UPSILON), ("WD", WD), ("ETAI", ETAI), ("ETAM", ETAM)]:
    if np.isnan(arr).any():
        print(f"Warning: {name} has {np.isnan(arr).sum()} NaN value(s)")
