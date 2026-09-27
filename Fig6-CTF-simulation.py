"""Simulation data for Fig. 6 (CTF protocol, LdF ensemble).

The force is loaded at r_F until the first unfolding is observed, then at r_U up to f_max.
Upsilon, <W_d> and <tau> are computed with the hairpyn CTFProtocol formulas for 3 loading
rates r_F and 11 ratios r_U/r_F:
  - Upsilon: analytic Bell-Evans survival of F and k_UtoF, with the survival of U at -r_U
    measured on 10000 backward (unloading) trajectories per r_U (NaN at large r_U/r_F, where
    the rare events that dominate the integral are not sampled);
  - <W_d>: <W_d>_0 at r_F minus DeltaWd, from 3000 no-feedback trajectories started in U
    at each of 50 forces, for r_F and for each r_U;
  - <tau>: analytic (first-unfolding density times the duration).

Saves the results to `Fig6-CTF-data.npz`, which `Fig6-CTF-plot.py` loads.
"""
import os
import warnings

# numerical overflow / empty-slice warnings from hairpyn would bury the progress bar
# (set via env var so the joblib worker processes inherit it)
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=RuntimeWarning)

import hairpyn as hp
import numpy as np
from tqdm import tqdm

from none_Wd0 import Wd0_T

HERE = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(HERE, "Fig6-CTF-data.npz")

rF_list = np.array([1, 4, 10])                                  # pN/s
ratio_list = np.concatenate([np.logspace(0, 1, 10), [np.inf]])   # r_U/r_F

UPSILON = np.zeros((len(rF_list), len(ratio_list)))
T = np.zeros((len(rF_list), len(ratio_list)))
WD = np.zeros((len(rF_list), len(ratio_list)))

with tqdm(total=3 * len(rF_list), unit="step") as pbar:
    for k in range(len(rF_list)):
        simulation = hp.Simulation(ensemble='LdF')
        simulation.protocol = hp.CTFProtocol(rF=rF_list[k])   # CTFProtocol sets its own N
        tag = f"rF={rF_list[k]} ({k + 1}/{len(rF_list)})"

        pbar.set_description(f"{tag} Upsilon")
        UPSILON[k] = simulation.protocol.Upsilon(simulation, ratio_list)
        pbar.update()

        pbar.set_description(f"{tag} avg_time")
        T[k] = simulation.protocol.avg_time(simulation, ratio_list)
        pbar.update()

        # <W_d> = no-feedback <W_d>_0 at r_F minus the reduction DeltaWd
        pbar.set_description(f"{tag} DeltaWd")
        WD[k] = Wd0_T((simulation.fmax - simulation.fmin) / rF_list[k]) - simulation.protocol.DeltaWd(simulation, ratio_list)
        pbar.update()

        simulation.reset()

WD0 = Wd0_T(T)   # no-feedback <W_d>_0 at the same mean duration <tau>

ETAI = (WD0 - WD)/(WD0 + UPSILON)
ETAM = ((WD0 - WD) - UPSILON)/WD0

np.savez(OUTFILE, rF_list=rF_list, ratio_list=ratio_list, UPSILON=UPSILON,
         WD=WD, WD0=WD0, ETAI=ETAI, ETAM=ETAM, T=T)
print(f"Saved results to {OUTFILE}")
for name, arr in [("UPSILON", UPSILON), ("ETAI", ETAI), ("ETAM", ETAM)]:
    if np.isnan(arr).any():
        print(f"Warning: {name} has {np.isnan(arr).sum()} NaN value(s)")
