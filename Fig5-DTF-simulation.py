"""Simulation data for Fig. 5 (DTF protocol, LdF ensemble).

The force is loaded at r_F up to the decision force f_1, where the state is observed:
if U, loading continues at r_U, otherwise at r_F. Upsilon, <W_d> and <tau> are computed
with the hairpyn DTFProtocol formulas for 200 values of f_1 and 4 ratios r_U/r_F.

Saves the results to `Fig5-DTF-data.npz`, which `Fig5-DTF-plot.py` loads.
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
OUTFILE = os.path.join(HERE, "Fig5-DTF-data.npz")

rF = 4                                   # pN/s
ratio_list = np.array([1, 4, 40, np.inf])   # r_U/r_F
ensemble = 'LdF'
SEED = 0                                 # random seed

hp.seed(SEED)

simulation = hp.Simulation(ensemble=ensemble)
cp1_list = np.linspace(simulation.fmin, simulation.fmax, 200)   # decision forces f_1 [pN]

UPSILON = np.zeros((len(ratio_list), len(cp1_list)))
DELTAWD = np.zeros((len(ratio_list), len(cp1_list)))
T = np.zeros((len(ratio_list), len(cp1_list)))

with tqdm(total=3 * len(ratio_list), unit="step") as pbar:
    for k in range(len(ratio_list)):
        simulation.protocol = hp.DTFProtocol(rF=rF, rU=ratio_list[k]*rF)
        tag = f"rU/rF={ratio_list[k]:g} ({k + 1}/{len(ratio_list)})"

        pbar.set_description(f"{tag} Upsilon")
        UPSILON[k] = simulation.protocol.Upsilon(simulation, cp1_list)
        pbar.update()

        pbar.set_description(f"{tag} avg_time")
        T[k] = simulation.protocol.avg_time(simulation, cp1_list)
        pbar.update()

        pbar.set_description(f"{tag} DeltaWd")
        DELTAWD[k] = simulation.protocol.DeltaWd(simulation, cp1_list)
        pbar.update()

        simulation.reset()

# no-feedback reference: r_U = r_F (first ratio) is the no-feedback protocol at r_F
Wd0 = Wd0_T(T[0,0])

WD = Wd0 - DELTAWD
WD0 = Wd0_T(T)   # no-feedback <W_d>_0 at the same mean duration <tau>
ETAI = (WD0 - WD)/(WD0 + UPSILON)
ETAM = ((WD0 - WD) - UPSILON)/WD0

np.savez(OUTFILE, ensemble=ensemble, rF=rF, cp1_list=cp1_list, ratio_list=ratio_list,
         UPSILON=UPSILON, WD=WD, WD0=WD0, ETAI=ETAI, ETAM=ETAM, T=T)
print(f"Saved results to {OUTFILE}")
for name, arr in [("UPSILON", UPSILON), ("ETAI", ETAI), ("ETAM", ETAM)]:
    if np.isnan(arr).any():
        print(f"Warning: {name} has {np.isnan(arr).sum()} NaN value(s)")
