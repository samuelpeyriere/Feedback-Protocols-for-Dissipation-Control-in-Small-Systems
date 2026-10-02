"""Simulation data for Fig. 7 (constant-DS protocol, "CDS" in file names; LdF = force ensemble).

The force is loaded at r_F up to the decision force f_1, where the state is observed:
if U, the force goes to f_max at r_U; if F, loading continues at r_F' until the next
unfolding, then goes to f_max at r_U. Upsilon comes from the hairpyn Strategy formula,
<W_d> and <tau> directly from simulations, for 200 values of f_1 and 3 ratios r_F'/r_F.

Saves the results to `Fig7-CDS-data.npz`, which `Fig7-CDS-plot.py` loads.
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

from none_Wd0 import Wd0_T

HERE = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(HERE, "Fig7-CDS-data.npz")

rF = 4                                 # pN/s
rU = np.inf                            # pN/s
ratio_list = np.array([1/4, 1/2, 1])   # values of r_F'/r_F
f1_list = np.linspace(8, 22, 200)      # decision forces f_1 [pN]
N = 3000                               # trajectories per f_1
ensemble = 'LdF'
SEED = 0                               # random seed

hp.seed(SEED)

simulation = hp.Simulation(ensemble=ensemble)
Wd0 = Wd0_T((simulation.fmax - simulation.fmin)/rF)   # no-feedback <W_d>_0 at r_F

UPSILON = np.zeros((len(ratio_list), len(f1_list)))
WD = np.zeros((len(ratio_list), len(f1_list)))
T = np.zeros((len(ratio_list), len(f1_list)))

with tqdm(total=len(ratio_list) * (1 + len(f1_list)), unit="step") as pbar:
    for k in range(len(ratio_list)):
        simulation = hp.Simulation(N=N, ensemble=ensemble)
        simulation.protocol = hp.Strategy(rF=rF, rFprime=ratio_list[k]*rF, rU=rU)
        tag = f"rF'/rF={ratio_list[k]:g} ({k + 1}/{len(ratio_list)})"

        pbar.set_description(f"{tag} Upsilon")
        UPSILON[k] = simulation.protocol.Upsilon(simulation, f1_list)
        pbar.update()

        def func(f1):
            simulation.protocol.f1 = f1
            simulation.run()
            Wd = simulation.dissipated_work().mean()
            t, _ = simulation.avg_time()
            simulation.reset()
            return Wd, t

        # one progress step per f_1 value simulated
        pbar.set_description(f"{tag} runs")
        results = []
        for res in Parallel(n_jobs=-1, return_as="generator")(hp.seeded(delayed(func)(f1) for f1 in f1_list)):
            results.append(res)
            pbar.update()
        WD[k], T[k] = map(np.array, zip(*results))

WD0 = Wd0_T(T)   # no-feedback <W_d>_0 at the same mean duration <tau>
ETAI = (WD0 - WD)/(WD0 + UPSILON)
ETAM = ((WD0 - WD) - UPSILON)/WD0

np.savez(OUTFILE, ensemble=ensemble, rF=rF, rU=rU, Wd0=Wd0, f1_list=f1_list,
         ratio_list=ratio_list, UPSILON=UPSILON, WD=WD, WD0=WD0, ETAI=ETAI, ETAM=ETAM, T=T)
print(f"Saved results to {OUTFILE}")
for name, arr in [("UPSILON", UPSILON), ("WD", WD), ("ETAI", ETAI), ("ETAM", ETAM)]:
    if np.isnan(arr).any():
        print(f"Warning: {name} has {np.isnan(arr).sum()} NaN value(s)")
