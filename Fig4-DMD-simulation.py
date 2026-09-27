"""Simulation data for Fig. 4 (discrete Maxwell demon, MD protocol, LdF ensemble).

The molecule is equilibrated at f_0 and its state is measured: the force then jumps to
f_0 + Deltaf if U is observed, to f_0 - Deltaf if F is observed, and returns quasi-statically
to f_0 at rate r_QS. The cycle is repeated for 50 values of f_0.

Saves the results to `Fig4-DMD-data.npz`, which `Fig4-DMD-plot.py` loads.
Row 0 of UPSILON / WD / ETAI is the simulation, row 1 the theory.
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
OUTFILE = os.path.join(HERE, "Fig4-DMD-data.npz")

N = 5000
Deltat = 1e-2   # s
Deltaf = 1      # pN
rQS = 0.01      # pN/s, quasi-static return rate
f0_list = np.linspace(13, 16.25, 50)


def computation(f0):
    simulation = hp.Simulation(N=N, ensemble='LdF')
    simulation.protocol = hp.MDProtocol(f0=f0, Deltaf=Deltaf, rQS=rQS)
    simulation.Deltat = Deltat

    simulation.run(pre_thermalise=True)
    # The MD cycle starts and ends at f0, so the equilibrium free-energy difference
    # is zero and the dissipated work is the total work W. Do NOT use
    # simulation.dissipated_work() here: it subtracts the *state-dependent*
    # DeltaG(sigma_end) - DeltaG(sigma_start), which adds rare +-beta*(DeltaG0 - f0*xm)
    # (~5 kT) jumps that dominate the Jarzynski average log<exp(-Wd)> and push
    # Upsilon far above theory away from f_c.
    Wd_forward = simulation.work()
    Upsilon, _ = simulation.Upsilon(Wd_forward, Wd_forward, equation='J')
    Wd = Wd_forward.mean()
    simulation.reset()

    # equilibrium probabilities of U, for the theory
    pU_f0 = simulation.thermalise(f0, 1).mean()
    pU_f0mDf = simulation.thermalise(f0 - Deltaf, 1).mean()
    pU_f0pDf = simulation.thermalise(f0 + Deltaf, 1).mean()
    return Upsilon, Wd, pU_f0, pU_f0mDf, pU_f0pDf


results = Parallel(n_jobs=-1, return_as='generator')(delayed(computation)(f0) for f0 in f0_list)
Upsilon, Wd, pU_f0, pU_f0mDf, pU_f0pDf = map(
    np.array, zip(*tqdm(results, total=len(f0_list), unit="f0")))

# theory: Upsilon = ln(P_U(f0+Deltaf) + P_F(f0-Deltaf)),  <Wd> = -S(P_U(f0))
with np.errstate(divide='ignore', invalid='ignore'):
    minus_entropy = np.nan_to_num(pU_f0 * np.log(pU_f0) + (1 - pU_f0) * np.log(1 - pU_f0))
UPSILON = np.vstack((Upsilon, np.log(pU_f0pDf + (1 - pU_f0mDf))))
WD = np.vstack((Wd, minus_entropy))
ETAI = -WD / UPSILON

np.savez(OUTFILE, f0_list=f0_list, Deltat=Deltat, Deltaf=Deltaf, rQS=rQS,
         UPSILON=UPSILON, WD=WD, ETAI=ETAI, PU=pU_f0)
print(f"Saved results to {OUTFILE}")
