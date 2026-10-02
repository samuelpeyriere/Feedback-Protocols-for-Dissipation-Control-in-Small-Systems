"""Simulation data for Supp. Fig. 3 (CTF protocol, approximations).

Compares, as a function of r_U/r_F, three ways of computing Upsilon, <W_d> and <tau>:
  - simulation:                   forward runs, Upsilon from the Jarzynski equality (Eq. 20b);
  - formula + partial simulation: the hairpyn CTFProtocol formulas;
  - single hopping:               analytical estimate in the single-hopping approximation.

Saves the results to `Supp-Fig3-CTF-approx-data.npz`, which `Supp-Fig3-CTF-approx-plot.py` loads.
"""
import os
import warnings

# numerical overflow / empty-slice warnings from hairpyn would bury the output
# (set via env var so the joblib worker processes inherit it)
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=RuntimeWarning)

import hairpyn as hp
import numpy as np
from joblib import Parallel, delayed

from none_Wd0 import Wd0_T
from single_hopping import single_hopping_Wd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(HERE, "Supp-Fig3-CTF-approx-data.npz")

rF = 2                               # pN/s
ratio_list = np.logspace(0, 1, 50)   # r_U/r_F
SEED = 0                             # random seed

hp.seed(SEED)

# --------------------------------------------------------------------------- simulation
simulation = hp.Simulation(N=5000, ensemble='LdF')
simulation.protocol = hp.CTFProtocol(rF=rF)


def computation(ratio):
    simulation.protocol.rU = ratio * simulation.protocol.rF

    simulation.run()
    Wd_forward = simulation.dissipated_work()

    # the Jarzynski estimate only uses the forward work
    Upsilon, _ = simulation.Upsilon(Wd_forward, Wd_forward, equation='J')
    T, _ = simulation.avg_time()
    simulation.reset()
    return Upsilon, Wd_forward.mean(), T


Upsilon_JZ, Wd, T = map(np.array, zip(*Parallel(n_jobs=-1)(hp.seeded(delayed(computation)(ratio) for ratio in ratio_list))))

# --------------------------------------------------------------------------- single hopping estimation
model = simulation.model
fmin = simulation.fmin
fmax = simulation.fmax

force_list = np.linspace(fmin, fmax, 1000)
Deltaf = force_list[1] - force_list[0]

Wd0_estimation, _ = simulation.Wd0_estimation()

PsF_rF, _, Wd_rF = single_hopping_Wd(model, force_list, rF)

DeltaWd_estimation = np.zeros_like(ratio_list, dtype=float)
Upsilon_estimation = np.zeros_like(ratio_list, dtype=float)
T_estimation = np.zeros_like(ratio_list, dtype=float)
for k, ratio in np.ndenumerate(ratio_list):
    rU = ratio * rF
    _, PsU_rU, Wd_rU = single_hopping_Wd(model, force_list, rU)

    # average over the first unfolding force f, with density k_FtoU(f)/r_F * PsF(f_min, f):
    # the protocol switches to r_U there
    density = PsF_rF[0] * model.kFtoU(force_list) / rF
    DeltaWd_estimation[k] = (density * (Wd_rF - Wd_rU)).sum() * Deltaf
    Upsilon_estimation[k] = np.log(np.nansum(1 / PsF_rF[:, 0] * model.kUtoF(force_list) * 1 / PsU_rU[-1, :]) * Deltaf / rF)
    T_estimation[k] = (density * ((force_list - fmin) / rF + (fmax - force_list) / rU)).sum() * Deltaf

# --------------------------------------------------------------------------- formula + partial simulation
simulation = hp.Simulation(ensemble='LdF')
simulation.protocol = hp.CTFProtocol(rF=rF)

Wd0_formula, _ = simulation.Wd0()

T_formula = simulation.protocol.avg_time(simulation, ratio_list)
Upsilon_formula = simulation.protocol.Upsilon(simulation, ratio_list)
DeltaWd_formula = simulation.protocol.DeltaWd(simulation, ratio_list)

# --------------------------------------------------------------------------- comparison at equal mean cycle time
# Each method is compared with the no-feedback protocol of the same mean duration <tau>.
# The formula / single-hopping <W_d> are taken relative to the no-feedback protocol at r_F.
T0 = (simulation.fmax - simulation.fmin) / rF
Wd0_rF = Wd0_T(T0)
DeltaWdT_JZ = Wd0_T(T) - Wd
DeltaWdT_formula = Wd0_T(T_formula) - (Wd0_rF - DeltaWd_formula)
DeltaWdT_estimation = Wd0_T(T_estimation) - (Wd0_rF - DeltaWd_estimation)

np.savez(OUTFILE, rF=rF, ratio_list=ratio_list,
         Upsilon_JZ=Upsilon_JZ, Wd=Wd, T=T,
         Upsilon_formula=Upsilon_formula, Wd0_formula=Wd0_formula,
         DeltaWd_formula=DeltaWd_formula, T_formula=T_formula,
         Upsilon_estimation=Upsilon_estimation, Wd0_estimation=Wd0_estimation,
         DeltaWd_estimation=DeltaWd_estimation, T_estimation=T_estimation,
         Wd0_rF=Wd0_rF, DeltaWdT_JZ=DeltaWdT_JZ,
         DeltaWdT_formula=DeltaWdT_formula, DeltaWdT_estimation=DeltaWdT_estimation)
print(f"Saved results to {OUTFILE}")
