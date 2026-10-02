"""Simulation data for Supp. Fig. 2 (DTF protocol, approximations).

Compares, as a function of the decision force f_1, three ways of computing Upsilon,
<W_d> and <tau>:
  - simulation:                   forward runs, Upsilon from the Jarzynski equality (Eq. 20b);
  - formula + partial simulation: the hairpyn DTFProtocol formulas;
  - single hopping:               analytical estimate in the single-hopping approximation.

Saves the results to `Supp-Fig2-DTF-approx-data.npz`, which `Supp-Fig2-DTF-approx-plot.py` loads.
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
OUTFILE = os.path.join(HERE, "Supp-Fig2-DTF-approx-data.npz")

rF = 4      # pN/s
rU = 17     # pN/s
cp1_list = np.linspace(8, 22, 100)   # decision forces f_1 [pN]
SEED = 0                             # random seed

hp.seed(SEED)

# --------------------------------------------------------------------------- simulation
simulation = hp.Simulation(N=10000, ensemble='LdF')
simulation.protocol = hp.DTFProtocol(rF=rF, rU=rU)

Wd0, _ = simulation.Wd0()   # no-feedback <W_d> at r_F, simulated


def computation(f1):
    simulation.protocol.cp1 = f1

    simulation.run()
    Wd_forward = simulation.dissipated_work()

    # the Jarzynski estimate only uses the forward work
    Upsilon, _ = simulation.Upsilon(Wd_forward, Wd_forward, equation='J')
    T, _ = simulation.avg_time()
    simulation.reset()
    return Upsilon, Wd_forward.mean(), T


Upsilon_JZ, Wd, T = map(np.array, zip(*Parallel(n_jobs=-1)(hp.seeded(delayed(computation)(f1) for f1 in cp1_list))))

# --------------------------------------------------------------------------- single hopping estimation
model = simulation.model
fmin = simulation.fmin
fmax = simulation.fmax

force_list = np.linspace(8, 22, 5000)   # f_1 grid of the estimation

Wd0_estimation, _ = simulation.Wd0_estimation()

PsF_rF, PsU_rF, Wd_rF = single_hopping_Wd(model, force_list, rF)
_, PsU_rU, Wd_rU = single_hopping_Wd(model, force_list, rU)

# U observed at f_1 (probability 1 - PsF(f_min, f_1)): the rest of the ramp runs at r_U
DeltaWd_estimation = (1 - PsF_rF[0]) * (Wd_rF - Wd_rU)
Upsilon_estimation = np.log(1 / PsU_rU[-1] + 1 - 1 / PsU_rF[-1])
T_estimation = (force_list - fmin) / rF \
    + (fmax - force_list) * (1 / rU + (1 / rF - 1 / rU) * model.PsF(fmin, force_list, rF))

# --------------------------------------------------------------------------- formula + partial simulation
simulation = hp.Simulation(N=1000, ensemble='LdF')
simulation.protocol = hp.DTFProtocol(rF=rF, rU=rU)

Upsilon_formula = simulation.protocol.Upsilon(simulation, cp1_list)
T_formula = simulation.protocol.avg_time(simulation, cp1_list)
DeltaWd_formula = simulation.protocol.DeltaWd(simulation, cp1_list)
Wd_formula = Wd0 - DeltaWd_formula

# --------------------------------------------------------------------------- comparison at equal mean cycle time
# Each method is compared with the no-feedback protocol of the same mean duration <tau>.
# The formula / single-hopping <W_d> are taken relative to the no-feedback protocol at r_F.
T0 = (simulation.fmax - simulation.fmin) / rF
Wd0_rF = Wd0_T(T0)
DeltaWdT_JZ = Wd0_T(T) - Wd
DeltaWdT_formula = Wd0_T(T_formula) - (Wd0_rF - DeltaWd_formula)
DeltaWdT_estimation = Wd0_T(T_estimation) - (Wd0_rF - DeltaWd_estimation)

np.savez(OUTFILE, rF=rF, rU=rU, cp1_list=cp1_list, force_list=force_list,
         Upsilon_JZ=Upsilon_JZ, Wd=Wd, T=T,
         Upsilon_formula=Upsilon_formula, Wd_formula=Wd_formula, T_formula=T_formula,
         Upsilon_estimation=Upsilon_estimation, Wd0_estimation=Wd0_estimation,
         DeltaWd_estimation=DeltaWd_estimation, T_estimation=T_estimation,
         DeltaWd_formula=DeltaWd_formula, Wd0_rF=Wd0_rF, DeltaWdT_JZ=DeltaWdT_JZ,
         DeltaWdT_formula=DeltaWdT_formula, DeltaWdT_estimation=DeltaWdT_estimation)
print(f"Saved results to {OUTFILE}")
