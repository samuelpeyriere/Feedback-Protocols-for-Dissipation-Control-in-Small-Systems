"""Simulation data for Supp. Fig. 4 (constant-DS protocol, "CDS" in file names; approximations).

Compares, as a function of the decision force f_1, three ways of computing Upsilon,
<W_d> and <tau>:
  - simulation:                   forward runs, Upsilon from the Jarzynski equality (Eq. 20b);
  - formula + partial simulation: formulas fed with no-feedback simulations;
  - single hopping:               analytical estimate in the single-hopping approximation.
r_U = inf as in Fig. 7.

Saves the results to `Supp-Fig4-CDS-approx-data.npz`, which `Supp-Fig4-CDS-approx-plot.py` loads.
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
from single_hopping import single_hopping_Wd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(HERE, "Supp-Fig4-CDS-approx-data.npz")

rF = 4          # pN/s, loading rate up to the decision force f_1
rFprime = 1     # pN/s, loading rate after F was observed at f_1
rU = np.inf     # pN/s, rate once U is observed

f1_list = np.linspace(8.01, 22, 100)   # starts just above f_min = 8 pN

# --------------------------------------------------------------------------- simulation
simulation = hp.Simulation(N=1000, ensemble='LdF')
simulation.protocol = hp.Strategy(rF=rF, rFprime=rFprime, rU=rU)

Wd0, _ = simulation.Wd0()   # no-feedback <W_d> at r_F, simulated


def computation(f1):
    simulation.protocol.f1 = f1

    simulation.run()
    Wd_forward = simulation.dissipated_work()

    # the Jarzynski estimate only uses the forward work
    Upsilon, _ = simulation.Upsilon(Wd_forward, Wd_forward, equation='J')
    T, _ = simulation.avg_time()
    simulation.reset()
    return Upsilon, Wd_forward.mean(), T


Upsilon_JZ, Wd, T = map(np.array, zip(*Parallel(n_jobs=-1)(delayed(computation)(f1) for f1 in f1_list)))

# --------------------------------------------------------------------------- formula + partial simulation
model = simulation.model
fmin = simulation.fmin
fmax = simulation.fmax

f = np.linspace(fmin, fmax, 500)   # f_1 grid of the formula
df = f[1] - f[0]
bin_edges = np.concatenate([[f[0] - df / 2], f[:-1] + df / 2, [f[-1] + df / 2]])
X, Y = np.meshgrid(f, f)

PsF_rF = model.PsF(Y, X, rF)
PsF_rFprime = model.PsF(Y, X, rFprime)

simulation = hp.Simulation(N=1000, ensemble='LdF')
simulation.protocol = hp.Strategy(rF=rF, rFprime=rFprime, rU=rU)
Upsilon_formula = simulation.protocol.Upsilon(simulation, f)

# PhiU[k]: probability of being unfolded at f[k] when pulling at r_F without feedback,
# from the fraction of unfolded trajectories at each time step, binned by force
simulation = hp.Simulation(N=1000, ensemble='LdF')
simulation.protocol = hp.NOProtocol(rF=rF)
simulation.run()
pU = np.sum(simulation.state_list, axis=1) / simulation.N
bin_indices = np.digitize(simulation.force_list[:, 0], bin_edges, right=True)
simulation.reset()
PhiU = np.zeros_like(f, dtype=float)
for i in range(1, len(bin_edges)):
    bin_elements = pU[bin_indices == i]
    PhiU[i - 1] = np.mean(bin_elements) if len(bin_elements) > 0 else 0

# WDrF[k] / WDrU[k]: simulated <W_d> of a no-feedback pull at r_F / r_U from f[k] to f_max,
# starting in U
simulation = hp.Simulation(N=1000, initial_state=True, ensemble='LdF')
simulation.protocol = hp.NOProtocol(rF=rF)
WDrF = np.zeros_like(f)
for k, f0 in tqdm(np.ndenumerate(f), total=len(f), desc='W_d at r_F'):
    simulation.fmin = f0
    simulation.run(pre_thermalise=False)
    WDrF[k] = simulation.dissipated_work().mean()
    simulation.reset()

WDrU = np.zeros_like(f)
if np.isfinite(rU):
    simulation = hp.Simulation(N=1000, initial_state=True, ensemble='LdF')
    simulation.protocol = hp.NOProtocol(rF=rU)
    for k, f0 in tqdm(np.ndenumerate(f), total=len(f), desc='W_d at r_U'):
        simulation.fmin = f0
        simulation.run(pre_thermalise=False)
        WDrU[k] = simulation.dissipated_work().mean()
        simulation.reset()
# rU = inf: instantaneous jump to f_max in U, no transition, hence no dissipation (WDrU = 0)

DeltaWd_formula = np.zeros_like(f, dtype=float)
T_formula = np.zeros_like(f, dtype=float)

# F at f_1 = f[k]: unfolding at f' = f[k:] dissipates beta*(f'-f_c)*x_m, then WD(f') from U at f'.
# Both terms must be taken at the unfolding force f', not at f_1 (WDrF[k] starts in U at f_1 < f_c,
# refolds immediately and dissipates ~beta*(f_c-f_1)*x_m, which does not happen on the F branch).
for k in range(len(f)):
    DeltaWd_formula[k] = PhiU[k]*(WDrF[k] - WDrU[k]) + (1-PhiU[k])*(np.nansum(PsF_rF[k,k:]*model.kFtoU(f[k:])*(WDrF[k:]+model.beta*(f[k:]-model.fc)*model.xm))*df/rF
                                                                    - np.nansum(PsF_rFprime[k,k:]*model.kFtoU(f[k:])*(WDrU[k:]+model.beta*(f[k:]-model.fc)*model.xm))*df/rFprime)
    T_formula[k] = (f[k]-fmin)/rF + PhiU[k]*(fmax - f[k])/rU + (1-PhiU[k])*np.nansum(PsF_rFprime[k,k:]*model.kFtoU(f[k:])*((f[k:]-f[k])/rFprime + (fmax - f[k:])/rU))*df/rFprime

# --------------------------------------------------------------------------- single hopping estimation
force_list = np.linspace(fmin, fmax, 10000)   # f_1 grid of the estimation
Deltaf = force_list[1] - force_list[0]
X, Y = np.meshgrid(force_list, force_list)

DeltaWd_estimation = np.zeros_like(force_list, dtype=float)
Upsilon_estimation = np.zeros_like(force_list, dtype=float)
T_estimation = np.zeros_like(force_list, dtype=float)

PsF_rFprime = model.PsF(Y, X, rFprime)
PsF_rF, _, Wd_rF = single_hopping_Wd(model, force_list, rF)
_, PsU_rU, Wd_rU = single_hopping_Wd(model, force_list, rU)

# same structure as the formula above, with the zero-hopping probability 1 - PsF(f_min, f_1)
# in place of PhiU and the single-hopping <W_d> in place of the simulated one
for k in range(len(force_list)):
    DeltaWd_estimation[k] = (1 - PsF_rF[0,k])*(Wd_rF[k] - Wd_rU[k]) + PsF_rF[0,k]*(np.nansum(PsF_rF[k,k:]*model.kFtoU(force_list[k:])*(Wd_rF[k:]+model.beta*(force_list[k:]-model.fc)*model.xm))*Deltaf/rF
                                                                                             - np.nansum(PsF_rFprime[k,k:]*model.kFtoU(force_list[k:])*(Wd_rU[k:]+model.beta*(force_list[k:]-model.fc)*model.xm))*Deltaf/rFprime)
    Upsilon_estimation[k] = np.log(1/PsU_rU[-1,k] + np.nansum(1/PsF_rFprime[k:,k] * model.kUtoF(force_list[k:]) * 1/PsU_rU[-1,k:])*Deltaf/rFprime)
    T_estimation[k] = (force_list[k]-fmin)/rF + (1-PsF_rF[0,k])*(fmax - force_list[k])/rU + PsF_rF[0,k]*np.nansum(PsF_rFprime[k,k:]*model.kFtoU(force_list[k:])*((force_list[k:]-force_list[k])/rFprime + (fmax - force_list[k:])/rU))*Deltaf/rFprime

# --------------------------------------------------------------------------- comparison at equal mean cycle time
# Each method is compared with the no-feedback protocol of the same mean duration <tau>.
# The formula / single-hopping <W_d> are taken relative to the no-feedback protocol at r_F.
T0 = (fmax - fmin) / rF
Wd0_rF = Wd0_T(T0)
DeltaWdT_JZ = Wd0_T(T) - Wd
DeltaWdT_formula = Wd0_T(T_formula) - (Wd0_rF - DeltaWd_formula)
DeltaWdT_estimation = Wd0_T(T_estimation) - (Wd0_rF - DeltaWd_estimation)

np.savez(OUTFILE, rF=rF, rFprime=rFprime, rU=rU, f1_list=f1_list, f=f, force_list=force_list,
         Wd0=Wd0, Upsilon_JZ=Upsilon_JZ, Wd=Wd, T=T,
         Upsilon_formula=Upsilon_formula, DeltaWd_formula=DeltaWd_formula, T_formula=T_formula,
         Upsilon_estimation=Upsilon_estimation, DeltaWd_estimation=DeltaWd_estimation,
         T_estimation=T_estimation,
         Wd0_rF=Wd0_rF, DeltaWdT_JZ=DeltaWdT_JZ,
         DeltaWdT_formula=DeltaWdT_formula, DeltaWdT_estimation=DeltaWdT_estimation)
print(f"Saved results to {OUTFILE}")
