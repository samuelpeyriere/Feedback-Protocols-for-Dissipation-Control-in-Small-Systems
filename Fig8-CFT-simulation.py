"""Simulation data for Fig. 8 (Crooks fluctuation theorem with the DTF protocol).

Saves the results to `Fig8-CFT-data.npz`, which `Fig8-CFT-plot.py` loads.

Backward process and "ghost" trajectories
-----------------------------------------
The backward process replays a forward protocol branch sigma_1 in reverse (r_U branch if U
was observed at f_1, r_F branch otherwise) and checks the state again at f_1. A backward
trajectory whose observation at f_1 differs from its branch is a ghost trajectory: it has no
time-reversed forward counterpart, so it is discarded.

- Conditional CFT:  rho_->(W_d | s) / rho_<-(-W_d | s) = exp(W_d + I_s),
  with I_s = log(p_<-(s | s) / p_->(s)), where rho_<- only keeps the non-ghost trajectories
  run on branch s. It does not depend on how often each branch is replayed.
- Total CFT:  rho_->(W_d) / rho_<-(-W_d) = exp(W_d + Upsilon), Upsilon = log(sum_s p_<-(s | s)),
  where rho_<- is the non-ghost part of a backward ensemble in which every branch is replayed
  EQUALLY often. Replaying the branches with the forward probabilities p_->(s) (i.e. simply
  reversing each forward trajectory) and then dropping the ghosts gives a biased Upsilon.

So the backward run replays N/2 trajectories of each branch, drawn from the forward run.
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
OUTFILE = os.path.join(HERE, "Fig8-CFT-data.npz")

N = 10000
N_JOBS = 12          # each job holds ~2 GB of trajectories
ENSEMBLE = 'LdF'
RF, RU = 4, 17       # pN/s
CP1_EXAMPLE = 14.5   # decision force of the histograms in panel (a)
cp1_list = np.linspace(13.5, 15.5, 100)
EQUATIONS = ('J', 'B', 'C')   # Jarzynski, Bennett, Crooks


def observation_index(sim):
    """Time index of the measurement at f_1 (common to all forward trajectories)."""
    p = sim.protocol
    cp = sim.force_list[:, 0] if sim.model.ensemble == 'LdF' else sim.length_list[:, 0]
    return int(np.argmax((p.cp1 - p.half_interval < cp) & (cp <= p.cp1 + p.half_interval)))


def keep_columns(sim, cols):
    """Replace the forward trajectories by the columns `cols` (N of them, with repeats)."""
    sim.force_list = sim.force_list[:, cols]
    sim.length_list = sim.length_list[:, cols]
    sim.state_list = sim.state_list[:, cols]
    sim.rate_list = sim.rate_list[:, cols]


def run_cft(cp1, seed=None):
    """Forward run, then balanced backward run. Returns the dissipated works split by branch."""
    rng = np.random.default_rng(seed)
    np.random.seed(rng.integers(2**32))   # hairpyn draws from the global numpy RNG

    sim = hp.Simulation(N=N, ensemble=ENSEMBLE)
    sim.protocol = hp.DTFProtocol(rF=RF, rU=RU, cp1=cp1)

    # ---- forward
    sim.run()
    Wd_fw = sim.dissipated_work()
    index = observation_index(sim)
    branch_U = sim.state_list[index].copy()   # sigma_1 = U  <->  r_U branch
    T = sim.avg_time()[0]

    # ---- backward: replay each branch equally often (a missing branch is simply left out)
    idx_U, idx_F = np.flatnonzero(branch_U), np.flatnonzero(~branch_U)
    if len(idx_U) == 0 or len(idx_F) == 0:
        cols = rng.choice(idx_U if len(idx_U) else idx_F, N)
    else:
        cols = np.concatenate([rng.choice(idx_U, N // 2), rng.choice(idx_F, N - N // 2)])
    keep_columns(sim, cols)
    sim.run_backward()
    Wd_bw = sim.dissipated_work(backward=True)

    branch_U_bw = branch_U[cols]
    obs_U_bw = sim.state_list_backward[-1 - index]    # backward observation at f_1
    real_U = branch_U_bw & obs_U_bw                  # non-ghost trajectories of each branch
    real_F = ~branch_U_bw & ~obs_U_bw

    with np.errstate(invalid='ignore', divide='ignore'):
        out = dict(
            Wd_fw=Wd_fw, Wd_fw_U=Wd_fw[branch_U], Wd_fw_F=Wd_fw[~branch_U],
            Wd_bw=Wd_bw[real_U | real_F], Wd_bw_U=Wd_bw[real_U], Wd_bw_F=Wd_bw[real_F],
            n_ghost=int((~(real_U | real_F)).sum()),
            pU_fw=branch_U.mean(),
            pU_bw=real_U.sum() / branch_U_bw.sum(),       # p_<-(U | r_U branch)
            pF_bw=real_F.sum() / (~branch_U_bw).sum(),    # p_<-(F | r_F branch)
            T=T,
        )
    sim.reset()
    return out


def estimate(sim, run):
    """I_U, I_F and Upsilon from each of the equalities in EQUATIONS."""
    res = {}
    for eq in EQUATIONS:
        res['I_U_' + eq] = sim.Upsilon(run['Wd_fw_U'], run['Wd_bw_U'], equation=eq)[0]
        res['I_F_' + eq] = sim.Upsilon(run['Wd_fw_F'], run['Wd_bw_F'], equation=eq)[0]
        res['UPSILON_' + eq] = sim.Upsilon(run['Wd_fw'], run['Wd_bw'], equation=eq)[0]
    return res


def computation(cp1, seed):
    run = run_cft(cp1, seed)
    res = estimate(hp.Simulation(ensemble=ENSEMBLE), run)
    for key in ('pU_fw', 'pU_bw', 'pF_bw', 'T', 'n_ghost'):
        res[key] = run[key]
    return res


if __name__ == "__main__":
    seeds = np.random.SeedSequence().spawn(len(cp1_list) + 1)
    seeds = [int(s.generate_state(1)[0]) for s in seeds]

    results = Parallel(n_jobs=N_JOBS, return_as='generator')(
        delayed(computation)(cp1, s) for cp1, s in zip(cp1_list, seeds[1:]))
    results = list(tqdm(results, total=len(cp1_list), desc='f_1 sweep', unit='f1'))
    sweep = {key: np.array([r[key] for r in results]) for key in results[0]}

    # theoretical values: I_s = log(p_<-(s|s) / p_->(s)), Upsilon from the protocol
    with np.errstate(invalid='ignore', divide='ignore'):
        sweep['I_U_theory'] = np.log(sweep['pU_bw'] / sweep['pU_fw'])
        sweep['I_F_theory'] = np.log(sweep['pF_bw'] / (1 - sweep['pU_fw']))
    sim = hp.Simulation(ensemble=ENSEMBLE)
    sim.protocol = hp.DTFProtocol(rF=RF, rU=RU)
    sweep['UPSILON_theory'] = sim.protocol.Upsilon(sim, cp1_list)

    # example work distributions for panel (a)
    print(f"Example run at f_1 = {CP1_EXAMPLE} pN")
    example = run_cft(CP1_EXAMPLE, seeds[0])
    example = {'ex_' + key: val for key, val in example.items()}

    np.savez(OUTFILE, ensemble=ENSEMBLE, rF=RF, rU=RU, cp1_list=cp1_list,
             cp1_example=CP1_EXAMPLE, **sweep, **example)
    print(f"Saved results to {OUTFILE}")
    print(f"Example: {example['ex_n_ghost']} ghost trajectories discarded out of {N}")
    for key, arr in sweep.items():
        if np.isnan(arr).any():
            print(f"Warning: {key} has {np.isnan(arr).sum()} NaN value(s)")
