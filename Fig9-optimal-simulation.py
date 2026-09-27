"""Simulation data for Fig. 9 (bias of the free-energy estimator for the optimal protocols).

For each protocol, the Jarzynski estimator over the first N pulls,
    B_N = -ln( (1/N) sum_{i<=N} exp(-W_d,i) ) + Upsilon,
is averaged over M independent sets of pulls and stored against the
cumulative measurement time N <tau>.

Saves the results to `Fig9-optimal-data.npz`, which `Fig9-optimal-plot.py` loads.
Protocol keys given on the command line (e.g. `python Fig9-optimal-simulation.py DTF`) are
re-simulated alone; the other protocols keep their values from the existing data file.
"""
import os
import sys
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
OUTFILE = os.path.join(HERE, "Fig9-optimal-data.npz")

M = 1000            # number of independent sets of pulls averaged over
BATCH = 1000        # pulls simulated at once; the trajectories of a run take ~90 MB per
                    # 1000 pulls, so a set of N pulls is run in N/BATCH batches
ensemble = 'LdF'


def no_feedback(sim):
    """rF chosen so that <tau> is close to that of the optimal feedback protocols"""
    sim.protocol = hp.NOProtocol(rF=6)
    return 0., (sim.fmax - sim.fmin) / sim.protocol.rF


def optimal_DTF(sim):
    """optimal eta_M (Fig. 5: r_U = inf, f_1 = 16.1 pN)"""
    sim.protocol = hp.DTFProtocol(rF=4, rU=np.inf, cp1=16.1)
    f1 = np.array([sim.protocol.f1])
    return sim.protocol.Upsilon(sim, f1)[0], sim.protocol.avg_time(sim, f1)[0]


def optimal_CTF(sim):
    """not optimal (eta_M < 0 for all CTF): r_U chosen for a <tau> close to the other protocols"""
    sim.protocol = hp.CTFProtocol(rF=4, rU=17)
    ratio = np.array([sim.protocol.rU / sim.protocol.rF])
    return sim.protocol.Upsilon(sim, ratio)[0], sim.protocol.avg_time(sim, ratio)[0]


def optimal_CDS(sim):
    """constant-DS, optimal eta_M (Fig. 7: r_F' = 1 pN/s, f_1 = 14.3 pN)"""
    sim.protocol = hp.Strategy(rF=4, rFprime=1, rU=np.inf, cp1=14.3)
    f1 = np.array([sim.protocol.f1])
    return sim.protocol.Upsilon(sim, f1)[0], sim.protocol.avg_time(sim, f1)[0]


# key: (set-up function returning (Upsilon, <tau>), number of pulls N)
PROTOCOLS = {
    'none': (no_feedback, 1000),
    'DTF':  (optimal_DTF, 10000),
    'CTF':  (optimal_CTF, 10000),
    'CDS':  (optimal_CDS, 1000),
}


def bias_curve(sim, N, Upsilon):
    """B_N for N = 1..N from one set of N pulls, simulated in batches of sim.N pulls."""
    assert N % sim.N == 0, "N must be a multiple of the batch size"
    W = []
    for _ in range(N // sim.N):
        sim.run()
        W.append(sim.dissipated_work())
        sim.reset()
    W = np.concatenate(W)
    return -np.log(np.cumsum(np.exp(-W)) / np.arange(1, N + 1)) + Upsilon


run_keys = sys.argv[1:] or list(PROTOCOLS)
unknown = set(run_keys) - set(PROTOCOLS)
if unknown:
    sys.exit(f"unknown protocol(s) {sorted(unknown)}, choose among {list(PROTOCOLS)}")

# protocols not re-simulated keep their stored values
results = {}
if set(run_keys) != set(PROTOCOLS):
    with np.load(OUTFILE) as old:
        results = {k: old[k] for k in old.files if k not in ('ensemble', 'M')}

with tqdm(total=M * len(run_keys), unit="set") as pbar:
    for key in run_keys:
        setup, N = PROTOCOLS[key]
        pbar.set_description(key)
        simulation = hp.Simulation(N=BATCH, ensemble=ensemble)
        Upsilon, tau = setup(simulation)

        # one progress step per set of N pulls
        B = np.zeros(N)
        for B_one in Parallel(n_jobs=-1, return_as="generator")(
                delayed(bias_curve)(simulation, N, Upsilon) for _ in range(M)):
            B += B_one / M
            pbar.update()

        results[f'T_{key}'] = np.arange(1, N + 1) * tau
        results[f'B_{key}'] = B
        results[f'Upsilon_{key}'] = Upsilon

np.savez(OUTFILE, ensemble=ensemble, M=M, **results)
print(f"Saved results to {OUTFILE}")
for key in PROTOCOLS:
    if np.isnan(results[f'B_{key}']).any():
        print(f"Warning: B_{key} has {np.isnan(results[f'B_{key}']).sum()} NaN value(s)")
