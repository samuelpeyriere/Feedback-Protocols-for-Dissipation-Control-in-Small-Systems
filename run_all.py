"""Run all simulations, then redraw all figures.

`Supp-Fig6-none-simulation.py` runs first, since the simulations of Figs. 5-7 and Supp.
Figs. 2-4 and 7 need its data; the other simulations follow, then all plot scripts. Each
script runs with the Python interpreter that runs this one, and the run stops at the first
script that fails.

    python run_all.py           # simulations, then plots
    python run_all.py sims      # simulations only
    python run_all.py plots     # plots only (from the existing .npz files)
"""
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = "Supp-Fig6-none-simulation.py"


def scripts(pattern):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, pattern)))


def fmt(seconds):
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


what = sys.argv[1:] or ["sims", "plots"]
if not set(what) <= {"sims", "plots"}:
    sys.exit(__doc__)

todo = []
if "sims" in what:
    todo += [FIRST] + [s for s in scripts("*-simulation.py") if s != FIRST]
if "plots" in what:
    todo += scripts("*-plot.py")

t0 = time.time()
for i, script in enumerate(todo, 1):
    print(f"\n=== [{i}/{len(todo)}] {script}  (started {time.strftime('%H:%M:%S')}, "
          f"total elapsed {fmt(time.time() - t0)})", flush=True)
    ts = time.time()
    if subprocess.run([sys.executable, "-u", os.path.join(HERE, script)], cwd=HERE).returncode:
        sys.exit(f"FAILED: {script}")
    print(f"--- {script} done in {fmt(time.time() - ts)}", flush=True)

print(f"\nAll {len(todo)} scripts done in {fmt(time.time() - t0)}")
