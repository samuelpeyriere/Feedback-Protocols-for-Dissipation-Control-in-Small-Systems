# Simulation and plotting code for the figures

Each figure has a `*-plot.py` script that draws it. Figures based on simulations also
have a `*-simulation.py` script, which writes the data to a `*-data.npz` file that
the plot script then loads. The `.npz` files used for the paper are included, so
each figure can be redrawn without running its simulation again.

| Figure | Simulation script | Data file | Plot script | Output |
|---|---|---|---|---|
| Fig. 1 (ghost trajectories) | – | – | `Fig1-force-displacement-plot.py` | `Fig1-force-displacement.pdf` |
| Fig. 2 (force ensemble) | – | – | `Fig2-force-ensemble-plot.py` | `Fig2-force-ensemble.pdf` |
| Fig. 3 (single hopping) | – | – | `Fig3-single-hopping-plot.py` | `Fig3-single-hopping.pdf` |
| Fig. 4 (discrete Maxwell demon) | `Fig4-DMD-simulation.py` | `Fig4-DMD-data.npz` | `Fig4-DMD-plot.py` | `Fig4-DMD.pdf` |
| Fig. 5 (DTF protocol) | `Fig5-DTF-simulation.py` | `Fig5-DTF-data.npz` | `Fig5-DTF-plot.py` | `Fig5-DTF.pdf` |
| Fig. 6 (CTF protocol) | `Fig6-CTF-simulation.py` | `Fig6-CTF-data.npz` | `Fig6-CTF-plot.py` | `Fig6-CTF.pdf` |
| Fig. 7 (constant-DS protocol) | `Fig7-CDS-simulation.py` | `Fig7-CDS-data.npz` | `Fig7-CDS-plot.py` | `Fig7-CDS.pdf` |
| Fig. 8 (Crooks fluctuation theorem) | `Fig8-CFT-simulation.py` | `Fig8-CFT-data.npz` | `Fig8-CFT-plot.py` | `Fig8-CFT.pdf` |
| Fig. 9 (optimal protocols) | `Fig9-optimal-simulation.py` | `Fig9-optimal-data.npz` (+ Figs. 5–7 data) | `Fig9-optimal-plot.py` | `Fig9-optimal.pdf` |
| Supp. Fig. 1 (hopping cases) | – | – | `Supp-Fig1-single-hopping-plot.py` | `Supp-Fig1-single-hopping.pdf` |
| Supp. Fig. 2 (DTF approximations) | `Supp-Fig2-DTF-approx-simulation.py` | `Supp-Fig2-DTF-approx-data.npz` | `Supp-Fig2-DTF-approx-plot.py` | `Supp-Fig2-DTF-approx.pdf` |
| Supp. Fig. 3 (CTF approximations) | `Supp-Fig3-CTF-approx-simulation.py` | `Supp-Fig3-CTF-approx-data.npz` | `Supp-Fig3-CTF-approx-plot.py` | `Supp-Fig3-CTF-approx.pdf` |
| Supp. Fig. 4 (constant-DS approximations) | `Supp-Fig4-CDS-approx-simulation.py` | `Supp-Fig4-CDS-approx-data.npz` | `Supp-Fig4-CDS-approx-plot.py` | `Supp-Fig4-CDS-approx.pdf` |
| Supp. Fig. 5 (simulation step) | – | – | `Supp-Fig5-simulation-structure-plot.py` | `Supp-Fig5-simulation-structure.pdf` |
| Supp. Fig. 6 (no feedback) | `Supp-Fig6-none-simulation.py` | `Supp-Fig6-none-data.npz` | `Supp-Fig6-none-plot.py` | `Supp-Fig6-none.pdf` |
| Supp. Fig. 7 (dependence on r_F) | `Supp-Fig7-rF-simulation.py` | `Supp-Fig7-rF-data.npz` | `Supp-Fig7-rF-plot.py` | `Supp-Fig7-rF.pdf` |

File names use `DMD`, `DTF`, `CTF` and `CDS`; `CDS` is the constant Dual-Strategy (constant-DS)
of the paper. `LdF` in the code is the force ensemble.

Shared modules:

- `hairpyn/`: the simulation library (hairpin model, feedback protocols, force-ramp simulators).
- `wiley_style.py`: matplotlib style (fonts, sizes, colorblind-safe palette).
- `plot_helpers.py`: axis helpers (broken axis for `x = inf`, slope markers, the layout of Supp. Figs. 2–4).
- `none_Wd0.py`: `Wd0_T(T)`, the dissipated work without feedback at mean duration `T`, from the exact
  expression (Eq. 25 of the main text). This is the reference that the feedback protocols are compared
  to, interpolated from `Supp-Fig6-none-data.npz`.
- `single_hopping.py`: the single-hopping estimate of the dissipated work used in Supp. Figs. 2–4.

## Setup

Python ≥ 3.10 with the packages in `requirements.txt`:

```sh
pip install -r requirements.txt
```

The `hairpyn` simulation library is included in `hairpyn/` (MIT license, see `hairpyn/LICENSE`)
and is imported from there when the scripts are run, so it needs no separate installation.

The figures use the STIX Two Text fonts, included in `fonts/stix2_ttf/` (SIL Open Font
License, see `fonts/stix2_ttf/OFL.txt`; source: <https://github.com/stipub/stixfonts>).
To use another copy, point `$STIX2_TTF_DIR` to its directory.

## Running

Run each script from any directory, e.g.

```sh
python Fig5-DTF-simulation.py   # writes Fig5-DTF-data.npz
python Fig5-DTF-plot.py         # writes Fig5-DTF.pdf
```

To run all simulations and then redraw all figures, showing the progress (`[i/N]` script
counter, elapsed time, and the progress bars of the scripts):

```sh
python run_all.py           # simulations, then plots
python run_all.py sims      # simulations only
python run_all.py plots     # plots only, from the existing .npz files
```

It runs `Supp-Fig6-none-simulation.py` first, then the other simulations, then the plot
scripts, with the same Python interpreter, and stops at the first script that fails.

- The simulations of Figs. 5–7 and Supp. Figs. 2–4 and 7 need `Supp-Fig6-none-data.npz`. If it
  is missing, `none_Wd0.py` runs `Supp-Fig6-none-simulation.py` first to create it.
- Panel (a) of Fig. 9 also loads the data of Figs. 5–7. `Fig9-optimal-simulation.py` accepts protocol keys
  (`none`, `DTF`, `CTF`, `CDS`) to re-simulate only those, e.g. `python Fig9-optimal-simulation.py DTF`;
  the others keep their values from `Fig9-optimal-data.npz`.
- The simulations use all CPU cores through joblib. `Fig8-CFT-simulation.py` holds about 2 GB
  per job. Lower `N_JOBS` there if memory is short.
- Each simulation script fixes its random seed (`SEED` at the top of the script), so a run gives
  the same data every time, whatever the number of CPU cores. Each joblib job gets its own seed,
  drawn from `SEED` (`hp.seeded` in `hairpyn/rng.py`). The `.npz` files used for the paper were
  produced before the seeds were fixed, so a new run reproduces them only up to statistical noise.
