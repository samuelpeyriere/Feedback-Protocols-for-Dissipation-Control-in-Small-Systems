"""Single-hopping estimate of the dissipated work, shared by the Supp. Fig. 2-4 simulations.

In the single-hopping approximation, a trajectory that is unfolded (U) at force f_i
refolds at most once, at some f > f_i, and then unfolds again at f' > f before f_max.
Each such refolding/unfolding event dissipates beta * x_m * (f' - f).
"""
import numpy as np
from scipy.integrate import quad


def single_hopping_Wd(model, force_list, r):
    """Mean dissipated work [k_B T] after starting in U at each force of `force_list`,
    pulling at rate r up to force_list[-1], in the single-hopping approximation.

    Returns (PsF, PsU, Wd), where PsF[i, j] = PsF(f_i, f_j, r) and PsU[i, j] = PsU(f_i, f_j, r)
    are the survival probabilities of F and U from f_i to f_j (callers reuse them), and
    Wd[i] is the dissipated work starting from U at f_i. The last entry of Wd is 0.
    """
    N = len(force_list)
    Deltaf = force_list[1] - force_list[0]
    X, Y = np.meshgrid(force_list, force_list)
    PsF = model.PsF(Y, X, r)
    PsU = model.PsU(Y, X, r)

    # inside[i]: integral over f' in [f_i, f_max] of PsF(f_i, f'), i.e. the mean force
    # increase f' - f_i before a molecule folded at f_i unfolds again
    inside = np.zeros(N, dtype=float)
    for i in range(N - 1):
        inside[i] = PsF[i, i:].sum() * Deltaf

    # outer integral over the refolding force f in [f_i, f_max] of the refolding density
    # k_UtoF(f)/r * PsU(f_i, f), times inside(f). The first grid cell [f_i, f_{i+1}] is
    # integrated with quad (`fine`), the rest with a Riemann sum (`coarse`).
    coarse = np.zeros(N, dtype=float)
    fine = np.zeros(N, dtype=float)
    for i in range(N - 1):
        fine[i] = quad(lambda f: model.kUtoF(f) / r * model.PsU(force_list[i], f, r),
                       force_list[i], force_list[i + 1])[0] * inside[i]
        coarse[i] = (model.kUtoF(force_list[i:]) / r * PsU[i, i:] * inside[i:])[1:].sum() * Deltaf

    Wd = model.beta * (1 - PsU[:, -1]) * (coarse + fine) * model.xm
    return PsF, PsU, Wd
