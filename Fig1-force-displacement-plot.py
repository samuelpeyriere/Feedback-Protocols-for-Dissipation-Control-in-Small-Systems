"""Fig. 1: ghost trajectories in a discrete feedback pulling protocol.

Protocol (force ensemble: force is the control parameter, the extension jumps
by x_m when the hairpin unfolds):

    forward   f: fmin -> fmax.  Ramp at r_F up to f1, measure the state of the
                                hairpin at f1.  If unfolded, switch to r_U for
                                the rest of the ramp; if folded, keep r_F.

    reverse   f: fmax -> fmin.  The measurement outcome m is drawn *first* and
                                fixes the protocol: the segment fmax -> f1 is
                                run at -r_U if m = U and at -r_F if m = F.  The
                                segment f1 -> fmin is always run at -r_F.  The
                                state is merely *observed* when the ramp crosses
                                f1.

A reverse trajectory is the time reverse of a forward one only if the state
observed at f1 agrees with the label m that selected the protocol.  Reverse
trajectories where the two disagree have no forward counterpart at all: they
are the *ghost* trajectories, and they are what the fluctuation theorem for
this protocol has to pay for (the "thermodynamic information" term).

The forward process cannot produce such a mismatch -- there the outcome *is*
what selects the rate -- so ghosts only ever appear in the reverse column.

The trajectories below are hand-specified illustrations, not samples: each
class is drawn once or twice so the reader can see it, which says nothing
about how likely that class actually is.  (Refolding above f1 is in reality
quite rare at these rates, as is starting the reverse ramp folded at fmax.)

Pure schematic (no simulation data needed).
"""
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

import wiley_style as ws

# --------------------------------------------------------------------------- protocol
FMIN, F1, FMAX = 8.0, 14.0, 18.0   # [pN] start, measurement and end force
RF, RU = 5.0, 17.0                 # [pN/s] slow (folded) and fast (unfolded) pulling rates

KX = 0.10       # [pN/nm] effective stiffness of trap + handles
XM = 18.0       # [nm] extension released by unfolding
SIGMA_X = 1.1   # [nm] rms of the (smoothed) extension fluctuations

# --------------------------------------------------------------------------- figure settings
COLOR_U = ws.OKABE_ITO['blue']         # fast branch, outcome "unfolded"
COLOR_F = ws.OKABE_ITO['vermillion']   # slow branch, outcome "folded"
COLOR_GHOST = ws.OKABE_ITO['black']    # edge of the f_1 marker on a ghost, and its arrows
COLOR_GUIDE = ws.OKABE_ITO['gray']     # branch rails, f_1 guides, annotation leaders
COLOR_BAND = '#e6e6e6'                 # shading of the r_U/r_F-dependent force range
GHOST_ALPHA = 0.5                      # ghosts keep their branch color, faded
GHOST_DASH = (0, (4, 2.2))
FS_NOTE = ws.FS_TICK                   # multi-line annotations


@dataclass
class Trajectory:
    """One realisation, specified by the forces at which the state flips."""

    branch: str   # "U" or "F": which rate is used between f1 and fmax
    n0: int       # state at the start of the process (0 folded, 1 unfolded)
    flips: tuple  # forces [pN] at which the hairpin changes state, in path order
    seed: int     # noise seed


# --------------------------------------------------------------------------- trajectories
def state_at(force, n0, flips, forward):
    """State along a monotone force ramp (0 = folded, 1 = unfolded)."""
    n = np.full(np.shape(force), n0, dtype=int)
    for f_flip in flips:
        n = n + ((force >= f_flip) if forward else (force <= f_flip))
    return n % 2


def fluctuations(size, seed, width=41):
    """Smooth, zero-mean noise so the extension does not look like a ruler."""
    raw = np.random.default_rng(seed).normal(size=size + width)
    smooth = np.convolve(raw, np.ones(width) / width, mode='valid')[:size]
    return SIGMA_X * (smooth - smooth.mean()) / smooth.std()


def ramp(traj, forward, n_pts=1500):
    """Force and time along the two-segment ramp of a single trajectory."""
    rate = RU if traj.branch == 'U' else RF
    if forward:
        t_meas = (F1 - FMIN) / RF
        duration = t_meas + (FMAX - F1) / rate
    else:
        t_meas = (FMAX - F1) / rate
        duration = t_meas + (F1 - FMIN) / RF

    t = np.concatenate([np.linspace(0.0, t_meas, n_pts // 2, endpoint=False),
                        np.linspace(t_meas, duration, n_pts // 2)])
    if forward:
        force = np.where(t <= t_meas, FMIN + RF * t, F1 + rate * (t - t_meas))
    else:
        force = np.where(t <= t_meas, FMAX - rate * t, F1 - RF * (t - t_meas))
    return t, force, t_meas


def realise(traj, forward):
    """Time, force, extension and the state observed at f1."""
    t, force, t_meas = ramp(traj, forward)
    n = state_at(force, traj.n0, traj.flips, forward)
    x = force / KX + n * XM + fluctuations(t.size, traj.seed)
    i_meas = int(np.searchsorted(t, t_meas))
    observed = 'U' if n[i_meas] else 'F'
    return t, force, x, i_meas, observed


# Forward: `branch` is *determined* by the state at f1 -- checked in build().
FORWARD = [
    Trajectory('U', 0, (11.3,), 11),             # unfolds early, measured U -> speed up
    Trajectory('U', 0, (13.4,), 13),             # unfolds just below f1, still measured U
    Trajectory('F', 0, (11.0, 12.8, 15.4), 14),  # unfolds, refolds, measured F
    Trajectory('F', 0, (14.9,), 15),             # unfolds just above f1: too late to count
    Trajectory('F', 0, (), 17),                  # never unfolds
]

# Reverse: `branch` is the protocol label m, drawn *before* the run.  Whether it
# matches the observed state is the whole question.
REVERSE = [
    Trajectory('U', 1, (11.4,), 21),   # observed U  -> conjugate exists
    Trajectory('U', 1, (9.6,), 22),    # observed U  -> conjugate exists
    Trajectory('F', 1, (14.9,), 23),   # refolds above f1, observed F -> conjugate
    Trajectory('U', 1, (15.4,), 25),   # ramped at -r_U yet observed F -> GHOST
    Trajectory('F', 1, (12.6,), 26),   # ramped at -r_F yet observed U -> GHOST
]


def build(trajectories, forward):
    """Realise every trajectory and flag the ghosts."""
    out = []
    for traj in trajectories:
        t, force, x, i_meas, observed = realise(traj, forward)
        ghost = observed != traj.branch
        if forward and ghost:
            raise AssertionError(
                f"forward trajectory {traj.flips} is measured {observed} "
                f"but declares branch {traj.branch}: the forward protocol cannot do that")
        out.append((traj, t, force, x, i_meas, observed, ghost))
    return out


# --------------------------------------------------------------------------- drawing helpers
def branch_color(branch):
    return COLOR_U if branch == 'U' else COLOR_F


def line_style(traj, ghost):
    """A ghost keeps the color of the branch it was run on, faded and dashed."""
    color = branch_color(traj.branch)
    if ghost:
        return dict(color=color, lw=ws.LW_THIN, ls=GHOST_DASH, alpha=GHOST_ALPHA, zorder=2)
    return dict(color=color, lw=ws.LW_THIN, ls='-', zorder=3)


def draw_branch_rails(ax_x, forward):
    """The two elastic branches, x = f/kx and x = f/kx + x_m, as background rails.

    One pair per pulling rate, since f(t) -- and hence the rail -- depends on
    which branch of the protocol is being run.
    """
    for branch in ('U', 'F'):
        t, force, _ = ramp(Trajectory(branch, 0, (), 0), forward)
        for offset in (0.0, XM):
            ax_x.plot(t, force / KX + offset, color=COLOR_GUIDE, lw=1,
                      ls=(0, (1, 2.6)), zorder=0)


def draw_protocol(ax_f, forward):
    """The control parameter itself: f(t) depends on the branch and nothing else.

    A ghost run is *indistinguishable* here from a legitimate one on the same
    branch -- only the state trace below tells them apart.
    """
    for branch in ('U', 'F'):
        color = branch_color(branch)
        t, force, t_meas = ramp(Trajectory(branch, 0, (), 0), forward)
        ax_f.plot(t, force, color=color, lw=ws.LW_THICK, solid_capstyle='round', zorder=3)
        # forward: both branches cross f1 at the same instant, so mark it once
        mark = COLOR_GUIDE if forward else color
        ax_f.axvline(t_meas, color=mark, lw=ws.LW_THIN, ls=(0, (1, 2)), zorder=1)
        ax_f.plot(t_meas, F1, marker='o', color=mark, mec='white', mew=1.5,
                  ls='none', zorder=5)


def draw_trajectories(ax_x, runs):
    """Extension vs time: the branch you sit on, and the rip when you leave it."""
    for traj, t, force, x, i_meas, observed, ghost in runs:
        style = line_style(traj, ghost)
        ax_x.plot(t, x, **style)

        # state observed at f1: filled marker = unfolded, open marker = folded
        edge = style['color']
        ax_x.plot(t[i_meas], x[i_meas], marker='o',
                  mfc=edge if observed == 'U' else 'white',
                  mec=COLOR_GHOST if ghost else edge, mew=2.5 if ghost else 2,
                  ls='none', zorder=6)


def style_axes(ax_f, ax_x, title, t_max):
    ax_f.axhspan(F1, FMAX, color=COLOR_BAND, lw=0, zorder=0)
    ax_f.axhline(F1, color=COLOR_GUIDE, lw=1, ls='--', zorder=1)
    ax_f.set_yticks([FMIN, F1, FMAX])
    ax_f.set_yticklabels([r'$f_{\rm min}$', r'$f_1$', r'$f_{\rm max}$'])
    ax_f.set_ylim(FMIN - 1.0, FMAX + 1.2)
    ax_f.set_ylabel('Force $f$ [pN]')
    ax_f.set_title(title, loc='left')
    ax_f.tick_params(labelbottom=False)

    ax_x.set_ylim(FMIN / KX - 16, FMAX / KX + XM + 16)
    ax_x.set_ylabel('Extension $x$ [nm]')
    ax_x.set_xlabel('Time [s]')

    for ax in (ax_f, ax_x):
        ax.set_xlim(-0.05, t_max)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)


def annotate(axes):
    (fwd_f, rev_f), (fwd_x, rev_x) = axes

    t_meas_f = (F1 - FMIN) / RF
    t_meas_rU, t_meas_rF = (FMAX - F1) / RU, (FMAX - F1) / RF
    leader = dict(arrowstyle='-', color=COLOR_GUIDE, lw=1, shrinkA=3, shrinkB=3)

    # which rate is applied where
    fwd_f.annotate('$r_F$', xy=(0.60, 11.4), ha='right')
    fwd_f.annotate('$r_U$', xy=(1.44, 17.4), color=COLOR_U, ha='left', va='center')
    fwd_f.annotate('$r_F$', xy=(1.80, 16.2), color=COLOR_F, ha='left', va='center')
    fwd_f.annotate('state measured\nat $f_1$', xy=(t_meas_f, F1), xytext=(0.62, 16.4),
                   fontsize=FS_NOTE, ha='center', va='center',
                   arrowprops={**leader, 'shrinkB': 6})
    rev_f.annotate('$-r_U$', xy=(0.13, 15.8), xytext=(0.40, 17.1), color=COLOR_U,
                   ha='left', va='center', arrowprops={**leader, 'color': COLOR_U})
    rev_f.annotate('$-r_F$', xy=(0.62, 15.1), color=COLOR_F, ha='left', va='bottom')

    # where the two rails of the r_F protocol sit at time t (forward process);
    # branch F runs at r_F throughout, so its force is a single ramp from fmin
    def rail(t, offset):
        return (FMIN + RF * t) / KX + offset

    # the extension jump that makes this the mixed ensemble: the arrow spans
    # exactly the gap between the two branches at t_arrow, tip to tip
    t_arrow = 1.84
    fwd_x.annotate('', xy=(t_arrow, rail(t_arrow, XM)), xytext=(t_arrow, rail(t_arrow, 0.0)),
                   arrowprops=dict(arrowstyle='<->', color='k', lw=1.5, shrinkA=0, shrinkB=0))
    fwd_x.annotate('$x_m$', xy=(t_arrow + 0.04, rail(t_arrow, XM / 2 + 2.0)),
                   ha='left', va='center')
    fwd_x.annotate('unfolded branch', xy=(0.30, rail(0.30, XM)), xytext=(0.16, 148),
                   color=COLOR_GUIDE, fontsize=FS_NOTE, ha='left', va='center',
                   arrowprops=leader)
    fwd_x.annotate('folded branch', xy=(0.55, rail(0.55, 0.0)), xytext=(0.72, 80),
                   color=COLOR_GUIDE, fontsize=FS_NOTE, ha='left', va='center',
                   arrowprops=leader)

    # the two ways a reverse run fails to be anybody's time reverse
    ghost_arrow = dict(arrowstyle='->', color=COLOR_GHOST, lw=1.5, shrinkA=3, shrinkB=7)
    rev_x.annotate('ramped at $-r_U$\nbut observed folded:\nno forward partner',
                   xy=(t_meas_rU, F1 / KX), xytext=(0.10, 96.0),
                   fontsize=FS_NOTE, ha='left', va='center', arrowprops=ghost_arrow)
    rev_x.annotate('ramped at $-r_F$\nbut observed unfolded:\nno forward partner',
                   xy=(t_meas_rF, F1 / KX + XM), xytext=(1.02, 192.0),
                   fontsize=FS_NOTE, ha='left', va='center', arrowprops=ghost_arrow)


def draw_legend(fig):
    branch_u = Line2D([], [], color=COLOR_U, lw=ws.LW_THIN, label=r'branch $r_U$ (outcome $U$)')
    branch_f = Line2D([], [], color=COLOR_F, lw=ws.LW_THIN, label=r'branch $r_F$ (outcome $F$)')
    # ghosts keep their branch color, so the handle shows both, faded
    ghost = tuple(Line2D([], [], color=color, lw=ws.LW_THIN, ls=GHOST_DASH, alpha=GHOST_ALPHA)
                  for color in (COLOR_U, COLOR_F))

    def marker(face):
        return Line2D([], [], color='k', marker='o', mfc=face, mew=2, ls='none')

    # fig.legend fills column-major, so interleave to get the lines on the top
    # row and the two f1 markers underneath them
    handles = [branch_u, marker('k'), branch_f, marker('white'), ghost]
    labels = [branch_u.get_label(), 'unfolded at $f_1$', branch_f.get_label(),
              'folded at $f_1$', r'ghost: observed state $\neq$ protocol label']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=3, frameon=False, handlelength=2.4, columnspacing=2.4,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})


# --------------------------------------------------------------------------- figure
def make_figure(outfile=None):
    """Build the full figure."""
    forward_runs = build(FORWARD, forward=True)
    reverse_runs = build(REVERSE, forward=False)
    t_max = max(t[-1] for _, t, *_ in forward_runs + reverse_runs) * 1.04

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey='row', sharex='col')

    for col, (forward, runs) in enumerate(((True, forward_runs), (False, reverse_runs))):
        draw_protocol(axes[0, col], forward)
        draw_branch_rails(axes[1, col], forward)
        draw_trajectories(axes[1, col], runs)

    style_axes(axes[0, 0], axes[1, 0],
               r'Forward $f_{\rm min}\!\rightarrow\!f_{\rm max}$: outcome at $f_1$ sets the rate',
               t_max)
    style_axes(axes[0, 1], axes[1, 1],
               r'Reverse $f_{\rm max}\!\rightarrow\!f_{\rm min}$: rate set before the run',
               t_max)
    for ax in axes[:, 1]:
        ax.set_ylabel('')
    annotate(axes)

    for (row, col), letter in zip(np.ndindex(axes.shape), 'abcd'):
        ws.panel_label(axes[row, col], letter + ')', x=-0.1 if col == 0 else -0.02)

    fig.tight_layout(w_pad=1)
    draw_legend(fig)

    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ws.use(verbose=True)
    make_figure(outfile=os.path.join(here, 'Fig1-force-displacement.pdf'))
