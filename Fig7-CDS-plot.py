"""Fig. 7: constant-DS protocol ("CDS" in file names) in the LdF (force) ensemble.
(a) Example force trajectories and the corresponding observed state sequences.
    The force is loaded at r_F up to the decision force f_1, where the state is observed:
      - U -> the force goes to f_max at r_U                       (trajectory A)
      - F -> loading continues at r_F' until unfolding at f_2,
             then the force goes to f_max at r_U                  (trajectory B)
(b) Thermodynamic quantities vs f_1 for several r_F'/r_F.

Run `Fig7-CDS-simulation.py` first: it writes `Fig7-CDS-data.npz`, which this script loads.
You can also import `make_figure` and call it directly.
"""
import os

import hairpyn as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import blended_transform_factory

import wiley_style as ws
from plot_helpers import text_width_in_data, draw_slope_marker

# --------------------------------------------------------------------------- figure settings
# Light gray + dark blue: the two trajectories also differ in lightness (grayscale-safe)
COLOR_A = ws.OKABE_ITO['gray']      # unfolds at the checkpoint
COLOR_B = ws.OKABE_ITO['blue']      # passes the checkpoint folded, unfolds later at f_2
COLOR_CP = ws.OKABE_ITO['purple']   # checkpoint (force f_1 and the observation made there)
COLOR_FC = ws.OKABE_ITO['black']    # coexistence force f_c in panel (b)

SKIP_FIRST = 1            # number of leading f_1 points left out of panel (b)

F2_EXAMPLE = 16           # unfolding force of trajectory B in panel (a)

LEGEND_SPACE = 0.11       # fraction of figure height reserved for the legend at the bottom


def panels_b(rU_finite):
    """Panel (b) layout:  (row, col), key in `quantities`, title, unit, hide_x, ylim"""
    return [
        ((0, 0), 'UPSILON', r'$k_B T \Upsilon_{\rm DS}$',        r'[$k_B T$]', True,
         (-0.1, 2.8) if rU_finite else (-0.5, 15)),
        ((0, 1), 'WD',      r'$\langle W_d \rangle$',   r'[$k_B T$]', True,
         (-0.1, 2.8) if rU_finite else (-1, 2.8)),
        ((0, 2), 'WD0',     r'$\langle W_d \rangle_0$', r'[$k_B T$]', True,  None),
        ((1, 0), 'ETAI',    r'$\eta_I$',                '',           False, (-0.05, 0.75)),
        ((1, 1), 'ETAM',    r'$\eta_M$',                '',           False, (-0.05, 0.75)),
        ((1, 2), 'T',       r'$\langle \tau \rangle$',  '[s]',        False, None),
    ]


# --------------------------------------------------------------------------- panel (a) helpers
def protocol_times(sim, f1, f2):
    """Checkpoint time t1, time tU at which f_max is reached if U is observed at f_1,
    unfolding time t2 at f_2 if F is observed at f_1, and time tF at which f_max is then reached."""
    p = sim.protocol
    t1 = (f1 - sim.fmin) / p.rF
    tU = t1 + (sim.fmax - f1) / p.rU
    t2 = t1 + (f2 - f1) / p.rFprime
    tF = t2 + (sim.fmax - f2) / p.rU
    return t1, tU, t2, tF


def make_examples(sim, f2):
    """The two example trajectories of panel (a): A unfolds at the checkpoint, B at f2."""
    f1 = sim.protocol.f1
    t1, tU, t2, tF = protocol_times(sim, f1, f2)
    return [
        dict(vertices=[(0, sim.fmin), (t1, f1), (tU, sim.fmax)],
             t_unfold=t1, f_unfold=f1, t_end=tU, color=COLOR_A, y=0.65),
        dict(vertices=[(0, sim.fmin), (t1, f1), (t2, f2), (tF, sim.fmax)],
             t_unfold=t2, f_unfold=f2, t_end=tF, color=COLOR_B, y=0.25),
    ]


def draw_trajectory(ax, vertices, color):
    """Piecewise-linear force protocol through the given (t, f) vertices."""
    t, f = zip(*vertices)
    ax.plot(t, f, linewidth=ws.LW_THICK, c=color)


def draw_checkpoint(ax, t1, f1):
    """Checkpoint dot, dashed guides to both axes, and the f_1 label."""
    xl, yl = ax.get_xlim(), ax.get_ylim()   # freeze limits so the guides reach the spines
    ax.plot([xl[0], t1], [f1, f1], ls='--', c=COLOR_CP, linewidth=1.5, zorder=1)
    ax.plot([t1, t1], [yl[0], f1], ls='--', c=COLOR_CP, linewidth=1.5, zorder=1)
    ax.scatter([t1], [f1], c=COLOR_CP, s=100, zorder=5)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.text(0.03, f1 + 0.2, r'$f_1 = %g$ pN' % f1, ha='left', va='bottom', color=COLOR_CP,
            transform=blended_transform_factory(ax.transAxes, ax.transData))


def pack_F(ax, x_left, x_right, y, w_F, **kw):
    """As many 'F's as fit between x_left and x_right, right-aligned at x_right."""
    n_F = int((x_right - x_left) // w_F)
    if n_F > 0:
        ax.text(x_right, y, 'F' * n_F, ha='right', **kw)


def draw_state_row(ax, ex, t_check, rU_finite, w_F, w_U):
    """Row of observed states aligned with the protocol.

    The observation at the checkpoint is centered on t_check (in COLOR_CP), the first
    'U' is centered on the unfolding time, the 'F's are packed in between, and the
    arrow runs to a final 'U' centered on the time at which f_max is reached.
    """
    y, color = ex['y'], ex['color']
    t_unf, t_end = ex['t_unfold'], ex['t_end']
    kw = dict(va='center')

    if np.isclose(t_unf, t_check):
        # unfolding observed at the checkpoint
        ax.text(t_check, y, 'U', ha='center', color=COLOR_CP, **kw)
        pack_F(ax, 0, t_check - w_U / 2, y, w_F, color=color, **kw)
    else:
        # folded at the checkpoint, unfolds later
        ax.text(t_check, y, 'F', ha='center', color=COLOR_CP, **kw)
        pack_F(ax, 0, t_check - w_F / 2, y, w_F, color=color, **kw)
        ax.text(t_unf, y, 'U', ha='center', color=color, **kw)
        pack_F(ax, t_check + w_F / 2, t_unf - w_U / 2, y, w_F, color=color, **kw)

    if rU_finite:
        gap = 0.2 * w_U
        x0, x1 = t_unf + w_U / 2 + gap, t_end - w_U / 2 - gap
        if x1 > x0:
            ax.arrow(x0, y, x1 - x0, 0, length_includes_head=True, width=.06,
                     head_width=.2, head_length=.05, color=color)
            ax.text(t_end, y, 'U', ha='center', color=color, **kw)


def draw_jump_guide(ax_traj, ax_state, t, f, y, color):
    """Dotted line from the unfolding point in the force trace down to the 'U' in the state row."""
    con = ConnectionPatch(xyA=(t, f), coordsA=ax_traj.transData,
                          xyB=(t, y + 0.2), coordsB=ax_state.transData,
                          color=color, linestyle=':', linewidth=1.5, zorder=0.5)
    ax_traj.figure.add_artist(con)


# --------------------------------------------------------------------------- panel (b) helper
def plot_quantity(fig, spec, x, ratio_list, curves, title, unit,
                  fc=None, ylim=None, hide_x=False):
    """One sub-panel of (b): curves[k] vs f_1, one line per r_F'/r_F in ratio_list.

    Returns (ax, handles, labels).
    """
    ax = fig.add_subplot(spec)
    x = np.asarray(x)[SKIP_FIRST:]

    handles, labels = [], []
    for k, (ratio, curve) in enumerate(zip(ratio_list, curves)):
        line, = ax.plot(x, np.asarray(curve)[SKIP_FIRST:], **ws.curve_style(k, marker=False))
        handles.append(line)
        labels.append(r"$r_F' = r_F$" if ratio == 1 else r"$r_F' = %g\, r_F$" % ratio)

    if fc is not None:
        h = ax.axvline(fc, color=COLOR_FC, linewidth=ws.LW_THIN, zorder=0)
        handles.append(h)
        labels.append(r'$f_c$')

    ax.set_title(title, loc='left')
    ax.set_title(unit, loc='right')
    if hide_x:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(r'$f_1$ [pN]')
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax, handles, labels


# --------------------------------------------------------------------------- panels
def draw_panel_a(fig, spec, sim, examples):
    """Force trajectories (top) and state axis (bottom). The state letters are
    drawn later by `draw_state_rows`, once the layout is final."""
    gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=spec,
                                          height_ratios=[6, 1], hspace=0.1)
    ax_traj = fig.add_subplot(gs[0])
    ax_state = fig.add_subplot(gs[1], sharex=ax_traj)

    for ex in examples:
        draw_trajectory(ax_traj, ex['vertices'], ex['color'])

    # rate markers; dy = rate * dx so the staircase matches the actual slope
    p = sim.protocol
    t1, f1 = examples[0]['t_unfold'], examples[0]['f_unfold']   # A unfolds at the checkpoint
    t2, f2 = examples[1]['t_unfold'], examples[1]['f_unfold']
    draw_slope_marker(ax_traj, t1 / 2, (sim.fmin + f1) / 2, dx=0.25, dy=p.rF * 0.25,
                      label=r'$r_F$', color=COLOR_B)
    draw_slope_marker(ax_traj, (t1 + t2) / 2, (f1 + f2) / 2, dx=0.25, dy=p.rFprime * 0.25,
                      label=r"$r_F'$", color=COLOR_B, label_dx=0.2, label_dy=-1)
    if np.isfinite(p.rU):
        for ex in examples:
            t0, f0 = ex['t_unfold'], ex['f_unfold']
            draw_slope_marker(ax_traj, (t0 + ex['t_end']) / 2, (f0 + sim.fmax) / 2,
                              dx=0.125, dy=p.rU * 0.125, label=r'$r_U$', color=ex['color'])

    draw_checkpoint(ax_traj, t1, f1)

    ax_traj.set_title('Example trajectories')
    ax_traj.set_ylabel('Pull force [pN]')
    ax_traj.tick_params(labelbottom=False)
    ax_traj.text(.8, .02, r'$\Delta t = %d$ ms' % (sim.Deltat * 1e3),
                 ha='center', transform=ax_traj.transAxes)

    ax_state.set_ylim(-0.05, 1.05)
    ax_state.set_ylabel('Observed\n' r'state $\sigma$')
    ax_state.set_xlabel('Time [s]')
    ax_state.tick_params(left=False, labelleft=False)
    return ax_traj, ax_state


def draw_state_rows(ax_traj, ax_state, sim, examples):
    """Draw the state letters. Call only after the final layout is fixed.

    The letters are sized in data units, so tight_layout would rescale the axis under them.
    """
    ax_state.set_xlim(ax_state.get_xlim())
    w_F = text_width_in_data(ax_state, 'F')
    w_U = text_width_in_data(ax_state, 'U')
    t1 = examples[0]['t_unfold']   # trajectory A unfolds at the checkpoint
    rU_finite = np.isfinite(sim.protocol.rU)
    for ex in examples:
        draw_state_row(ax_state, ex, t1, rU_finite, w_F, w_U)
        draw_jump_guide(ax_traj, ax_state, ex['t_unfold'], ex['f_unfold'], ex['y'], ex['color'])


def draw_panel_b(fig, spec, sim, f1_list, ratio_list, quantities):
    """Grid of quantities vs f_1. Returns legend handles and labels."""
    gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=spec, wspace=0.35, hspace=0.3)
    for (row, col), key, title, unit, hide_x, ylim in panels_b(np.isfinite(sim.protocol.rU)):
        _, handles, labels = plot_quantity(fig, gs[row, col], f1_list, ratio_list,
                                           quantities[key], title, unit,
                                           fc=sim.model.fc, ylim=ylim, hide_x=hide_x)

    return handles, labels


# --------------------------------------------------------------------------- figure
def make_figure(sim, f1_list, ratio_list, quantities, f2=F2_EXAMPLE, outfile=None):
    """Build the full figure.

    `quantities` maps the keys of `panels_b` to arrays of shape (len(ratio_list), len(f1_list)).
    `f2` is the unfolding force of example trajectory B in panel (a).
    """
    examples = make_examples(sim, f2)

    # taller figure so the subplots keep ~ their size despite the legend strip
    fig = plt.figure(figsize=(17, 6 / (1 - LEGEND_SPACE)))
    outer = gridspec.GridSpec(1, 2, width_ratios=[5, 12], wspace=0.12)

    ax_traj, ax_state = draw_panel_a(fig, outer[0], sim, examples)
    handles, labels = draw_panel_b(fig, outer[1], sim, f1_list, ratio_list, quantities)

    # both panel labels are positioned relative to panel (a)
    ws.panel_label(ax_traj, 'a)', x=-0.1)
    ws.panel_label(ax_traj, 'b)', x=1.1)

    # layout: tight_layout ignores `rect` with nested gridspecs, so free the bottom strip by hand
    fig.tight_layout()
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + LEGEND_SPACE)

    # legend in one row, centered under panel (b)
    box_b = outer[1].get_position(fig)
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=((box_b.x0 + box_b.x1) / 2, 0.005))

    draw_state_rows(ax_traj, ax_state, sim, examples)

    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(here, 'Fig7-CDS-data.npz'))

    ws.use(verbose=True)
    # protocol of the example trajectories in panel (a); r_F and r_U as in the simulation
    sim = hp.Simulation(ensemble='LdF')
    sim.protocol = hp.Strategy(rF=float(data['rF']), rFprime=1, rU=float(data['rU']), cp1=15)
    quantities = {key: data[key] for key in ('UPSILON', 'WD', 'WD0', 'ETAI', 'ETAM', 'T')}
    make_figure(sim, data['f1_list'], data['ratio_list'], quantities,
                outfile=os.path.join(here, 'Fig7-CDS.pdf'))
