"""Fig. 5: DTF protocol.
(a) Example force trajectories and the corresponding observed state sequences.
(b) Thermodynamic quantities vs the decision force f_1, one curve per r_U/r_F.

Run `Fig5-DTF-simulation.py` first: it writes `Fig5-DTF-data.npz`, which this script loads.
You can also import `make_figure` and call it directly.
"""
import os

import hairpyn as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.transforms import blended_transform_factory

import wiley_style as ws
from plot_helpers import text_width_in_data, draw_slope_marker

# --------------------------------------------------------------------------- figure settings
FS_STATE = 18             # state letters, slightly smaller than labels

# Panel (a): loading segment, the two branches after f_1, and the decision point f_1
COLOR_LOAD = ws.OKABE_ITO['orange']
COLOR_F = ws.OKABE_ITO['skyblue']     # observed F at f_1 -> keep pulling at r_F
COLOR_U = ws.OKABE_ITO['gray']        # observed U at f_1 -> pull at r_U
COLOR_F1 = ws.OKABE_ITO['purple']     # decision force f_1

# Panel (b): reference lines
COLOR_FC = ws.OKABE_ITO['black']      # vertical line at the coexistence force f_c
COLOR_LOG2 = ws.OKABE_ITO['gray']     # ln(2) reference line in the Upsilon panel

# y-positions of the rows in the state panel
Y_U, Y_LOAD, Y_F = 0.65, 0.45, 0.25

LEGEND_SPACE = 0.11       # fraction of figure height reserved for the legend at the bottom

# Panel (b) layout:  (row, col), key in `quantities`, title, unit, minor_y, hide_x
PANELS_B = [
    ((0, 0), 'UPSILON', r'$k_B T \Upsilon_2$',        r'[$k_B T$]', 0.1,   True),
    ((0, 1), 'WD',      r'$\langle W_d \rangle$',   r'[$k_B T$]', 0.01,  True),
    ((0, 2), 'WD0',     r'$\langle W_d \rangle_0$', r'[$k_B T$]', None,  True),
    ((1, 0), 'ETAI',    r'$\eta_I$',                '',           0.025, False),
    ((1, 1), 'ETAM',    r'$\eta_M$',                '',           0.05,  False),
    ((1, 2), 'T',       r'$\langle \tau \rangle$',  '[s]',        0.1,   False),
]


# --------------------------------------------------------------------------- panel (a) helpers
def dtf_times(sim):
    """Time of the decision at f_1, and times at which f_max is reached on each branch."""
    p = sim.protocol
    t1 = (p.cp1 - sim.fmin) / p.rF
    t_end_F = t1 + (sim.fmax - p.cp1) / p.rF    # F observed: continue at r_F
    t_end_U = t1 + (sim.fmax - p.cp1) / p.rU    # U observed: continue at r_U
    return t1, t_end_F, t_end_U


def draw_arrow(ax, x_from, x_to, y, w, color):
    """Horizontal arrow between two letters of width w centered at x_from and x_to."""
    gap = 0.2 * w
    x0, x1 = x_from + w / 2 + gap, x_to - w / 2 - gap
    if x1 > x0:
        ax.arrow(x0, y, x1 - x0, 0, length_includes_head=True, width=.06,
                 head_width=.2, head_length=.05, color=color)


def draw_f1_guides(ax_traj, sim):
    """Dashed lines from both axes to the decision point (t_1, f_1).
    Call after the layout (axis limits) is fixed."""
    t1, _, _ = dtf_times(sim)
    f1 = sim.protocol.cp1
    x_left, y_bottom = ax_traj.get_xlim()[0], ax_traj.get_ylim()[0]
    ax_traj.hlines(f1, x_left, t1, colors=COLOR_F1, linestyles='dashed')
    ax_traj.vlines(t1, y_bottom, f1, colors=COLOR_F1, linestyles='dashed')


# --------------------------------------------------------------------------- panel (b) helpers
def ratio_label(ratio):
    if np.isinf(ratio):
        return r'$r_U = \infty$'
    return r'$r_U = r_F$' if ratio == 1 else r'$r_U = %g \cdot r_F$' % ratio


def plot_vs_f1(fig, spec, cp1_list, ratio_list, curves, title, unit, fc,
               minor_y=None, hide_x=False):
    """One sub-panel of (b): curves[k] vs f_1, one line per r_U/r_F in ratio_list,
    plus a vertical line at f_c. Returns (ax, handles, labels)."""
    ax = fig.add_subplot(spec)
    handles, labels = [], []
    for k, (ratio, curve) in enumerate(zip(ratio_list, curves)):
        line, = ax.plot(cp1_list, curve, **ws.curve_style(k, marker=False))
        handles.append(line)
        labels.append(ratio_label(ratio))

    fc_line = ax.axvline(fc, color=COLOR_FC, linewidth=ws.LW_THIN)
    handles.append(fc_line)
    labels.append(r'$f_c$')

    ax.set_title(title, loc='left')
    ax.set_title(unit, loc='right')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_y) if minor_y
                               else ticker.AutoMinorLocator())
    if hide_x:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(r'$f_1$ [pN]')
    return ax, handles, labels


# --------------------------------------------------------------------------- panels
def draw_panel_a(fig, spec, sim):
    """Force trajectories (top) and state axis (bottom). The state letters and f_1
    guides are drawn later by `draw_state_rows`, once the layout is final."""
    gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=spec,
                                          height_ratios=[6, 1], hspace=0.1)
    ax_traj = fig.add_subplot(gs[0])
    ax_state = fig.add_subplot(gs[1], sharex=ax_traj)

    f1 = sim.protocol.cp1
    t1, t_end_F, t_end_U = dtf_times(sim)
    ax_traj.plot([0, t1], [sim.fmin, f1], linewidth=ws.LW_THICK, c=COLOR_LOAD)
    ax_traj.plot([t1, t_end_F], [f1, sim.fmax], linewidth=ws.LW_THICK, c=COLOR_F)
    ax_traj.plot([t1, t_end_U], [f1, sim.fmax], linewidth=ws.LW_THICK, c=COLOR_U)
    ax_traj.scatter([t1], [f1], c=COLOR_F1, s=100, zorder=5)

    # rate markers: r_F on the loading and on the F branch, r_U on the U branch
    draw_slope_marker(ax_traj, t1 / 2, (sim.fmin + f1) / 2, dx=0.25, dy=1,
                      label=r'$r_F$', color=COLOR_LOAD)
    draw_slope_marker(ax_traj, (t1 + t_end_F) / 2, (f1 + sim.fmax) / 2, dx=0.25, dy=1,
                      label=r'$r_F$', color=COLOR_F)
    if np.isfinite(sim.protocol.rU):
        draw_slope_marker(ax_traj, (t1 + t_end_U) / 2, (f1 + sim.fmax) / 2, dx=0.125, dy=1.7,
                          label=r'$r_U$', color=COLOR_U)

    ax_traj.set_title('Example trajectories')
    ax_traj.set_ylabel('Pull force [pN]')
    ax_traj.tick_params(labelbottom=False)

    # f_1 label sits just above its dashed line (x in axes coords, y in data coords)
    ax_traj.text(.03, f1 + 0.15, r'$f_1 = %d$ pN' % f1, ha='left', va='bottom', color=COLOR_F1,
                 transform=blended_transform_factory(ax_traj.transAxes, ax_traj.transData))
    ax_traj.text(.83, .11, r'$r_F = %d$ pN/s' % sim.protocol.rF,
                 ha='center', transform=ax_traj.transAxes)
    ax_traj.text(.8, .02, r'$\Delta t = %d$ ms' % (sim.Deltat * 1e3),
                 ha='center', transform=ax_traj.transAxes)

    ax_state.set_ylim(-0.05, 1.05)
    ax_state.set_ylabel('Observed\n' r'state $\sigma$')
    ax_state.set_xlabel('Time [s]')
    ax_state.tick_params(left=False, labelleft=False)
    return ax_traj, ax_state


def draw_state_rows(ax_traj, ax_state, sim):
    """Observed states: F during loading, the observation at f_1 (U or F), and the
    subsequent branch up to f_max, where the molecule is unfolded (U). Also draws
    the f_1 guides. Call only after the final layout is fixed.

    The letters are sized in data units, so tight_layout would rescale the axis under them.
    """
    ax_traj.set_ylim(ax_traj.get_ylim())
    ax_state.set_xlim(ax_state.get_xlim())
    w = max(text_width_in_data(ax_state, s, FS_STATE) for s in 'FU')
    t1, t_end_F, t_end_U = dtf_times(sim)
    kw = dict(fontsize=FS_STATE, ha='center', va='center')

    # loading: F ... -> decision at f_1
    ax_state.text(0, Y_LOAD, 'F', color=COLOR_LOAD, **kw)
    draw_arrow(ax_state, 0, t1, Y_LOAD, w, COLOR_LOAD)

    # observation at f_1
    ax_state.text(t1, Y_U, 'U', color=COLOR_F1, **kw)
    ax_state.text(t1, Y_F, 'F', color=COLOR_F1, **kw)

    # F branch: keep pulling at r_F until f_max
    draw_arrow(ax_state, t1, t_end_F, Y_F, w, COLOR_F)
    ax_state.text(t_end_F, Y_F, 'U', color=COLOR_F, **kw)

    # U branch: pull at r_U until f_max (instantaneous if r_U = inf)
    if np.isfinite(sim.protocol.rU):
        draw_arrow(ax_state, t1, t_end_U, Y_U, w, COLOR_U)
        ax_state.text(t_end_U, Y_U, 'U', color=COLOR_U, **kw)

    draw_f1_guides(ax_traj, sim)


def draw_panel_b(fig, spec, sim, cp1_list, ratio_list, quantities, panels=PANELS_B):
    """Grid of quantities vs f_1. Returns legend handles and labels."""
    gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=spec, wspace=0.35, hspace=0.3)
    axes = {}
    for (row, col), key, title, unit, minor_y, hide_x in panels:
        axes[row, col], handles, labels = plot_vs_f1(
            fig, gs[row, col], cp1_list, ratio_list, quantities[key], title, unit,
            fc=sim.model.fc, minor_y=minor_y, hide_x=hide_x)

    # ln(2) reference in the Upsilon panel, labelled at the right end just below the line
    ax_ups = axes[0, 0]
    ax_ups.axhline(np.log(2), color=COLOR_LOG2, linestyle='--', linewidth=ws.LW_THIN, zorder=0)
    ax_ups.annotate(r'$\ln 2$', xy=(0.97, np.log(2)),
                    xycoords=blended_transform_factory(ax_ups.transAxes, ax_ups.transData),
                    xytext=(0, -4), textcoords='offset points', ha='right', va='top',
                    color=COLOR_LOG2, fontsize=ws.FS_TICK)

    return handles, labels


# --------------------------------------------------------------------------- figure
def make_figure(sim, cp1_list, ratio_list, quantities, outfile=None):
    """Build the full figure.

    `quantities` maps the keys of PANELS_B to arrays of shape (len(ratio_list), len(cp1_list)).
    """
    # taller figure so the subplots keep ~ their size despite the legend strip
    fig = plt.figure(figsize=(17, 6 / (1 - LEGEND_SPACE)))
    outer = gridspec.GridSpec(1, 2, width_ratios=[2, 3], wspace=0.12)

    ax_traj, ax_state = draw_panel_a(fig, outer[0], sim)
    handles, labels = draw_panel_b(fig, outer[1], sim, cp1_list, ratio_list, quantities)

    # both panel labels are positioned relative to panel (a)
    ws.panel_label(ax_traj, 'a)', x=-0.1)
    ws.panel_label(ax_traj, 'b)', x=1.1)

    # layout: tight_layout ignores `rect` with nested gridspecs, so free the bottom strip by hand
    fig.tight_layout()
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + LEGEND_SPACE)

    # legend in one row, right-aligned with panel (b): it is wider than panel (b), and
    # centered it would stick out past the right edge and widen the saved figure
    box_b = outer[1].get_position(fig)
    fig.legend(handles, labels, loc='lower right', ncol=len(labels),
               bbox_to_anchor=(box_b.x1, 0.005))

    draw_state_rows(ax_traj, ax_state, sim)

    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(here, 'Fig5-DTF-data.npz'))
    ensemble = str(data['ensemble'])

    ws.use(verbose=True)
    # protocol of the example trajectories in panel (a)
    sim = hp.Simulation(ensemble=ensemble)
    sim.protocol = hp.DTFProtocol(rF=float(data['rF']), rU=16, cp1=15)
    quantities = {key: data[key] for key in ('UPSILON', 'WD', 'WD0', 'ETAI', 'ETAM', 'T')}
    make_figure(sim, data['cp1_list'], data['ratio_list'], quantities,
                outfile=os.path.join(here, 'Fig5-DTF.pdf'))
