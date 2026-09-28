"""Fig. 6: CTF protocol in the LdF ensemble.
(a) Example force trajectories and the corresponding observed state sequences.
(b) Thermodynamic quantities vs r_U/r_F for several loading rates r_F.

Run `Fig6-CTF-simulation.py` first: it writes `Fig6-CTF-data.npz`, which this script loads.
You can also import `make_figure` and call it directly.
"""
import os

import hairpyn as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch

import wiley_style as ws
from plot_helpers import text_width_in_data, draw_slope_marker, plot_with_inf_axis

# --------------------------------------------------------------------------- figure settings
# Light gray + dark blue: the two trajectories also differ in lightness (grayscale-safe)
COLOR_A = ws.OKABE_ITO['gray']
COLOR_B = ws.OKABE_ITO['blue']

LEGEND_SPACE = 0.11       # fraction of figure height reserved for the legend at the bottom

# Panel (b) layout:  (row, col), key in `quantities`, title, unit, hide_x, share_inf_y
PANELS_B = [
    ((0, 0), 'UPSILON', r'$k_B T \Upsilon_{\infty}$',        r'[$k_B T$]', True,  False),
    ((0, 1), 'WD',      r'$\langle W_d \rangle$',   r'[$k_B T$]', True,  True),
    ((0, 2), 'WD0',     r'$\langle W_d \rangle_0$', r'[$k_B T$]', True,  True),
    ((1, 0), 'ETAI',    r'$\eta_I$',                '',           False, True),
    ((1, 1), 'ETAM',    r'$\eta_M$',                '',           False, False),
    ((1, 2), 'T',       r'$\langle \tau \rangle$',  '[s]',        False, True),
]


# --------------------------------------------------------------------------- panel (a) helpers
def switch_times(sim, f_switch):
    """Time at which the protocol switches from r_F to r_U, and time at which f_max is reached."""
    t_switch = (f_switch - sim.fmin) / sim.protocol.rF
    t_end = t_switch + (sim.fmax - f_switch) / sim.protocol.rU
    return t_switch, t_end


def draw_trajectory(ax, sim, f_switch, color):
    """Piecewise-linear force protocol: f_min -> f_switch at r_F, then -> f_max at r_U."""
    t_switch, t_end = switch_times(sim, f_switch)
    ax.plot([0, t_switch, t_end], [sim.fmin, f_switch, sim.fmax],
            linewidth=ws.LW_THICK, c=color)


def draw_state_row(ax, sim, f_switch, y, color, w_F, w_U):
    """Row of observed states aligned with the protocol.

    The first 'U' is centered exactly on the switching time (the jump). The 'F's
    are packed immediately to its left. The arrow runs to a final 'U' centered on
    the time at which f_max is reached.
    """
    t_switch, t_end = switch_times(sim, f_switch)
    kw = dict(va='center', color=color)

    ax.text(t_switch, y, 'U', ha='center', **kw)

    # as many 'F's as fit between t=0 and the left edge of that 'U'
    x_right = t_switch - w_U / 2
    n_F = int(x_right // w_F)
    if n_F > 0:
        ax.text(x_right, y, 'F' * n_F, ha='right', **kw)

    if np.isfinite(sim.protocol.rU):
        gap = 0.2 * w_U
        x0, x1 = t_switch + w_U / 2 + gap, t_end - w_U / 2 - gap
        if x1 > x0:
            ax.arrow(x0, y, x1 - x0, 0, length_includes_head=True, width=.06,
                     head_width=.2, head_length=.05, color=color)
            ax.text(t_end, y, 'U', ha='center', **kw)


def draw_jump_guide(ax_traj, ax_state, sim, f_switch, y, color):
    """Dotted line from the jump in the force trace down to the 'U' in the state row."""
    t_switch, _ = switch_times(sim, f_switch)
    con = ConnectionPatch(xyA=(t_switch, f_switch), coordsA=ax_traj.transData,
                          xyB=(t_switch, y + 0.2), coordsB=ax_state.transData,
                          color=color, linestyle=':', linewidth=1.5, zorder=0.5)
    ax_traj.figure.add_artist(con)


# --------------------------------------------------------------------------- panels
def draw_panel_a(fig, spec, sim, examples):
    """Force trajectories (top) and state axis (bottom). The state letters are
    drawn later by `draw_state_rows`, once the layout is final."""
    gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=spec,
                                          height_ratios=[6, 1], hspace=0.1)
    ax_traj = fig.add_subplot(gs[0])
    ax_state = fig.add_subplot(gs[1], sharex=ax_traj)

    for f_switch, color, _ in examples:
        draw_trajectory(ax_traj, sim, f_switch, color)

    # r_F marker on the loading segment (common to all trajectories)
    f_first = examples[0][0]
    t_first, _ = switch_times(sim, f_first)
    draw_slope_marker(ax_traj, t_first / 2, (sim.fmin + f_first) / 2, dx=0.25, dy=1,
                      label=r'$r_F$', color=COLOR_B)

    # r_U markers on each unloading segment
    if np.isfinite(sim.protocol.rU):
        for f_switch, color, _ in examples:
            t_switch, t_end = switch_times(sim, f_switch)
            draw_slope_marker(ax_traj, (t_switch + t_end) / 2, (f_switch + sim.fmax) / 2,
                              dx=0.125, dy=1.9, label=r'$r_U$', color=color)

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
    for f_switch, color, y in examples:
        draw_state_row(ax_state, sim, f_switch, y, color, w_F, w_U)
        draw_jump_guide(ax_traj, ax_state, sim, f_switch, y, color)


def draw_panel_b(fig, spec, rF_list, ratio_list, quantities, panels=PANELS_B):
    """Grid of quantities vs r_U/r_F. Returns legend handles and labels."""
    gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=spec, wspace=0.35, hspace=0.3)
    labels = [r'$r_F = %d$ pN/s' % rF for rF in rF_list]
    handles = []
    for (row, col), key, title, unit, hide_x, share_inf_y in panels:
        _, _, handles, _ = plot_with_inf_axis(
            fig, gs[row, col], ratio_list, quantities[key], labels,
            title=title, unit=unit, xlabel=r'$r_U$/$r_F$',
            hide_x=hide_x, share_inf_y=share_inf_y)
    return handles, labels


# --------------------------------------------------------------------------- figure
def make_figure(sim, rF_list, ratio_list, quantities, examples=None, outfile=None):
    """Build the full figure.

    `quantities` maps the keys of PANELS_B to arrays of shape (len(rF_list), len(ratio_list)).
    `examples` is a list of (switch force, color, y of state row).
    """
    if examples is None:
        examples = [(13, COLOR_A, 0.65),
                    (15, COLOR_B, 0.25)]

    # taller figure so the subplots keep ~ their size despite the legend strip
    fig = plt.figure(figsize=(17, 6 / (1 - LEGEND_SPACE)))
    outer = gridspec.GridSpec(1, 2, width_ratios=[5, 12], wspace=0.12)

    ax_traj, ax_state = draw_panel_a(fig, outer[0], sim, examples)
    handles, labels = draw_panel_b(fig, outer[1], rF_list, ratio_list, quantities)

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
    data = np.load(os.path.join(here, 'Fig6-CTF-data.npz'))

    ws.use(verbose=True)
    # protocol of the example trajectories in panel (a)
    sim = hp.Simulation(ensemble='LdF')
    sim.protocol = hp.CTFProtocol(rF=4, rU=16)
    quantities = {key: data[key] for key in ('UPSILON', 'WD', 'WD0', 'ETAI', 'ETAM', 'T')}
    make_figure(sim, data['rF_list'], data['ratio_list'], quantities,
                outfile=os.path.join(here, 'Fig6-CTF.pdf'))
