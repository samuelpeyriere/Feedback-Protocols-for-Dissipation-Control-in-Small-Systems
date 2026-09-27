"""Fig. 4: discrete Maxwell demon (MD protocol, LdF ensemble).
(a) Example force trajectories and the corresponding observed state sequences.
(b) Thermodynamic quantities vs the measurement force f_0, simulation vs theory.

Run `Fig4-DMD-simulation.py` first: it writes `Fig4-DMD-data.npz`, which this script loads.
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

# Panel (a): the two branches after the measurement at f_0, and the measurement point
COLOR_U = ws.OKABE_ITO['blue']        # observed U at f_0 -> jump to f_0 + Deltaf
COLOR_F = ws.OKABE_ITO['gray']        # observed F at f_0 -> jump to f_0 - Deltaf
COLOR_F0 = ws.OKABE_ITO['purple']     # measurement force f_0

# Panel (b): reference lines
COLOR_FC = ws.OKABE_ITO['black']      # vertical line at the coexistence force f_c
COLOR_LOG2 = ws.OKABE_ITO['gray']     # +-ln(2) reference lines

# y-positions of the rows in the state panel
Y_U, Y_F = 0.65, 0.25

# Panel (b) layout:  (row, col), key in `quantities`, title, unit, hide_x
PANELS_B = [
    ((0, 0), 'UPSILON', r'$k_B T \Upsilon$',                     r'[$k_B T$]', True),
    ((0, 1), 'WD',      r'$\langle W_d \rangle$',                r'[$k_B T$]', True),
    ((1, 0), 'ETAI',    r'$\eta_I$',                             '',           False),
    ((1, 1), 'WDUPS',   r'$\langle W_d \rangle + k_B T \Upsilon$', r'[$k_B T$]', False),
]
LABELS_B = ['simulation', 'theory']


# --------------------------------------------------------------------------- panel (a) helpers
def md_times(sim):
    """Time of the jump away from f_0, and time at which f_0 is reached again."""
    p = sim.protocol
    t_jump = sim.Deltat
    return t_jump, t_jump + p.Deltaf / p.rQS


def draw_arrow(ax, x_from, x_to, y, w, color):
    """Horizontal arrow from a letter of width w centered at x_from, up to x_to."""
    x0 = x_from + w / 2 + 0.2 * w
    if x_to > x0:
        ax.arrow(x0, y, x_to - x0, 0, length_includes_head=True, width=.06,
                 head_width=.2, head_length=2, color=color)


# --------------------------------------------------------------------------- panel (b) helpers
def plot_vs_f0(fig, spec, f0_list, curves, title, unit, fc, hide_x=False, sharex=None):
    """One sub-panel of (b): simulation (curves[0]) and theory (curves[1]) vs f_0,
    plus a vertical line at f_c. Returns (ax, handles)."""
    ax = fig.add_subplot(spec, sharex=sharex)
    handles = [ax.plot(f0_list, curve, **ws.curve_style(k, marker=False))[0]
               for k, curve in enumerate(curves)]
    handles.append(ax.axvline(fc, color=COLOR_FC, linewidth=ws.LW_THIN))

    ax.set_title(title, loc='left')
    ax.set_title(unit, loc='right')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(.1))
    if hide_x:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(r'$f_0$ [pN]')
    return ax, handles


def draw_ref_line(ax, y, label, above):
    """Horizontal reference line with its label at the left end."""
    ax.axhline(y, color=COLOR_LOG2, linestyle='--', linewidth=ws.LW_THIN, zorder=0)
    ax.annotate(label, xy=(0.03, y),
                xycoords=blended_transform_factory(ax.transAxes, ax.transData),
                xytext=(0, 4 if above else -4), textcoords='offset points', ha='left',
                va='bottom' if above else 'top', color=COLOR_LOG2, fontsize=ws.FS_TICK)


# --------------------------------------------------------------------------- panels
def draw_panel_a(fig, spec, sim):
    """Force trajectories (top) and state axis (bottom). The state letters are
    drawn later by `draw_state_rows`, once the layout is final."""
    gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=spec,
                                          height_ratios=[6, 1], hspace=0.1)
    ax_traj = fig.add_subplot(gs[0])
    ax_state = fig.add_subplot(gs[1], sharex=ax_traj)

    p = sim.protocol
    f0, Df = p.f0, p.Deltaf
    t_jump, t_end = md_times(sim)
    for sign, color in ((+1, COLOR_U), (-1, COLOR_F)):
        ax_traj.plot([0, t_jump, t_end], [f0, f0 + sign * Df, f0],
                     linewidth=ws.LW_THICK, c=color)
    ax_traj.axhline(f0, ls='dashed', c=COLOR_F0, linewidth=ws.LW_THIN * 0.75, zorder=0)
    ax_traj.scatter([0], [f0], c=COLOR_F0, s=100, zorder=5, clip_on=False)

    # quasi-static return rate on each branch (10 s wide staircase) at 1/3 of the branch,
    # clear of the corner text; labelled on the side of the bar away from the trajectory
    t_mark = t_jump + (t_end - t_jump) / 3
    f_mark = Df * (1 - (t_mark - t_jump) / (t_end - t_jump))
    draw_slope_marker(ax_traj, t_mark, f0 + f_mark, dx=10, dy=-10 * p.rQS,
                      label=r'$-r_0$', color=COLOR_U, label_dx=6, label_dy=0.06,
                      ha='center', va='bottom')
    draw_slope_marker(ax_traj, t_mark, f0 - f_mark, dx=10, dy=10 * p.rQS,
                      label=r'$r_0$', color=COLOR_F, label_dx=6, label_dy=-0.05,
                      ha='center', va='top')

    ax_traj.set_title('Example trajectories')
    ax_traj.set_ylabel('Pull force [pN]')
    ax_traj.tick_params(labelbottom=False)

    # f_0 label just above its dashed line, clear of the vertical jumps at t = 0
    ax_traj.text(.1, f0 + 0.04, r'$f_0 = %g$ pN' % f0, ha='left', va='bottom', color=COLOR_F0,
                 transform=blended_transform_factory(ax_traj.transAxes, ax_traj.transData))
    # protocol parameters in the empty bottom-right corner, below the F branch
    ax_traj.text(.96, .03, r'$r_0 = %g$ pN/s' '\n' r'$\Delta t = %d$ ms' % (p.rQS, sim.Deltat * 1e3),
                 ha='right', va='bottom', linespacing=1.4, transform=ax_traj.transAxes)

    ax_state.set_ylim(-0.05, 1.05)
    ax_state.set_ylabel('Observed\n' r'state $\sigma$')
    ax_state.set_xlabel('Time [s]')
    ax_state.tick_params(left=False, labelleft=False)
    return ax_traj, ax_state


def draw_state_rows(ax_state, sim):
    """Observed state at f_0 (U or F) and the branch that follows until f_0 is reached
    again. Call only after the final layout is fixed.

    The letters are sized in data units, so tight_layout would rescale the axis under them.
    """
    ax_state.set_xlim(ax_state.get_xlim())
    w = max(text_width_in_data(ax_state, s, FS_STATE) for s in 'FU')
    _, t_end = md_times(sim)
    kw = dict(fontsize=FS_STATE, ha='center', va='center', color=COLOR_F0)

    ax_state.text(0, Y_U, 'U', **kw)
    ax_state.text(0, Y_F, 'F', **kw)
    draw_arrow(ax_state, 0, t_end, Y_U, w, COLOR_U)
    draw_arrow(ax_state, 0, t_end, Y_F, w, COLOR_F)


def draw_panel_b(fig, spec, sim, f0_list, quantities, pU, panels=PANELS_B):
    """Grid of quantities vs f_0, with P_U(f_0) bottom right and the legend top right."""
    gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=spec, wspace=0.35, hspace=0.3)
    fc = sim.model.fc
    axes = {}
    for (row, col), key, title, unit, hide_x in panels:
        axes[row, col], handles = plot_vs_f0(
            fig, gs[row, col], f0_list, quantities[key], title, unit, fc, hide_x=hide_x,
            sharex=axes.get((0, 0)))

    draw_ref_line(axes[0, 0], np.log(2), r'$\ln 2$', above=False)
    draw_ref_line(axes[0, 1], -np.log(2), r'$-\ln 2$', above=True)

    plot_vs_f0(fig, gs[1, 2], f0_list, [pU], r'$P_U(f_0)$', '', fc, sharex=axes[0, 0])

    legend_ax = fig.add_subplot(gs[0, 2])
    legend_ax.axis('off')
    legend_ax.legend(handles, LABELS_B + [r'$f_c$'], loc='center')


# --------------------------------------------------------------------------- figure
def make_figure(sim, f0_list, quantities, pU, outfile=None):
    """Build the full figure.

    `quantities` maps the keys of PANELS_B to arrays of shape (2, len(f0_list)):
    row 0 is the simulation, row 1 the theory. `pU` is P_U(f_0) from the simulation.
    """
    fig = plt.figure(figsize=(17, 6))
    outer = gridspec.GridSpec(1, 2, width_ratios=[5, 12], wspace=0.12)

    ax_traj, ax_state = draw_panel_a(fig, outer[0], sim)
    draw_panel_b(fig, outer[1], sim, f0_list, quantities, pU)

    # both panel labels are positioned relative to panel (a)
    ws.panel_label(ax_traj, 'a)', x=-0.1)
    ws.panel_label(ax_traj, 'b)', x=1.1)

    fig.tight_layout()
    draw_state_rows(ax_state, sim)

    if outfile is not None:
        fig.savefig(outfile)   # transparent / tight / pad from ws.rc_params()
    return fig


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(here, 'Fig4-DMD-data.npz'))

    ws.use(verbose=True)
    # protocol of the example trajectories in panel (a)
    sim = hp.Simulation(ensemble='LdF')
    sim.protocol = hp.MDProtocol(f0=15, Deltaf=float(data['Deltaf']), rQS=float(data['rQS']))
    sim.Deltat = float(data['Deltat'])
    quantities = {key: data[key] for key in ('UPSILON', 'WD', 'ETAI')}
    quantities['WDUPS'] = data['WD'] + data['UPSILON']
    make_figure(sim, data['f0_list'], quantities, data['PU'],
                outfile=os.path.join(here, 'Fig4-DMD.pdf'))
