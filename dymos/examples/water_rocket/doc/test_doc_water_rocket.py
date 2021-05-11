import unittest
from collections import namedtuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from dymos.examples.water_rocket.phases import (new_water_rocket_trajectory,
                                                set_sane_initial_guesses)

from dymos.utils.doc_utils import save_for_docs
from dymos.utils.testing_utils import require_pyoptsparse


@require_pyoptsparse(optimizer='IPOPT')
@use_tempdirs
class TestWaterRocketForDocs(unittest.TestCase):

    @save_for_docs
    def test_water_rocket_height_for_docs(self):
        p = om.Problem(model=om.Group())

        traj, phases = new_water_rocket_trajectory(objective='height')
        traj = p.model.add_subsystem('traj', traj)

        p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
        p.driver.opt_settings['print_level'] = 4
        p.driver.opt_settings['max_iter'] = 1000
        p.driver.declare_coloring()

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup()
        set_sane_initial_guesses(p, phases)

        p.run_driver()

        summary = summarize_results(p)
        for key, entry in summary.items():
            print(f'{key}: {entry.value:6.4f} {entry.unit}')

        exp_out = traj.simulate(times_per_seg=200)

        # NOTE: only the last figure is shown in the generated docs
        plot_propelled_ascent(p, exp_out)
        plot_trajectory(p, exp_out)
        plot_states(p, exp_out)

        plt.show()

        # Check results (tolerance is relative unless value is zero)
        assert_near_equal(summary['Launch angle'].value, 85, .02)
        assert_near_equal(summary['Flight angle at end of propulsion'].value, 85, .02)
        assert_near_equal(summary['Empty mass'].value, 0.1425114, 1e-3)
        assert_near_equal(summary['Water volume'].value, 0.868281, 1e-3)
        assert_near_equal(summary['Maximum range'].value, 15.78, 5)
        assert_near_equal(summary['Maximum height'].value, 54.133184, 1e-3)
        assert_near_equal(summary['Maximum velocity'].value, 47.320298, 1e-3)

    @save_for_docs
    def test_water_rocket_range_for_docs(self):
        p = om.Problem(model=om.Group())

        traj, phases = new_water_rocket_trajectory(objective='range')
        traj = p.model.add_subsystem('traj', traj)

        p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
        p.driver.opt_settings['print_level'] = 4
        p.driver.opt_settings['max_iter'] = 1000
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
        p.driver.declare_coloring(tol=1.0E-12)

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()
        # p.driver.add_recorder(om.SqliteRecorder('ex_water_rocket.db'))

        p.setup()
        set_sane_initial_guesses(p, phases)

        p.run_driver()

        summary = summarize_results(p)
        for key, entry in summary.items():
            print(f'{key}: {entry.value:6.4f} {entry.unit}')

        exp_out = traj.simulate(times_per_seg=200)

        # NOTE: only the last figure is shown in the generated docs
        plot_propelled_ascent(p, exp_out)
        plot_trajectory(p, exp_out)
        plot_states(p, exp_out)

        plt.show()

        # Check results (tolerance is relative unless value is zero)
        assert_near_equal(summary['Launch angle'].value, 46, 0.02)
        assert_near_equal(summary['Flight angle at end of propulsion'].value, 38, 0.02)
        assert_near_equal(summary['Empty mass'].value, 0.189, 1e-2)
        assert_near_equal(summary['Water volume'].value, 1.026, 1e-2)
        assert_near_equal(summary['Maximum range'].value, 85.11, 1e-2)
        assert_near_equal(summary['Maximum height'].value, 23.08, 1e-2)
        assert_near_equal(summary['Maximum velocity'].value, 41.31, 1e-2)


def plot_trajectory(p, exp_out):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 8))

    time_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.time'),
                'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.time'),
                'descent': p.get_val('traj.descent.timeseries.time')}

    time_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.time'),
                'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.time'),
                'descent': exp_out.get_val('traj.descent.timeseries.time')}

    r_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.states:r'),
             'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.states:r'),
             'descent': p.get_val('traj.descent.timeseries.states:r')}

    r_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.states:r'),
             'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.states:r'),
             'descent': exp_out.get_val('traj.descent.timeseries.states:r')}

    h_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.states:h'),
             'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.states:h'),
             'descent': p.get_val('traj.descent.timeseries.states:h')}

    h_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.states:h'),
             'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.states:h'),
             'descent': exp_out.get_val('traj.descent.timeseries.states:h')}

    axes.plot(r_imp['propelled_ascent'], h_imp['propelled_ascent'], 'ro', markerfacecolor='None')
    axes.plot(r_imp['ballistic_ascent'], h_imp['ballistic_ascent'], 'mo', markerfacecolor='None')
    axes.plot(r_imp['descent'], h_imp['descent'], 'bo', markerfacecolor='None')

    axes.plot(r_exp['propelled_ascent'], h_exp['propelled_ascent'], 'r-')
    axes.plot(r_exp['ballistic_ascent'], h_exp['ballistic_ascent'], 'm-')
    axes.plot(r_exp['descent'], h_exp['descent'], 'b-')

    axes.set_xlabel('r (m)')
    axes.set_ylabel('h (m)')
    axes.set_aspect('equal', 'box')

    fig.tight_layout()


def plot_states(p, exp_out):
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8), sharex=True)

    states = ['r', 'h', 'v', 'gam']
    units = ['m', 'm', 'm/s', 'deg']
    phases = ['propelled_ascent', 'ballistic_ascent', 'descent']

    time_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.time'),
                'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.time'),
                'descent': p.get_val('traj.descent.timeseries.time')}

    time_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.time'),
                'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.time'),
                'descent': exp_out.get_val('traj.descent.timeseries.time')}

    x_imp = {phase: {state: p.get_val(f"traj.{phase}.timeseries.states:{state}", unit)
                     for state, unit in zip(states, units)
                     }
             for phase in phases
             }

    x_exp = {phase: {state: exp_out.get_val(f"traj.{phase}.timeseries.states:{state}", unit)
                     for state, unit in zip(states, units)
                     }
             for phase in phases
             }

    for i, (state, unit) in enumerate(zip(states, units)):
        axes[i].set_ylabel(f"{state} ({unit})" if state != 'gam' else f'$\gamma$ ({unit})')

        axes[i].plot(time_imp['propelled_ascent'], x_imp['propelled_ascent'][state], 'ro', markerfacecolor='None')
        axes[i].plot(time_imp['ballistic_ascent'], x_imp['ballistic_ascent'][state], 'mo', markerfacecolor='None')
        axes[i].plot(time_imp['descent'], x_imp['descent'][state], 'bo', markerfacecolor='None')
        axes[i].plot(time_exp['propelled_ascent'], x_exp['propelled_ascent'][state], 'r-', label='Propelled Ascent')
        axes[i].plot(time_exp['ballistic_ascent'], x_exp['ballistic_ascent'][state], 'm-', label='Ballistic Ascent')
        axes[i].plot(time_exp['descent'], x_exp['descent'][state], 'b-', label='Descent')

        if state == 'gam':
            axes[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins='auto', steps=[1, 1.5, 3, 4.5, 6, 9, 10]))
            axes[i].set_yticks(np.arange(-90, 91, 45))

    axes[i].set_xlabel('t (s)')
    axes[0].legend()

    fig.tight_layout()


def plot_propelled_ascent(p, exp_out):
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(4, 8))
    t_imp = p.get_val('traj.propelled_ascent.time', 's')
    t_exp = exp_out.get_val('traj.propelled_ascent.time', 's')

    ax[0].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.states:p', 'bar'), 'ro', markerfacecolor='None')
    ax[0].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.states:p', 'bar'), 'r-')
    ax[0].set_ylabel('p (bar)')
    ax[0].set_ylim(bottom=0)

    ax[1].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.states:V_w', 'L'), 'ro', markerfacecolor='None')
    ax[1].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.states:V_w', 'L'), 'r-')
    ax[1].set_ylabel('$V_w$ (L)')
    ax[1].set_ylim(0, p.get_val('traj.parameters:V_b', 'L')[0])

    ax[2].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.T', 'N'), 'ro', markerfacecolor='None')
    ax[2].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.T', 'N'), 'r-')
    ax[2].set_ylabel('T (N)')
    ax[2].set_ylim(bottom=0)

    ax[3].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.states:v', 'm/s'), 'ro', markerfacecolor='None')
    ax[3].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.states:v', 'm/s'), 'r-')
    ax[3].set_ylabel('v (m/s)')
    ax[3].set_ylim(bottom=0)

    ax[4].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.states:gam', 'deg'), 'ro', markerfacecolor='None')
    ax[4].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.states:gam', 'deg'), 'r-')
    ax[4].set_ylabel('$\gamma$ (deg)')
    ax[4].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins='auto', steps=[1, 1.5, 3, 4.5, 6, 9, 10]))

    ax[-1].set_xlabel('t (s)')

    fig.tight_layout()


def summarize_results(water_rocket_problem):
    p = water_rocket_problem
    Entry = namedtuple('Entry', 'value unit')
    summary = {
        'Launch angle': Entry(p.get_val('traj.propelled_ascent.timeseries.states:gam',  units='deg')[0, 0], 'deg'),
        'Flight angle at end of propulsion': Entry(p.get_val('traj.propelled_ascent.timeseries.states:gam',
                                                   units='deg')[-1, 0], 'deg'),
        'Empty mass': Entry(p.get_val('traj.parameters:m_empty', units='kg')[0], 'kg'),
        'Water volume': Entry(p.get_val('traj.propelled_ascent.timeseries.states:V_w', 'L')[0, 0], 'L'),
        'Maximum range': Entry(p.get_val('traj.descent.timeseries.states:r', units='m')[-1, 0], 'm'),
        'Maximum height': Entry(p.get_val('traj.ballistic_ascent.timeseries.states:h', units='m')[-1, 0], 'm'),
        'Maximum velocity': Entry(p.get_val('traj.propelled_ascent.timeseries.states:v', units='m/s')[-1, 0], 'm/s'),
    }

    return summary


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
