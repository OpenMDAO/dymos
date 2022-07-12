import unittest
import numpy as np
from numpy.polynomial import Polynomial as P
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
import dymos as dm
from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse


def min_time_climb(optimizer='SLSQP', num_seg=3, transcription='gauss-lobatto',
                   transcription_order=3, force_alloc_complex=False, add_rate=False):

    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.declare_coloring()

    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['Function precision'] = 1.0E-12
        p.driver.opt_settings['Linesearch tolerance'] = 0.1
        p.driver.opt_settings['Major step limit'] = 0.5
    elif optimizer == 'IPOPT':
        p.driver.opt_settings['tol'] = 1.0E-5
        p.driver.opt_settings['print_level'] = 5
        p.driver.opt_settings['mu_strategy'] = 'monotone'
        p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
        p.driver.opt_settings['mu_init'] = 0.01

    t = {'gauss-lobatto': dm.GaussLobatto(num_segments=num_seg, order=transcription_order),
         'radau-ps': dm.Radau(num_segments=num_seg, order=transcription_order)}

    traj = dm.Trajectory()

    phase = dm.Phase(ode_class=MinTimeClimbODE, transcription=t[transcription])
    traj.add_phase('phase0', phase)

    p.model.add_subsystem('traj', traj)

    phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                           duration_ref=100.0)

    phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6,
                    ref=1.0E3, defect_ref=1.0E3, units='m',
                    rate_source='flight_dynamics.r_dot')

    phase.add_state('h', fix_initial=True, lower=0, upper=20000.0,
                    ref=1.0E2, defect_ref=1.0E2, units='m',
                    rate_source='flight_dynamics.h_dot', targets=['h'])

    phase.add_state('v', fix_initial=True, lower=10.0,
                    ref=1.0E2, defect_ref=1.0E2, units='m/s',
                    rate_source='flight_dynamics.v_dot', targets=['v'])

    phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5,
                    ref=1.0, defect_ref=1.0, units='rad',
                    rate_source='flight_dynamics.gam_dot', targets=['gam'])

    phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5,
                    ref=1.0E3, defect_ref=1.0E3, units='kg',
                    rate_source='prop.m_dot', targets=['m'])

    phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      rate_continuity=True, rate_continuity_scaler=100.0,
                      rate2_continuity=False, targets=['alpha'])

    phase.add_parameter('S', val=49.2386, units='m**2', opt=False, targets=['S'])
    phase.add_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
    phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

    phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3)
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0)
    phase.add_boundary_constraint('gam', loc='final', equals=0.0)

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

    # Unnecessary but included to test capability
    phase.add_path_constraint(name='alpha', lower=-8, upper=8)
    phase.add_path_constraint(name='time', lower=0, upper=400)
    phase.add_path_constraint(name='time_phase', lower=0, upper=400)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', ref=1.0)

    # test mixing wildcard ODE variable expansion and unit overrides
    phase.add_timeseries_output(['aero.*', 'prop.thrust', 'prop.m_dot'],
                                units={'aero.f_lift': 'lbf', 'prop.thrust': 'lbf'})

    # test adding rate as timeseries output
    if add_rate:
        phase.add_timeseries_output('aero.mach', rate=True)

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True, force_alloc_complex=force_alloc_complex)

    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = 300.0

    p['traj.phase0.states:r'] = phase.interp('r', [0.0, 111319.54])
    p['traj.phase0.states:h'] = phase.interp('h', [100.0, 20000.0])
    p['traj.phase0.states:v'] = phase.interp('v', [135.964, 283.159])
    p['traj.phase0.states:gam'] = phase.interp('gam', [0.0, 0.0])
    p['traj.phase0.states:m'] = phase.interp('m', [19030.468, 16841.431])
    p['traj.phase0.controls:alpha'] = phase.interp('alpha', [0.0, 0.0])

    dm.run_problem(p, simulate=True)

    return p


@use_tempdirs
class TestMinTimeClimb(unittest.TestCase):

    def _test_results(self, p):
        """ Verify the results of the optimization. """
        # Verify that ODE output mach is added to the timeseries
        assert_near_equal(p.get_val('traj.phase0.timeseries.mach')[-1], 1.0, tolerance=1.0E-2)

        # Check that time matches to within 1% of an externally verified solution.
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 321.0, tolerance=0.02)

    def _test_wilcard_outputs(self, p):
        """ Test that all wilcard outputs are provided. """
        output_dict = dict(p.model.list_outputs(units=True, out_stream=None))
        ts = {k: v for k, v in output_dict.items() if 'timeseries.' in k}
        for c in ['mach', 'CD0', 'kappa', 'CLa', 'CL', 'CD', 'q', 'f_lift', 'f_drag', 'thrust', 'm_dot']:
            assert(any([True for t in ts if 'timeseries.' + c in t]))

    def _test_timeseries_units(self, p):
        """ Test that the units from the timeseries are correct. """
        output_dict = dict(p.model.list_outputs(units=True, out_stream=None))
        assert(output_dict['traj.phases.phase0.timeseries.thrust']['units'] == 'lbf')  # no wildcard, from units dict
        assert(output_dict['traj.phases.phase0.timeseries.m_dot']['units'] == 'kg/s')  # no wildcard, from ODE
        assert(output_dict['traj.phases.phase0.timeseries.f_drag']['units'] == 'N')    # wildcard, from ODE
        assert(output_dict['traj.phases.phase0.timeseries.f_lift']['units'] == 'lbf')  # wildcard, from units dict

    def _test_mach_rate(self, p, plot=False):
        """ Test that the mach rate is provided by the timeseries and is accurate. """
        # Verify correct timeseries output of mach_rate
        output_dict = dict(p.model.list_outputs(units=True, out_stream=None))
        ts = {k: v for k, v in output_dict.items() if 'timeseries.' in k}
        self.assertTrue('traj.phases.phase0.timeseries.mach_rate' in ts)

        time = p['traj.phases.phase0.timeseries.time'][:, 0]
        mach = p['traj.phases.phase0.timeseries.mach'][:, 0]
        mach_rate = p['traj.phases.phase0.timeseries.mach_rate'][:, 0]

        # Fit a numpy polynomial segment by segment to mach vs time, and compare the derivatives to mach_rate
        gd = p.model.traj.phases.phase0.options['transcription'].grid_data
        seg_idxs = gd.subset_segment_indices['all']
        num_seg = seg_idxs.shape[0]

        if plot:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            fig, axes = plt.subplots(2, 1, sharex=True)
            color = iter(cm.viridis(np.linspace(0, 1, num_seg)))
            axes[0].set_ylabel('Mach')
            axes[1].set_ylabel('Mach rate (1/s)')
            axes[1].set_xlabel('time (s)')

        for i in range(num_seg):
            time_seg = time[seg_idxs[i, 0]: seg_idxs[i, 1]]
            mach_seg = mach[seg_idxs[i, 0]: seg_idxs[i, 1]]
            order = len(time_seg) - 1
            p = P.fit(time_seg, mach_seg, order)
            deriv = p.deriv(1)

            time_nodes = time[seg_idxs[i, 0]: seg_idxs[i, 1]]
            mach_nodes = mach[seg_idxs[i, 0]: seg_idxs[i, 1]]
            mach_rate_nodes = mach_rate[seg_idxs[i, 0]: seg_idxs[i, 1]]

            if plot:
                c = next(color)
                t_plot = np.linspace(time_seg[0], time_seg[-1], 20)
                m_plot = p(t_plot)
                m_rate_plot = deriv(t_plot)
                axes[0].plot(t_plot, m_plot, c=c)
                axes[1].plot(t_plot, m_rate_plot, '--', c=c)

                axes[0].plot(time[seg_idxs[i, 0]: seg_idxs[i, 1]], mach[seg_idxs[i, 0]: seg_idxs[i, 1]], 'o', c=c)
                axes[1].plot(time[seg_idxs[i, 0]: seg_idxs[i, 1]], mach_rate[seg_idxs[i, 0]: seg_idxs[i, 1]], 'o', c=c)

            assert_near_equal(mach_nodes, p(time_nodes), tolerance=1.0E-9)
            assert_near_equal(mach_rate_nodes, deriv(time_nodes), tolerance=1.0E-9)

        if plot:
            plt.show()

    @require_pyoptsparse(optimizer='SLSQP')
    def test_results_gauss_lobatto(self):
        NUM_SEG = 12
        ORDER = 3
        p = min_time_climb(optimizer='IPOPT', num_seg=NUM_SEG, transcription_order=ORDER,
                           transcription='gauss-lobatto', add_rate=True)

        self._test_results(p)

        self._test_wilcard_outputs(p)

        self._test_timeseries_units(p)

        self._test_mach_rate(p)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_results_radau(self):
        NUM_SEG = 12
        ORDER = 3
        p = min_time_climb(optimizer='SLSQP', num_seg=NUM_SEG, transcription_order=ORDER,
                           transcription='radau-ps', add_rate=True)

        self._test_results(p)

        self._test_wilcard_outputs(p)

        self._test_timeseries_units(p)

        self._test_mach_rate(p)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
