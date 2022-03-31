import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from dymos import Trajectory, GaussLobatto, Phase, Radau, run_problem
from dymos.examples.shuttle_reentry.shuttle_ode import ShuttleODE

# Expected results from Betts
expected_results = {'constrained': {'time': 2198.67, 'theta': 30.6255},
                    'unconstrained': {'time': 2008.59, 'theta': 34.1412}}


@use_tempdirs
class TestReentry(unittest.TestCase):

    def make_problem(self, constrained=True, transcription=GaussLobatto, optimizer='SLSQP',
                     numseg=50):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver(optimizer=optimizer)
        p.driver.declare_coloring()

        if optimizer == 'IPOPT':
            p.driver.opt_settings['max_iter'] = 500
            p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
            p.driver.opt_settings['print_level'] = 5
            p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
            p.driver.opt_settings['tol'] = 1.0E-7
            p.driver.opt_settings['mu_strategy'] = 'monotone'

        traj = p.model.add_subsystem('traj', Trajectory())
        phase0 = traj.add_phase('phase0',
                                Phase(ode_class=ShuttleODE,
                                      transcription=transcription(num_segments=numseg, order=3)))

        phase0.set_time_options(fix_initial=True, units='s', duration_ref=200)
        phase0.add_state('h', fix_initial=True, fix_final=True, units='ft', rate_source='hdot',
                         targets=['h'], lower=0, ref0=75000, ref=300000, defect_ref=1000)
        phase0.add_state('gamma', fix_initial=True, fix_final=True, units='rad',
                         rate_source='gammadot', targets=['gamma'],
                         lower=-89. * np.pi / 180, upper=89. * np.pi / 180)
        phase0.add_state('phi', fix_initial=True, fix_final=False, units='rad',
                         rate_source='phidot', lower=0, upper=89. * np.pi / 180)
        phase0.add_state('psi', fix_initial=True, fix_final=False, units='rad',
                         rate_source='psidot', targets=['psi'], lower=0, upper=90. * np.pi / 180)
        phase0.add_state('theta', fix_initial=True, fix_final=False, units='rad',
                         rate_source='thetadot', targets=['theta'],
                         lower=-89. * np.pi / 180, upper=89. * np.pi / 180)
        phase0.add_state('v', fix_initial=True, fix_final=True, units='ft/s',
                         rate_source='vdot', targets=['v'], lower=500, ref0=2500, ref=25000)
        phase0.add_control('alpha', units='rad', opt=True,
                           lower=-np.pi / 2, upper=np.pi / 2, targets=['alpha'])
        phase0.add_control('beta', units='rad', opt=True,
                           lower=-89 * np.pi / 180, upper=1 * np.pi / 180, targets=['beta'])

        if constrained:
            phase0.add_path_constraint('q', lower=0, upper=70, ref=70)

        phase0.add_objective('theta', loc='final', ref=-0.01)

        p.setup(check=True, force_alloc_complex=True)

        p.set_val('traj.phase0.states:h',
                  phase0.interp('h', [260000, 80000]), units='ft')
        p.set_val('traj.phase0.states:gamma',
                  phase0.interp('gamma', [-1 * np.pi / 180, -5 * np.pi / 180]),
                  units='rad')
        p.set_val('traj.phase0.states:phi',
                  phase0.interp('phi', [0, 75 * np.pi / 180]), units='rad')
        p.set_val('traj.phase0.states:psi',
                  phase0.interp('psi', [90 * np.pi / 180, 10 * np.pi / 180]),
                  units='rad')
        p.set_val('traj.phase0.states:theta',
                  phase0.interp('theta', [0, 25 * np.pi / 180]), units='rad')
        p.set_val('traj.phase0.states:v', phase0.interp('v', [25600, 2500]), units='ft/s')
        p.set_val('traj.phase0.t_initial', 0, units='s')
        p.set_val('traj.phase0.t_duration', 2000, units='s')
        p.set_val('traj.phase0.controls:alpha',
                  phase0.interp('alpha', [17.4 * np.pi / 180, 17.4 * np.pi / 180]), units='rad')
        p.set_val('traj.phase0.controls:beta',
                  phase0.interp('beta', [-75 * np.pi / 180, 0 * np.pi / 180]), units='rad')

        return p

    @require_pyoptsparse(optimizer='SLSQP')
    def test_partials(self):
        p = self.make_problem(constrained=True, transcription=Radau, optimizer='SLSQP', numseg=5)
        p.run_model()
        cpd = p.check_partials(method='cs', compact_print=True, out_stream=None)
        assert_check_partials(cpd, atol=1.0E-4, rtol=1.1)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_constrained_radau(self):
        p = self.make_problem(constrained=True, transcription=Radau, optimizer='IPOPT')

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['constrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.states:theta', units='deg')[-1],
                          expected_results['constrained']['theta'],
                          tolerance=1e-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_constrained_gauss_lobatto(self):
        p = self.make_problem(constrained=True, transcription=GaussLobatto, optimizer='IPOPT')

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['constrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.states:theta', units='deg')[-1],
                          expected_results['constrained']['theta'],
                          tolerance=1e-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_unconstrained_radau(self):
        p = self.make_problem(constrained=False, transcription=Radau, optimizer='IPOPT')

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['unconstrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.states:theta', units='deg')[-1],
                          expected_results['unconstrained']['theta'],
                          tolerance=1e-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_unconstrained_gauss_lobatto(self):
        p = self.make_problem(constrained=False, transcription=GaussLobatto, optimizer='IPOPT')

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['unconstrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.states:theta', units='deg')[-1],
                          expected_results['unconstrained']['theta'],
                          tolerance=1e-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_mixed_controls(self):

        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring(tol=1.0E-12)
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['print_level'] = 5
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
        p.driver.opt_settings['mu_strategy'] = 'monotone'

        traj = p.model.add_subsystem('traj', Trajectory())
        phase0 = traj.add_phase('phase0',
                                Phase(ode_class=ShuttleODE,
                                      transcription=Radau(num_segments=30, order=3)))

        phase0.set_time_options(fix_initial=True, units='s', duration_ref=200)
        phase0.add_state('h', fix_initial=True, fix_final=True, units='ft', rate_source='hdot',
                         targets=['h'], lower=0, ref0=75000, ref=300000, defect_ref=1000)
        phase0.add_state('gamma', fix_initial=True, fix_final=True, units='rad',
                         rate_source='gammadot', targets=['gamma'],
                         lower=-89. * np.pi / 180, upper=89. * np.pi / 180)
        phase0.add_state('phi', fix_initial=True, fix_final=False, units='rad',
                         rate_source='phidot', lower=0, upper=89. * np.pi / 180)
        phase0.add_state('psi', fix_initial=True, fix_final=False, units='rad',
                         rate_source='psidot', targets=['psi'], lower=0,
                         upper=90. * np.pi / 180)
        phase0.add_state('theta', fix_initial=True, fix_final=False, units='rad',
                         rate_source='thetadot', targets=['theta'],
                         lower=-89. * np.pi / 180, upper=89. * np.pi / 180)
        phase0.add_state('v', fix_initial=True, fix_final=True, units='ft/s',
                         rate_source='vdot', targets=['v'], lower=500, ref0=2500, ref=25000)
        phase0.add_control('alpha', units='rad', opt=True,
                           lower=-np.pi / 2, upper=np.pi / 2)
        phase0.add_polynomial_control('beta', order=9, units='rad', opt=True,
                                      lower=-89 * np.pi / 180, upper=1 * np.pi / 180)

        phase0.add_objective('theta', loc='final', ref=-0.01)
        phase0.add_path_constraint('q', lower=0, upper=70, ref=70)

        p.setup(check=True, force_alloc_complex=True)

        p.set_val('traj.phase0.states:h',
                  phase0.interp('h', [260000, 80000]), units='ft')
        p.set_val('traj.phase0.states:gamma',
                  phase0.interp('gamma', [-1 * np.pi / 180, -5 * np.pi / 180]),
                  units='rad')
        p.set_val('traj.phase0.states:phi',
                  phase0.interp('phi', [0, 75 * np.pi / 180]),
                  units='rad')
        p.set_val('traj.phase0.states:psi',
                  phase0.interp('psi', [90 * np.pi / 180, 10 * np.pi / 180]),
                  units='rad')
        p.set_val('traj.phase0.states:theta',
                  phase0.interp('theta', [0, 25 * np.pi / 180]),
                  units='rad')
        p.set_val('traj.phase0.states:v',
                  phase0.interp('v', [25600, 2500]),
                  units='ft/s')
        p.set_val('traj.phase0.t_initial', 0, units='s')
        p.set_val('traj.phase0.t_duration', 2000, units='s')
        p.set_val('traj.phase0.controls:alpha',
                  phase0.interp('alpha', [17.4, 17.4]), units='deg')
        p.set_val('traj.phase0.polynomial_controls:beta',
                  phase0.interp('beta', [-20, 0]), units='deg')

        run_problem(p, simulate=True)

        sol = om.CaseReader('dymos_solution.db').get_case('final')
        sim = om.CaseReader('dymos_simulation.db').get_case('final')

        from scipy.interpolate import interp1d

        t_sol = sol.get_val('traj.phase0.timeseries.time')
        beta_sol = sol.get_val('traj.phase0.timeseries.polynomial_controls:beta', units='deg')

        t_sim = sim.get_val('traj.phase0.timeseries.time')
        beta_sim = sim.get_val('traj.phase0.timeseries.polynomial_controls:beta', units='deg')

        sol_interp = interp1d(t_sol.ravel(), beta_sol.ravel())
        sim_interp = interp1d(t_sim.ravel(), beta_sim.ravel())

        t = np.linspace(0, t_sol.ravel()[-1], 1000)

        assert_near_equal(sim_interp(t), sol_interp(t), tolerance=0.01)

        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['constrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.states:theta', units='deg')[-1],
                          expected_results['constrained']['theta'],
                          tolerance=1e-2)


if __name__ == '___main__':
    unittest.main()
