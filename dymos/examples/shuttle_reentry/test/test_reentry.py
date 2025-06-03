import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from dymos.examples.shuttle_reentry.shuttle_ode import ShuttleODE


# Expected results from Betts
expected_results = {'constrained': {'time': 2198.67, 'theta': 30.6255},
                    'unconstrained': {'time': 2008.59, 'theta': 34.1412}}


@use_tempdirs
class TestReentry(unittest.TestCase):

    def make_problem(self, constrained=True, transcription=dm.GaussLobatto, optimizer='SLSQP',):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver(optimizer=optimizer)
        p.driver.declare_coloring()

        if optimizer == 'IPOPT':
            p.driver.opt_settings['max_iter'] = 500
            p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
            p.driver.opt_settings['print_level'] = 0
            p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
            p.driver.opt_settings['tol'] = 1.0E-2
            p.driver.opt_settings['constr_viol_tol'] = 1.0E-6
            p.driver.opt_settings['mu_strategy'] = 'monotone'
            p.driver.opt_settings['mu_init'] = 0.01
            p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
        elif optimizer == 'SNOPT':
            p.driver.declare_coloring()
            p.driver.opt_settings['iSumm'] = 6
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-5
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-3

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0',
                                dm.Phase(ode_class=ShuttleODE,
                                         transcription=transcription))

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

        phase0.set_time_val(initial=0, duration=2000, units='s')
        phase0.set_state_val('h', [260000, 80000], units='ft')
        phase0.set_state_val('gamma', [-1 * np.pi / 180, -5 * np.pi / 180], units='rad')
        phase0.set_state_val('phi', [0, 75 * np.pi / 180], units='rad')
        phase0.set_state_val('psi', [90 * np.pi / 180, 10 * np.pi / 180], units='rad')
        phase0.set_state_val('theta', [0, 25 * np.pi / 180], units='rad')
        phase0.set_state_val('v', [25600, 2500], units='ft/s')
        phase0.set_control_val('alpha', [17.4 * np.pi / 180, 17.4 * np.pi / 180], units='rad')
        phase0.set_control_val('beta', [-75 * np.pi / 180, 0 * np.pi / 180], units='rad')

        return p

    @require_pyoptsparse(optimizer='SLSQP')
    def test_partials(self):
        tx = dm.Radau(num_segments=5, order=3)
        p = self.make_problem(constrained=True, transcription=tx, optimizer='SLSQP')
        p.run_model()
        cpd = p.check_partials(method='cs', compact_print=True, out_stream=None)
        assert_check_partials(cpd, atol=1.0E-4, rtol=1.1)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_constrained_radau(self):
        tx = dm.Radau(num_segments=50, order=3)
        p = self.make_problem(constrained=True, transcription=tx, optimizer='IPOPT')

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['constrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.theta', units='deg')[-1],
                          expected_results['constrained']['theta'],
                          tolerance=1e-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_constrained_gauss_lobatto(self):
        tx = dm.GaussLobatto(num_segments=50, order=3)
        p = self.make_problem(constrained=True, transcription=tx, optimizer='IPOPT')

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['constrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.theta', units='deg')[-1],
                          expected_results['constrained']['theta'],
                          tolerance=1e-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_constrained_birkhoff_lgl(self):
        tx = dm.Birkhoff(num_nodes=30, grid_type='lgl')
        p = self.make_problem(constrained=True, transcription=tx, optimizer='IPOPT')

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['constrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.theta', units='deg')[-1],
                          expected_results['constrained']['theta'],
                          tolerance=1e-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_constrained_birkhoff_cgl(self):
        tx = dm.Birkhoff(num_nodes=30, grid_type='cgl')
        p = self.make_problem(constrained=True, transcription=tx, optimizer='IPOPT')

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['constrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.theta', units='deg')[-1],
                          expected_results['constrained']['theta'],
                          tolerance=1e-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_unconstrained_radau(self):
        tx = dm.Radau(num_segments=50, order=3)
        p = self.make_problem(constrained=False, transcription=tx, optimizer='IPOPT')

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['unconstrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.theta', units='deg')[-1],
                          expected_results['unconstrained']['theta'],
                          tolerance=1e-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_unconstrained_gauss_lobatto(self):
        tx = dm.GaussLobatto(num_segments=50, order=3)
        p = self.make_problem(constrained=False, transcription=tx, optimizer='IPOPT')

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['unconstrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.theta', units='deg')[-1],
                          expected_results['unconstrained']['theta'],
                          tolerance=1e-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_unconstrained_birkhoff_lgl(self):
        tx = dm.Birkhoff(num_nodes=60, grid_type='lgl')
        p = self.make_problem(constrained=False, transcription=tx, optimizer='IPOPT')

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['unconstrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.theta', units='deg')[-1],
                          expected_results['unconstrained']['theta'],
                          tolerance=1e-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_unconstrained_birkhoff_cgl(self):
        tx = dm.Birkhoff(num_nodes=60, grid_type='cgl')
        p = self.make_problem(constrained=False, transcription=tx, optimizer='IPOPT')

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['unconstrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.theta', units='deg')[-1],
                          expected_results['unconstrained']['theta'],
                          tolerance=1e-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_reentry_mixed_controls(self):

        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring(tol=1.0E-12)
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['print_level'] = 0
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
        p.driver.opt_settings['mu_strategy'] = 'monotone'
        p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
        p.driver.opt_settings['mu_init'] = 0.01
        p.driver.opt_settings['tol'] = 1.0E-4

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0',
                                dm.Phase(ode_class=ShuttleODE,
                                         transcription=dm.Radau(num_segments=30, order=3)))

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
        phase0.add_control('beta', order=9, units='rad', opt=True,
                           lower=-89 * np.pi / 180, upper=1 * np.pi / 180, control_type='polynomial')

        phase0.add_objective('theta', loc='final', ref=-0.01)
        phase0.add_path_constraint('q', lower=0, upper=70, ref=70)

        p.setup()

        phase0.set_state_val('h', [260000, 80000], units='ft')
        phase0.set_state_val('gamma', [-1 * np.pi / 180, -5 * np.pi / 180], units='rad')
        phase0.set_state_val('phi', [0, 75 * np.pi / 180], units='rad')
        phase0.set_state_val('psi', [90 * np.pi / 180, 10 * np.pi / 180], units='rad')
        phase0.set_state_val('theta', [0, 25 * np.pi / 180], units='rad')
        phase0.set_state_val('v', [25600, 2500], units='ft/s')
        phase0.set_time_val(initial=0, duration=2000, units='s')

        phase0.set_control_val('alpha', [17.4, 17.4], units='deg')
        phase0.set_control_val('beta', [-20.0, 0.0], units='deg')

        dm.run_problem(p, simulate=True)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol = om.CaseReader(sol_db).get_case('final')
        sim = om.CaseReader(sim_db).get_case('final')

        from openmdao.components.interp_util.interp import InterpND

        t_sol = sol.get_val('traj.phase0.timeseries.time')
        beta_sol = sol.get_val('traj.phase0.timeseries.beta', units='deg')

        t_sim = sim.get_val('traj.phase0.timeseries.time')
        beta_sim = sim.get_val('traj.phase0.timeseries.beta', units='deg')

        # need unique (monotonically increasing) times for interpolation
        t_sol_u, t_sol_idx = np.unique(t_sol, return_index=True)
        t_sim_u, t_sim_idx = np.unique(t_sim, return_index=True)

        sol_interp = InterpND('slinear', t_sol_u.ravel(), beta_sol[t_sol_idx].ravel()).interpolate
        sim_interp = InterpND('slinear', t_sim_u.ravel(), beta_sim[t_sim_idx].ravel()).interpolate

        t = np.linspace(0, t_sol.ravel()[-1], 1000)

        assert_near_equal(sim_interp(t), sol_interp(t), tolerance=0.01)

        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1],
                          expected_results['constrained']['time'],
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.theta', units='deg')[-1],
                          expected_results['constrained']['theta'],
                          tolerance=1e-2)


if __name__ == '___main__':
    unittest.main()
