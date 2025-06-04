import os
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos.examples.double_integrator.breakwell_ode import BreakwellODE


@require_pyoptsparse(optimizer='IPOPT')
def double_integrator_direct_collocation(transcription='gauss-lobatto', compressed=True, optimizer='IPOPT'):

    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()
    p.driver.options['optimizer'] = optimizer

    if optimizer == 'IPOPT':
        p.driver.opt_settings['max_iter'] = 5000
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['print_level'] = 0
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
        p.driver.opt_settings['tol'] = 1.0E-3
        p.driver.opt_settings['constr_viol_tol'] = 1.0E-6
    elif optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-3
        p.driver.opt_settings['iSumm'] = 6

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=30, order=3, compressed=compressed)
    elif transcription == "radau-ps":
        t = dm.Radau(num_segments=30, order=3, compressed=compressed)
    elif transcription == 'birkhoff':
        t = dm.Birkhoff(num_nodes=51)
    elif transcription.startswith('picard'):
        grid_type = transcription.split('-')[-1]
        t = dm.PicardShooting(num_segments=1, nodes_per_seg=50, grid_type=grid_type)
    else:
        raise ValueError('invalid transcription')

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    phase = traj.add_phase('phase0', dm.Phase(ode_class=BreakwellODE, transcription=t))

    phase.set_time_options(fix_initial=True, fix_duration=True, units='s')

    phase.add_state('v', fix_initial=True, rate_source='u', units='m/s')
    phase.add_state('x', fix_initial=True, rate_source='v', units='m', shape=(1, ))
    phase.add_state('J', fix_initial=True, rate_source='J_dot')

    phase.add_control('u', units='m/s**2', scaler=0.1, continuity=False, rate_continuity=False,
                      rate2_continuity=False, shape=(1, ))

    # Maximize distance travelled in one second.
    phase.add_objective('J', loc='final')

    phase.add_path_constraint('x', upper=0.1)
    phase.add_boundary_constraint('x', loc='final', equals=0.0)
    phase.add_boundary_constraint('v', loc='final', equals=-1.0)

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True, force_alloc_complex=True)

    phase.set_time_val(initial=0.0, duration=1.0)
    phase.set_state_val('x', [0, 0])
    phase.set_state_val('v', [1, 0])
    phase.set_state_val('J', [0, 4])
    phase.set_control_val('u', [-5, -5])

    return p


@use_tempdirs
class TestBreakwellExample(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    @staticmethod
    def _assert_results(p, tol=1.0E-4):
        x = p.get_val('traj.phase0.timeseries.x')
        v = p.get_val('traj.phase0.timeseries.v')
        J = p.get_val('traj.phase0.timeseries.J')
        J_expected = 2*v[0]**3/(9*(0.1 - x[0])) - 2*v[-1]**3/(9*(0.1-x[-1]))

        assert_near_equal(x[-1], 0.0, tolerance=tol)
        assert_near_equal(v[-1], -1.0, tolerance=tol)
        assert_near_equal(J[-1], J_expected, tolerance=tol)

    def test_ex_double_integrator_gl_compressed(self):
        p = double_integrator_direct_collocation('gauss-lobatto',
                                                 compressed=True)
        dm.run_problem(p)
        self._assert_results(p)

    def test_ex_double_integrator_gl_uncompressed(self):
        p = double_integrator_direct_collocation('gauss-lobatto',
                                                 compressed=False)
        dm.run_problem(p)
        self._assert_results(p)

    def test_ex_double_integrator_radau_compressed(self):
        p = double_integrator_direct_collocation('radau-ps',
                                                 compressed=True, optimizer='IPOPT')
        dm.run_problem(p)
        self._assert_results(p)

    def test_ex_double_integrator_radau_uncompressed(self):
        p = double_integrator_direct_collocation('radau-ps',
                                                 compressed=False)
        dm.run_problem(p)
        self._assert_results(p)

    def test_ex_double_integrator_birkhoff(self):
        p = double_integrator_direct_collocation('birkhoff', optimizer='IPOPT')
        dm.run_problem(p)
        self._assert_results(p)

    def test_ex_double_integrator_picard_shooting(self):
        p = double_integrator_direct_collocation('picard-cgl', optimizer='IPOPT')
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=True)
        self._assert_results(p)

    def test_check_partials(self):
        p = double_integrator_direct_collocation('birkhoff', optimizer='SLSQP')
        p.run_model()
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)


if __name__ == "__main__":
    unittest.main()
