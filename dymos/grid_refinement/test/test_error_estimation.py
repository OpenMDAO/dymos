import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

from openmdao.utils.general_utils import set_pyoptsparse_opt

from dymos.grid_refinement.error_estimation import eval_ode_on_grid, compute_state_quadratures

OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP', fallback=True)


class TestBrachistochroneExample(unittest.TestCase):

    def _run_brachistochrone(self, transcription_class=dm.Radau, control_type='control', g=9.80665):
        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER
        p.driver.declare_coloring(tol=1.0E-12)

        tx = transcription_class(num_segments=15,
                                 order=3,
                                 compressed=False)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        # p.model.add_subsystem('traj0', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        units=BrachistochroneODE.states['x']['units'],
                        fix_initial=True, fix_final=True, solve_segments=False)
        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        units=BrachistochroneODE.states['y']['units'],
                        fix_initial=True, fix_final=True, solve_segments=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        units=BrachistochroneODE.states['v']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        if control_type == 'control':
            phase.add_control('theta', continuity=True, rate_continuity=True,
                              units='deg', lower=0.01, upper=179.9)
        elif control_type == 'polynomial_control':
            phase.add_polynomial_control('theta', units='deg', lower=0.01, upper=179.9, order=3)

        phase.add_parameter('g', units='m/s**2', val=1.0)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.setup()

        p['traj0.phase0.t_initial'] = 0.0
        p['traj0.phase0.t_duration'] = 2.0

        p['traj0.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['traj0.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['traj0.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        if control_type == 'control':
            p['traj0.phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        else:
            p['traj0.phase0.polynomial_controls:theta'] = [5, 15, 80, 100]
        p['traj0.phase0.parameters:g'] = g

        return p

    def test_eval_ode_on_grid(self):
        """ Tests that the eval_ode_on_grid method works correctly against a variety of problems. """

        for tx_class in (dm.Radau, dm.GaussLobatto):
            for control_type in ('control', 'polynomial_control'):
                for g in (9.80665, 1.62):
                    with self.subTest(msg=f'{tx_class.__name__} - {control_type} - g = {g}',):
                        p = self._run_brachistochrone(transcription_class=tx_class, control_type=control_type, g=g)
                        p.run_driver()

                        phase = p.model.traj0.phases.phase0

                        x, f = eval_ode_on_grid(phase, phase.options['transcription'])

                        # Check that the computed state rates in the grid refinement ODE are equal to those
                        # in the original problem. (on the same grid they should be within machine precision)
                        for name, options in phase.state_options.items():
                            phase_rate_path = f'timeseries.state_rates:{name}'
                            x_solution = phase.get_val(f'timeseries.states:{name}')
                            f_solution = phase.get_val(phase_rate_path)
                            assert_near_equal(x[name].ravel(), x_solution.ravel())
                            assert_near_equal(f[name].ravel(), f_solution.ravel())

    def test_compute_state_quadratures(self):
        """ Tests that the eval_ode_on_grid method works correctly against a variety of problems. """

        for tx_class in (dm.Radau, dm.GaussLobatto):
            for control_type in ('control', 'polynomial_control'):
                for g in (9.80665, 1.62):
                    with self.subTest(msg=f'{tx_class.__name__} - {control_type} - g = {g}',):
                        p = self._run_brachistochrone(transcription_class=tx_class,
                                                      control_type=control_type,
                                                      g=g)
                        p.run_driver()

                        phase = p.model.traj0.phases.phase0

                        x, f = eval_ode_on_grid(phase, phase.options['transcription'])

                        t_duration = phase.get_val('t_duration')

                        x_hat = compute_state_quadratures(x, f, t_duration, phase.options['transcription'])

                        assert_near_equal(x_hat['x'], x['x'], tolerance=1.0E-4)
                        assert_near_equal(x_hat['y'], x['y'], tolerance=1.0E-4)
                        assert_near_equal(x_hat['v'], x['v'], tolerance=1.0E-4)
