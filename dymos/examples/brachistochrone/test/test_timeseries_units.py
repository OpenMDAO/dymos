import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal
import dymos as dm


@use_tempdirs
class TestTimeseriesUnits(unittest.TestCase):

    def _make_problem(self, transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                      compressed=True, optimizer='SLSQP', run_driver=True, force_alloc_complex=False,
                      solve_segments=False):

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.declare_coloring(tol=1.0E-12)

        if transcription == 'gauss-lobatto':
            t = dm.GaussLobatto(num_segments=num_segments,
                                order=transcription_order,
                                compressed=compressed)
        elif transcription == 'radau-ps':
            t = dm.Radau(num_segments=num_segments,
                         order=transcription_order,
                         compressed=compressed)

        ode = lambda num_nodes: om.ExecComp(['vdot = g * cos(theta)',
                                             'xdot = v * sin(theta)',
                                             'ydot = -v * cos(theta)'],
                                            g={'value': 9.80665, 'units': 'm/s**2'},
                                            v={'shape': (num_nodes,), 'units': 'm/s'},
                                            theta={'shape': (num_nodes,), 'units': 'rad'},
                                            vdot={'shape': (num_nodes,),
                                                  'units': 'm/s**2',
                                                  'tags': ['state_rate_source:v']},
                                            xdot={'shape': (num_nodes,),
                                                  'units': 'degR',
                                                  'tags': ['state_rate_source:x']},
                                            ydot={'shape': (num_nodes,),
                                                  'units': 'degK',
                                                  'tags': ['state_rate_source:y']},
                                            has_diag_partials=True)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=ode, transcription=t)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False, solve_segments=solve_segments)
        phase.add_state('y', fix_initial=True, fix_final=False, solve_segments=solve_segments)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=solve_segments)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', static_target=True)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        phase.add_timeseries_output('xdot', units='degF')
        phase.add_timeseries_output('ydot', units='degC')

        p.setup(check=['unconnected_inputs'], force_alloc_complex=force_alloc_complex)

        p['traj0.phase0.t_initial'] = 0.0
        p['traj0.phase0.t_duration'] = 2.0

        p['traj0.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj0.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj0.phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['traj0.phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['traj0.phase0.parameters:g'] = 9.80665

        dm.run_problem(p, run_driver=run_driver, simulate=True)

        sol_case = om.CaseReader('dymos_solution.db').get_case('final')
        sim_case = om.CaseReader('dymos_simulation.db').get_case('final')

        assert_near_equal(sol_case.get_val('traj0.phase0.timeseries.xdot', units='degF'),
                          sol_case.get_val('traj0.phase0.timeseries.state_rates:x', units='degF'))

        assert_near_equal(sol_case.get_val('traj0.phase0.timeseries.ydot', units='degC'),
                          sol_case.get_val('traj0.phase0.timeseries.state_rates:y', units='degC'))

        assert_near_equal(sim_case.get_val('traj0.phase0.timeseries.xdot', units='degF'),
                          sim_case.get_val('traj0.phase0.timeseries.state_rates:x', units='degF'))

        assert_near_equal(sim_case.get_val('traj0.phase0.timeseries.ydot', units='degC'),
                          sim_case.get_val('traj0.phase0.timeseries.state_rates:y', units='degC'))

        return p

    def test_ex_brachistochrone_radau_uncompressed(self):
        self._make_problem(transcription='radau-ps', compressed=False)

    def test_ex_brachistochrone_gl_uncompressed(self):
        self._make_problem(transcription='gauss-lobatto', compressed=False)
