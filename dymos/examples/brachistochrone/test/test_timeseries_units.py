import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.assert_utils import assert_near_equal
import dymos as dm


@use_tempdirs
class TestTimeseriesUnits(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
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

        def ode(num_nodes):
            return om.ExecComp(['vdot = g * cos(theta)',
                                'xdot = v * sin(theta)',
                                'ydot = -v * cos(theta)'],
                               g={'val': 9.80665, 'units': 'm/s**2'},
                               v={'shape': (num_nodes,), 'units': 'm/s'},
                               theta={'shape': (num_nodes,), 'units': 'rad'},
                               vdot={'shape': (num_nodes,),
                                     'units': 'm/s**2',
                                     'tags': ['dymos.state_rate_source:v']},
                               xdot={'shape': (num_nodes,),
                                     'units': 'degR',
                                     'tags': ['dymos.state_rate_source:x']},
                               ydot={'shape': (num_nodes,),
                                     'units': 'degK',
                                     'tags': ['dymos.state_rate_source:y']},
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

        phase.add_parameter('g', units='m/s**2', static_target=True, include_timeseries=True)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        phase.add_timeseries_output('xdot', units='degF')
        phase.add_timeseries_output('ydot', units='degC')

        phase.timeseries_options['include_state_rates'] = True

        p.setup(check=['unconnected_inputs'], force_alloc_complex=force_alloc_complex)

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])

        phase.set_control_val('theta', [5, 100])
        phase.set_parameter_val('g', 9.80665)

        dm.run_problem(p, run_driver=run_driver, simulate=True)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj0.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol_case = om.CaseReader(sol_db).get_case('final')
        sim_case = om.CaseReader(sim_db).get_case('final')

        assert_near_equal(sol_case.get_val('traj0.phase0.timeseries.xdot', units='degF'),
                          sol_case.get_val('traj0.phase0.timeseries.xdot', units='degF'))

        assert_near_equal(sol_case.get_val('traj0.phase0.timeseries.ydot', units='degC'),
                          sol_case.get_val('traj0.phase0.timeseries.ydot', units='degC'))

        assert_near_equal(sim_case.get_val('traj0.phase0.timeseries.xdot', units='degF'),
                          sim_case.get_val('traj0.phase0.timeseries.xdot', units='degF'))

        assert_near_equal(sim_case.get_val('traj0.phase0.timeseries.ydot', units='degC'),
                          sim_case.get_val('traj0.phase0.timeseries.ydot', units='degC'))

        return p

    def test_ex_brachistochrone_radau_uncompressed(self):
        self._make_problem(transcription='radau-ps', compressed=False)

    def test_ex_brachistochrone_gl_uncompressed(self):
        self._make_problem(transcription='gauss-lobatto', compressed=False)
