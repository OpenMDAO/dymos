import unittest

import openmdao.api as om
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.utils.testing_utils import require_pyoptsparse


@use_tempdirs
class TestPhaseParameterPromotion(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    def test_promotes_parameter(self):
        transcription = 'radau-ps'
        optimizer = 'SLSQP'
        num_segments = 10
        transcription_order = 3
        compressed = False

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        OPT, OPTIMIZER = set_pyoptsparse_opt(optimizer, fallback=True)
        p.driver.options['optimizer'] = OPTIMIZER
        p.driver.declare_coloring()

        if transcription == 'gauss-lobatto':
            t = dm.GaussLobatto(num_segments=num_segments,
                                order=transcription_order,
                                compressed=compressed)
        elif transcription == 'radau-ps':
            t = dm.Radau(num_segments=num_segments,
                         order=transcription_order,
                         compressed=compressed)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)

        traj.add_phase('phase0', phase, promotes_inputs=['t_initial', 't_duration', 'parameters:g'])

        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

        phase.add_state('x', fix_initial=True, fix_final=False, solve_segments=False,
                        units='m', rate_source='xdot')
        phase.add_state('y', fix_initial=True, fix_final=False, solve_segments=False,
                        units='m', rate_source='ydot')
        phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=False,
                        units='m/s', rate_source='vdot', targets=['v'])

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9, targets=['theta'])

        phase.add_parameter('g', units='m/s**2', val=9.80665, targets=['g'])

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p.set_val('traj.t_initial', 0.0)
        p.set_val('traj.t_duration', 2.0)

        p.set_val('traj.phase0.states:x', phase.interp('x', ys=[0, 10]))
        p.set_val('traj.phase0.states:y', phase.interp('y', ys=[10, 5]))
        p.set_val('traj.phase0.states:v', phase.interp('v', ys=[0, 9.9]))
        p.set_val('traj.phase0.controls:theta', phase.interp('theta', ys=[5, 100]))
        p.set_val('traj.parameters:g', 9.80665)

        p.run_driver()

        assert_near_equal(p['traj.t_duration'], 1.8016, tolerance=1.0E-4)


if __name__ == '__main__':
    unittest.main()
