import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


@use_tempdirs
class TestPicardShootingSharedTranscription(unittest.TestCase):

    def test_one_transcription_two_phases(self):
        g = 9.80665
        phase_data = {'phase1': (0.0, 1.0, 30.0),
                      'phase2': (1.0, 0.8, 60.0)}

        tx = dm.PicardShooting(num_segments=2, nodes_per_seg=5)
        tx.options['ode_nonlinear_solver'] = om.NonlinearBlockGS(maxiter=150, use_aitken=True, iprint=0)

        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())
        for name in phase_data:
            phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
            phase.set_time_options(fix_initial=True, fix_duration=True)
            phase.add_state('x', fix_initial=True)
            phase.add_state('y', fix_initial=True)
            phase.add_state('v', fix_initial=True)
            phase.add_control('theta', units='deg', rate_continuity=False)
            phase.add_parameter('g', units='m/s**2', opt=False, val=g)
            traj.add_phase(name, phase)

        p.setup()

        for name, (t0, duration, theta) in phase_data.items():
            phase = traj.phases._get_subsystem(name)
            phase.set_time_val(initial=t0, duration=duration)
            phase.set_state_val('x', [0.0, 0.0])
            phase.set_state_val('y', [10.0, 10.0])
            phase.set_state_val('v', [0.0, 0.0])
            phase.set_control_val('theta', [theta, theta])

        p.run_model()

        solvers = []
        for name, (t0, duration, theta) in phase_data.items():
            # Each phase has its own copy of the solvers, with the options set on the transcription.
            ode_iter_group = traj.phases._get_subsystem(name).ode_iter_group
            nl_solver = ode_iter_group.segment_prop_group.nonlinear_solver
            self.assertEqual(nl_solver.options['maxiter'], 150)
            solvers.extend([nl_solver, ode_iter_group.segment_prop_group.linear_solver,
                            ode_iter_group.nonlinear_solver, ode_iter_group.linear_solver])

            # With a constant angle and a start from rest, the states have closed forms.
            th = np.radians(theta)
            expected = {'x': 0.5 * g * np.cos(th) * np.sin(th) * duration ** 2,
                        'y': 10.0 - 0.5 * g * np.cos(th) ** 2 * duration ** 2,
                        'v': g * np.cos(th) * duration}
            for state, value in expected.items():
                assert_near_equal(p.get_val(f'traj.{name}.timeseries.{state}')[-1, 0], value,
                                  tolerance=1.0E-9)

        self.assertEqual(len({id(solver) for solver in solvers}), len(solvers))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
