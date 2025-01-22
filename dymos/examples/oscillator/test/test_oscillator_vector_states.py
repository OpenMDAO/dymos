import unittest
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal

import openmdao.api as om
import dymos as dm
import numpy as np


@use_tempdirs
class TestDocOscillator(unittest.TestCase):

    def test_matrix_param(self):

        from dymos.examples.oscillator.oscillator_ode import OscillatorVectorODE

        radau = dm.Radau(num_segments=2, order=3)
        birkhoff = dm.Birkhoff(num_nodes=7)

        # Instantiate an OpenMDAO Problem instance.
        for tx in (radau, birkhoff):
            for static_params in (True, False):
                with self.subTest(f'{static_params=}, {tx=}'):
                    prob = om.Problem()
                    prob.driver = om.ScipyOptimizeDriver()
                    prob.driver.options["optimizer"] = 'SLSQP'

                    phase = dm.Phase(ode_class=OscillatorVectorODE, transcription=tx,
                                    ode_init_kwargs={'static_params': static_params})

                    phase.set_time_options(fix_initial=True, duration_bounds=(1, 2), duration_ref=1)
                    phase.add_state("x", fix_initial=True, rate_source="x_dot")

                    A_mat = np.array(
                        [
                            [0, 1],
                            [-1, 0]
                        ]
                    )

                    phase.add_parameter("A", val=A_mat, targets=["A"], static_target=static_params)
                    phase.add_objective("time", loc="final", scaler=1)

                    traj = dm.Trajectory()
                    traj.add_phase("phase0", phase)

                    prob.model.add_subsystem("traj", traj)

                    prob.driver.declare_coloring()
                    prob.setup(force_alloc_complex=True)
                    phase.set_state_val('x', vals=[[1, 0], [1, 0]])

                    dm.run_problem(prob, run_driver=True, simulate=True, make_plots=True)
                    t_f = prob.get_val('traj.phase0.timeseries.time')[-1]
                    final_state = prob.get_val('traj.phase0.timeseries.x')[-1, :]
                    assert_near_equal(final_state, np.array([np.cos(t_f), -np.sin(t_f)]).ravel(),
                                    tolerance=1e-5)
