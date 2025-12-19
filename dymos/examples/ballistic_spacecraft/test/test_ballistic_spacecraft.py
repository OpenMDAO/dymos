"""
This example minimizes the launch C3 of a spacecraft from Earth to Mars assuming
ballistic flight.

This tests exists to make sure that we get correct behavior when connecting to
the initial and final states within a phase to "pin" the initial states to those values.

This test also serves as an example of:
- setting custom OpenMDAO units (canonical astrodynamic distance and time units)
- Setting the initial and final state values from some upstream computation (in this case the ephemeris).
- Using Jax components for AD including a few common gotchas.

Performance
-----------

This test is less performant that it could be due to the feedback in the model.
It could be reformulated to be purely feed-forward but that wasn't the intent here.

We're also not providing dymos with an initial guess for the state histories here,
which will result in more work to find the solution.

Jax Gotchas
-----------

In the ballistic spacecraft ODE, want the radius magnitude computed on a per-node basis.
However, the default behavior of jnp.linalg.norm is to compute the norm of the entire
vector, so we need to be sure to specify the axis. We also specify keepdims to maintain
the 2D shape of the resulting vector so that the shapes are compatible when dividing by r_mag.

```
r_mag = jnp.linalg.norm(r, axis=-1, keepdims=True)
```

In the C3 component, which computes the square of the hyperbolic excess velocity (energy),
The C3 is computed as the square of v-infinity.

```
c3 = jnp.dot(v_inf, v_inf)
```

This results in a scalar value of c3, so we must take care to set the shape of c3 to ().

Alternatively, we could have left the shape as (1,) and explicitly shaped c3 in the
compute primal method.
```
c3 = jnp.array(jnp.dot(v_inf, v_inf))
```

"""
import unittest

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None

import openmdao.api as om
import openmdao.utils.units as om_units
from openmdao.utils.general_utils import env_truthy
from openmdao.utils.testing_utils import require_pyoptsparse
import dymos as dm

from dymos.examples.ballistic_spacecraft.ephem_comp import EphemerisComp, KMPAU, MU_SUN
from dymos.examples.ballistic_spacecraft.ballistic_ode_comp import BallisticODEComp

# Add our specific DU and TU to OpenMDAO's recognized units.
om_units.add_unit('DU', f'{KMPAU}*1000*m')
period = 2 * np.pi * np.sqrt(KMPAU ** 3 / MU_SUN)
om_units.add_unit('TU', f'{period}*s')


class C3Comp(om.JaxExplicitComponent):

    def setup(self):
        self.add_input('v0', shape=(3,), units='km/s')
        self.add_input('v_earth', shape=(3,), units='km/s')

        # NOTE: jnp.dot returns a scalar, so we either have to make this
        # shape=() to force it to be a scalar, or make this shape (1,)
        # and then explicitly reshape the c3 output to be of shape (1,).
        self.add_output('c3', shape=(), units='km**2/s**2')

    def compute_primal(self, v0, v_earth):
        v_inf = v0 - v_earth
        c3 = jnp.dot(v_inf, v_inf)
        return c3


class TestBallisticSpacecraft(unittest.TestCase):

    @unittest.skipIf(jax is None, 'requires jax and jaxlib')
    @require_pyoptsparse('IPOPT')
    def test_ballistic_spacecraft(self):
        txs = {'birkhoff': dm.Birkhoff(num_nodes=20)}

        if env_truthy('DYMOS_2'):
            txs['radau'] = dm.Radau(num_segments=5, order=5)

        for tx_name, tx in txs.items():

            with self.subTest(f'{tx_name}'):

                N = tx.grid_data.subset_num_nodes['all']

                p = om.Problem()

                traj = dm.Trajectory()

                phase = traj.add_phase('phase0',
                                        dm.Phase(ode_class=BallisticODEComp,
                                                 transcription=tx,
                                                 ode_init_kwargs={'mu': MU_SUN}),
                                        promotes_inputs=['initial_states:v'])

                phase.set_time_options(units='TU', fix_initial=False, fix_duration=False, duration_bounds=(0.01, None))

                phase.set_state_options('r', rate_source='r_dot', units='DU',
                                        fix_initial=True, fix_final=True)

                phase.set_state_options('v', rate_source='v_dot', units='DU/TU',
                                        fix_initial=False, fix_final=False)

                phase.add_timeseries_output('r_dot', units='km/s')
                phase.add_timeseries_output('v_dot', units='km/s**2')

                p.model.add_subsystem('traj', traj, promotes_inputs=[('initial_states:v', 'v0')])

                p.model.add_subsystem('ephem', EphemerisComp(num_nodes=N))

                p.model.add_subsystem('c3_comp', C3Comp(), promotes_inputs=['v0'], promotes_outputs=['c3'])

                p.model.connect('traj.phase0.timeseries.time', 'ephem.t')

                # Force the inital position to that of Earth
                p.model.connect('ephem.r', 'traj.phase0.initial_states:r', src_indices=om.slicer[0, 0, :])

                # Force the final position to that of Mars
                p.model.connect('ephem.r', 'traj.phase0.final_states:r', src_indices=om.slicer[-1, 1, :])

                # Connect variables necessary to compute departure C3
                p.model.connect('ephem.v', 'c3_comp.v_earth', src_indices=om.slicer[0, 0, :])
                # p.model.connect('traj.phase0.initial_states:v', 'c3_comp.v0')

                # Resolve ambiguties for AutoIVC
                p.model.set_input_defaults('v0', units='km/s', val=np.array([30, 0.01, 0.001]))

                # The phase computes times at the nodes bvased on t_initial and t_duration.
                # These are then fed back to the ephemeris to compute the positions of Earth and Mars at the initial and final times.
                # We need a solver to resolve the residuals created by this feedback.
                # In this case, since the downstream component computes the input for the previous component, NLBGS will do.
                p.model.nonlinear_solver = om.NonlinearBlockGS()

                # Since the model is not feed-forward, the default "run once" linear solver will not work.
                p.model.linear_solver = om.DirectSolver()

                # The phase already manages the time and state design variables for the problem.
                # The objective is computed downstream from the trajectory.
                p.model.add_objective('c3')

                p.driver = om.pyOptSparseDriver(optimizer='IPOPT', print_results=False)
                p.driver.opt_settings['print_level'] = 0
                p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
                p.driver.declare_coloring()

                p.setup()

                # Turn off convergence messages.
                p.set_solver_print(-1)

                p.model.set_val('traj.phase0.t_initial', 0, units='d')
                p.model.set_val('traj.phase0.t_duration', 300.0, units='d')

                result = p.run_driver()

                r = phase.get_val('timeseries.r', units='km')
                r_earth = p.get_val('ephem.r', units='km')[:, 0, :]
                r_mars = p.get_val('ephem.r', units='km')[:, 1, :]

                # Check that there are approximately 180 degrees between the departure and arrival positions
                r_departure = r_earth[0, :]
                r_arrival = r_mars[-1, :]
                angular_distance = np.arccos(np.dot(r_departure, r_arrival) /
                                             (np.linalg.norm(r_departure) * np.linalg.norm(r_arrival)))

                # Assert the optimization was successful
                self.assertTrue(result.success)

                # Assert the result is near a Hohmann transfer
                self.assertTrue(175. < jnp.degrees(angular_distance) < 185., msg=f'{angular_distance}')


if __name__ == '__main__':
    unittest.main()
