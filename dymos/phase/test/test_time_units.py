import unittest

import openmdao.api as om
import dymos as dm


class ODENoTaggedStateUnits(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', shape=(nn,), desc='velocity', units='m/s')

        self.add_input('g', val=9.80665, desc='grav. acceleration', units='m/s/s', tags=['dymos.static_target'])

        self.add_input('theta', shape=(nn,), desc='angle of wire', units='rad')

        self.add_output('xdot', shape=(nn,), desc='velocity component in x', units='m/s',
                        tags=['dymos.state_rate_source:x'])

        self.add_output('ydot', shape=(nn,), desc='velocity component in y', units='m/s',
                        tags=['dymos.state_rate_source:y', 'dymos.state_units:m'])

        self.add_output('vdot', shape=(nn,), desc='acceleration magnitude', units='m/s**2',
                        tags=['dymos.state_rate_source:v', 'dymos.state_units:m/s'])

    def compute(self, inputs, outputs):
        pass

    def compute_partials(self, inputs, partials):
        pass


class TestTimeUnits(unittest.TestCase):

    def test_time_units_none(self):
        p = om.Problem(model=om.Group())

        transcription = dm.Radau(num_segments=10, order=3, compressed=True)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=ODENoTaggedStateUnits,
                         transcription=transcription)
        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj0', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units=None)

        with self.assertRaises(RuntimeError) as e:
            p.setup(check=True, force_alloc_complex=False)

        expected = "Unable to infer the units of state variable `x` from\nthe rate units because the " \
                   "time units of the phase are set to None.\nChange the time units to something other " \
                   "than None, or explicitly\nset the state units using one of the following options:\n" \
                   "- Tag the state rate source `xdot` with `dymos.state_units:{units}`\n" \
                   "- Use the `set_state_options('x', units={units})` method on the phase."

        self.assertEqual(str(e.exception.__cause__), expected)
