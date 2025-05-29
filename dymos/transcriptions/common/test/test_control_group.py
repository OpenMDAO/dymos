
import unittest

import openmdao.api as om

import dymos as dm


class MyODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('polycon', shape=(nn,))
        self.add_input('con', shape=(nn,))
        self.add_input('state', shape=(nn,))
        self.add_output('out', shape=(nn,))
        self.add_output('state_deriv', shape=(nn,))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pass


class TestControlInterpComp(unittest.TestCase):

    def test_poly_con_bug(self):
        prob = om.Problem()

        phase = dm.Phase(ode_class=MyODE,
                         transcription=dm.GaussLobatto(num_segments=3))

        phase.add_state('state', rate_source='state_deriv')

        phase.add_control('polycon', order=1, val=0.0, opt=True, control_type='polynomial')
        phase.add_control('con', val=0.0, opt=True)

        phase.add_objective('out', loc='final', ref=1000.0)

        traj = dm.Trajectory()
        prob.model.add_subsystem('traj', traj)
        traj.add_phase('phase', phase)

        # Make sure we raise no exceptions.
        prob.setup()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
