import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm


class ODEComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # z is the state vector, a nn x 2 x 2 in the form of [[x, y], [vx, vy]]
        self.add_input('param', shape=3, units=None)
        self.add_input('z', shape=(nn, 2, 2), units=None)
        self.add_output('zdot', shape=(nn, 2, 2), units=None)

        self.declare_partials(of='zdot', wrt='z', method='cs')
        self.declare_coloring(wrt=['z'], method='cs', num_full_jacs=5, tol=1.0E-12)

    def compute(self, inputs, outputs):
        print('param', inputs['param'])
        outputs['zdot'][:, 0, 0] = inputs['z'][:, 1, 0]
        outputs['zdot'][:, 0, 1] = inputs['z'][:, 1, 1]
        outputs['zdot'][:, 1, 0] = 0.0
        outputs['zdot'][:, 1, 1] = -9.81


def add_parameter_test(testShape=None):
    p = om.Problem()

    tx = dm.Radau(num_segments=10, order=3, solve_segments=False)

    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=ODEComp, transcription=tx)

    if testShape is None:
        phase.add_parameter('param', dynamic=False)
    else:
        phase.add_parameter('param', shape=testShape, dynamic=False)

    traj.add_phase('phase', phase)

    p.model.add_subsystem('traj', traj)

    phase.set_time_options(fix_initial=True, duration_bounds=(1, 5), units=None)
    phase.add_state('z', rate_source='zdot', fix_initial=True, units=None)

    phase.add_boundary_constraint('z', loc='final', lower=0, upper=0, indices=[1])
    phase.add_objective('time', loc='final')

    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring(tol=1.0E-12)

    p.setup()

    p['traj.phase.parameters:param'] = [2.5, 3.5, 4.5]

    p.set_val('traj.phase.t_initial', 0)
    p.set_val('traj.phase.t_duration', 5)
    p.set_val('traj.phase.states:z', phase.interpolate(ys=[[[0, 0], [10, 10]], [[10, 0], [10, -10]]], nodes='state_input'))

    p.run_driver()


@use_tempdirs
class TestParameterTypes(unittest.TestCase):
    def test_tuple(self):
        add_parameter_test((3, ))

    def test_list(self):
        add_parameter_test([3, ])

    def test_scaler(self):
        add_parameter_test(3)

    def test_nothing(self):
        add_parameter_test()
