from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent
from openmdao.utils.assert_utils import assert_rel_error

from dymos import ExplicitPhase, declare_time, declare_state
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


@declare_time(targets=['t'], units='s')
@declare_state('y', targets=['y'], rate_source='ydot', units='m')
class TestODE(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        self.add_input('t', val=np.ones(self.options['num_nodes']), units='s')
        self.add_input('y', val=np.ones(self.options['num_nodes']), units='m')
        self.add_output('ydot', val=np.ones(self.options['num_nodes']), units='m/s')

        ar = np.arange(self.options['num_nodes'])
        self.declare_partials(of='ydot', wrt='t', rows=ar, cols=ar)
        self.declare_partials(of='ydot', wrt='y', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        t = inputs['t']
        y = inputs['y']
        outputs['ydot'] = y - t ** 2 + 1

    def compute_partials(self, inputs, partials):
        partials['ydot', 't'] = -2 * inputs['t']


class TestExplicitPhase(unittest.TestCase):

    def test_simple_integration(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      ExplicitPhase(num_segments=1, transcription_order=3,
                                                    num_steps=4, ode_class=TestODE))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:y'] = 0.5

        p.run_model()

        np.set_printoptions(linewidth=1024)
        p.check_partials()

        # from openmdao.api import view_model
        # view_model(p.model)

        # t = phase.get_values('time').ravel()
        # y = phase.get_values('y').ravel()
        #
        # p.model.list_outputs(print_arrays=True)
        # p.model.list_inputs(print_arrays=True)
        #
        # print(t)
        # print(y)

        # assert_rel_error(self,
        #                  t,
        #                  [0.0, 0.5, 1.0, 1.5, 2.0],
        #                  tolerance=1.0E-12)
        #
        # assert_rel_error(self,
        #                  y,
        #                  [0.5, 1.425130208333333, 2.639602661132812, 4.006818970044454,
        #                   5.301605229265987],
        #                  tolerance=1.0E-12)

    def test_with_controls(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      ExplicitPhase(num_segments=1, transcription_order=3,
                                                    num_steps=10, ode_class=BrachistochroneODE))

        phase.set_time_options(fix_initial=True, fix_duration=False)
        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)
        phase.add_control('theta', lower=0.0, upper=180.0, units='deg')
        phase.add_design_parameter('g', opt=False, val=9.80665)

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        from time import time
        st = time()
        p.run_model()
        print('time:', time() - st)

        t = phase.get_values('time', nodes='solution').ravel()
        x = phase.get_values('x', nodes='solution').ravel()
        y = phase.get_values('y', nodes='solution').ravel()

        print(phase.grid_data.num_nodes)
        print(phase.grid_data.subset_node_indices['solution'])

        print(t)
        print(y)

        # p.model.list_outputs(print_arrays=True)

        import matplotlib.pyplot as plt
        plt.plot(x, y, 'ro')
        plt.show()

        # assert_rel_error(self,
        #                  t,
        #                  [0.0, 0.5, 1.0, 1.5, 2.0],
        #                  tolerance=1.0E-12)

        # op = p.model.list_outputs(print_arrays=True)
        # print(op)
        # from openmdao.api import view_model
        # view_model(p.model)

        # xs = []
        # ys = []
        #
        # for i in range(phase.grid_data.num_steps_per_segment[0]):
        #     ys.append(p['phase0.segments.seg_0.step_{0}.advance.states:y_f'.format(i)].tolist())
        #     xs.append(p['phase0.segments.seg_0.step_{0}.advance.states:x_f'.format(i)].tolist())
        #
        # import matplotlib.pyplot as plt
        # plt.plot(xs, ys, 'ro')
        # plt.show()

        # assert_rel_error(self,
        #                  y,
        #                  [0.5, 1.425130208333333, 2.639602661132812, 4.006818970044454,
        #                   5.301605229265987],
        #                  tolerance=1.0E-12)


if __name__ == '__main__':

    T = TestExplicitPhase()
    T.test_simple_integration()