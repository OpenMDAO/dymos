from __future__ import print_function, division, absolute_import

import unittest

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

SHOW_PLOTS = True

import numpy as np
import openmdao.api as om
import dymos as dm


@dm.declare_time(units='s')
@dm.declare_state('S', rate_source='Sdot', units='m')
@dm.declare_parameter('v', targets='v', units='m/s')
@dm.declare_parameter('theta', targets='theta', units='rad')
class BrachistochroneArclengthODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')
        self.add_input('theta', val=np.zeros(nn), desc='angle of wire', units='rad')
        self.add_output('Sdot', val=np.zeros(nn), desc='rate of change of arclength', units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='Sdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='Sdot', wrt='theta', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        v = inputs['v']
        outputs['Sdot'] = np.sqrt(1.0 + 1.0/np.tan(theta)) * v * np.sin(theta)

    def compute_partials(self, inputs, jacobian):
        theta = inputs['theta']
        v = inputs['v']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        tan_theta = np.tan(theta)
        cot_theta = 1.0 / tan_theta
        csc_theta = 1.0 / sin_theta

        jacobian['Sdot', 'v'] = sin_theta * np.sqrt(1.0 + 1.0/tan_theta)
        jacobian['Sdot', 'theta'] = v * (2 * cos_theta * (cot_theta + 1) - csc_theta) / (2 * np.sqrt(cot_theta + 1))


def make_brachistochrone_phase(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                               compressed=True):

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=num_segments,
                            order=transcription_order,
                            compressed=compressed)
    elif transcription == 'radau-ps':
        t = dm.Radau(num_segments=num_segments,
                     order=transcription_order,
                     compressed=compressed)
    elif transcription == 'runge-kutta':
        t = dm.RungeKutta(num_segments=num_segments,
                          order=transcription_order,
                          compressed=compressed)

    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)

    return phase


class TestSimultaneousPhases(unittest.TestCase):

    def test_simultaneous_phases_radau(self):

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        #
        # First Phase: Standard Brachistochrone
        #

        transcription = 'radau-ps'
        num_segments = 10
        transcription_order = 3
        compressed = False

        if transcription == 'gauss-lobatto':
            tx0 = dm.GaussLobatto(num_segments=num_segments,
                                  order=transcription_order,
                                  compressed=compressed)
            tx1 = dm.GaussLobatto(num_segments=num_segments, order=transcription_order,
                                  compressed=compressed)

        elif transcription == 'radau-ps':
            tx0 = dm.Radau(num_segments=num_segments,
                           order=transcription_order,
                           compressed=compressed)
            tx1 = dm.Radau(num_segments=num_segments*2,
                           order=transcription_order,
                           compressed=compressed)

        elif transcription == 'runge-kutta':
            tx0 = dm.RungeKutta(num_segments=num_segments,
                                order=transcription_order,
                                compressed=compressed)
            tx1 = dm.RungeKutta(num_segments=num_segments, order=transcription_order,
                                compressed=compressed)

        phase0 = dm.Phase(ode_class=BrachistochroneODE, transcription=tx0)

        p.model.add_subsystem('phase0', phase0)

        phase0.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase0.set_state_options('x', fix_initial=True, fix_final=False, solve_segments=False)
        phase0.set_state_options('y', fix_initial=True, fix_final=False, solve_segments=False)
        phase0.set_state_options('v', fix_initial=True, fix_final=False, solve_segments=False)

        phase0.add_control('theta', continuity=True, rate_continuity=True,
                           units='deg', lower=0.01, upper=179.9)

        phase0.add_input_parameter('g', units='m/s**2', val=9.80665)

        phase0.add_boundary_constraint('x', loc='final', equals=10)
        phase0.add_boundary_constraint('y', loc='final', equals=5)

        # Add alternative timeseries output to provide control inputs for the next phase
        phase0.add_timeseries('timeseries2', transcription=tx1, subset='control_input')

        #
        # Second Phase: Integration of ArcLength
        #

        phase1 = dm.Phase(ode_class=BrachistochroneArclengthODE, transcription=tx1)

        p.model.add_subsystem('phase1', phase1)

        phase1.set_time_options(fix_initial=True, input_duration=True)

        phase1.set_state_options('S', fix_initial=True, fix_final=False)

        phase1.add_control('theta', opt=False, units='deg')
        phase1.add_control('v', opt=False, units='m/s')

        #
        # Connect the two phases
        #
        p.model.connect('phase0.t_duration', 'phase1.t_duration')

        p.model.connect('phase0.timeseries2.controls:theta', 'phase1.controls:theta')
        p.model.connect('phase0.timeseries2.states:v', 'phase1.controls:v')

        # Minimize time at the end of the phase
        # phase1.add_objective('S', loc='final', scaler=10)
        phase1.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase0.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase0.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase0.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase0.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.input_parameters:g'] = 9.80665

        p['phase1.states:S'] = 0.0

        p.run_driver()

        # print(phase0.options['transcription'].grid_data.subset_node_indices['control_input'])

        import matplotlib.pyplot as plt
        plt.plot(p['phase1.timeseries.time'], p['phase1.timeseries.states:S'])
        plt.show()

if __name__ == '__main__':
    unittest.main()
