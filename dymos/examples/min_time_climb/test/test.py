# LIBRARY IMPORTS
from datetime import datetime
import matplotlib

matplotlib.use('agg')  # used for plotting
import os
import numpy as np

import matplotlib.pyplot as plt
import openmdao.api as om
import dymos as dm


class EVA_ODE(om.ExplicitComponent):
    # describes how a spacewalker translates in 1D based on jetpack thrust

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('m', val=np.ones(nn), units='kg')  # mass
        self.add_input('T', val=np.ones(nn), units='N')  # jetpack thrust

        self.add_output('a', val=np.ones(nn), units='m/s**2')  # resulting acceleration

        arange = np.arange(nn)
        self.declare_partials('a', ['m', 'T'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        m = inputs['m']
        T = inputs['T']

        outputs['a'] = T * m ** (-1)

    def compute_partials(self, inputs, J):
        m = inputs['m']
        T = inputs['T']

        J['a', 'm'] = - T * m ** (-2)
        J['a', 'T'] = m ** (-1)


p = om.Problem(model=om.Group())

p.driver = om.pyOptSparseDriver()

# =======================================================================================================================
# PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM
# =======================================================================================================================
# This works:
IVC = p.model.add_subsystem('IVC', om.IndepVarComp(), promotes=['*'])
IVC.add_output('m', val=500.0, units='kg')  # astronaut mass

# This is broken:
# p.model.set_input_defaults('m', val=100.0, units='kg')  # astronaut mass
# =======================================================================================================================


hop0 = dm.Trajectory()
p.model.add_subsystem('hop0', hop0)

main_phase = hop0.add_phase('main_phase',
                            dm.Phase(ode_class=EVA_ODE,
                                     transcription=dm.Radau(num_segments=1, order=7)))

main_phase.set_time_options(fix_initial=True, fix_duration=False, units='s',
                            duration_bounds=(1.0, 500))

main_phase.add_state('x', fix_initial=True, fix_final=True, units='m',
                     rate_source='v')  # position

main_phase.add_state('v', fix_initial=True, fix_final=True, units='m/s',
                     rate_source='a')  # velocity

main_phase.add_control('T', units='N', targets='T', lower=-3, upper=3)  # jetpack thrust

main_phase.add_objective('time', loc='final')  #

p.setup(mode='auto', check=['unconnected_inputs'], force_alloc_complex=True)

p['hop0.main_phase.t_initial'] = 0.0
p['hop0.main_phase.t_duration'] = 20.0

p['hop0.main_phase.states:x'] = main_phase.interpolate(ys=[0, 100], nodes='state_input')  # position
p['hop0.main_phase.states:v'] = main_phase.interpolate(ys=[0, 0], nodes='state_input')  # velocity

p['hop0.main_phase.controls:T'] = main_phase.interpolate(ys=[2, -2],
                                                         nodes='control_input')  # thrust

p.final_setup()

p.run_driver()

om.n2(p.model)

print('Where is my Astronaut?...')

print('\n Position (m): ==============================')
print(p.get_val('hop0.main_phase.states:x'))

print('\n Velocity (m/s): ==============================')
print(p.get_val('hop0.main_phase.states:v'))

print('Thrust (N): ==============================')
print(p.get_val('hop0.main_phase.controls:T'))

print('Time (s): ================================')
print(p.get_val('hop0.main_phase.timeseries.time'))