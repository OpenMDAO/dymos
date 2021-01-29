import unittest

import matplotlib
import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg')
plt.style.use('ggplot')

from dymos.utils.doc_utils import save_for_docs


# class TestBalancedFieldLengthForDocs(unittest.TestCase):
#
#     # @save_for_docs
#     def test_balanced_field_length_for_docs(self):
import matplotlib.pyplot as plt

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import dymos as dm
from dymos.examples.balanced_field.ground_roll_ode_comp import GroundRollODEComp
from dymos.examples.balanced_field.takeoff_climb_ode_comp import TakeoffClimbODEComp

#
# Instantiate the problem and configure the optimization driver
#
p = om.Problem(model=om.Group())

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.opt_settings['iSumm'] = 6
# p.driver.opt_settings['Verify level'] = 3
# p.driver.opt_settings['Major step limit'] = 0.1
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-5
p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
p.driver.opt_settings['Major iterations limit'] = 500
p.driver.opt_settings['Minor iterations limit'] = 100000
# p.driver.opt_settings['Linesearch tolerance'] = 0.5


# p.driver.options['optimizer'] = 'IPOPT'
# p.driver.opt_settings['print_level'] = 5
# p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'

p.driver.declare_coloring()

#
# Instantiate the trajectory and phase
#
traj = dm.Trajectory()

p.model.add_subsystem('traj', traj)

#
# First Phase - Brake Release to V1
# Operating with two engines
# We don't know what V1 is a priori, we're going to use this model to determine it.
#

# Break Release to V1
br_to_v1_phase = dm.Phase(ode_class=GroundRollODEComp, transcription=dm.Radau(num_segments=3))

br_to_v1_phase.set_time_options(initial_val=0.0, duration_val=20.0, fix_initial=True, duration_bounds=(0.1, 100))

br_to_v1_phase.set_state_options('r', fix_initial=True, val=br_to_v1_phase.interpolate(ys=[0, 2500.0], nodes='state_input'))

br_to_v1_phase.set_state_options('v', fix_initial=True, val=br_to_v1_phase.interpolate(ys=[0.0001, 100.0], nodes='state_input'))

br_to_v1_phase.add_parameter('h', val=0.0, units='ft', opt=False, dynamic=False)
br_to_v1_phase.add_parameter('T', val=2. * 27000, units='lbf', opt=False, dynamic=False)
br_to_v1_phase.add_parameter('m', val=174200, units='lbm', opt=False, dynamic=True)
br_to_v1_phase.add_parameter('mu_r', val=0.03, opt=False, dynamic=False)
br_to_v1_phase.add_parameter('alpha', val=0.0, opt=False, dynamic=True, units='deg')

br_to_v1_phase.add_boundary_constraint('v_over_v_stall', loc='final', equals=1)

# Rejected Takeoff Phase
# Thrust is zero and mu_r is changed to 0.3 to reflect braking
rto_phase = dm.Phase(ode_class=GroundRollODEComp, transcription=dm.Radau(num_segments=5))

rto_phase.set_time_options(initial_val=20.0, duration_val=20.0,
                           initial_bounds=(1, 100), duration_bounds=(0.1, 100))

rto_phase.add_state('r', val=rto_phase.interpolate(ys=[1000, 2000], nodes='state_input'))
rto_phase.add_state('v', fix_final=True, val=rto_phase.interpolate(ys=[100, 0.0001], nodes='state_input'))

rto_phase.add_parameter('h', val=0.0, units='ft', opt=False, dynamic=False)
rto_phase.add_parameter('T', val=0., units='lbf', opt=False, dynamic=False)
rto_phase.add_parameter('m', val=174200, units='lbm', opt=False, dynamic=True)
rto_phase.add_parameter('mu_r', val=0.3, opt=False, dynamic=False)
rto_phase.add_parameter('alpha', val=0.0, opt=False, dynamic=True, units='deg')

# rto_phase.add_boundary_constraint('v', loc='final', upper=0.001, units='kn')
rto_phase.add_objective('r', loc='final', ref=1000)

# Continue nominal takeoff until rotation (v1 to vr)
# Operating under a single engine
v1_to_vr_phase = dm.Phase(ode_class=GroundRollODEComp, transcription=dm.Radau(num_segments=3))

v1_to_vr_phase.set_time_options(initial_val=20.0, duration_val=20.0,
                                initial_bounds=(1, 100), duration_bounds=(0.1, 100))

v1_to_vr_phase.add_state('r', val=v1_to_vr_phase.interpolate(ys=[1000, 2000], nodes='state_input'))
v1_to_vr_phase.add_state('v', val=v1_to_vr_phase.interpolate(ys=[100, 120.0], nodes='state_input'))

v1_to_vr_phase.add_parameter('h', val=0.0, units='ft', opt=False, dynamic=False)
v1_to_vr_phase.add_parameter('T', val=27000, units='lbf', opt=False, dynamic=False)
v1_to_vr_phase.add_parameter('m', val=174200, units='lbm', opt=False, dynamic=True)
v1_to_vr_phase.add_parameter('mu_r', val=0.03, opt=False, dynamic=False)
v1_to_vr_phase.add_parameter('alpha', val=0.0, opt=False, dynamic=True, units='deg')

v1_to_vr_phase.add_boundary_constraint('v_over_v_stall', loc='final', equals=1.2)

# Rotate
# Increase alpha at a constant rate until the normal force on the landing gear is zero.
rotate_phase = dm.Phase(ode_class=GroundRollODEComp, transcription=dm.Radau(num_segments=3))

rotate_phase.set_time_options(initial_val=40.0, duration_val=5.0,
                              initial_bounds=(1, 100), duration_bounds=(5, 5))

rotate_phase.add_state('r', val=rotate_phase.interpolate(ys=[1000, 2000], nodes='state_input'))
rotate_phase.add_state('v', val=rotate_phase.interpolate(ys=[100, 120.0], nodes='state_input'), defect_ref=1)

rotate_phase.add_parameter('h', val=0.0, units='ft', opt=False, dynamic=False)
rotate_phase.add_parameter('T', val=27000, units='lbf', opt=False, dynamic=False)
rotate_phase.add_parameter('m', val=174200, units='lbm', opt=False, dynamic=True)
rotate_phase.add_parameter('mu_r', val=0.03, opt=False, dynamic=False)

rotate_phase.add_polynomial_control('alpha', order=1, opt=True, units='deg', lower=0, upper=10, ref=10)

rotate_phase.add_boundary_constraint('F_r', loc='final', equals=0, ref=100000)

rotate_phase.add_timeseries_output('*')

# Climb
# Ascent with a maximum flight path angle of 5 degrees until 35 ft above ground and v > 1.2 * v_stall

climb_phase = dm.Phase(ode_class=TakeoffClimbODEComp, transcription=dm.Radau(num_segments=5))

climb_phase.set_time_options(initial_bounds=(1, 200), duration_bounds=(1, 100), duration_ref=1.0)

climb_phase.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)

climb_phase.add_state('h', fix_initial=True, fix_final=True, lower=0.0, ref=1.0, defect_ref=1.0)

climb_phase.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)

climb_phase.add_state('gam', fix_initial=True, lower=0.0, upper=om.convert_units(5, 'deg', 'rad'), ref=0.05, defect_ref=0.05)

climb_phase.add_parameter('T', val=27000, opt=False, units='lbf', dynamic=False)
climb_phase.add_parameter('m', val=174200, opt=False, units='lbm')

climb_phase.add_control('alpha', opt=True, lower=-15, upper=15, rate_continuity=True, rate_continuity_scaler=1, units='deg')

climb_phase.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.2)

# Set up the trajectory.
traj.add_phase('br_to_v1', br_to_v1_phase)
traj.add_phase('rto', rto_phase)
traj.add_phase('v1_to_vr', v1_to_vr_phase)
traj.add_phase('rotate', rotate_phase)
# traj.add_phase('climb', climb_phase)

# Mass is constant throughout takeoff
traj.add_parameter('m', val=174200, units='lbm')

# Link the phases together for continuity in time, states, and controls
traj.link_phases(['br_to_v1', 'rto'], vars=['time', 'r', 'v'])
traj.link_phases(['br_to_v1', 'v1_to_vr'], vars=['time', 'r', 'v'])
traj.link_phases(['v1_to_vr', 'rotate'], vars=['time', 'r', 'v', 'alpha'])
# traj.link_phases(['rotate', 'climb'], vars=['time', 'r', 'v', 'alpha'])

# Add an additional linkage constraint to force the end of the rto phase to have the same range as the end of the climb phase
# traj.add_linkage_constraint(phase_a='rto', phase_b='climb',
#                             var_a='r', var_b='r',
#                             loc_a='final', loc_b='final',
#                             ref=1000, units='ft')

# Record all ODE outputs in all phases
for phs in [br_to_v1_phase, rto_phase, v1_to_vr_phase, rotate_phase, climb_phase]:
    phs.add_timeseries_output('*')

p.setup()




p.set_val('traj.br_to_v1.t_initial', 0)
p.set_val('traj.br_to_v1.t_duration', 25)

p.set_val('traj.br_to_v1.states:r', br_to_v1_phase.interpolate(ys=[0, 2500.0], nodes='state_input'))
p.set_val('traj.br_to_v1.states:v', br_to_v1_phase.interpolate(ys=[0.0001, 100.0], nodes='state_input'))

p.set_val('traj.br_to_v1.parameters:alpha', 0, units='deg')
p.set_val('traj.br_to_v1.parameters:h', 0.0)

#

p.set_val('traj.v1_to_vr.t_initial', 25)
p.set_val('traj.v1_to_vr.t_duration', 10)

p.set_val('traj.v1_to_vr.states:r', v1_to_vr_phase.interpolate(ys=[2500, 300.0], nodes='state_input'))
p.set_val('traj.v1_to_vr.states:v', v1_to_vr_phase.interpolate(ys=[100, 110.0], nodes='state_input'))

p.set_val('traj.v1_to_vr.parameters:alpha', 0.0, units='deg')

p.set_val('traj.v1_to_vr.parameters:h', 0.0)

#

p.set_val('traj.rto.t_initial', 25)
p.set_val('traj.rto.t_duration', 30)

p.set_val('traj.rto.states:r', rto_phase.interpolate(ys=[2500, 5000.0], nodes='state_input'))
p.set_val('traj.rto.states:v', rto_phase.interpolate(ys=[110, 0.0001], nodes='state_input'))

p.set_val('traj.rto.parameters:alpha', 0.0, units='deg')
p.set_val('traj.rto.parameters:h', 0.0)
p.set_val('traj.rto.parameters:T', 0.0)
p.set_val('traj.rto.parameters:mu_r', 0.3)

# #
#
# p.set_val('traj.rotate.t_initial', 35)
# p.set_val('traj.rotate.t_duration', 10)
#
# p.set_val('traj.rotate.states:r', rotate_phase.interpolate(ys=[5000, 5500.0], nodes='state_input'))
# p.set_val('traj.rotate.states:v', rotate_phase.interpolate(ys=[160, 170.0], nodes='state_input'))
#
# p.set_val('traj.rotate.polynomial_controls:alpha', 0.0, units='deg')
#
# p.set_val('traj.rotate.parameters:h', 0.0)
#
# #
#
# p.set_val('traj.climb.t_initial', 45)
# p.set_val('traj.climb.t_duration', 20)
#
# p.set_val('traj.climb.states:r', climb_phase.interpolate(ys=[5000, 7000.0], nodes='state_input'), units='ft')
# p.set_val('traj.climb.states:v', climb_phase.interpolate(ys=[100, 150.0], nodes='state_input'), units='kn')
# p.set_val('traj.climb.states:h', climb_phase.interpolate(ys=[0, 35.0], nodes='state_input'), units='ft')
# p.set_val('traj.climb.states:gam', climb_phase.interpolate(ys=[0, 5.0], nodes='state_input'), units='deg')
#
# p.set_val('traj.climb.controls:alpha', 5.0, units='deg')
# p.set_val('traj.climb.parameters:T', 27000.0, units='lbf')


dm.run_problem(p, run_driver=True, simulate=True, make_plots=True)



if __name__ == '__main__':  # pragma: no cover
    unittest.main()
