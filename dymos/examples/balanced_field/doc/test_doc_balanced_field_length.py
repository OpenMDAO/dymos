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
# p.driver.options['optimizer'] = 'SNOPT'
# p.driver.opt_settings['iSumm'] = 6
# # p.driver.opt_settings['Verify level'] = 3
# # p.driver.opt_settings['Major step limit'] = 0.1
# p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-5
# p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
# p.driver.opt_settings['Major iterations limit'] = 500
# p.driver.opt_settings['Minor iterations limit'] = 100000
# # p.driver.opt_settings['Linesearch tolerance'] = 0.5


p.driver.options['optimizer'] = 'IPOPT'
p.driver.opt_settings['print_level'] = 5
p.driver.opt_settings['derivative_test'] = 'first-order'
# p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'

p.driver.declare_coloring()



#
# First Phase - Brake Release to V1
# Operating with two engines
# We don't know what V1 is a priori, we're going to use this model to determine it.
#

br_to_v1 = dm.Phase(ode_class=GroundRollODEComp, transcription=dm.Radau(num_segments=3))

#
# Set the options on the optimization variables
#
br_to_v1.set_time_options(fix_initial=True, duration_bounds=(1, 1000), duration_ref=10.0)

br_to_v1.add_state('r', fix_initial=True, lower=0, ref=1000.0, defect_ref=1000.0)

br_to_v1.add_state('v', fix_initial=True, lower=0.0001, ref=100.0, defect_ref=100.0)

# br_to_v1.add_parameter('h', opt=False, units='m', dynamic=False)
# br_to_v1.add_parameter('T', val=27000 * 2, opt=False, units='lbf', dynamic=False)
br_to_v1.add_parameter('alpha', val=0.0, opt=False, units='deg')
# br_to_v1.add_parameter('mu_r', val=0.03, opt=False, units=None, dynamic=False)

br_to_v1.add_timeseries_output('*')

# Second Phase - V1 to Vr
# Operating with one engine
# Vr is taken to be 1.2 * the stall speed (v_stall)
#

v1_to_vr = dm.Phase(ode_class=GroundRollODEComp, transcription=dm.Radau(num_segments=3))

#
# Set the options on the optimization variables
#
v1_to_vr.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)

v1_to_vr.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)

v1_to_vr.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)

# v1_to_vr.add_parameter('h', val=0.0, opt=False, units='m', dynamic=False)
# v1_to_vr.add_parameter('T', val=27000, opt=False, units='lbf', dynamic=False)
# v1_to_vr.add_parameter('mu_r', val=0.03, opt=False, units=None, dynamic=False)
v1_to_vr.add_parameter('alpha', val=0.0, opt=False, units='deg')
v1_to_vr.add_timeseries_output('*')


rto = dm.Phase(ode_class=GroundRollODEComp, transcription=dm.Radau(num_segments=3))


#
# Set the options on the optimization variables
#
rto.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)

rto.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)

rto.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)

# rto.add_parameter('h', val=0.0, opt=False, units='m', dynamic=False)
# rto.add_parameter('T', val=0.0, opt=False, units='N', dynamic=False)
# rto.add_parameter('mu_r', val=0.3, opt=False, units=None, dynamic=False)
rto.add_parameter('alpha', val=0.0, opt=False, units='deg')

rto.add_timeseries_output('*')


# Minimize range at the end of the phase
rto.add_objective('r', loc='final', ref=1000.0) #  ref0=2000.0, ref=3000.0)


# Fourth Phase - Rotate for single engine takeoff
# v_rotate to runway normal force = 0
#

rotate = dm.Phase(ode_class=GroundRollODEComp, transcription=dm.Radau(num_segments=3))

rotate.set_time_options(fix_initial=False, duration_bounds=(1.0, 5), duration_ref=1.0)
rotate.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
rotate.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)

# rotate.add_parameter('h', val=0.0, opt=False, units='m', dynamic=False)
# rotate.add_parameter('T', val=27000, opt=False, units='lbf', dynamic=False)
# rotate.add_parameter('mu_r', val=0.05, opt=False, units=None, dynamic=False)

rotate.add_polynomial_control('alpha', order=1, opt=True, units='deg', lower=0, upper=10, ref=10)

rotate.add_timeseries_output('*')


climb = dm.Phase(ode_class=TakeoffClimbODEComp, transcription=dm.Radau(num_segments=5))

climb.set_time_options(fix_initial=False, duration_bounds=(1, 100), duration_ref=1.0)

climb.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
climb.add_state('h', fix_initial=True, lower=0.0, ref=1.0, defect_ref=1.0)
climb.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)
climb.add_state('gam', fix_initial=True, lower=0.0, ref=0.05, defect_ref=0.05)

# climb.add_parameter('T', val=27000, opt=False, units='lbf', dynamic=False)

climb.add_control('alpha', opt=True, units='deg', lower=-10, upper=15, ref=10, rate_continuity=True, rate_continuity_scaler=1)

climb.add_timeseries_output('*')



p.model.linear_solver = om.DirectSolver()
# p.model.linear_solver = om.PETScKrylov()
# p.model.linear_solver.options['maxiter'] = 4
# # p.model.linear_solver.options['iprint'] = 2
# p.model.linear_solver.precon = om.DirectSolver()


#
# Instantiate the trajectory and add phases
#
traj = dm.Trajectory()

p.model.add_subsystem('traj', traj)

traj.add_phase('br_to_v1', br_to_v1)
traj.add_phase('v1_to_vr', v1_to_vr)
traj.add_phase('rto', rto)
traj.add_phase('rotate', rotate)
traj.add_phase('climb', climb)


traj.add_parameter('m', val=174200., opt=False, units='lbm',
                   targets={'br_to_v1': ['m'], 'v1_to_vr': ['m'], 'rto': ['m'],
                            'rotate': ['m'], 'climb': ['m']})

traj.add_parameter('T_nominal', val=27000 * 2, opt=False, units='lbf', dynamic=False,
                   targets={'br_to_v1': ['T']})

traj.add_parameter('T_engine_out', val=27000, opt=False, units='lbf', dynamic=False,
                   targets={'v1_to_vr': ['T'], 'rotate': ['T'], 'climb': ['T']})

traj.add_parameter('T_shutdown', val=0.0, opt=False, units='lbf', dynamic=False,
                   targets={'rto': ['T']})

traj.add_parameter('mu_r_nominal', val=0.03, opt=False, units=None, dynamic=False,
                   targets={'br_to_v1': ['mu_r'], 'v1_to_vr': ['mu_r'],  'rotate': ['mu_r']})

traj.add_parameter('mu_r_braking', val=0.3, opt=False, units=None, dynamic=False,
                   targets={'rto': ['mu_r']})

# traj.add_parameter('h_runway', val=0., opt=False, units='ft',
#                    targets={'br_to_v1': ['h'], 'v1_to_vr': ['h'], 'rto': ['h'],
#                             'rotate': ['h']})

traj.add_parameter('rho', val=1.225, opt=False, units='kg/m**3', dynamic=False,
                   targets={'br_to_v1': ['rho'], 'v1_to_vr': ['rho'], 'rto': ['rho'],
                            'rotate': ['rho']})

#
# Link the phases
#

# Standard "end of first phase to beginning of second phase" linkages
traj.link_phases(['br_to_v1', 'v1_to_vr'], vars=['time', 'r', 'v'])
traj.link_phases(['v1_to_vr', 'rotate'], vars=['time', 'r', 'v', 'alpha'])
traj.link_phases(['rotate', 'climb'], vars=['time', 'r', 'v', 'alpha'])
traj.link_phases(['br_to_v1', 'rto'], vars=['time', 'r', 'v'])

# # Less common "final value of r must be the match at ends of two phases".
traj.add_linkage_constraint(phase_a='rto', var_a='r', loc_a='final',
                            phase_b='climb', var_b='r', loc_b='final',
                            ref=1000)

# Add the boundary constraints here

rto.add_boundary_constraint('v', loc='final', upper=0.001, ref=100, linear=True)

rotate.add_boundary_constraint('F_r', loc='final', equals=0, ref=100000)

climb.add_boundary_constraint('h', loc='final', equals=35, ref=35, units='ft', linear=True)
climb.add_boundary_constraint('gam', loc='final', equals=5, ref=5, units='deg', linear=True)
climb.add_path_constraint('gam', lower=0, upper=5, ref=5, units='deg')
climb.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.2, ref=1.2)

#
# Setup the problem and set the initial guess
#
p.setup(check=True)

p.set_val('traj.br_to_v1.t_initial', 0)
p.set_val('traj.br_to_v1.t_duration', 35)

p.set_val('traj.br_to_v1.states:r', br_to_v1.interpolate(ys=[0, 2500.0], nodes='state_input'))
p.set_val('traj.br_to_v1.states:v', br_to_v1.interpolate(ys=[0.0001, 100.0], nodes='state_input'))

p.set_val('traj.br_to_v1.parameters:alpha', 0, units='deg')
# p.set_val('traj.br_to_v1.parameters:h', 0.0)

#

p.set_val('traj.v1_to_vr.t_initial', 35)
p.set_val('traj.v1_to_vr.t_duration', 35)

p.set_val('traj.v1_to_vr.states:r', v1_to_vr.interpolate(ys=[2500, 300.0], nodes='state_input'))
p.set_val('traj.v1_to_vr.states:v', v1_to_vr.interpolate(ys=[100, 110.0], nodes='state_input'))

p.set_val('traj.v1_to_vr.parameters:alpha', 0.0, units='deg')

# p.set_val('traj.v1_to_vr.parameters:h', 0.0)

#

p.set_val('traj.rto.t_initial', 35)
p.set_val('traj.rto.t_duration', 1)

p.set_val('traj.rto.states:r', rto.interpolate(ys=[2500, 5000.0], nodes='state_input'))
p.set_val('traj.rto.states:v', rto.interpolate(ys=[110, 0.0001], nodes='state_input'))

p.set_val('traj.rto.parameters:alpha', 0.0, units='deg')



p.set_val('traj.rotate.t_initial', 35)
p.set_val('traj.rotate.t_duration', 5)

p.set_val('traj.rotate.states:r', rotate.interpolate(ys=[1750, 1800.0], nodes='state_input'))
p.set_val('traj.rotate.states:v', rotate.interpolate(ys=[80, 85.0], nodes='state_input'))

p.set_val('traj.rotate.polynomial_controls:alpha', 0.0, units='deg')

# p.set_val('traj.rotate.parameters:h', 0.0)

#
p.set_val('traj.climb.t_initial', 30)
p.set_val('traj.climb.t_duration', 20)

p.set_val('traj.climb.states:r', climb.interpolate(ys=[5000, 5500.0], nodes='state_input'), units='ft')
p.set_val('traj.climb.states:v', climb.interpolate(ys=[160, 170.0], nodes='state_input'), units='kn')
p.set_val('traj.climb.states:h', climb.interpolate(ys=[0, 35.0], nodes='state_input'), units='ft')
p.set_val('traj.climb.states:gam', climb.interpolate(ys=[0, 5.0], nodes='state_input'), units='deg')

p.set_val('traj.climb.controls:alpha', 5.0, units='deg')
# p.set_val('traj.climb.parameters:T', 27000.0, units='lbf')


dm.run_problem(p, run_driver=True, simulate=True, make_plots=True) #, restart='dymos_simulation.db')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
