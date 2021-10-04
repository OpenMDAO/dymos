import openmdao.api as om
import dymos as dm

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


prob = om.Problem()

prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')

tx = dm.transcriptions.ExplicitShooting(num_segments=5, grid='gauss-lobatto', method='rk4',
                                        order=3, num_steps_per_segment=10, compressed=True)

phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

# automatically discover states
phase.set_state_options('x', fix_initial=True)
phase.set_state_options('y', fix_initial=True)
phase.set_state_options('v', fix_initial=True)

phase.add_parameter('g', val=1.0, units='m/s**2', opt=True, lower=1, upper=9.80665)
phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9)

phase.add_boundary_constraint('x', loc='final', equals=10.0)
phase.add_boundary_constraint('y', loc='final', equals=5.0)

phase.add_timeseries_output('*')

prob.model.add_subsystem('phase0', phase)

phase.add_objective('time', loc='final')

prob.setup(force_alloc_complex=True)

prob.set_val('phase0.t_initial', 0.0)
prob.set_val('phase0.t_duration', 2)
prob.set_val('phase0.states:x', 0.0)
prob.set_val('phase0.states:y', 10.0)
prob.set_val('phase0.states:v', 1.0E-6)
prob.set_val('phase0.parameters:g', 1.0, units='m/s**2')
prob.set_val('phase0.controls:theta', phase.interp('theta', ys=[0.01, 90]), units='deg')

dm.run_problem(prob, run_driver=True)
