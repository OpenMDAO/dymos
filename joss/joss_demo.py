import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt


# First define a system which computes the equations of motion
class BrachistochroneEOM(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.zeros(nn), units='m/s',
                       desc='velocity')

        self.add_input('theta', val=np.zeros(nn), units='rad',
                       desc='angle of wire')

        self.add_output('xdot', val=np.zeros(nn), units='m/s',
                        desc='velocity component in x')

        self.add_output('ydot', val=np.zeros(nn), units='m/s',
                        desc='velocity component in y')

        self.add_output('vdot', val=np.zeros(nn), units='m/s**2',
                        desc='acceleration magnitude')

        # Setup partials for the analytic derivatives
        # These all have diagonal partial-derivative jacobians
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='vdot', wrt='theta', rows=ar, cols=ar)
        self.declare_partials(of='xdot', wrt='*', rows=ar, cols=ar)
        self.declare_partials(of='ydot', wrt='*', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        v, theta = inputs.values()

        outputs['vdot'] = 9.80665 * np.cos(theta)
        outputs['xdot'] = v * np.sin(theta)
        outputs['ydot'] = -v * np.cos(theta)

    def compute_partials(self, inputs, jacobian):
        v, theta = inputs.values()

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        jacobian['vdot', 'theta'] = -9.80665 * sin_theta

        jacobian['xdot', 'v'] = sin_theta
        jacobian['xdot', 'theta'] = v * cos_theta

        jacobian['ydot', 'v'] = -cos_theta
        jacobian['ydot', 'theta'] = v * sin_theta


# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with Radau Transcription
phase = dm.Phase(ode_class=BrachistochroneEOM,
                 transcription=dm.Radau(num_segments=10, order=3))
traj.add_phase(name='phase0', phase=phase)

# Set the time options
phase.set_time_options(fix_initial=True,
                       duration_bounds=(0.5, 10.0))

# Set the state options
phase.add_state('x', rate_source='xdot',
                fix_initial=True, fix_final=True)
phase.add_state('y', rate_source='ydot',
                fix_initial=True, fix_final=True)
phase.add_state('v', rate_source='vdot',
                fix_initial=True, fix_final=False)

# Define theta as a control.
phase.add_control(name='theta', units='rad',
                  lower=0, upper=np.pi)

# Minimize final time.
phase.add_objective('time', loc='final')

# Set the driver.
p.driver = om.ScipyOptimizeDriver()

# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significant speed up the execution of dymos.
p.driver.declare_coloring()

# Setup the problem
p.setup()

# Now that the OpenMDAO problem is setup, we can guess the
# values of time, states, and controls.
p.set_val('traj.phase0.t_duration', 2.0)

# States and controls here use a linearly interpolated
# initial guess along the trajectory.
p.set_val('traj.phase0.states:x',
          phase.interp('x', [0, 10]),
          units='m')

p.set_val('traj.phase0.states:y',
          phase.interp('y', [10, 5]),
          units='m')

p.set_val('traj.phase0.states:v',
          phase.interp('v', [0, 5]),
          units='m/s')

p.set_val('traj.phase0.controls:theta',
          phase.interp('theta', [5, 45]),
          units='deg')

# Run the driver to solve the problem
p.run_driver()

# Check the validity of our results by using
# scipy.integrate.solve_ivp to integrate the solution.
sim_out = traj.simulate()

# Plot the results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))

axes[0].plot(p.get_val('traj.phase0.timeseries.states:x'),
             p.get_val('traj.phase0.timeseries.states:y'),
             'ro', label='solution')

axes[0].plot(sim_out.get_val('traj.phase0.timeseries.states:x'),
             sim_out.get_val('traj.phase0.timeseries.states:y'),
             'b-', label='simulation')

axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m/s)')
axes[0].legend()
axes[0].grid()

axes[1].plot(p.get_val('traj.phase0.timeseries.time'),
             p.get_val('traj.phase0.timeseries.controls:theta',
                       units='deg'),
             'ro', label='solution')

axes[1].plot(sim_out.get_val('traj.phase0.timeseries.time'),
             sim_out.get_val('traj.phase0.timeseries.controls:theta',
                             units='deg'),
             'b-', label='simulation')

axes[1].set_xlabel('time (s)')
axes[1].set_ylabel(r'$\theta$ (deg)')
axes[1].legend()
axes[1].grid()

plt.show()
