Dymos:  Open Source Optimization of Dynamic Multidisciplinary Systems
=====================================================================

[![Build Status](https://travis-ci.com/OpenMDAO/dymos.svg?token=tUBGTjUY1qBbh4Htx3Sr&branch=master)](https://travis-ci.com/OpenMDAO/dymos) [![Coverage Status](https://coveralls.io/repos/github/OpenMDAO/dymos/badge.svg?branch=master&t=dJxu2Q)](https://coveralls.io/github/OpenMDAO/dymos?branch=master)


Dymos is a framework for the simulation and optimization of dynamical systems within the OpenMDAO Multidisciplinary Analysis and Optimization environment.
Dymos leverages implicit and explicit simulation techniques to simulate generic dynamic systems of arbitary complexity.

The software has two primary objectives:
-   Provide a generic ODE integration interface that allows for the analysis of dynamical systems.
-   Allow the user to solve optimal control problems involving dynamical multidisciplinary systems.

Installation
------------

The default installation of the developmental version of Dymos will install the minimum number of prerequisites:

```
python -m pip install git+https://github.com/OpenMDAO/dymos.git
```

Installation of a specific version can be accomplished with

```
python -m pip install git+https://github.com/OpenMDAO/dymos.git@RELEASENAME
```

See the [releases page](https://github.com/OpenMDAO/dymos/releases) for a listing of the latest releases.

If you plan on building the documentation or running all of the tests locally, you should install _all_ dependencies with:

```
pip install git+https://github.com/OpenMDAO/dymos.git#egg=project[all]
```

Documentation
-------------

Online documentation is available at [https://openmdao.github.io/dymos/](https://openmdao.github.io/dymos/)

Defining Ordinary Differential Equations
----------------------------------------

The first step in simulating or optimizing a dynamical system is to define the ordinary
differential equations to be integrated.  The user first builds an OpenMDAO model which has outputs
that provide the rates of the state variables.  This model can be an OpenMDAO model of arbitrary
complexity, including nested groups and components, layers of nonlinear solvers, etc.

Dymos solutions are constructed of one or more _Phases_.
When setting up a phase, we add state variables, dynamic controls, and parameters,
tell Dymos how the value of each should be connected to the ODE system, and tell Dymos
the variable paths in the system that contain the rates of our state variables that are to be
integrated.

```python
    import numpy as np
    from openmdao.api import ExplicitComponent

    class BrachistochroneEOM(ExplicitComponent):
        def initialize(self):
            self.options.declare('num_nodes', types=int)

        def setup(self):
            nn = self.options['num_nodes']

            # Inputs
            self.add_input('v',
                           val=np.zeros(nn),
                           desc='velocity',
                           units='m/s')

            self.add_input('g',
                           val=9.80665*np.ones(nn),
                           desc='gravitational acceleration',
                           units='m/s/s')

            self.add_input('theta',
                           val=np.zeros(nn),
                           desc='angle of wire',
                           units='rad')

            self.add_output('xdot',
                            val=np.zeros(nn),
                            desc='velocity component in x',
                            units='m/s')

            self.add_output('ydot',
                            val=np.zeros(nn),
                            desc='velocity component in y',
                            units='m/s')

            self.add_output('vdot',
                            val=np.zeros(nn),
                            desc='acceleration magnitude',
                            units='m/s**2')

            self.add_output('check',
                            val=np.zeros(nn),
                            desc='A check on the solution: v/sin(theta) = constant',
                            units='m/s')

            # Setup partials
            arange = np.arange(self.options['num_nodes'])

            self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange, val=1.0)

            self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange, val=1.0)

            self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange, val=1.0)

            self.declare_partials(of='check', wrt='v', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='check', wrt='theta', rows=arange, cols=arange, val=1.0)

        def compute(self, inputs, outputs):
            theta = inputs['theta']
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            g = inputs['g']
            v = inputs['v']

            outputs['vdot'] = g*cos_theta
            outputs['xdot'] = v*sin_theta
            outputs['ydot'] = -v*cos_theta
            outputs['check'] = v/sin_theta

        def compute_partials(self, inputs, jacobian):
            theta = inputs['theta']
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            g = inputs['g']
            v = inputs['v']

            jacobian['vdot', 'g'] = cos_theta
            jacobian['vdot', 'theta'] = -g*sin_theta

            jacobian['xdot', 'v'] = sin_theta
            jacobian['xdot', 'theta'] = v*cos_theta

            jacobian['ydot', 'v'] = -cos_theta
            jacobian['ydot', 'theta'] = v*sin_theta

            jacobian['check', 'v'] = 1/sin_theta
            jacobian['check', 'theta'] = -v*cos_theta/sin_theta**2
```

Integrating Ordinary Differential Equations
-------------------------------------------

Dymos's `RungeKutta` and solver-based pseudspectral transcriptions
provide the ability to numerically integrate the ODE system it is given.
Used in an optimal control context, these provide a shooting method in
which each iteration provides a physically viable trajectory.

Pseudospectral Methods
----------------------

Dymos currently supports the Radau Pseudospectral Method and high-order
Gauss-Lobatto transcriptions.  These implicit techniques rely on the
optimizer to impose "defect" constraints which enforce the physical
accuracy of the resulting trajectories.  To verify the physical
accuracy of the solutions, Dymos can explicitly integrate them using
variable-step methods.

Solving Optimal Control Problems
--------------------------------

Dymos uses the concept of _Phases_ to support optimal control of dynamical systems.
Users connect one or more Phases to construct trajectories.
Each Phase can have its own:

-   Optimal Control Transcription (Gauss-Lobatto, Radau Pseudospectral, or RungeKutta)
-   Equations of motion
-   Boundary and path constraints

Dymos Phases and Trajectories are ultimately just OpenMDAO Groups that can exist in
a problem along with numerous other models, allowing for the simultaneous
optimization of systems and dynamics.

Here is a Dymos solution for the Brachistochrone example shown above:

```python
import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt

# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
phase = dm.Phase(ode_class=BrachistochroneODE,
                 transcription=dm.GaussLobatto(num_segments=10, order=3))

traj.add_phase(name='phase0', phase=phase)

# Set the time options
# Time has no targets in our ODE.
# We fix the initial time so that the it is not a design variable in the optimization.
# The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

# Set the time options
# Initial values of positions and velocity are all fixed.
# The final value of position are fixed, but the final velocity is a free variable.
# The equations of motion are not functions of position, so 'x' and 'y' have no targets.
# The target of 'v' will be automatically found by Dymos since `v` is an input at the top-level of the ODE.
# The rate source points to the output in the ODE which provides the time derivative of the given state.
phase.add_state('x', fix_initial=True, fix_final=True, units='m', rate_source='xdot')
phase.add_state('y', fix_initial=True, fix_final=True, units='m', rate_source='ydot')
phase.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='vdot')

# Define theta as a control.
phase.add_control(name='theta', units='rad', lower=0, upper=np.pi, targets=['theta'])

# Minimize final time.
phase.add_objective('time', loc='final')

# Set the driver.
p.driver = om.ScipyOptimizeDriver()

# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significant speed up the execution of Dymos.
p.driver.declare_coloring()

# Setup the problem
p.setup(check=True)

# Now that the OpenMDAO problem is setup, we can set the values of the states.
p.set_val('traj.phase0.states:x',
          phase.interpolate(ys=[0, 10], nodes='state_input'),
          units='m')

p.set_val('traj.phase0.states:y',
          phase.interpolate(ys=[10, 5], nodes='state_input'),
          units='m')

p.set_val('traj.phase0.states:v',
          phase.interpolate(ys=[0, 5], nodes='state_input'),
          units='m/s')

p.set_val('traj.phase0.controls:theta',
          phase.interpolate(ys=[90, 90], nodes='control_input'),
          units='deg')

# Run the driver to solve the problem
p.run_driver()

# Check the validity of our results by using scipy.integrate.solve_ivp to integrate the solution.
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
             p.get_val('traj.phase0.timeseries.controls:theta', units='deg'),
             'ro', label='solution')

axes[1].plot(sim_out.get_val('traj.phase0.timeseries.time'),
             sim_out.get_val('traj.phase0.timeseries.controls:theta', units='deg'),
             'b-', label='simulation')

axes[1].set_xlabel('time (s)')
axes[1].set_ylabel(r'$\theta$ (deg)')
axes[1].legend()
axes[1].grid()

plt.show()
```

![alt text](brachistochroneSolution.png "Brachistochrone Solution")
