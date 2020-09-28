---
title: 'dymos: A Python package for optimal control of multidisciplinary systems'
tags:
  - Python
  - OpenMDAO
  - optimal control
  - trajectory optimization
  - multidisciplinary optimization
  - NASA
authors:
  - name: Rob Falck
    orcid: 0000-0001-9864-4928
    affiliation: 1
  - name: Justin Gray
    affiliation: 1
  - name: Kaushik Ponnapalli
    affiliation: 2
  - name: Ted Wright
    affiliation: 1
affiliations:
  - name: NASA Glenn Research Center
    index: 1
  - name: HX5 LLC
    index: 2
date: 26 June 2020
bibliography: paper.bib
---

# Summary

Dymos is an library for solving optimal-control type optimization problems. 
It contains capabilities typical of optimal control software and can handle a wide range of typical optimal control problems.
Its software design (built on top of NASA's OpenMDAO Framework [@Gray2019a] also allows users to tackle problems where the trajectory is not necessarily central to the optimization and where the system dynamics may include expensive, implicit calculations.
Building dymos as a library on top of OpenMDAO extends its capabilities due to the efficiency with which OpenMDAO can compute analytic derivatives, even in the presence of iterative behavior.
Optimal control software generally require that the dynamics of the system be defined as a set of explicit ordinary differential equations (ODE) which compute the rates of the state variables to be integrated.
Sometimes the dynamics are instead posed as a set of differential algebraic equations, where some equality constraint needs to be satisfied at the solution in addition to the ODE.
One application of this approach is the method of differential inclusions, in which the state time-history is posed as a dynamic control, and the traditional control variables needed to achieve that trajectory are found using a nonlinear solver within the ODE [@Seywald1994].
Support for implicit calculations gives users more freedom to pose dynamics in more natural ways, but typically causes numerical and computational cost challenges in an optimization context, especially when finite-differences are used to compute derivatives for the optimizer.
The ability to efficiently solve problems with nested implicit behavior without completely reformulating the problem sets dymos apart from many other optimal control software tools.

Dymos can be used stand-alone, or as a building block in a larger model where a significant portion of the optimization is focused on some non-controls aspect with a trajectory added to enforce some constraint upon the design.
In some contexts, you may hear this kind of problem referred to as co-design, controls-co-design, or multidisciplinary design optimization.

## The dymos perspective on optimal control

Dymos is similar to some other trajectory optimization tools in that the trajectory of a system is subdivided into chunks of time called _phases_.
Breaking the trajectory into phases provides several capabilities.
Intermediate constraints along a trajectory can be enforced by applying boundary constraint to a phase that begins or ends at the time of interest.
For instance, the optimal trajectory of a launch vehicle may be required to ascend vertically to clear a launch tower before it pitches over on its way to orbit.
Path constraints can be applied within each phase to bound some performance parameter within that phase.
For example, reentry vehicles may need to shape their trajectory to limit aerodynamic heating.
Phases also provide the ability to compose a trajectory of phases in which the dynamics may change from Phase to phase.
A aircraft with vertical takeoff and landing capability, for instance, may use different sets of dynamics to define its flight while in vertical flight and horizontal flight.
As another useful feature, dymos offers the ability to use variable order polynomial controls that are useful in forcing a lower-order control solution upon a phase of the trajectory.
This can help to reduce "chatter" in controls and provide more robust convergence.

Dymos is primarily focused on an implicit pseudospectral approach to optimal-control. 
It leverages two common direct collocation transcriptions: the high-order Gauss-Lobatto transcription [@Herman1996] and the Radau pseudospectral method [@Garg2009].
Dymos also provides explicit forms of both of these transcriptions, which provides a single or multiple-shooting approach.
All of these transcriptions are implemented in a way that is independent of the ODE implementation, nearly transparent to the user, and requires very minor code changes - typically a single line in the run-script.
ODE's are implemented as standard OpenMDAO systems which are passed to phases at instantiation time with some additional annotations to identify the states, state-rates, and control inputs.

Dymos does not ship with its own built in optimizer. 
It relies on whatever optimizers you have available in your OpenMDAO installation. 
OpenMDAO ships with an interface to the optimizers in SciPy [@2020SciPy-NMeth], and an additional wrapper for the pyoptsparse [@Perez2012a] library which has more powerful optimizer options such as SNOPT [@GilMS05] and IPOPT [@wachter2006].
For simple problems, Scipy's SLSQP optimizer generally works fine.
On more challenging optimal-control problems, higher quality optimizers are important for getting good performance.

## Statement of Need

Modeling complex multidisciplinary systems often involves the use of iterative nonlinear solvers to converge the design or operational parameters of interacting subsystems.
To speed the design optimization of such systems, NASA's OpenMDAO software was developed to generalize the calculation of derivatives across models for use in gradient-based optimization.
However, the optimal control of multidisciplinary systems may involve the use of nonlinear solvers within the ODE itself.
The implicit analysis could arise due to the need for some high fidelity physics model (e.g. a vortex lattice method for aerodynamic performance prediction) or from the formulation of a differential inclusion approach.
Despite the application of adjoint differentiation, shooting methods based on explicit time-marching still suffer from a performance standpoint due to the need to reconverge the solvers within the ODE at each time step.
Dymos was developed to leverage the advanced differentiation capabilities of OpenMDAO in combination with modern pseudospectral optimal control techniques to enable optimization of dynamic systems that feature complex interactions between subsystems.
Implicit pseudospectral approaches evaluate the ODE across an entire trajectory simultaneously.
While explicit time-marching requires the repeated convergence of small, dense systems of equations at a single instant in time, implicit pseudospectral methods converge a larger but more sparse system of equations once across the trajectory per evaluation of the ODE.
Computing the derivatives across iterative systems analytically does not require the systems to be reconverged, significantly reduces computational time during optimization.

Combining nonlinear solvers within the context of optimal-control problems grants the user a lot of flexibility in how to handle the direct collocation problems.
Typically, the collocation defects --- necessary to enforce the physics of the ODE --- are handled by assigning equality constraints to the optimizer.
However it is also possible to use a solver to converge the defects for every optimizer iteration, in effect creating a single or multiple shooting approach that may be beneficial for some problems.
If a solver-based collocation approach is used in combination with finite-differencing to approximate the derivatives needed by the optimizer, then the potential benefits are outweighed by numerical inaccuracies and high computational cost.
Dymos works around this problem by leveraging OpenMDAO's support for adjoint (reverse) differentiation to realize the benefits of shooting methods without a substantial performance penalty.

## Selected applications of dymos

Dymos has been used to demonstrate the coupling of flight dynamics and subsystem thermal constraints in electrical aircraft applications [@Falck2017a; @Hariton2020a].
NASA's X-57 "Maxwell" is using dymos for mission planning to maximize data collection while abiding the limits of battery storage capacity and subsystem temperatures [@Schnulo2018a; @Schnulo2019a].
Other authors have used dymos to perform studies of aircraft acoustics [@Ingraham2020a] and the the design of supersonic aircraft with thermal fuel management systems [@Jasa2018a].

## Example usage of dymos

As a simple use-case of dymos, consider the classic brachistochrone optimal control problem.
In this problem, we seek the shape of a frictionless wire strung between two points of different heights such that a bead sliding along the wire moves from the first point to the second point in minimum time.
To find the solution in dymos, we first define the ordinary differential equations that govern the motion of the bead.

```python
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
```

Having defined the ODE, we can now use Dymos to find the optimal time-history of the angle between the nadir and the wire (theta).

```python

p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
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
```

The resulting plots of the state and control histories are:

![Brachistochrone Solution](brach_plots.png)

# Acknowledgements

Dymos was developed with funding from NASA's Transformational Tools and Technologies ($T^3$) Project.

# References
