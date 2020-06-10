# Modeling Dynamic Systems with Dymos

!!! info "Things you'll learn through this example"
    - How to define a basic Dymos ODE system.
    - How to explictly propagate the system from some initial state.

Dymos is a library for modeling dynamic systems and performing optimal
control with the [OpenMDAO](https://github.com/OpenMDAO/OpenMDAO) framework.
Dynamic systems are typically defined by some set of ordinary
differential equations (the ODE) which governs their behavior.

Consider the path of a projectile moving in a constant, two-dimensional gravity field where gravity is the only force acting upon it.

[diagram]

There are a few ways we could define this system, but one of the simplest ways is to define the position and velocity of the projectile in terms of 2D Cartesian coordinates

\begin{align}
    \dot{x} &= v_x \\
    \dot{y} &= v_y \\
    \dot{v}_x &= 0 \\
    \dot{v}_y &= -g
\end{align}

In Dymos, the ODE is an OpenMDAO System (a Component, or a Group of components).
The following ExplicitComponent computes the state rates for this simple projectile.

More detail on the workings of an ExplicitComponent can be found in the OpenMDAO documentation.  In summary, an ExplicitComponent used as part of an ODE in Dymos should override the following methods:

- **initialize**:  Called at setup, and used to define options for the component.  **ALL** Dymos ODE components should have the property `num_nodes`, which defines the number of points at which the outputs are simultaneously computed.
- **setup**: Used to add inputs and outputs to the component, and declare which outputs (and indices of outputs) are dependent on each of the inputs.
- **compute**: Used to compute the outputs, given the inputs.
- **compute_partials**: Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically.  This method may be omitted if finite difference or complex-step approximations are used, though analytic is recommended.

=== "projectile_ode.py"
{{ inline_source('dymos.examples.simple_projectile.doc.projectile_ode',
include_def=True,  
include_docstring=True,
indent_level=0)
}}

!!! note "Things to note about the ODE system"
    - There is no input for the position states ($x$ and $y$).  The dynamics aren't functions of these states, so they aren't needed as inputs.
    - While $g$ is an input to the system, since it will never change throughout the trajectory, it can be an option on the system.  This way we don't have to define any partials w.r.t. $g$.

## Hello World: Propagating the Projectile

One of the first things one might be interested in regarding an ODE is propagating it from some given initial conditions.
This is known as solving the initial value problem (IVP), and there are many software packages that can do this.
The following is a minimal script that starts the system at some set of initial conditions and propagates them for some fixed duration.
Some elements of this code will be explained later, but we'll hit the highlights now.

{{ embed_test('dymos.examples.simple_projectile.doc.test_doc_projectile.TestDocProjectile.test_ivp') }}

## What happened?

This script consists of the following general steps:

1. After importing relevant packages, an OpenMDAO Problem is instantiated
2. A Dymos Trajectory object is instantiated, and a single Phase named `'phase0'` is added to it.
   1. That Phase takes the ODE _class_ as one of its arguments.  It will instantiate instances of it as needed.
   2. The transcription determines how the implicit integration and optimization are performed.  It's necessary but not particularly relevant in this example.
3. The states to be integrated are added to the phase.
   1. Each state needs a rate source - an ODE-relative path of the output which provides the time derivative of the state variable.
   2. Those states which are inputs to the ODE need to provide their targets in the ODE (again, with an ODE-relative path).
4. The problem is setup (this prepares the model for execution in OpenMDAO)
5. Default values are assigned to the states and time.
   1. Variables `t_initial` and `t_duration` (the initial time and duration of the Phase) are scalars.
   2. State variables are vectors whose values are provided throughout the Phase.  Here they're all being filled with a single value (0.0 for the position components, 100.0 for the velocity components)
6. `Problem.run_model` is called.  This executes the Problem's `model` one time.  This is a necessary step before using the `Trajectory.simulate` method.
7.  The trajectory is simulated and the results returned as an OpenMDAO Problem instance called `sim_out`.
8.  The states are plotted using values obtained from the _timeseries_.  Phases contain `_timeseries_` data that provides contiguous values regardless of the transcription used.

Method `simulate` exists on both Trajectory and Phase objects.
It uses the [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) function to propagate the states defined in each phase of the trajectory from their initial values at the initial time to some final value at `time = t_initial + t_duration`.

In Dymos, the `simulate` method is useful for testing the functionality of an ODE (making sure it behaves as expected) and for checking the validity of answers after optimization.
However, it cannot solve boundary value problems.
For instance, we may be interested in determining how long it takes this projectile to impact the ground given its initial conditions.
In the next step, we'll use the implicit techniques in Dymos to solve that problem.

## Why is the solution different from the simulation results?

The plots above display both the solution from the implicit transcription (blue dots) and the results of the simulation (orange line).
Here they do not match because we only performed a single execution of the model.
**The purpose of a model execution in Dymos is to calculate the objective and constraints for the optimizer.**
These constraints include the collocation _defect_ constraints, which (when driven to zero) indicate that the current polynomial representation of the state-time history matches the physically correct trajectory.
In this case, no iteration was performed, and thus the solution is not physically valid.


