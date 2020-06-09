# An Introduction to Modeling Dynamic Systems with Dymos

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

## What `simulate` does

Method `simulate` exists on both Trajectory and Phase objects.


