# Tutorial: Solving the brachistochrone optimal control problem with Dymos

The brachistochrone is one of the most well-known optimal control problems.
It was originally posed as a challenge by Johann Bernoulli.

!!! note "The brachistochrone problem"
    _Given two points A and B in a vertical plane, find the path AMB
    down which a movable point M must by virtue of its weight fall from
    A to B in the shortest possible time._

    -  Johann Bernoulli, Acta Eruditorum, June 1696

We seek to find the optimal shape of a wire between two points (A and B) such that a bead sliding
without friction along the wire moves from point A to point B in minimum time.

{{ embed_plot_from_script('scripts/brachistochrone_fbd.py') }}

## State variables

In this implementation, three _state_ variables are used to define the configuration of the system at any given instant in time.

- **x**: The horizontal position of the particle at an instant in time.
- **y**: The vertical position of the particle at an instant in time.
- **v**: The speed of the particle at an instant in time.

## System dynamics

From the free-body diagram above, the evolution of the state variables is given by the following ordinary differential equations (ODE).

\begin{align}
    \frac{d x}{d t} &= v \sin(\theta) \\
    \frac{d y}{d t} &= v \cos(\theta) \\
    \frac{d v}{d t} &= g \cos(\theta)
\end{align}

## Control variables

This system has a single control variable.

- **$\theta$**: The angle between the gravity vector and the tangent to the curve at the current instant in time.

## The initial and final conditions

In this case, starting point **A** is given as _(0, 10)_.
The point moving along the curve will begin there with zero initial velocity.

The initial conditions are:

\begin{align}
    x_0 &= 0 \\
    y_0 &= 10 \\
    v_0 &= 0
\end{align}

The end point **B** is given as _(10, 5)_.
The point will end there, but the velocity at that point is not constrained.

The final conditions are:

\begin{align}
    x_f &= 10 \\
    y_f &= 5 \\
    v_f &= \mathrm{free}
\end{align}

## Defining the ODE as an OpenMDAO System

In Dymos, the ODE is an OpenMDAO System (a Component, or a Group of components).
The following ExplicitComponent computes the state rates for the brachistochrone problem.

More detail on the workings of an ExplicitComponent can be found in the OpenMDAO documentation.  In summary:

- **initialize**:  Called at setup, and used to define options for the component.  **ALL** Dymos ODE components should have the property `num_nodes`, which defines the number of points at which the outputs are simultaneously computed.
- **setup**: Used to add inputs and outputs to the component, and declare which outputs (and indices of outputs) are dependent on each of the inputs.
- **compute**: Used to compute the outputs, given the inputs.
- **compute_partials**: Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically.  This method may be omitted if finite difference or complex-step approximations are used, though analytic is recommended.

=== "brachistochrone_ode.py"
{{ inline_source('dymos.examples.brachistochrone.doc.brachistochrone_ode',
include_def=True,  
include_docstring=True,
indent_level=0)
}}

!!! note "Things to note about the ODE system"
    - There is no input for the position states ($x$ and $y$).  The dynamics aren't functions of these states, so they aren't needed as inputs.
    - While $g$ is an input to the system, since it will never change throughout the trajectory, it can be an option on the system.  This way we don't have to define any partials w.r.t. $g$.
    - The output `check` is an _auxiliary_ output, not a rate of the state variables.  In this case, optimal control theory tells us that `check` should be constant throughout the trajectory, so it's a useful output from the ODE.

## Testing the ODE

Now that the ODE system is defined, it is strongly recommended to test the analytic partials before using it in optimization.
If the partials are incorrect, then the optimization will almost certainly fail.
Fortunately, OpenMDAO makes testing derivatives easy with the `check_partials` method.
The `assert_check_partials` method in `openmdao.utils.assert_utils` can be used in test frameworks to verify the correctness of the partial derivatives in a model.

The following is a test method which creates a new OpenMDAO problem whose model contains the ODE class.
The problem is setup with the `force_alloc_complex=True` argument to enable complex-step approximation of the derivatives.
Complex step typically produces derivative approximations with an error on the order of 1.0E-16, as opposed to ~1.0E-6 for forward finite difference approximations.

=== "test_brachistochrone_partials.py"  
{{ inline_source('dymos.examples.brachistochrone.doc.test_doc_brachistochrone.TestBrachistochrone.test_brachistochrone_partials',
include_def=True,  
include_docstring=True,  
indent_level=0)  
}}

## Solving the Problem

The following script fully defines the brachistochrone problem with Dymos and solves it.  In this section we'll walk through each step.

=== "brachistochrone.py"
{{ inline_source('dymos.examples.brachistochrone.doc.test_doc_brachistochrone.TestBrachistochrone.test_brachistochrone',
include_def=False,
include_docstring=False,
indent_level=0)
}}
