# The Van der Pol Oscillator

In dynamics, the Van Der Pol oscillator[1] is a non-conservative oscillator with non-linear damping.
It evolves in time according to the second-order differential equation:

\begin{align}
    \frac{d^2x}{dt^2} - u (1 - x^2) \frac{dx}{dt} + x &= 0
\end{align}

where $x$ is the position coordinate (a function of the time $t$), and $u$ is a scalar parameter
indicating the nonlinearity and the strength of the damping.

To make this an optimal control problem, we want to find the smallest control that will dampen the oscillation
(drive the state variables to zero). We can express this as an objective function $J$ to minimize:

\begin{align}
    J &= \int x^2_0 + x^2_1 + u^2
\end{align}

In other words, we want to find the optimal (smallest) trajectory of the control $u$ such that the oscillation
and the oscillation's rate of change are driven to zero.

## State Variables

There are three _state_ variables are used to define the configuration of the system at any given instant in time.

- $x_1$: The primary output of the oscillator.
- $x_0$: The rate of change of the primary output.
- $J$: The objective function to be minimized.

The objective function is included as a state variable so that Dymos will do the integration.

The $x_1$ and $x_0$ state variables are also inputs to the system, along with the control $u$.

## System Dynamics

The evolution of the state variables is given by the following ordinary differential equations (ODE):

\begin{align}
    \frac{dx_0}{dt} &= (1 - x^2_1) x_0 - x_1 + u \\
    \frac{dx_1}{dt} &= x_0 \\
    \frac{dJ}{dt} &= x^2_0 + x^2_1 + u^2
\end{align}

## Control Variables

This system has a single control variable:

- $u$: The control input.

The control variable has a constraint: $-0.75 \leq u \leq 1.0$

## The initial and final conditions

The initial conditions are:

\begin{align}
    x_0 &= 1 \\
    x_1 &= 1 \\
      u &= -0.75
\end{align}

The final conditions are:

\begin{align}
    x_0 &= 0 \\
    x_1 &= 0 \\
      u &= 0
\end{align}

## Defining the ODE as an OpenMDAO System

In Dymos, the ODE is an OpenMDAO System (a Component, or a Group of components).
The following _ExplicitComponent_ computes the state rates for the Van der Pol problem.

More detail on the workings of an _ExplicitComponent_ can be found in the OpenMDAO documentation.  In summary:

- **initialize**:  Called at setup, and used to define options for the component.  **ALL** Dymos ODE components
  should have the property `num_nodes`, which defines the number of points at which the outputs are simultaneously computed.
- **setup**: Used to add inputs and outputs to the component, and declare which outputs (and indices of outputs)
  are dependent on each of the inputs.
- **compute**: Used to compute the outputs, given the inputs.
- **compute_partials**: Used to compute the derivatives of the outputs with respect to each of the inputs analytically.
  This method may be omitted if finite difference or complex-step approximations are used, though analytic is recommended.

!!! note "Things to note about the Van der Pol ODE system"
    - Only the _vanderpol_ode_ class below is important for defining the basic problem. The other classes are
      used to demonstrate Message Passing Interface (MPI) parallel calculation of the system. They can be ignored.
    - $x_1$, $x_0$, and $u$ are inputs.
    - $\dot{x_1}$, $\dot{x_0}$, and $\dot{J}$ are outputs.
    - **declare_partials** is called for every output with respect to every input.
    - For efficiency, partial derrivatives that are constant have values specified in the **setup** method rather than
      the **compute_partials** method. So although 9 partials are declared, only 5 are computed in **compute_partials**.

=== "vanderpol_ode.py"
{{ inline_source('dymos.examples.vanderpol.vanderpol_ode',
include_def=True,
include_docstring=True,
indent_level=0)
}}

## Defining the Dymos Problem

Once the ODEs are defined, they are used to create a Dymos _Problem_ object that allows solution.

!!! note "Things to note about the Van der Pol Dymos Problem definition"
    - The **vanderpol** function creates and returns a Dymos _Problem_ instance that can be used
      for simulation or optimization.
    - The **vanderpol** function has optional arguments for specifying options for the
      type of transcription, number of segments, optimizer, etc. These can be ignored
      when first trying to understand the code.
    - The _Problem_ object has a _Trajectory_ object, and the trajectory has a single _Phase_.
      Most of the problem setup is performed by calling methods on the phase (**set_time_options**,
      **add_state**, **add_boundary_constraint**, **add_objective**).
    - The **add_state** and **add_control** calls include the _target_ parameter for $x_0$, $x_1$, and $u$.
      This is required so that the inputs are correctly calculated.
    - Initial (linear) guesses are supplied for the states and control.

=== "vanderpol_dymos.py"
{{ inline_source('dymos.examples.vanderpol.vanderpol_dymos',
include_def=True,
include_docstring=True,
indent_level=0)
}}

## Simulating the Problem (without control)

The following script creates an instance of the Dymos vanderpol problem and simulates it.

Since the problem was only simulated and not solved, the solution lines in the plots show only
the initial guesses for $x_0$, $x_1$, and $u$. The simulation lines shown in the plots are the
system response with the control variable $u$ held constant.

The first two plots shows the variables $x_0$ and $x_1$ vs time. The third plots shows $x_0$ vs. $x_1$
(which will be mostly circular in the case of undamped oscillation). The final plot is the (fixed)
control variable $u$ vs time.

{{ embed_test('dymos.examples.vanderpol.doc.test_doc_vanderpol.TestVanderpolForDocs.test_vanderpol_for_docs_simulation') }}

## Solving the Optimal Control Problem

The next example shows optimization followed by simulation.

With a successful optimization, the resulting plots show a good match between the simulated (with varying control)
and optimized results. The state variables $x_0$ and $x_1$ as well as the control variable $u$ are all driven to zero.

{{ embed_test('dymos.examples.vanderpol.doc.test_doc_vanderpol.TestVanderpolForDocs.test_vanderpol_for_docs_optimize') }}

## Solving the Optimal Control Problem with Grid Refinement

Repeating the optimization with grid refinement enabled requires changing only two lines in the code. For the sake
of grid refinement demonstration, the initial number of segments is also reduced by a factor of 5.

Optimization with grid refinement gets results similar to the example without grid refinement, but runs faster
and does not require supplying a good guess for the number segments.

{{ embed_test('dymos.examples.vanderpol.doc.test_doc_vanderpol.TestVanderpolForDocs.test_vanderpol_for_docs_optimize_refine') }}

## References
[1] Van Der Pol oscillator description from [Wikipedia](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator)
