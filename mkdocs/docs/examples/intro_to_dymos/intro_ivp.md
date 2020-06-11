# Modeling Dynamic Systems with Dymos

!!! info "Things you'll learn through this example"
    - How to define a basic Dymos ODE system.
    - How to explictly propagate the system from some initial state.

Dymos is a library for modeling dynamic systems and performing optimal
control with the [OpenMDAO](https://github.com/OpenMDAO/OpenMDAO) framework.
Dynamic systems are typically defined by some set of ordinary
differential equations (the ODE) which governs their behavior.

Consider a simple damped harmonic oscillator.


![Damped harmonic oscillator free-body diagram](figures/spring_mass_damper.png)


\begin{align}
    \ddot{x} &= -\frac{kx}{m} - - \frac{c \dot{x}}{m}
\end{align}

Converting this to a first order system results in an ODE system with two states:

\begin{align}
    \dot{x} &= v \\
    \dot{v} &= -\frac{kx}{m} - \frac{c \dot{x}}{m}
\end{align}

## The OpenMDAO model of the ODE

In Dymos, the ODE is an OpenMDAO System (a Component, or a Group of components).
The following ExplicitComponent computes the velocity rate for the damped harmonic oscillator.

More detail on the workings of an ExplicitComponent can be found in the OpenMDAO documentation.  In summary, an ExplicitComponent used as part of an ODE in Dymos should override the following methods:

- **initialize**:  Called at setup, and used to define options for the component.  **ALL** Dymos ODE components should have the property `num_nodes`, which defines the number of points at which the outputs are simultaneously computed.
- **setup**: Used to add inputs and outputs to the component, and declare which outputs (and indices of outputs) are dependent on each of the inputs.
- **compute**: Used to compute the outputs, given the inputs.
- **compute_partials**: Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically.  This method may be omitted if finite difference or complex-step approximations are used, though analytic is recommended.

=== "oscillator_ode.py"
{{ inline_source('dymos.examples.oscillator.doc.oscillator_ode',
include_def=True,  
include_docstring=True,
indent_level=0)
}}

!!! note "Things to note about the ODE system"
    - In this case, the ODE is a function of both states, but this isn't always the case.  If the dynamics aren't functions of some states, those states aren't needed as inputs.
    - This ODE only computes the rate of change of velocity.  Since the rate of change of displacement can directly be obtained from another state variable, it doesn't need to be computed by the ODE.
      This would also be true if the state's rate was a control, design parameter, or input parameter value.
    - It's possible that we might want to use parameters `c`, `k`, and `m` as design variables at some point, so they're also inclu;ded as inputs here.
      Alternatively, if we had no interest in ever treating them as design variables, we could add their values as options to the ODE system in the `initialize` method.

## Hello World: Propagating the dynamics

One of the first things one might be interested in regarding an ODE is propagating it from some given initial conditions.
This is known as solving the initial value problem (IVP), and there are many software packages that can do this.
The following is a minimal script that starts the system at some set of initial conditions and propagates them for some fixed duration.
Some elements of this code will be explained later, but we'll hit the highlights now.

{{ embed_test('dymos.examples.oscillator.doc.test_doc_oscillator.TestDocOscillator.test_ivp') }}

## What happened?

This script consists of the following general steps:

1. After importing relevant packages, an OpenMDAO Problem is instantiated
2. A Dymos Trajectory object is instantiated, and a single Phase named `'phase0'` is added to it.
   1. That Phase takes the ODE _class_ as one of its arguments.  It will instantiate instances of it as needed.
   2. The transcription determines how the implicit integration and optimization are performed.  It's necessary but not particularly relevant in this example.
3. The states to be integrated are added to the phase.
   1. Each state needs a rate source - an ODE-relative path of the output which provides the time derivative of the state variable.
      For the rate_source of `x`, we provide `v`.  Dymos understands this is the name of one of the other states (or time, or controls, or parameters).
   2. Those states which are inputs to the ODE need to provide their targets in the ODE (again, with an ODE-relative path).
4. The problem is setup (this prepares the model for execution in OpenMDAO).
5. Default values are assigned to the states and time.
   1. Variables `t_initial` and `t_duration` (the initial time and duration of the Phase) are scalars.
   2. State variables are vectors whose values are provided throughout the Phase.  Here they're all being filled with a single value (10.0 for displacement, 0.0 for the velocity)
6. `Problem.run_model` is called.  This executes the Problem's `model` one time.  This is a necessary step before using the `Trajectory.simulate` method.
7.  The trajectory is simulated using the `simulate()` method and the results returned as an OpenMDAO Problem instance called `sim_out`.
8.  The states are plotted using values obtained from the _timeseries_.  Phases contain `_timeseries_` data that provides contiguous values regardless of the transcription used.

Method `simulate` exists on both Trajectory and Phase objects.
It uses the [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) function to propagate the states defined in each phase of the trajectory from their initial values at the initial time to some final value at `time = t_initial + t_duration`.

In Dymos, the `simulate` method is useful for testing the functionality of an ODE (making sure it behaves as expected) and for checking the validity of answers after optimization.
However, it cannot solve boundary value problems.

## Why is the solution different from the simulation results?

The plots above display both the solution from the implicit transcription (blue dots) and the results of the simulation (orange line).
Here they do not match because we only performed a single execution of the model.
**The purpose of a model execution in Dymos is to calculate the objective and constraints for the optimizer.**
These constraints include the collocation _defect_ constraints, which (when driven to zero) indicate that the current polynomial representation of the state-time history matches the physically correct trajectory.
In this case, no iteration was performed, and thus the solution is not physically valid.

To be clear, the output of Dymos in this case is not a physcially valid trajectory.
The `simulate()` call after executing the model is the expected result using the variable step integrator from Scipy.

There are two ways to converge this solution using the implicit transcription techniques in Dymos.
1. We can run an optimization driver with some "dummy" objective to converge the collocation defect constraints.
2. We can have Dymos use a nonlinear solver to vary the state time-history until the colloction defect constraints are satisfied.

Traditionally, many collocation optimal control techniques have use an optimizer-based approach because it is extremely efficient.
OpenMDAO provides a lot of versatility in adding in nonlinear solvers within the optimization problem.
In our case, using a solver to converge state-time history means that the optimizer "sees" a physical trajectory at every iteration.
In an optimization context, we can use the solver-based convergence of defects to obtain a shooting method with analytic derivatives.

## Using a solver to converge the physical trajectory

We let Dymos know that one or more states should be converged using the `solve_segments=True` argument.
If passed to the transcription, it applies to all states.
Otherwise, we can pass it only to certain states as an argument to `add_state` or `set_state_options`.

{{ embed_test('dymos.examples.oscillator.doc.test_doc_oscillator.TestDocOscillator.test_ivp_solver') }}

## Using an optimization driver to converge the physical trajectory

Alternatively, we can use an optimization driver to converge the state time histories.


In the case of an initial value problem (fixed time duration, fixed initial states, and no controls or parameters as design variables) there are no
degrees of freedom to optimize the problem, just single possible trajectory which satisfies the collocation constraints.


In OpenMDAO (and thus Dymos) optimizers require an objective.
Even though the initial time and duration of the phase are fixed, we provide the final time as a "dummy" objective here.

{{ embed_test('dymos.examples.oscillator.doc.test_doc_oscillator.TestDocOscillator.test_ivp_driver') }}

## But the solution still doesn't match the simulation

If you look at the plots from the last two examples, you'll notice that the state time-history from the solution has some pretty significant deviations from the simulation results.
This is an important lesson in using implicit collocation techniques.

!!! warning "A converged trajectory isn't necessarily correct"

As we mentioned before, the `simulate()` method exists to provide a check on a converged trajectory.
In this case, the trajectory found using `simulate()` doesn't really interpolate the solution from the collocation technique.
In the next section, we'll explain how to deal with this.
