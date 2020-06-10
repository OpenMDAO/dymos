# Solving boundary-value problems with Dymos

!!! info "Things you'll learn through this example"
    - Targeting terminal conditions with Dymos using an optimization driver.

In the previous tutorial we propagated the state of a dynamic system for a fixed time.
In practical applications, it's more common to solve boundary-value problems in Dymos.

In the previous example the Phase object was used without explanation.
In Dymos, a _Phase_ represents a portion of a larger _Trajectory_.
Each _Phase_ of a trajectory may be subject to different equations of motion (a different ODE).
Also, _Phases_ may be used to pose boundary constraints on different parts of a Trajectory.

## Propagating the projectile until ground impact

The optimal control transcriptions used by Dymos have most commonly been used in conjunction with an optimizing driver.
Dymos has the capability to solve boundary value problems using only nonlinear solvers or through an optimization driver.
In this example we'll use an optimizer to vary the state history of the projectile until it:
1. Finds a state history which satisfies the ODE.
2. Finds a state history for which the final value of `y` is 0.

To do this, we need a single degree-of-freedom, some variable in the problem that can by varied to satisfy the constraint on the final value of `y`.
In this case, we're going to leave the initial state fixed and vary the duration of the phase such that the trajectory ends with a final `y` value of 0.

## Defining gradients of the ODE

The implicit optimization technique used by Dymos relies on gradients to reliably converge the solution.

!!! warning "**Warning: If gradients are not defined for the ODE, Dymos will not be able to solve the problem.**"

The following ODE is updated to include the partial derivatives of the ODE w.r.t. the inputs.
There are a few ways this could be done.
We could use finite differences or complex-step in the ODE system to approximate the partials, but in this case the partials are linear and it's easy to add them.
Since they're linear, the value of the derivative is provided in `declare_partials` and the `compute_partials` method is not necessary.

=== "projectile_ode.py"
{{ inline_source('dymos.examples.simple_projectile.doc.projectile_ode_with_partials',
include_def=True,
include_docstring=True,
indent_level=0,
highlight_lines=(23, 24, 25))
}}

## The run script to solve the boundary value problem

{{ embed_test('dymos.examples.simple_projectile.doc.test_doc_projectile.TestDocProjectile.test_bvp_driver_derivs') }}

Let's go over the key differences in this script and the previous script which solved the IVP.

1. This script now uses an optimizer to iterate the design variables of the problem such that the constraints are satisfied and the objective minimized.

    The design variables in this case are the duration of the phase (whose bounds are provided in the `set_time_options` method, and the state values at the state input nodes.

2. The `phase.set_time_options` method is used to inform the optimizer that the initial time is fixed (`fixed_initial=True`) and the duration of the phase is open but bounded (`duration_bounds=(5, 50)`)

    Dymos can propagate dynamics backwards in time by setting the duration to a negative value, but the collocation techniques break down when the duration of a phase is zero.

3. All state variables have fixed initial values, and so the `fix_initial` option is used to remove the initial state value as a design variable.

    State `y` also has a fixed final value.  Option `fix_final` is set to true to remove the final value as a design variable.
    Note that the values of the fixed ends will keep whatever value is initially provided to them by the user before running the driver.

3. Solving the problem using an optimizer requires an objective to be set.

    In this case it's acceptable to use the final `'time'` as the objective to be minimized, although the fact that the final altitude of the projectile is fixed means there should only be a single feasible time value.

4. In order to solve the problem, and not just compute the current objective and constraint values, the OpenMDAO `Problem.run_driver` method is used rather than `run_model`.

    Following the completion of the optimizer, the result should be a trajectory which propagates for 20.39 seconds obtains a final `y` value of 0.0 m.
