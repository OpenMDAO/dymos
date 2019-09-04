Constraints
-----------

Now that we've shown how to add degrees of freedom to a system with variables in the form of
time, states, and controls, we need to look at how to constrain the system.  In optimal control,
constraints typically come in two flavors:  boundary constraints and path constraints.

Boundary Constraints
~~~~~~~~~~~~~~~~~~~~

Boundary constraints are constraints on a variable value at the start or end of a phase.  There
are a few different ways to impose these in Dymos, each with slightly different behavior.

Lets consider that we want to solve for the elevation angle that results in the maximum possible
range flown by a cannonball.  In this situation we have some set of initial conditions that are
fixed.

.. math:

    t_0 = 0 s
    x_0 = 0 m
    y_0 = 0 m
    v_0 = 100 m/s

The first, most obvious way to constrain fixed values is to remove them from the optimization
problem altogether.

For time, this is done using the `opt_initial` or `opt_duration` arguments
to `set_time_options`.  This removes the OpenMDAO *IndepVarComp* which provides `t_initial` and
`t_duration` to the phase.  This also allows `t_initial` and `t_duration` to be provided from
an external source via connection, if so desired.

For states and controls, the situation is slightly different.  Rather than providing initial
and final values, similar to the way time is handled, the implicit simulation techniques must
be provided state values at the state discretization nodes and control values at *all* nodes.  The
option to specify initial and final values *by connection* therefore does not exist.  Instead,
for states and controls, the user specifies `fix_initial=True` or `fix_final=True`.  This leaves
the `IndepVarComp` in tact, but omits the first or last value of the variable as a design variable.

Removing constrained values from the optimization has the following pros and cons.  On the pro side,
we're making the optimization problem smaller by omitting them.  On the con side, the optimizer
has absolutely no freedom to move these values around even a little.  This can sometimes lead to
failure modes that aren't necessarily obvious, especially to new users.

The following example solves the brachistochrone problem by omitting the initial time and initial
state, as well as the final position state from the optimization.

[EMBED TEST 1]

The second method for bounding initial/final time, states, or controls is to leave them in the
optimization problem but to constrain only their initial or final values.  For time, this is
accomplished with the options `initial_bounds` and `duration_bounds`.  Each of these takes a tuple
of `(lower, upper)` values that the optimizer must obey when providing new variable values.  Note
that since states and controls may be vector valued, lower and upper may themselves be iterable.
To *pin* the value of a state, time, or control to a value just set lower and upper to the same
value.

As for the pros and cons of this technique, its largely similar to that for the first technique,
but it somewhat optimizer dependent.  Some optimizers *may* allow bounds on design variables to
be violated slightly (to some small tolerance).  In theory this could alleviate some of the issues
with omitting a design variable altogether, but in practice that's unlikely.

The first two options work by imposing bounds (or by not providing a variable to the optimizer
altogether).  The third option is to pose bound constraints as actual constraints on the NLP.
This is accomplished with the `add_boundary_constraint` method on Phases.

The downside of this technique is that it makes the NLP problem larger, though not by much.  On
the plus side, this method allows the user to constrain any output within the ODE.  If the user
needs to constrain an auxiliary output, this is the only option.  It may also behave somewhat better
in certain circumstances.  Depending on scaling, the NLP may ensure that collocation defects are
satisfied before forcing an infeasible boundary constraint to be satisfied, for instance.

[EMBED TEST 2]

In conclusion, while using `fix_initial=True` or `opt_initial=False` for problems with fixed initial
conditions is not a bad solution, the generality of `add_boundary_constraint`, especially for
terminal constraints that risk being overconstrained makes it a good first-choice in those.
situations.  Also, explicitly integrated phases fundamentally are solving an initial value problem.
As such simple bounds on final state values are not possible in those situations, and
`add_boundary_constraint` must be used instead.

Path Constraints
~~~~~~~~~~~~~~~~

The second class of constraints supported by Dymos are *path* constraints, so called because
they are imposed throughout the entire phase.  Like bound constraints, path constraints can be
imposed on design variables using simple bounds.  This is accomplished using the `lower` and `upper`
arguments to `add_state`, `add_control`, and `add_design_parameter`.
(Since time is monotonically increasing or decreasing the notion of a path constraint is
irrelevant for it).

For vector-valued states and controls, lower/upper should be dimensioned the same as state or
control.  If given as a scalar, it will be applied to all values in the state or control.

.. note::
    Bounds on states in Gauss-Lobatto Phases are **not** equivalent to path constraints.  The values
    of states in Gauss-Lobatto phases are provided at only the state-transcription nodes and then
    interpolated to the collocation nodes.  Therefore, the bounds will have no impact on these
    interpolated values which therefore may not satisfy the bounds, as one might expect.

Phases also support the `add_path_constraint` method, which imposes path constraints as constraints
in the NLP problem.  As with `add_bound_constraint`, the `add_path_constraint` method is the only
option for path constraining an output of the ODE.  The downside of path constraints is that they
add a considerable number of constraints to the NLP problem and thus may negatively impact
performance.

.. note::
    Path constraints are imposed at all nodes in the Phase.  This, in addition
    to dynamic control transcription, is the reason why the *grid* definition is important even in
    explicitly integrated Phases.
