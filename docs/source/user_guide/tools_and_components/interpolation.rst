=======================
Interpolation Utilities
=======================

The direct optimal control techniques employed by |project| are more likely to be
successful with a reasonable initial guess.  One way to accomplish this is to use
the following procedure for optimizing a case:

#. Provide guesses for phase times, *initial* state values, and a notional control history.
#. Use ``phase.simulate(nodes='disc')`` to simulate the phase and have the results populated at the state discretization nodes.
#. Copy these state values at the state discretization nodes back into the problem.

[TODO Example]

4. Call run-driver to optimize the problem

What this approach does:
------------------------

The implicit collocation techniques drive *defect constraints* to zero during the course of an
optimization.  When those constraints are zero, the time-histories of the states are *correct* to
the extent that the current grid (number of segments, segment orders, and segment distribution) allows.

By initializing the optimization with the results of a physical simuation, we are giving the optimize
a guess in which the defect constraints are largely satisfied.  In fact, if you use this technique and
find that the defect constraints are not initially satisfied, this suggests that the current grid
is not capable of accurately representing the dynamics of the problem.

Barring a miracle or a lot of luck, the guessed control profile will not be optimal, but the
initial trajectory will be physically compatible with those controls.

So how do you specify a control profile guess?
----------------------------------------------

|project| phases provide an interpolation method `phase.interpolate`.  Interpolate takes
three arguments:

- xs
    Optional.  Values of the independent variable at the given y-values.  If only two values
    are provided to ys this can be omitted and will assume the values [-1, 1] for linear
    interpolation.
- ys
    Values to be interpolated.  If only two values are given, linear interpolation is assumed.
- nodes
    Specifies the nodes onto which the interpolation should be performed.  Option 'all' returns
    interpolated values at all nodes in the phase.  Option 'disc' provides interpolated values
    at the state discretization nodes, and option 'col' provides interpolated values at the
    collocation nodes.

This method is necessary because the distribution of nodes across a phase is dependent on the
grid definition (number of segments, segment orders, and segment distribution).  Using something
like `numpy.linspace` would provide values linearly interpolated onto evenly spaced nodes, but
the nodes in the phase are typically not evenly spaced.

In practice, use `nodes='all'` for interpolating control values and `nodes='disc'` for interpolating
state values.

[TODO Example]

.. toctree::
    :maxdepth: 2
