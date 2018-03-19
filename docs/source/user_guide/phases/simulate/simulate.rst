==========================
Simulating ODEs with Scipy
==========================

|project| phases offer the ability to simulate the ODE using the current solution using the
methods available in `scipy.integrate`.  This has a variety of useful applications.

First, propagating the ODE as an initial value problem with a given control time-history can be
useful in demonstrating that the various components in the ODE have been coded correctly.  A system
that does not properly propagate the dynamics of the system as an IVP has virtually no chance of
successfully solving an optimal control problem.

Secondly, the implicit integration techniques employed in |project| often benefit from a reasonable
initial guess to the time-histories of the states and controls.  By providing a numerically
integrated time-history of the states in the system, the collocation defect constraints should all
be satisfied at the driver's starting point.  In fact, if any defect constraints are initially
unsatisfied here it is an indication that the phase has an insufficient grid (the number of
segments is too low or the transcription order of the segments is too low).  The ability to start
from numerically propagated solutions is particularly useful for the high-order Gauss-Lobatto
method, where invalid dynamics at the discretization nodes can lead to wildly inaccurate state
values at the collocation nodes due to the interpolation step.

Finally, the accuracy of implicit integration schemes is a function of the collocation grid (the
number of segments and the polynomial order on each segment).  In this regard, using simulate
with a variable step integrator can be seen as a "truth model" of sorts.  If the state values at
the discrete points used in the collocation technique overlaps the explicitly integrated solution,
it provides confidence in the accuracy of the implicit solution.  If the solutions diverge over time
or simply fail to match at certain points, it's an indication that grid refinement is necessary.

.. warning::

    The simulate method of Phase should **not** be used in an optimization context.  The purpose
    of the phase transcriptions used in |project| is to provide analytic derivatives **across** the
    integration.  This capability is important when applying gradient-based optimization.  For
    instance, the influence of a control value at a particular time on the objective function can
    be quite small.  If this influence is small, it can easily be drowned out by
    errors when trying to apply finite differencing across the integration, leading the optimizer
    to converge slowly or, worse, not converge at all.  Using analytic derivatives typically
    allows convergence in fewer iterations.

Example:  Using `simulate` to check the solution
------------------------------------------------

Consider the single-stage-to-orbit (SSTO) example.  In the following code, we solve the problem.
We then call simulate and capture the returned `SimulationOutput` object as `exp_out`.

The SimulationObject supports the same `get_values` interface as the Phase class, so visually
comparing the results is easy.

.. code-block:: python

    t0 = p['phase0.t_initial']
    tf = t0 + p['phase0.t_duration']
    exp_out = phase.simulate(times=np.linspace(t0, tf, 100))
    print(exp_out.get_values('vy'))

By default, the filename of the simulation is `<phase_name>_sim.db` where `phase_name` is the
name assigned to the phase object when it was added to the problem (`phase0` in the example above).
The user can override this filename by explicitly providing one via the `record_file` to the
simulate method.

.. code-block:: python

    t0 = p['phase0.t_initial']
    tf = t0 + p['phase0.t_duration']
    phase.simulate(times=np.linspace(t0, tf, 50), record_file='my_simulation.db')

Note that the simulate method also saves an OpenMDAO recording file of the explicitly integrated
solution.  We can load data from a previous explicit simulation by instantiating the
SimulationResults with the filename as its only argument.

.. code-block:: python

    exp_out = SimulationResults('my_simulation.db')
    print(exp_out.get_values('x'))

In the following example, we minimize the time required for a single stage vehicle to reach orbit.
Since it is a constant thrust/mass flow case, this is also the minimum propellant solution.

.. embed-code::
    dymos.examples.ssto.ex_ssto_earth
    :layout: interleave
