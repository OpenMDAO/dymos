Objective
---------

Optimal control problems in Dymos are posed within the context of *phases*.

When using an optimizer-based phase, like *Gauss-Lobatto* or *Radau-Pseudospectral*,
the user is **required** to provide an objective, even for a simple initial value problem.

To support objectives, phases use an overridden version of OpenMDAO's `add_objective` method.
This method handles a few complexities that would be involved in the standard OpenMDAO
`add_objective` method.

First, since the path to a variable within the phase might depend on the transcription used, the
`add_objective` method uses the same name-detection capability as other phase methods like
`add_boundary_constraint` and `add_path_constraint`.  The name provided should be one of

- `time`
- the name of a state varaible
- the name of a control variable
- the name of a control rate or rate2 (second derivative)
- the path of an output in the ODE relative to the top level of the ODE

Dymos will find the full path of the given variable and add it to the problem as an objective.

.. note::

    Many optimizers do not support multiple objective functions.  When constructing a composite
    objective you may need to form the objective as an output of a component in your ODE system.

Second, unlike typical OpenMDAO problems where the `index` can be used to effectively specify
the first or last value of a variable, optimal control problems have two competing notions of index:
the first is the location in time where the objective is to be measured, and the second is the index of a
vector valued variable that is to be considered the objective value, which must be scalar.

To remove this ambiguity, the `add_objective` method on phase has an argument `loc`, which may
have value `initial` or `final`, specifying whether the objectve is to be quantified at the
start or end of the phase.  The `index` option gives the index into a non-scalar variable value
to be used as the objective, which must be scalar.

Example: Minimizing Final Time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    phase.add_objective('time', loc='final')


Example: Maximizing Final Mass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example assumes that the phase has a state variable named *mass*.

.. code-block:: python

    phase.add_objective('mass', loc='final', scaler=-1)
