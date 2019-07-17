==========================================================
Tandem Phases:  Using different ODE simultaneously in time
==========================================================

Complex models sometimes encounter state variables which are best simulated on different time
scales, with some state variables changing quickly (fast variables) and some evolving slowly (slow variables).
For instance, and aircraft trajectory optimization which includes vehicle component temperatures might
see relatively gradual changes in altitude over the course of a two hour flight while temperatures of
some components seem to exhibit step-function-like behavior on the same scale.

To accommodate both fast and slow variables in the same ODE, one would typically need to use a _dense_
grid (with many segments/higher order segments).  This can be unnecessarily burdensome when there are
many slow variables or evaluating their rates is particularly expensive.

As a solution, Dymos allows the user to run two phases over the same range of times, where one
phase may have a more sparse grid to accommodate the slow variables, and one has a more dense grid
for the fast variables.

To connect the two phases, state variable values are passed from the first (slow) phase to the second
(fast) phase as non-optimal dynamic control variables.  These values are then used to evaluate the
rates of the fast variables.  Since outputs from the first phase in generally will not fall on the
appropriate grid points to be used by the second phase, interpolation is necessary.  This is one
application of the interpolating timeseries component.

In the following example, we solve the brachistochrone problem but do so to minimize the arclength
of the resulting wire instead of the time required for the bead to travel along the wire.
This is a trivial solution which should find a straight line from the starting point to the ending point.
There are two phases involved, the first utilizes the standard ODE for the brachistochrone problem.
The second integrates the arclength (:math:`S`) of the wire using the equation:

.. math::

    S = \int v \sin \theta  \sqrt{1 + \frac{1}{\tan^2 \theta}} \, dt

.. embed-code::
    dymos.examples.brachistochrone.doc.test_doc_brachistochrone_tandem_phases.BrachistochroneArclengthODE
    :layout: code

The trick is that the bead velocity (:math:`v`) is a state variable solved for in the first phase,
and the wire angle (:math:`\theta`) is a control variable "owned" by the first phase.  In the
second phase they are used as control variables with option ``opt=False`` so that their values are
expected as inputs for the second phase.  We need to connect their values from the first phase
to the second phase, at the :code:`control_input` node subset of the second phase.

In the following example, we instantiate two phases and add an interpolating timeseries to the first phase
which provides outputs at the :code:`control_input` nodes of the second phase.  Those values are
then connected and the entire problem run. The result is that the position and velocity variables
are solved on a relatively coarse grid while the arclength of the wire is solved on a much denser grid.

.. embed-code::
    dymos.examples.brachistochrone.doc.test_doc_brachistochrone_tandem_phases.TestDocTandemPhases.test_tandem_phases_for_docs
    :layout: code, plot
