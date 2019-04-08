=================
Double Integrator
=================

In the double integrator problem, we seek to maximize the distance traveled by a block,
sliding without friction along a horizontal surface, with acceleration as the control.

We minimize the final time, :math:`t_f`, by varying the dynamic control, :math:`\theta`, subject to the dynamics:

.. math ::
    \frac{d x}{d t} &= v \\
    \frac{d v}{d t} &= u \\

The initial conditions are

.. math ::
    x_0 &= 0 \\
    v_0 &= 0 \\

and the final conditions are

.. math ::
    x_f &= \mathrm{free} \\
    v_f &= 0

The control :math:`u` is constrained to fall between -1 and 1.  Due to the fact that the
control appears linearly in the equations of motion, we should expect "bang-bang" behavior
in the control.

1. The ODE System: double_integrator_ode.py
-------------------------------------------

This problem is unique in that we don't actually have to calculate anything in the ODE.  For the
sake of Dymos, we create an ExplicitComponent and provide it with the `num_nodes` option, but
it has no inputs and no outputs.  The rates for the states are entirely provided by the other states
and controls.

.. embed-code::
    dymos.examples.double_integrator.double_integrator_ode
    :layout: code

2. Building and running the problem
-----------------------------------

In order to facilitate the bang-bang behavior in the control we disable continuity and rate continuity
in the control value.

.. embed-code::
    dymos.examples.double_integrator.doc.test_doc_double_integrator.TestDoubleIntegratorForDocs.test_double_integrator_for_docs
    :layout: code, output, plot
