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

.. embed-code::
    dymos.examples.double_integrator.double_integrator_ode
    :layout: code

2. Building and running the problem
-----------------------------------

In the following code we follow the following process to solve the problem:

* Instantiate a problem and set up the driver.

* Create a Phase, give it our ODE system class, and add it to the problem.

* Set the time options (bounds on initial time and duration, scaling, etc).

* Set the state options (bounds, scaling, etc).  In this case we use :code:`fix_initial` and :code:`fix_final` to specify whether the initial/final values are included as design variables.

* Add controls to the problem.  We can add design parameters or dynamic controls and tie them (or their derivatives) to the parameters in the ODE system.

* Set the objective.  Here we seek to maximize the final value of :math:`x`.

* Call setup and then set the initial values of our variables.  We use interpolation routines to allow us to specify values of the states at the state discretization nodes and controls at all nodes.

* Run the driver.

* Simulate the control time history using scipy.ode.  This serves as a check that our optimization resulted in a valid solution.

.. embed-code::
    dymos.examples.double_integrator.test.test_doc_double_integrator.TestDoubleIntegratorForDocs.test_double_integrator_for_docs
    :layout: code, output, plot
