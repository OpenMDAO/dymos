=================================================
Brachistochrone: A simple optimal control example
=================================================

We seek to find the optimal shape of a wire between two points (A and B) such that a bead sliding
without friction along the wire moves from point A to point B in minimum time.

..  image:: figures/brachistochrone_fbd.png
   :scale: 100 %
   :alt: The free-body-diagram of the brachistochrone problem.
   :align: center

We minimize the final time, :math:`t_f`, by varying the dynamic control, :math:`\theta`, subject to the dynamics:

.. math ::
    \frac{d x}{d t} &= v \sin(\theta) \\
    \frac{d y}{d t} &= v \cos(\theta) \\
    \frac{d v}{d t} &= g \cos(\theta)

The initial conditions are

.. math ::
    x_0 &= 0 \\
    y_0 &= 10 \\
    v_0 &= 0, \\

and the final conditions are

.. math ::
    x_f &= 10 \\
    y_f &= 5 \\
    v_f &= \mathrm{free}

1. The ODE System: brachistochrone_ode.py
-----------------------------------------

..  embed-code::
    examples.brachistochrone.brachistochrone_ode
    :layout: code

There are a few things to note about the ODE system.  First, it is just a standard OpenMDAO system,
in this case an :code:`ExplicitComponent`.  The :code:`declare_time`, :code:`declare_state`, and
:code:`declare_parameter` decorators are used to inform Dymos as to where the time, states, and
potential control variables should be connected to the system.  The :code:`rate_source` parameter
of :code:`declare_state` dictates the output in the system that provides the time-derivative of
the corresponding state variable.

The second important feature is the :code:`num_nodes` metadata.  This informs the component as to
the number of time points for which it will be computing its values, which varies depending on the
transcription method.  Performance of Dymos is significantly improved by using vectorized operations,
as opposed to for-loops, to compute the outputs at all times simultaneously.

Finally, note that we are specifying rows and columns when declaring the partial derivatives.
Since our inputs and outputs are scalars *at each point in time*, and the value at an input at
one time only directly impacts the values of an output at the same point in time, the partial
derivative jacobian will be diagonal.  Specifying the partial derivatives as being sparse can
greatly improve the performance of Dymos.

2. Building and running the problem
-----------------------------------

In the following code we follow the following process to solve the problem:

* Instantiate a problem and set up the driver.

* Create a Phase, give it our ODE system class, and add it to the problem.

* Set the time options (bounds on initial time and duration, scaling, etc).

* Set the state options (bounds, scaling, etc).  In this case we use :code:`fix_initial` and :code:`fix_final` to specify whether the initial/final values are included as design variables.

* Add controls and design parameters to the phase.  We can add design parameters or dynamic controls and tie them (or their derivatives) to the parameters in the ODE system.

* Set the objective.  Here we seek to minimize the final time.

* Call setup and then set the initial values of our variables.  We use interpolation routines to allow us to specify values of the states at the state discretization nodes and controls at all nodes.

* Run the driver.

* Simulate the control time history using scipy.ode.  This serves as a check that our optimization resulted in a valid solution.

.. embed-code::
    dymos.examples.brachistochrone.test.test_doc_brachistochrone.TestBrachistochroneExample.test_brachistochrone_for_docs_gauss_lobatto
    :layout: code, output, plot
