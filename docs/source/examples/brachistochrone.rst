A simple optimal control example
================================

We illustrate how to use ozone to solve an optimal control problem - the Brachistochrone problem.
We seek to find the curve along which a ball rolls from one point to another in the shortest amount of time.
We minimize the final time, :math:`t_f`, by varying the dynamic control, :math:`\theta`, subject to the dynamics,

.. math ::
  \frac{\partial x}{\partial t} &= v \sin(\theta) \\
  \frac{\partial y}{\partial t} &= v \cos(\theta) \\
  \frac{\partial v}{\partial t} &= g \cos(\theta). \\

The initial conditions are

.. math ::
  x(0) &= 0 \\
  y(0) &= 0 \\
  v(0) &= 0, \\

and the transversality constraints are

.. math ::
  x(t_f) &= 2.0 \\
  y(t_f) &= -2.0 \\

Here, we use the 6th order Gauss--Legendre collocation method with 20 time steps.

1. Defining the system
----------------------

Here, our ODE function is defined by a single OpenMDAO system, an :code:`ExplicitComponent`.


2. Defining the ODE function class
----------------------------------

Here, we define the :code:`ODEFunction`, where we declare the 3 states and the control variable,
which is called a parameter in :code:`ODEFunction`.

---


3. Building and running the problem
-----------------------------------

Here, we pass call :code:`ScipyODEIntegrator` to build our integration model and run it.
The run script and resulting plot are shown below.  In the code we follow these
general steps:

* Instantiate a problem and set up the driver.

* Create a Phase (here we use GaussLobattoPhase), give it our ODE function, and add it to the problem.

* Set the time options (bounds on initial time and duration, scaling, etc).

* Set the state options (bounds, scaling, etc).  In this case we use :code:`fix_initial` and :code:`fix_final` to specify whether the initial/final values are included as design variables.

* Add controls to the problem.  Recall that :code:`ODEFunction` can include parameters which impact the results.  We can add static or dynamic controls and tie them (or their derivatives) to these parameters.

* Set the objective.  Here we seek to minimize the final time.

* Call setup and then set the initial values of our variables.  We use interpolation routines to allow us to specify all values of states and controls at the discretization nodes.

* Run the driver.

* Simulate the control time history using scipy.ode.  This serves as a check that our optimization resulted in a valid solution.



