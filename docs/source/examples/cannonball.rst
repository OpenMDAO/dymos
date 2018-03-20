Pseudospectral Time Integration of a Ballistic Cannonball
===========================================================

First, we begin with an example on the use of implicit simulation of a simple system without
dynamic controls: the flight of a cannonball.

[fig]

1. Defining the system
----------------------

We can express the equations of motion of a simple cannonball in a 2D rectilinear gravity field as:

.. math::
  \frac{\partial x}{\partial t} &= v_x \\
  \frac{\partial y}{\partial t} &= v_y \\
  \frac{\partial v_x}{\partial t} &= 0.0 \\
  \frac{\partial v_y}{\partial t} &= -g \\

These equations of motion are computed by the following OpenMDAO ExplicitComponent.  There
are a few key points to note about the equations of motion:

- while *x* and *y* are state variables, they are not inputs to the above equations.
- in addition to computing rates of the state variables, the equations of motion compute the velocity magnitude and flight path angle (:math:`\gamma`)
- we use OpenMDAO's ``compute_partials`` method to provide derivatives for all outputs.
- since the function is vectorized (computes values at all nodes simultaneously), and the outputs at one node is independent of the other nodes, the partial derivatives are in the form of a diagonal matrix.

---

To make the above system work with |project|, we wrap it in an ``ODEFunction`` object.
In the following code, we inherit from ``|project|2.ODEFunction`` and use ``declare_time``
and ``declare_state`` to provide |project| with information about the time variable and
state variables, respectively.

---

1. Solving an initial value problem
-----------------------------------

With fixed initial states and time we have a so-called *initial value problem* (IVP).
If we also assume that the final time is determined a-priori then there is a single physical trajectory for the system.
We can find the corresponding trajectory by setting up a simple optimal control problem.

Having defined our ODEFunction, we now go about setting up an OpenMDAO problem to solve the
corresponding optimal control problem.

2. Solving a two point boundary value problem
---------------------------------------------

Typically when predicting the trajectory of a cannonball, it's useful to know the impact point of the shot.
In this case we have fixed initial conditions for the state of the cannonball, and require that the
final altitude (:math:`y`) of the cannonball is zero.  To find this solution, we must allow the
duration of the flight (:math:`t_p`) to be a free variable.  The problem can be states as:

.. math::
  t_0 &= 0 \, s \\
  t_f &= \mathrm{free} \\
  x(0) &= 0 \, \mathrm{m} \\
  y(0) &= 0 \, \mathrm{m} \\
  v_x(0) &= 100 \, \mathrm{m/s} \\
  v_y(0) &= 50 \, \mathrm{m/s} \\
  y(t_f) &= 0 \, \mathrm{m}

To solve this problem we go through the following steps:

  #. Create an OpenMDAO problem and assign an optimization driver.
  #. Instantiate a |project| *Phase* - Here we use a ``GaussLobattoPhase``

     * The phase is responsible for integrating the ODE

     * The number of polynomial segments and the state transcription order
       of the segments affects the speed and accuracy of the solution.

  #. Set the time, state, and control options appropriately for the given problem.
  #. Set the objective function appropriately

     * All problems must have an objective, even if it has no free variables (pure simulation)

     * The ``set_objective`` method on Phases provides some convenience over the default
       OpenMDAO ``add_objective`` method, but either one would work.

  #. Provide initial guesses

     * State values are provided at discretization nodes

     * Control values are provided at all nodes.

     * Initial time (:math:`t_0`) and phase duration (:math:`t_p`) guess are provided

  #. Run the problem driver.
  #. Examine the results.

     * Did the optimizer converge?  The answer is likely nonsense in nonconverged runs.

     * Even if the optimizer converged, was the transcription grid (number of segments and segment order) capable of providing accurate results?

---

3. Maximizing the range of the cannonball
-----------------------------------------

The previous examples demonstrate some basic capabilities of |project| but don't actually
optimize anything - the optimization driver is essentially being used as a solver.  In this
case we seek to maximize the range travelled by the cannonball subject to a limit on its
initial velocity.  This example will demonstrate the following:

  * Using ``set_objective`` to maximize the final value of a state variable.
  * Using ``add_boundary_constarint`` to impose a limit on the initial velocity of the cannonball.

---