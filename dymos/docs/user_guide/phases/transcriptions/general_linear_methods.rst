General Linear Methods
----------------------

The general linear methods (GLM) equations unify all linear multistep methods
(e.g., Adams--Bashforth) and all Runge--Kutta methods (e.g., RK4, Gauss--Legendre)
using a single equation [Butcher2006]_.
The :code:`GLMPhase` in Dymos implements the GLM equations, enabling a large library of methods
to be available (currently only Runge--Kutta methods are implemented).
The use of the GLM equations has enabled this large library to be implemented,
even with the requirement for derivatives across the integrator
so that we can use a gradient-based optimizer [HwangMunster2018]_.

Formulations
~~~~~~~~~~~~

The :code:`GLMPhase` has three formulations (i.e., solution approaches) available:
optimizer-based, solver-based [HwangMunster2018]_, and time-marching.
The difference between these formulations is in how they solve the ODE,
i.e., how they compute the state variables being integrated.
In general, the solver-based formulation is the best (fastest) option of the three,
but in stiff ODEs, the time-marching formulation with an implicit Runge--Kutta scheme
is the most reliable.

1. Optimizer-based formulation.

  In the optimizer-based formulation, the ODE state variables are included in the
  optimization problem as design variables and the relevant Runge--Kutta equations
  are imposed as optimization constraints.
  Therefore, the responsibility for solving the ODE is assigned to the optimizer.
  This formulation is typically slower than the solver-based formulation,
  and it can suffer from convergence problems with stiff ODEs.

2. Solver-based formulation.

  In the solver-based formulation, the ODE state variables are treated as the unknowns
  of a nonlinear system of equations. The entire time history is then solved for at once,
  using a block Gauss--Seidel iteration, as explained by Hwang and Munster [HwangMunster2018]_.
  In most problems, this formulation is the most efficient of the three because it benefits from
  vectorization across time instances (which the time-marching formulation does not have) and the
  fact that the block Gauss--Seidel solver converges in a very small number of iterations.
  The primary disadvantage of this formulation is that it does not always converge when the time
  interval of the integration is too large and with stiff ODEs.

3. Time-marching formulation.

  In the time-marching formulation, the ODE state variables are integrated one time step at a time,
  as is typically done in solving ODEs in general.
  The advantage of this approach is that time-marching with an implicit Runge--Kutta scheme
  is the most reliable for stiff ODEs.
  The disadvantage is that for non-stiff ODEs, all three formulations converge, but time-marching
  is the slowest because of the lack of vectorization---the typical cost of a Python for loop.

List of available methods
~~~~~~~~~~~~~~~~~~~~~~~~~

Because the :code:`GLMPhase` is implemented using the GLM equations, adding a new Runge--Kutta
method is simple and quick, which has allowed a large library to be implemented.
For recommendations on which method to use, the reader is referred to
Hwang and Munster [HwangMunster2018]_
who show a Pareto fronts with error and computation time as the two axes for a set of test problems.
The list of available methods is given below.

===========================  ========  =====
'method_name' in GLMPhase    Type      Order
===========================  ========  =====
:code:`ForwardEuler`         Explicit    1
:code:`BackwardEuler`        Implicit    1
:code:`ExplicitMidpoint`     Explicit    2
:code:`ImplicitMidpoint`     Implicit    2
:code:`KuttaThirdOrder`      Explicit    3
:code:`RK4`                  Explicit    4
:code:`RK6`                  Explicit    6
:code:`RalstonsMethod`       Explicit    2
:code:`HeunsMethod`          Explicit    2
:code:`GaussLegendre2`       Implicit    2
:code:`GaussLegendre4`       Implicit    4
:code:`GaussLegendre6`       Implicit    6
:code:`Lobatto2`             Implicit    2
:code:`Lobatto4`             Implicit    4
:code:`RadauI3`              Implicit    3
:code:`RadauI5`              Implicit    5
:code:`RadauII3`             Implicit    3
:code:`RadauII5`             Implicit    5
:code:`Trapezoidal`          Implicit    2
===========================  ========  =====


References
^^^^^^^^^^
.. [Butcher2006] Butcher, J. C., “General linear methods,” Acta Numerica, Vol. 15, 2006, pp. 157–256.
.. [HwangMunster2018] Hwang, John T, and Drayton Munster. “Solution of Ordinary Differential Equations in Gradient-Based Multidisciplinary Design Optimization.” 2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference. American Institute of Aeronautics and Astronautics, 2018. Web. AIAA SciTech Forum.
