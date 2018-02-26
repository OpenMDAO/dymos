==============
Transcriptions
==============

|project| implements several different *transcriptions* which are used to convert a continuous
optimal control problem into an analogous nonlinear programming (NLP) problem.

|project| implements multi-phase trajectory optimization.  That is to say, in each phase
the state and control time histories are continuous.  At the beginning or end of each phase,
a discrete jump in the value of a state variable is permitted.  This comes in useful when
modelling things like stage or store (mass) jettisons or impulsive maneuvers (:math:`\Delta`Vs).
By linking phases together with continuity constraints, a trajectory or series of trajectories
can be assembled.

Different transcription methods are supported via different Phase types in |project|.  Currently,
the software supports the following techniques:

High-Order Gauss-Lobatto Collocation
------------------------------------

High-order Gauss-Lobatto is a generalization of the Hermite-Simpson optimization scheme developed
by Herman and Conway [HermanConway1996]_.

**Step 1:  Phase Segmentation**

Each phase is divided into polynomial segments.  The span and polynomial order of each segment
define the "grid" within each segment.  The grid can be specified by the user or set by an
automatic grid refinement algorithm, depending on the accuracy required.

.. image:: figures/01_segments.png
   :scale: 100 %
   :alt: A phase divided into four equal segments
   :align: center


**Step 2:  Discretization**

Each segment is discretized by applying the Legendre-Gauss-Lobatto nodes to normalized
segment space (:math:`\tau_{s}`)

.. image:: figures/02_nodes.png
   :scale: 100 %
   :alt: The Legendre Gauss Lobatto nodes in each segment.
   :align: center

**Step 3:  Input**

The time of the phase is specified using the initial time (:math:`t_{initial}`) and duration
(:math:`t_duration`) of the phase.  This sets the value of time at each *node* within the phase.

The value of each state is given at each state discretization node (the even-index LGL nodes of
each segment).  The user typically provides an initial guess for these values.  During optimization
these are design variables.

The value of each dynamic control is given at every node in the segment.
The user typically provides an initial guess for these values.  During optimization these
are design variables.

.. image:: figures/03_inputs.png
   :scale: 100 %
   :alt: The discretized state and control values
   :align: center

.. note::

    At each segment boundary the value of each state and control is given twice by default.
    The phase will automatically add constraints to ensure continuity at the segment boundary.
    Optionally, the user may provide the argument `compressed=True` upon Phase initialization,
    which specifies that state values are not to be given twice at each segment boundary.
    While this removes some variables from the problem, our experience has shown that the penalty
    is minimal, while using an 'uncompressed' transcription provides for better parallelization.

**Step 4:  Control Rate Interpolation**

With the control values known at all nodes in the grid, we can form a Lagrange interpolation
polynomial to compute the derivatives of the control at all nodes.  This allows us to use
control rates as inputs to the ODE.  In |project| the first and second derivatives of the
controls are automatically computed.

.. note::

   In a segment with 3 LGL nodes, the states are interpolated from values *and rates* at the two
   endpoint nodes, resulting in a 3th order polynomial.  The controls are interpolated from their
   values at all three nodes, resulting in a 2nd order polynomial.  Enforcing continuity in the
   second derivative of a control at segment bounds is not advisable when using High Order
   Gauss Lobatto with low-order segments.

.. image:: figures/04_control_rate_interpolation.png
   :scale: 100 %
   :alt: Control rates are interpolated.
   :align: center


**Step 5:  Evaluation of the ODE at the State Discretization Nodes**

On each segment, the state and control time histories are assumed to be polynomials.  For a
segment with three nodes, the two endpoints are the state discretization nodes.  The given ODE
is evaluated here (remember, thus far we know the values of the states at the discretization
nodes and the values of time and the controls at all nodes).

.. image:: figures/05_ode_eval_disc.png
   :scale: 100 %
   :alt: The slope of the state-time history at the discretization nodes has been evaluated.
   :align: center

**Step 6:  State Interpolation**

Using the state and state rates at the discretization nodes, form a Hermite interpolating
polynomial, giving the approximate state values and rates ath the collocation nodes.

.. image:: figures/06_interpolation.png
   :scale: 100 %
   :alt: The state time histories are interpolated.
   :align: center

**Step 7:  Evaluation of the ODE at the Collocation Nodes**

The given ODE is evaluated a second time, this time at the collocation nodes
(the odd-index LGL nodes), giving the computed state rates.

.. image:: figures/07_ode_eval_col.png
   :scale: 100 %
   :alt: The time-derivative of the states is evaluated at the collocation nodes.
   :align: center

**Step 7:  Evaluation of the Collocation Defects**

The collocation *defects* (:math:`\Delta`) are computed by finding the difference between the approximate state values
at the collocation nodes (from Step 5) and the computed values at the collocation nodes (from Step 6).

.. math::

   \Delta = f_{ode}(x_{col}, t_{col}, u_{col}) - x'_{col}

If uncompressed transcription is used, the continuity defects involving the states, controls, and
(optionally) control rates are also evaluated here.

**Step 8:  Iterate Steps 3 Through 7**

During optimization, the optimizer updates the design variables (initial time, final time,
state values at the discretization nodes, and control values at all nodes) until the constraints
posed in Step 7 are satisfied.  At this point the time-history of the state variable is *feasible*
,provided the accuracy of our grid is sufficient.  If we have one or more control variable then
we may have an infinite number of feasible trajectories.  In that case the optimizer will work to
find the time history that minimizes our objective function.

Advantages of High-Order Gauss-Lobatto Collocation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Provides collocated dyanamics at the endpoints of the segment.
  No node has an "undefined" control value as is the case in the Radau Pseudospectral Method.


Disdvantages of High-Order Gauss-Lobatto Collocation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Requires an interpolation step that can sometimes make it less amenable to poor initial guesses.
- Requires two steps to evaluate the dynamics at all nodes in the phase (first the discretization
  nodes, then the collocation nodes).  This poses a performance bottleneck when using
  parallelization to evaluate the dynamics.
- State path constraints cannot be imposed with simple bounds on states, since the interpolation
  step may interpolate a state value beyond the limits imposed on the design values provided at
  the state discretization nodes.



Radau Pseudospectral Method
---------------------------

The Radau-Pseudospectral method performs collocation of an optimal control problem by collocating
the dynamics at the Legendre Gauss Radau nodes [Garg2010]_.  The general procedure
for this method is as follows:

#. The phase is divided into polynomial segments on which the dynamics are assumed to be continuous.
#. The states and controls are provided by the optimizer at the LGR nodes *plus the endpoint* of each segment.
#. Given the state values, form a Lagrange polynomial on each segment and take its derivative to compute the approximate state rates at the collocation nodes.
#. The dynamics are evaluated at the collocation nodes (the LGR nodes not including the endpoint), giving the computed state rates.
#. The difference between the approximated state rates and computed state rates are given to the optimizer as constraints.
#. The optimizer iterates on the state and control values until the optimality conditions are satisfied.

Advantages
- No interpolation of states or controls is necessary, since the collocation nodes are a subset of the state discretization nodes.
- This method can evaluate the dynamics at all nodes in a phase in a single pass, while the Gauss-Lobatto method requires two passes (evaluate, interpolate, evaluate).  This removes a bottleneck when using parallelization to evaluate the dynamics.

Disadvantages
- One point in a phase is not subject to collocation (either the initial point or the end point).  As a result,
the control values at that node have less (or zero) impact on the collation defect constraints and are meaningless.  Various methods
exist for working around this deficiency, such as constraining the control value or derivatives at the endpoint, or by running the
optimization with both in LGR and reversed LGR (rLGR) modes and then taking the valid control from each.

General Linear Methods
----------------------

The general linear methods is a generalization that encapsulates both linear multistep integration methods (such as Adams Bashforth)
and the Runge-Kutta methods [Butcher2006]_.  These methods have been implemented in a way that allows them to be used in both
implicit and explicit forms, and allowing the implicit form to be optionally converged by a solver or the optimizer [HwangMunster2018]_.

In the Gauss-Lobatto and Radau Pseudospectral methods, the accuracy of the dynamics is enforced as a constraint on the optimizer.  Until
the collocation defect constraints are satisfied, the trajectory is non-physical.  By using time-marching or solver-based GLM methods,
the optimizer effectively sees a phyically valid trajectory at each iteration (until it nears convergence, however, it is unlikely to satisfy design constraints
posed in the form of boundary and path constraints).  These methods, then, are analogous to direct shooting techniques:

#. The optimizer guesses initial state values, time, and control time histories
#. The trajectory is integrated across the phase.
#. The optimizer iterates on the state and control values until the optimality conditions are satisfied.

When implicit GLMs are used and the convergence is managed by the optimizer via constraints, these
techniques are similar to the Gauss-Lobatto and Radau Pseudospectral methods.

[TODO: Expand upon this]



References
----------
.. [Butcher2006] Butcher, J. C., “General linear methods,” Acta Numerica, Vol. 15, 2006, pp. 157–256.
.. [Garg2010] Garg, Divya et al. “A Unified Framework for the Numerical Solution of Optimal Control Problems Using Pseudospectral Methods.” Automatica 46.11 (2010): 1843–1851.
.. [HermanConway1996] Herman, Albert L, and Bruce A Conway. “Direct Optimization Using Collocation Based on High-Order Gauss-Lobatto Quadrature Rules.” Journal of Guidance, Control, and Dynamics 19.3 (1996): 592–599.
.. [HwangMunster2018] Hwang, John T, and Drayton Munster. “Solution of Ordinary Differential Equations in Gradient-Based Multidisciplinary Design Optimization.” 2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference. American Institute of Aeronautics and Astronautics, 2018. Web. AIAA SciTech Forum.