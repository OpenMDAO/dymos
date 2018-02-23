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

#. The phase is divided into polynomial segments on which the dynamics are assumed to be continuous.
#. The states are provided by the optimizer at the discretization nodes (the even-index LGL nodes of each segment).
#. The controls are provided by the optimizer at *all* nodes in the phase.
#. The dynamics are evaluated at the discretization nodes, giving the computed state rates.
#. Using the state and state rates at the discretization nodes, form a Hermite interpolating polynomial, giving the approximate state values and rates ath the collocation nodes.
#. The dynamics are evaluated at the collocation nodes (the odd-index LGL nodes), giving the computed state rates.
#. The difference between the approximated state rates and computed state rates are given to the optimizer as constraints.
#. The optimizer iterates on the state and control values until the optimality conditions are satisfied.

Advantages
- Provides collocated dyanamics at the endpoints of the segment.  No node has an "undefined" control
value as is the case in the Radau Pseudospectral Method.

Disadvantages
- Requires an interpolation step that can sometimes make it less amenable to poor initial guesses.
- Requires two steps to evaluate the dynamics at all nodes in the phase (first the discretization nodes, then the collocation nodes).  This
poses a performance bottleneck when using parallelization to evaluate the dynamics.


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
- No interpolation of states or controls is necessary, since the collocation nodes are a subset of
the state discretization nodes.
- This method can evaluate the dynamics at all nodes in a phase in a single pass, while the Gauss-Lobatto
method requires two passes (evaluate, interpolate, evaluate).  This removes a bottleneck when using
parallelization to evaluate the dynamics.

Disadvantages
- One point in a phase is not subject to collocation (either the initial point or the end point).  As a result,
the control values at that node have less (or zero) impact on the collation defect constraints and are meaningless.  Various methods
exist for working around this deficiency, such as constraining the control value or derivatives at the endpoint, or by running the
optimization with both in LGR and reversed LGR (rLGR) modes and then taking the valid control from each.

The implementation-specific details vary by phase, but a 3rd-order polynomial requires 4 pieces of information to be uniquely defined.
In GaussLobatto phases, a transcription order of 3 results in two state *discretization* nodes an a single *collocation* node.
The state, time, and control values at the discretization nodes are used to evaluate the phase's `ODEFunction` and compute the state *rates* at the discretization nodes.
Then, a hermite interpolation scheme takes the values at rates of the states at the discretization nodes and computes the
value *and rate* of the states at the collocation node(s).  Now, with values for time, the controls, and the interpolated
states at the collocation nodes, enough data exists to evaluate the ODE function a second time, a the collocation node(s).
The difference between the interpolated rate and the rate calculated by the ODE function at the collocation node is called
the *collocation defect*.  These defects are driven to zero (either as a constraint for the optimizer, or in the case of the
GLM phase, as residuals for the solver).


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