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
^^^^^^^^^^
.. [Butcher2006] Butcher, J. C., “General linear methods,” Acta Numerica, Vol. 15, 2006, pp. 157–256.
.. [HwangMunster2018] Hwang, John T, and Drayton Munster. “Solution of Ordinary Differential Equations in Gradient-Based Multidisciplinary Design Optimization.” 2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference. American Institute of Aeronautics and Astronautics, 2018. Web. AIAA SciTech Forum.