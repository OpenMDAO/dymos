==========================================
SSTO Lunar Ascent with Polynomial Controls
==========================================

This example demonstrates the use of polynomial controls in Dymos. Polynomial controls define the
control profile as a *single* polynomial across the entire phase where the control values are specified
at the Legendre Gauss Lobatto (LGL) nodes in *phase dimensionless time*.  These controls can be of
any arbitrary order greater than 1 (linear).

We've already demonstrated that the optimal single stage ascent in the absense of an atmosphere
follows the linear tangent guidance law.  In this example, we'll change the control parameterization
such that :math:`\tan \theta` is provided by a polynomial control of order 1.  The LGL nodes
of a first order polynomial are the endpoints of the phase, thus the optimizer will be governing
the value of :math:`\tan \theta` at the initial and final times of the phase, and the Dymos will
interpolate the values of :math:`\tan \theta` to all other nodes in the Phase.

This example is equivalent to the previous linear tangent example in that we've reduced the problem
from finding the appropriate control value at all nodes to that of finding the optimal value of just
two quantities.  But instead of optimizing the slope and intercept given by the parameters
:math:`a` and :math:`b`, we're parameterizing the control using the endpoint values of the linear
polynomial.

Now the guidance comp needs to convert the inverse tangent of the current value of the polynomial
controls.

.. math::

    \theta = \arctan{p}

-------------------
Solving the problem
-------------------

.. embed-code::
    dymos.examples.ssto.doc.test_doc_ssto_polynomial_control.TestDocSSTOPolynomialControl.test_doc_ssto_polynomial_control
    :layout: code, plot

References
----------
[Longuski2016] Longuski, James M., José J. Guzmán, and John E. Prussing. Optimal control with aerospace applications. Springer, 2016.
