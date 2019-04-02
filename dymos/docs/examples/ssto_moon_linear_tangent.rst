==============================================
SSTO Lunar Ascent with Linear Tangent Guidance
==============================================

The following example implements a minimum time, single-stage to orbit ascent problem for
launching from the lunar surface.  Unlike the SSTO Earth Ascent example, here we use knowledge of
the solution to simplify the optimization.

Instead of optimizing the thrust angle at any point in time as a dynamic control, we use our
knowledge that the form of the solution is a "linear tangent".  See section 4.6 of [Longuski2016]_ for
more explanation.  In short, we've simplified the problem by finding the optimal value of :math:`\theta`
at many points into optimizing the value of just two scalar parameters, :math:`a` and :math:`b`.

.. math::

    \theta = \arctan{\left(a * t + b\right)}

Implementing this modified constrol scheme requires only a few changes.  Rather than declaring
:math:`\theta` as a controllable parameter for the ODE system, we implement a new component,
LinearTangentGuidanceComp that accepts :math:`a` and :math:`b` as design parameters to be optimized.  It
calculates :math:`theta`, which is then connected to the equations of motion component.

--------------------------------
Extended Design Structure Matrix
--------------------------------

.. figure:: figures/ssto_linear_tangent_xdsm.png

    The XDSM for the ODE system in the SSTO linear tangent problem.  The only significant change
    is that we have a new component, `guidance`, which accepts :math:`a`, :math:`b`,
    and :math:`time`, and computes :math:`\theta`.

-------------------
Solving the problem
-------------------

.. embed-code::
    dymos.examples.ssto.doc.test_doc_ssto_linear_tangent_guidance.TestDocSSTOLinearTangentGuidance.test_doc_ssto_linear_tangent_guidance
    :layout: code, output, plot

References
----------
[Longuski2016] Longuski, James M., José J. Guzmán, and John E. Prussing. Optimal control with aerospace applications. Springer, 2016.
