# SSTO Lunar Ascent with Linear Tangent Guidance

The following example implements a minimum time, single-stage to orbit
ascent problem for launching from the lunar surface. Unlike the SSTO
Earth Ascent example, here we use knowledge of the solution to simplify
the optimization.

Instead of optimizing the thrust angle at any point in time as a dynamic
control, we use our knowledge that the form of the solution is a
_linear tangent_. See section 4.6 of Longiski[@longuski2014optimal] for more
explanation. In short, we've simplified the problem by finding the
optimal value of $\theta$ at many points into optimizing the value of
just two scalar parameters, $a$ and $b$.

$$\theta = \arctan{\left(a * t + b\right)}$$

Implementing this modified constrol scheme requires only a few changes.
Rather than declaring $\theta$ as a controllable parameter for the ODE system, we implement a new component, _LinearTangentGuidanceComp_ that accepts $a$ and $b$ as parameters to be optimized.
It calculates $\theta$, which is then connected to the equations of motion component.

## Extended Design Structure Matrix

![The XDSM diagram for the ODE system in the SSTO inear tangent problem.](ssto_linear_tangent_xdsm.png)

In the XDSM for the ODE system for the SSTO linear tangent problem, the
only significant change is that we have a new component,
_guidance_, which accepts $a$, $b$, and $time$, and computes
$\theta$.

## Solving the problem

{{ embed_test('dymos.examples.ssto.doc.test_doc_ssto_linear_tangent_guidance.TestDocSSTOLinearTangentGuidance.test_doc_ssto_linear_tangent_guidance') }}

## References

\bibliography
