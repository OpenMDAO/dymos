=================
SSTO Lunar Ascent
=================

The following example implements a minimum time, single-stage to orbit ascent problem for
launching from the lunar surface.  This example assumes constant acceleration and a
rectilinear gravity field.  This example is based on the the example presented Appendix A.3 of [Longuski2016]_

To maximize code reuse, this example is implemented using the same general dyamics as the
SSTO Earth Ascent example, but the atmospheric model returns a density of zero
:math:`\frac{kg}{m^3}` regardless of altitude.

.. embed-code::
    dymos.examples.ssto.test.test_ex_ssto_moon.TestExampleSSTOMoon.test_plot
    :layout: code, output, plot

References
----------
[Longuski2016] Longuski, James M., José J. Guzmán, and John E. Prussing. Optimal control with aerospace applications. Springer, 2016.
