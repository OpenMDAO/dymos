=============
ph-Refinement
=============

The refinement strategy is p-then-h. That is, increasing the degree of the interpolating polynomials is the first choice,
followed by increasing the number of segments in the grid.

Assume the initial grid of a phase consists of :math:`K` segments. If the tolerance (:math:`\epsilon`) is not met for
a given segment, that segment is refined either by increasing the order or by splitting it. Let :math:`N_{min}` and
:math:`N_{max}` are the minimum and maximum allowed order for any given segment. If a segment :math:`k` requiring
refinement initially has order :math:`n_k`, it is increased to order :math:`n_k+p_k` where :math:`p` is computed as

.. math::

    p_k = \left[\text{log}_{n_k}\left(\frac{e_{max}^{(k)}}{\epsilon}\right)\right]

The expression is rounded up to ensure an integer value is returned.

If the new value for the segment order (:math:`\tilde{n}_k = n_k + p_k`) is less than the maximum order, then order of
that segment is increased on the new grid. If it exceeds the maximum value, the segment :math:`k` is divided into
:math:`B_k` segments, where

.. math::
    B_k = max\left(\left[\frac{\tilde{n}_k}{N_{min}}\right], 2\right)

Each of the newly formed segments is set to have the lowest allowed order :math:`N_{min}`