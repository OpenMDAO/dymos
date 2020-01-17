================
Error Estimation
================

The error is computed by comparing two approximations of the solutions

Each phase in the solved problem is evaluated individually. Assume that the problem was solved on the interval
[:math:`t_{0}`, :math:`t_{f}`] which is divided into :math:`K` segments as [:math:`t_{k-1}, t_{k}`],
:math:`k = 0, 1,..., K-1`. Each segments is contains :math:`n` nodes (:math:`\tau_0`,..., :math:`\tau_{n-1}`). A new
grid on which the error is to be evaluated is created. The new grid has order :math:`m` such that :math:`m > n` with
nodes (:math:`\widehat{\tau}_{0}`,..., :math:`\widehat{\tau}_{m-1}`). For the Radau method :math:`m = n + 1` and for the
Gauss-Lobatto method :math:`m = n + 2`. This difference is due to the fact that the implementation of Gauss-Lobatto
transcription requires an odd order.

The Lagrange interpolating polynomial :math:`\ell` is used to obtain values for the state and control on the new grid

.. math::

    \widehat{\ell}_j(\tau) = \prod_{{l=0, l \neq j}}_^{n-1}\,\frac{\tau - \tau_l^{(k)}}{\tau_j^{(k)} - \tau_l^{(k)}}

The state and control are then interpolated as

.. math::

   x^{(k)}(\widehat{\tau}) = \sum_{j=0}^{n-1}x_j^{(k)}\widehat{l}_j(\widehat{\tau}),\\
   u^{(k)}(\widehat{\tau}) = \sum_{j=0}^{n-1}u_j^{(k)}\widehat{l}_j(\widehat{\tau}),\:

Quadrature integration is then performed on the grid for each of the segments as

.. math::

    \widehat{x}^{(k)}(\widehat{\tau}_j^{(k)}) = \widehat{x}^{(k)}(\tau_{0}) + \frac{t_f - t_0}{2}\sum_{l=1}^{m-1}
    \widehat{I}_{jl}\,f_{ode}(x^{(k)}(\widehat{\tau}_l^{(k)}), u^{(k)}(\widehat{\tau}_l^{(k)}), \widehat{\tau}_l^{(k)}),
    \: j = 1, ..., m

where :math:`\widehat{I}_{jl}^{(k)}` is the :math:`m \times m` integration matrix corresponding the collocation points
of the transcription.

The absolute and relative errors are then computed as

.. math::
    E_i^{(k)}(\widehat{\tau}_l^{(k)}) = |\widehat{x}_i^{(k)}(\widehat{\tau}_l^{(k)}) - x(\widehat{\tau}_l^{(k)})|, \\
    e_i^{(k)}(\widehat{\tau}_l^{(k)}) = \frac{E_i^{(k)}(\widehat{\tau}_l^{(k)})}{1 + \max_{j \in [0, m]}
    |x_i(\widehat{\tau}_j^{(k)})|}, \\
    l = 0,..., m,\:i = 1,..., N

where :math:`N` is the number of states. Then the maximum relative error is computed as

.. math::
    e_{max}^{(k)} = \max_{i, l} \, e_i^{(k)}(\widehat{\tau}_l^{(k)})

The maximum relative error of each phase is checked against the tolerance.

This approach assumes the solution for the given problem to be smooth. In this case, increasing the number of nodes should cause
the numerical solution to more accurately reflect the system dynamics. So comparing these two solutions is a a good
estimate for the error in the numerical solution