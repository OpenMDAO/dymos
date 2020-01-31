=======================
Hyper-Sensitive Problem
=======================

This example is based on the Hyper-Sensitive problem given in [PattersonHagerRao2015]_. In this problem we seek to
minimize both the distance traveled when moving between fixed boundary conditions. We seek also to minimize the control
:math:`u` used. The system is subject to the dynamic constraints:

.. math ::
    \frac{d x}{d t} &= -x + u \\

The boundary conditions are

.. math ::
    x(t_0) &= 1.5 \\
    x(t_f) &= 1

The control :math:`u` is unconstrained while the final time :math:`t_f` is fixed. Due to the nature of dynamics, for
sufficiently large values of :math:`t_f`, the problem exhibits a "dive", "cruise", and "resurface" type structure. This
problem has a known analytic optimal solution

.. math ::
    x*(t) = c_1\text{exp}(t\sqrt{2}) + c_2\text{exp}(-t\sqrt{2}) \\
    u*(t) = \frac{d x*}{d t}(t) + x*(t)

where

.. math ::
    c_1 = \frac{1.5\text{exp}(-t_f\sqrt{2}) - 1}{\text{exp}(-t_f\sqrt{2}) - \text{exp}(t_f\sqrt{2})} \\
    c_2 = \frac{1 - 1.5\text{exp}(t_f\sqrt{2})}{\text{exp}(-t_f\sqrt{2}) - \text{exp}(t_f\sqrt{2})}


1. The ODE System: hyper_sensitive_ode.py
-----------------------------------------

..  embed-code::
    examples.hyper_sensitive.hyper_sensitive_ode
    :layout: code

2. Building and running the problem
-----------------------------------

The following code shows the procedure for solving the problem

..  embed-code::
    examples.hyper_sensitive.doc.test_doc_hyper_sensitive
    :layout: code
