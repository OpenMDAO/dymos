=================================================
Brachistochrone: A simple optimal control example
=================================================

We seek to find the optimal shape of a wire between two points (A and B) such that a bead sliding
without friction along the wire moves from point A to point B in minimum time.

..  image:: figures/brachistochrone_fbd.png
   :scale: 100 %
   :alt: The free-body-diagram of the brachistochrone problem.
   :align: center

We minimize the final time, :math:`t_f`, by varying the dynamic control, :math:`\theta`, subject to the dynamics:

.. math ::
    \frac{d x}{d t} &= v \sin(\theta) \\
    \frac{d y}{d t} &= v \cos(\theta) \\
    \frac{d v}{d t} &= g \cos(\theta)

The initial conditions are

.. math ::
    x_0 &= 0 \\
    y_0 &= 10 \\
    v_0 &= 0, \\

and the final conditions are

.. math ::
    x_f &= 10 \\
    y_f &= 5 \\
    v_f &= \mathrm{free}

1. The ODE System: brachistochrone_ode.py
-----------------------------------------

..  embed-code::
    examples.brachistochrone.brachistochrone_ode
    :layout: code

There are a few things to note about the ODE system.  First, it is just a standard OpenMDAO system,
in this case an :code:`ExplicitComponent`.  The :code:`declare_time`, :code:`declare_state`, and
:code:`declare_parameter` decorators are used to inform Dymos as to where the time, states, and
potential control variables should be connected to the system.  These decorators are *optional* and
only provide defaults for these values.  Otherwise they can be set via the Phase API (``set_time_options``,
``add_state``, ``add_control``, ``add_design_parameter``, ``add_input_parameter`` etc.
The :code:`rate_source` parameter of :code:`declare_state` dictates the output in the system
that provides the time-derivative of the corresponding state variable.

The second important feature is the :code:`num_nodes` option.  This informs the component as to
the number of time points for which it will be computing its values, which varies depending on the
transcription method.  Performance of Dymos is significantly improved by using vectorized operations,
as opposed to for-loops, to compute the outputs at all times simultaneously.  **All systems used
as ODEs must support this option and size their inputs and outputs appropriately.**

Finally, note that we are specifying rows and columns when declaring the partial derivatives.
Since our inputs and outputs are scalars *at each point in time*, and the value at an input at
one time only directly impacts the values of an output at the same point in time, the partial
derivative jacobian will be diagonal.  Specifying the partial derivatives as being sparse
greatly improves the performance of Dymos when the driver function ``declare_coloring`` is used.
Using a sparse optimizer like SNOPT or IPOPT can provide significant addition improvements in
performance.

2. Building and running the problem using high-order Gauss-Lobatto transcription
--------------------------------------------------------------------------------

The following code solves the Brachistochrone problem using Dymos.  Using Gauss-Lobatto transcription
is just a matter of picking the ``GaussLobatto`` transcription from the Dymos namespace.

.. embed-code::
    dymos.examples.brachistochrone.doc.test_doc_brachistochrone.TestBrachistochroneForDocs.test_brachistochrone_for_docs_gauss_lobatto
    :layout: code, output, plot

3. Building and running the problem using high-order Radau Pseudospectral transcription
---------------------------------------------------------------------------------------

Solving the problem using the Radau pseudospectral method is just a matter of changing to the
``Radau`` transcription from the Dymos namespace.

.. embed-code::
    dymos.examples.brachistochrone.doc.test_doc_brachistochrone.TestBrachistochroneForDocs.test_brachistochrone_for_docs_radau
    :layout: code, output, plot

4. Building and running the problem using high-order Runge-Kutta shooting transcription
---------------------------------------------------------------------------------------

We can also use a Runge-Kutta based shooting method, simply by selecting the ``RungeKutta``
transcription from the Dymos namespace.  Note in shooting methods the final state values are not
design variables and therefore we need to enforce their final values with nonlinear boundary
constraints.

.. embed-code::
    dymos.examples.brachistochrone.doc.test_doc_brachistochrone.TestBrachistochroneForDocs.test_brachistochrone_for_docs_runge_kutta
    :layout: code, output, plot

5. Using a phase-wide polynomial to control the brachistochrone
---------------------------------------------------------------

Looking at the plots for the previous cases it becomes apparent that the angle for the brachistochrone
might more easily be modeled as a linear polynomial spanning the phase.  That is, rather than having
the optimizer find the appropriate angle :math:`\theta` at each node, its task is significantly
simplified if we have it find the best initial and final value of :math:`\theta` and then linearly
interpolate between those two values.  We can achieve this by adding a :math:`\theta` as a
*polynomial control*.

.. automethod:: dymos.phase.Phase.add_polynomial_control
    :noindex:

.. embed-code::
    dymos.examples.brachistochrone.doc.test_doc_brachistochrone.TestBrachistochroneForDocs.test_brachistochrone_for_docs_runge_kutta_polynomial_controls
    :layout: code, output, plot