========================================================
Van Der Pol Oscillator: A simple optimal control example
========================================================

In dynamics, the Van Der Pol oscillator[1] is a non-conservative oscillator with non-linear damping.
It evolves in time according to the second-order differential equation:

.. math ::
    \frac{d^2x}{d t^2} - u(1 - x^2)\frac{d x}{d t} + x = 0

where :math:`x` is the position coordinate—which is a function of the time :math:`t`, and :math:`u`
is a scalar parameter indicating the nonlinearity and the strength of the damping.

In this Dymos Van Der Pol Oscillator optimal control example, we minimize the objective function, :math:`J`:

.. math ::
    J = \int x_0^2 + x_1^2 + u^2

by varying the dynamic control, :math:`u`, subject to the dynamics:

.. math ::
    \frac{d x_0}{d t} &= (1 - x_1^2) x_0 - x_1 + u \\
    \frac{d x_1}{d t} &= x_0 \\
    -0.75 &\le u \le 1.0

The initial conditions are

.. math ::
    x_0 &= 1 \\
    x_1 &= 1 \\
    u &=-0.75, \\

and the final conditions are

.. math ::
    x_0 &= 0 \\
    x_1 &= 0 \\
    u &= 0

In other words, we want to find the optimal trajectory of the control :math:`u` such that the
oscillation, its rate of change, and the control are all driven to zero.


1. The ODE System: vanderpol.py
-------------------------------

The ODE system is a standard OpenMDAO `ExplicitComponent` object. It has three inputs: the
two state variables :math:`x_0` and :math:`x_1`, plus the control :math:`u`.

The derivatives of the state variables are outputs. The include the two expected outputs
:math:`\frac{d x_0}{d t}` and :math:`\frac{d x_1}{d t}`, but also the derivative of the objective
function :math:`\frac{d J}{d t}` is also present. The objective function is treated as a state
variable so that Dymos will do the integration.

Note that some of the partial derivatives are declared with provided constant values. When
this is the case, the variables can be skipped in the `compute_partials` method.


..  embed-code::
    examples.vanderpol.vanderpol_ode
    :layout: code

2. Optimizing the system control: vanderpol_dymos.py
----------------------------------------------------
The following code creates and returns the Van Der Pol `Problem` using Dymos.
Th returned `Problem` object is ready for simulation or solution.

..  embed-code::
    examples.vanderpol.vanderpol_dymos
    :layout: code

The `vanderpol` function that sets up this Dymos problem has several optional arguments
that can set non-default values for the transcription type, number of segments, optimizer, etc.
The OpenMDAO `ScipyOptimizeDriver` is used by default because it has compact output suitable
for this example. When developing a new solution, more detailed output can be helpful and can
be enabled by using OpenMDAO `pyOptSparseDriver` instead.

Some other noteworthy issues are:

- the `add_state` and `add_control` calls include the `target` parameter for :math:`x_0`,
  :math:`x_1`, and :math:`u`. This is required so that the inputs are correctly calculated.
- Near the bottom there is a section that provides a linearly interpolated initial guess
  for the state and control curves. Some initial values for these guesses are required.


3. Example test runs
---------------------------------------

Simulation
**********

The first example test run shows the creation and simulation of the Dymos Van Der Pol `Problem`.

..  embed-code::
    dymos.examples.vanderpol.doc.test_doc_vanderpol.TestVanderpolForDocs.test_vanderpol_for_docs_simulation
    :layout: code, output, plot

Since the problem was only simulated and not solved, the *solution* lines in the plots show only the
initial guesses for :math:`x_0`, :math:`x_1`, and :math:`u`. The *simulation* lines shown in the plots
are the system response with the control variable :math:`u` held constant.

The first two plots  shows the variables :math:`x_0` and :math:`x_1`, vs time.
The third plots shows :math:`x_0` vs :math:`x_1` (which will be mostly circular in
the case of undamped oscillation).
The final plot is the (fixed) control variable :math:`u` vs time.

Optimize
********

The next example shows optimization in addition to simulation.

..  embed-code::
    dymos.examples.vanderpol.doc.test_doc_vanderpol.TestVanderpolForDocs.test_vanderpol_for_docs_optimize
    :layout: code, output, plot

With a successful optimization, the resulting plots show a good match between the simulated and optimized
results. The state variables :math:`x_0` and :math:`x_1` as well as the control variable :math:`u` are
all driven to zero.

Optimize with grid refinement
*****************************

Repeating the optimization with grid refinement enabled requires changing only two lines in the
code. For the sake of grid refinement demonstration, the initial number of segments is also reduced
by a factor of 5.

..  embed-code::
    dymos.examples.vanderpol.doc.test_doc_vanderpol.TestVanderpolForDocs.test_vanderpol_for_docs_optimize_refine
    :layout: code, output, plot

Optimization with grid refinement gets results similar to the example without grid refinement, but
runs faster and does not require supplying a good guess for the number of required segments.

4. References
-------------
[1] Van Der Pol oscillator description from `Wikipedia <https://en.wikipedia.org/wiki/Van_der_Pol_oscillator>`_
