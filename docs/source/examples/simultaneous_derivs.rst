=============================================
Brachistochrone with Simultaneous Derivatives
=============================================

A key feature of collocation algorithms such as high-order Gauss-Lobatto collocation or the
Radau pseudospectral method is that they exhibit a large degree of *sparsity* in the total
Jacobian.  By modeling the state-time histories as a series of polynomial segments, the collocation
defect constraints within each segment is largely dependent only on the state and control values
within the same segment.  State-of-the-art pseudospectral optimization tools such as SOCS, OTIS,
and GPOPS use the notion of *sparse finite differences* to perturb multiple independent variables
simultaneously when approximating the constraint Jacobian.  This can significantly reduce the
computational effort required to approximate the entire Jacobian via finite difference.

The unique way in which OpenMDAO assembles analytic derivatives across the problem allows us to
use a similar approach to provide the analytic constraint Jacobian.  This approach can reduce the time
required to solve moderately-sized optimal control problems by orders of magnitude and make the
convergence more robust.

OpenMDAO uses a simultaneous "coloring" algorithm to determine which variables can be perturbed
simultaneously to determine the total constraint Jacobian.  Variables in the same "color" each
impact a unique constraint, such that when all the variables are perturbed we can be assured that
any change in the constraint vector is due to at most one scalar variable.

Step 1: Using OpenMDAO's Simul-Coloring Capability
==================================================

The first step in the process is to construct an optimal control problem. To demonstrate this
capability we'll instantiate the brachistochrone problem with 200 3rd-order Gauss-Lobatto segments.

.. embed-code::
    dymos.examples.brachistochrone.test.test_doc_brach_setup_simul_derivs.TestBrachistochroneSimulDerivsSetupExample.test_brachistochrone_for_docs_gauss_lobatto_simul_derivs
    :layout: code

Having defined our code, we invoke the coloring algorithm via the OpenMDAO command line interface.
Save the above problem in a file `brach.py`.  The coloring is then invoked via:

.. code-block:: none
    openmdao simul_coloring -n 5 -t 1.0E-15 -o -s coloring.json brach.py

This runs five passes of the algoirthm (`-n 5`).  Each time, random data is inserted for the
design variables and the total Jacobian computed.  After each pass, a running sum of the Jacobian
matrix is maintained.  Elements of the total Jacobian that are truly nonzero will accumulate
to values above the threshold tolerance (`-t 1.0E-15`), and OpenMDAO will determine a "color"
for each design variable.  The sparsity pattern is also saved so that it can be used by
"sparse-aware" optimizers like SNOPT and IPOPT (`-s`).  The resulting coloring metadata will be stored
in the specified output file (`-o coloring.json`).

The simul_coloring script outputs the following information about our problem:

.. code-block:: none

    1 uncolored columns
    5 columns in color 1
    100 columns in color 2
    100 columns in color 3
    101 columns in color 4
    101 columns in color 5
    195 columns in color 6
    197 columns in color 7
    201 columns in color 8
    Sparsity structure has been computed for all response/design_var sub-jacobians.
    Total colors vs. total size: 9 vs 1001  (99.1% improvement)

Step 2: Giving pyoptsparse the coloring information
===================================================

To provide the problem with the coloring information we use the driver `set_simul_deriv_color`
method, providing it with the file to which the coloring data was saved.

.. code-block:: python

    p.driver.set_simul_deriv_color('coloring.json')

Inserting the above statement into the code above will run the brachistochrone using SNOPT and
take into account the sparsity infromation.

Performance comparison with and without simultaneous derivatives
================================================================

The following chart demonstrates the difference in timing that is achieved by
using simultaneous derivatives.  In this case the time required to solve the problem
dropped by over an order of magnitude.  For comparison, the performance of the
legacy optimal control software OTIS4 is also shown.  When using simul-derivs, the
performance of Dymos comes well within an order of magnitude (about 1.1x in this case) of
the performance of OTIS. This is despite the fact that Dymos supports analytic derivatives,
parallelization, and a non-conservative sparsity pattern which should close the gap further as
problem size grows.

This case has 600 constraints and 1001 variables, giving a total constraint Jaobian size of 600600.
Using a conservative sparsity pattern (assuming any variable in a segment can impact any constraint
in the same segment), OTIS computes that there are 7794 nonzero elements in the Jacobian.  The
non-conservative sparsity pattern calculated by OpenMDAO gives 4393 nonzero elements.

.. embed-code::
    source/examples/figures/simul_derivs_perf_chart.py
    :layout: plot

General Performance Tips Using Dymos
====================================

1. Use the CSCJacobian as the top-level Jacobian where possible
---------------------------------------------------------------

The CSCJacobian is a sparse Jacobian format used internally by OpenMDAO that can significantly
reduce memory requirements and signficantly improve performance of the Jacobian calculation.

2. Use DirectSolver as the top-level linear solver where possible
-----------------------------------------------------------------

Unless the problem grows extremely large, using DirectSolver to solve the linear system which
computes the Jacobian can yield significant performance improvements.

3. Use simultaneous derivatives
-------------------------------

As we've shown above, handling sparsity and simultaneous derivatives can significantly
improve performance.


4. Use "compressed" transcription when parallelization is not a concern
-----------------------------------------------------------------------

When providing the state and control values at segment boundaries, there are two options.
If a phase is declared with `compressed=True` (the default), the one value for the state/control
will be provided at the boundary, and used at the shared endpoint by both segments.
If `compressed=False`, then then two unique values are provided as design variables, with
state and control value continuity at the segment bound being enforced via a linear constraint.
Experience has shown that using compressed transcription signficantly improves performance by
reducing the number of variables and constraints given to the optimizer.  On the other hand,
when attempting to distribute the analysis across more than one processor using the separable
uncompressed transcription may give better performance.