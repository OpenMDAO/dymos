
=====================================
dymos.utils.matrix_partials_wizard
=====================================

Using the matrix and vector-compatible components described in this section can reduce the
number of components needed in the ODE system, but defining partial derivatives when populating
a matrix or vector can be a challenge.

To make this easier, |project| includes a utility called the `MatrixPartialsWizard`.

The `MatrixPartialsWizard` is a class that, on instantiation, takes the following arguments:

