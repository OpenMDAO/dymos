"""
This module houses constants that are used throughout dymos

Attributes
----------
INF_BOUND : float
    The value of infinity used for unbounded variables or constraints.  Some optimizers treat
    bounds as infinity if their value exceeds some threshold.  The value used in Dymos is set
    such that it satisfies the default value for both IPOPT (1.0E19) and SNOPT (1.0E20).
"""


INF_BOUND = 1.0E21
