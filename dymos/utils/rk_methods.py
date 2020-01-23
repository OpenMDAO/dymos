"""RK Methods

Contains data used by the Runge-Kutta methods in Dymos.

Attributes
----------
rk_methods : dict of dict of {str: object}
    A dictionary, keyed by lower-case Runge-Kutta method name, which contains fields needed
    to perform numerical integration using the Runge-Kutta methods.  Entries are as follows:
    num_stages (int) - The number of stages in the RK method.
    c (ndarray) - Normalized times (on [0, 1]) across each step at which the ODE is evaluated.
    b (ndarray) - The weighting factors used to compute the final state value across each step (segment) of the phase.
    b_err (ndarray) -  The weighting factors used to compute a guess of a different order, for the variable step
    methods.  This can be used to compute the error across each step.  If b_err = b, then
    the resulting errors will be zero and the method provides no error approximation.
    A (ndarray) - A num_stages x num_stages matrix used to compute the predicted state values at each stage
    within a segment.
"""

import numpy as np

rk_methods = {'RK4': {'num_stages': 4,
                      'c': np.array([0., 0.5, 0.5, 1.0]),
                      'control_disc_indices': [0, 1, 3],
                      'b': np.array([1/6, 1/3, 1/3, 1/6]),
                      'b_err': np.array([1/6, 1/3, 1/3, 1/6]),
                      'A': np.array([[0.0, 0.0, 0.0, 0.0],
                                     [0.5, 0.0, 0.0, 0.0],
                                     [0.0, 0.5, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.0]])}}
