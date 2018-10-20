from __future__ import print_function, division, absolute_import

import numpy as np

rk_methods = {'rk4': {'num_stages': 4,
                      'c': np.array([0., 0.5, 0.5, 1.0]),
                      'b': np.array([1/6, 1/3, 1/3, 1/6]),
                      'A': np.array([[0.0, 0.0, 0.0, 0.0],
                                     [0.5, 0.0, 0.0, 0.0],
                                     [0.0, 0.5, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.0]])}}
