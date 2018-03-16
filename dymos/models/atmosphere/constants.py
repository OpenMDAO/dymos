import numpy as np

epsilon = 500.0
h_trans = 11000.0
h_lower = h_trans - epsilon
h_upper = h_trans + epsilon
matrix = np.array([[h_lower**3,      h_lower**2, h_lower, 1.0],
                   [h_upper**3,      h_upper**2, h_upper, 1.0],
                   [3 * h_lower**2, 2 * h_lower,     1.0, 0.0],
                   [3 * h_upper**2, 2 * h_upper,     1.0, 0.0]])
