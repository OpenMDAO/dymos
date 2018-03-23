from __future__ import print_function, absolute_import, division

import numpy as np

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def brachistochrone_fbd():
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.axis('off')

    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)

    circle = plt.Circle((0, 10), radius=0.1, fc='k')
    ax.add_patch(circle)
    plt.text(0.2, 10.2, 'A')

    circle = plt.Circle((10, 5), radius=0.1, fc='k')
    ax.add_patch(circle)
    plt.text(10.2, 5.2, 'B')

    # Parabola that connects (0, 10) and (10, 5)
    # y = a*x**2 + b*x + c
    # y(0) = c = 10
    # y(10) = a*x**2 + b*x + 10 = 5
    #         x*(a*x + b) = -5
    #         10*a + b = -1/2
    # Choose a to suite, compute b
    a = 0.1
    b = -0.5 - 10*a
    c = 10

    def y_wire(x):
        return a*x**2 + b*x + c, 2*a*x + b

    x = np.linspace(0, 10, 100)
    y, _ = y_wire(x)
    plt.plot(x, y, 'b-')

    # Add the bead to the wire
    x = 3
    y, dy_dx = y_wire(x)
    plt.plot(x, y, 'ro', ms=10)

    # Draw and label the gravity vector
    gvec = FancyArrowPatch((x, y), (x, y-2), arrowstyle='->', mutation_scale=10)
    ax.add_patch(gvec)
    plt.text(x - 0.5, y-1, 'g')

    # Draw and label the velocity vector
    dx = 2
    dy = dy_dx * dx
    vvec = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    ax.add_patch(vvec)
    plt.text(x+dx-0.25, y+dy-0.25, 'v')

    # Draw the axes
    x = y = 0
    dx = 5
    dy = 0
    xhat = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    ax.add_patch(xhat)
    plt.text(x+dx/2-0.5, y+dy/2-0.5, 'x')

    dx = 0
    dy = 5
    yhat = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    ax.add_patch(yhat)
    plt.text(x+dx/2-0.5, y+dy/2-0.5, 'y')


    plt.show()


if __name__ == '__main__':
    brachistochrone_fbd()