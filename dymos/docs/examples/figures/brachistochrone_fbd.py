import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc


def get_angle_plot(line1, line2, radius=1, color=None, origin=(0, 0),
                   len_x_axis=1, len_y_axis=1):

    l1xy = line1.get_xydata()
    # Angle between line1 and x-axis
    l1_xs = l1xy[:, 0]
    l1_ys = l1xy[:, 1]
    angle1 = np.degrees(np.arctan2(np.diff(l1_ys), np.diff(l1_xs)))

    l2xy = line2.get_xydata()
    # Angle between line2 and x-axis
    l2_xs = l2xy[:, 0]
    l2_ys = l2xy[:, 1]
    angle2 = np.degrees(np.arctan2(np.diff(l2_ys), np.diff(l2_xs)))

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)

    angle = theta2 - theta1

    if color is None:
        color = line1.get_color()  # Uses the color of line 1 if color parameter is not passed.
    return Arc(origin, len_x_axis * radius, len_y_axis * radius, 0, theta1, theta2, color=color)
    # label =str(angle) + u"\u00b0")


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
    lv_line = plt.Line2D((x, x), (y, y-2), visible=False)  # Local vertical
    ax.add_patch(gvec)
    plt.text(x - 0.5, y-1, 'g')

    # Draw and label the velocity vector
    dx = 2
    dy = dy_dx * dx
    vvec = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    ax.add_patch(vvec)
    plt.text(x+dx-0.25, y+dy-0.25, 'v')

    # Draw angle theta
    vvec_line = plt.Line2D((x, x+dx), (y, y+dy), visible=False)
    angle_plot = get_angle_plot(lv_line, vvec_line, color='k', origin=(x, y), radius=3)
    ax.add_patch(angle_plot)
    ax.text(x+0.25, y-1.25, r'$\theta$')

    # Draw the axes
    x = y = 0
    dx = 5
    dy = 0
    xhat = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    ax.add_patch(xhat)
    plt.text(x+dx/2.0-0.5, y+dy/2.0-0.5, 'x')

    dx = 0
    dy = 5
    yhat = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    ax.add_patch(yhat)
    plt.text(x+dx/2.0-0.5, y+dy/2.0-0.5, 'y')

    plt.show()


if __name__ == '__main__':
    brachistochrone_fbd()
