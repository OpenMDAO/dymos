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
    # , label =str(angle) + u"\u00b0")


def ssto_fbd(include_drag=True):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.axis('off')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # plt.plot((0, 200), (200, 200), linestyle='--', color='#CCCCCC')

    def y_ssto(x):
        return -(x-1)**2+1, 2-2*x

    x = np.linspace(0.0, 1, 100)
    y, _ = y_ssto(x)

    plt.plot(x, y, 'b-', alpha=0.3)

    # Add the axes
    x = x[0]
    y = y[0]
    dx = 0.5
    dy = 0
    xhat = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    ax.add_patch(xhat)
    plt.text(x+dx/2.0, y+dy/2.0-0.05, 'x')
    dx = 0
    dy = 0.5
    yhat = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    ax.add_patch(yhat)
    plt.text(x+dx/2.0-0.05, y+dy/2.0, 'y')

    # Add the launch vehicle
    x = 0.5
    y, dy_dx = y_ssto(x)
    plt.plot(x, y, 'ro', ms=10)

    # Draw and label the gravity vector
    L = 0.2
    gvec = FancyArrowPatch((x, y), (x, y-L), arrowstyle='->', mutation_scale=10)
    lv_line = plt.Line2D((x, x), (y, y-L), visible=False)  # Local vertical
    ax.add_patch(gvec)
    plt.text(x-0.05, y-L, 'g')

    # Draw and label the velocity vector
    dx = 0.3
    dy = dy_dx * dx
    vvec = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    vvec_line = plt.Line2D((x, x+dx), (y, y+dy), visible=False)
    ax.add_patch(vvec)
    plt.text(x+dx, y+dy-0.05, 'v')

    # Draw and label the drag vector
    if include_drag:
        dx = -0.2
        dy = dy_dx * dx
        dvec = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
        dvec_line = plt.Line2D((x, x+dx), (y, y+dy), visible=False)
        ax.add_patch(dvec)
        plt.text(x+dx, y+dy+0.05, 'D')

    # Draw and label the thrust vector
    dx = 0.2
    dy = 0.6 * dy_dx * dx
    tvec = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    tvec_line = plt.Line2D((x, x + dx), (y, y + dy), visible=False)
    ax.add_patch(tvec)
    plt.text(x+dx, y+dy-0.05, 'T')

    # Draw the local horizon
    dx = 0.4
    dy = 0
    lh_line, = plt.plot((x, x+dx), (y, y+dy), linestyle='--', color='k', zorder=-1000)

    # Draw and label the thrust angle
    theta_plot = get_angle_plot(lh_line, vvec_line, color='k', origin=(x, y), radius=0.6)
    ax.add_patch(theta_plot)
    ax.text(x+0.1, y+0.02, r'$\theta$')

    # Draw and label the flight path angle
    gamma_plot = get_angle_plot(lh_line, tvec_line, color='k', origin=(x, y), radius=0.3)
    ax.add_patch(gamma_plot)
    ax.text(x+0.26, y+0.06, r'$\gamma$')

    plt.savefig('ssto_fbd.png')

    plt.show()


if __name__ == '__main__':
    ssto_fbd()
