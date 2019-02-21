import numpy as np

import matplotlib
matplotlib.use('Agg')
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def get_angle_plot(line1, line2, radius=1, color=None, origin=(0, 0),
                   len_x_axis=1, len_y_axis=1):  # pragma: no cover

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


def min_time_climb_fbd(include_drag=True):  # pragma: no cover
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.axis('off')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    def y_min_time_climb(x):
        return x, np.ones_like(x)*0.2

    x = np.linspace(0.0, 1, 100)
    y, _ = y_min_time_climb(x)
    #
    # plt.plot(x, y, 'b-', alpha=0.3)

    # Add the axes
    x = x[0]
    y = y[0]
    dx = 0.5
    dy = 0
    xhat = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    ax.add_patch(xhat)
    plt.text(x+dx/2.0, y+dy/2.0-0.05, 'r')
    dx = 0
    dy = 0.5
    yhat = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    ax.add_patch(yhat)
    plt.text(x+dx/2.0-0.05, y+dy/2.0, 'h')

    # Add the aircraft
    x = 0.5
    y, dy_dx = y_min_time_climb(x)

    im = ndimage.rotate(plt.imread('f4_profile.png'), 30)
    im = ndimage.gaussian_filter(im, sigma=1)  # blur image to soften edges
    oi = OffsetImage(im, zoom=0.1)

    box = AnnotationBbox(oi, (x, y + 0.015), frameon=False)
    ax.add_artist(box)

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
    plt.text(x+dx, y+dy, 'v')

    # Draw and label the drag vector
    dx = -0.2
    dy = dy_dx * dx
    dvec = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    dvec_line = plt.Line2D((x, x+dx), (y, y+dy), visible=False)
    ax.add_patch(dvec)
    plt.text(x+dx, y+dy+0.05, 'D')

    # Draw and label the lift vector
    dy_dx = -1.0 / dy_dx
    dx = -0.05
    dy = dy_dx * dx
    dvec = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    dvec_line = plt.Line2D((x, x+dx), (y, y+dy), visible=False)
    ax.add_patch(dvec)
    plt.text(x+dx, y+dy+0.05, 'L')

    # Draw and label the thrust vector
    dy_dx = np.tan(np.radians(40))
    dx = 0.35
    dy = 0.6 * dy_dx * dx
    tvec = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=10)
    tvec_line = plt.Line2D((x, x + dx), (y, y + dy), visible=False)
    ax.add_patch(tvec)
    plt.text(x+dx, y+dy, 'T')

    # Draw the local horizon
    dx = 0.4
    dy = 0
    lh_line, = plt.plot((x, x+dx), (y, y+dy), linestyle='--', color='k', zorder=-1000)

    # Draw and label the flight path angle
    theta_plot = get_angle_plot(lh_line, vvec_line, color='k', origin=(x, y), radius=0.45)
    ax.add_patch(theta_plot)
    ax.text(x+0.25, y+0.02, r'$\gamma$')

    # Draw and label the angle of attack
    gamma_plot = get_angle_plot(vvec_line, tvec_line, color='k', origin=(x, y), radius=0.5)
    ax.add_patch(gamma_plot)
    ax.text(x+0.25, y+0.08, r'$\alpha$')

    plt.savefig('min_time_climb_fbd.png')

    plt.show()


if __name__ == '__main__':  # pragma: no cover
    min_time_climb_fbd()
