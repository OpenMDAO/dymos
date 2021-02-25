import numpy as np
from scipy import interpolate

# A set of functions that fit splines to the track.


def get_track_points(track, initial_direction=np.array([1, 0])):
    # given a track description, place nodes along the centerlines in order to fit a spline
    # through them. Nodes are denser around corners
    pos = np.array([0, 0])
    direction = initial_direction

    points = [[0, 0]]

    for i in range(len(track.segments)):
        radius = track.get_segment_radius(i)
        length = track.get_segment_length(i)
        if radius == 0:
            # on a straight
            endpoint = pos + direction * length

            for j in range(1, length.astype(int) - 1):
                if j % 5 == 0:
                    points.append(pos + direction * j)

            pos = endpoint
        else:
            # corner
            # length is sweep in radians
            side = track.get_corner_direction(i)
            if side == 0:
                normal = np.array([-direction[1], direction[0]])
            else:
                normal = np.array([direction[1], -direction[0]])

            xc = pos[0] + radius * normal[0]
            yc = pos[1] + radius * normal[1]
            theta_line = np.arctan2(direction[1], direction[0])
            theta_0 = np.arctan2(pos[1] - yc, pos[0] - xc)
            if side == 0:
                theta_end = theta_0 + length
                direction = np.array(
                    [np.cos(theta_line + length), np.sin(theta_line + length)]
                )
            else:
                theta_end = theta_0 - length
                direction = np.array(
                    [np.cos(theta_line - length), np.sin(theta_line - length)]
                )
            theta_vector = np.linspace(theta_0, theta_end, 100)

            x, y = parametric_circle(theta_vector, xc, yc, radius)

            for j in range(len(x)):
                if j % 10 == 0:
                    points.append([x[j], y[j]])

            pos = np.array([x[-1], y[-1]])

    return np.array(points)


def parametric_circle(t, xc, yc, R):
    x = xc + R * np.cos(t)
    y = yc + R * np.sin(t)
    return x, y


def get_spline(points, interval=0.0001, s=0.0):
    # this function fits the spline
    tck, u = interpolate.splprep(points.transpose(), s=s, k=5)
    unew = np.arange(0, 1.0, interval)
    finespline = interpolate.splev(unew, tck)

    gates = interpolate.splev(u, tck)
    gatesd = interpolate.splev(u, tck, der=1)

    single = interpolate.splev(unew, tck, der=1)
    double = interpolate.splev(unew, tck, der=2)
    curv = (single[0] * double[1] - single[1] * double[0]) / (
        single[0] ** 2 + single[1] ** 2
    ) ** (3 / 2)

    return finespline, gates, gatesd, curv, single


def get_gate_normals(gates, gatesd):
    normals = []
    for i in range(len(gates[0])):
        der = [gatesd[0][i], gatesd[1][i]]
        mag = np.sqrt(der[0] ** 2 + der[1] ** 2)
        normal1 = [-der[1] / mag, der[0] / mag]
        normal2 = [der[1] / mag, -der[0] / mag]

        normals.append([normal1, normal2])

    return normals


def transform_gates(gates):
    # transforms from [[x positions],[y positions]] to [[x0, y0],[x1, y1], etc..]
    newgates = []
    for i in range(len(gates[0])):
        newgates.append(([gates[0][i], gates[1][i]]))
    return newgates


def reverse_transform_gates(gates):
    # transforms from [[x0, y0],[x1, y1], etc..] to [[x positions],[y positions]]
    newgates = np.zeros((2, len(gates)))
    for i in range(len(gates)):
        newgates[0, i] = gates[i][0]
        newgates[1, i] = gates[i][1]
    return newgates


def set_gate_displacements(gate_displacements, gates, normals):
    # does not modify original gates, returns updated version
    newgates = np.copy(gates)
    for i in range(len(gates[0])):
        if i > len(gate_displacements) - 1:
            disp = 0
        else:
            disp = gate_displacements[i]
        # if disp>0:
        normal = normals[i][0]  # always points outwards
        # else:
        # 	normal = normals[i][1] #always points inwards
        newgates[0][i] = newgates[0][i] + disp * normal[0]
        newgates[1][i] = newgates[1][i] + disp * normal[1]
    return newgates
