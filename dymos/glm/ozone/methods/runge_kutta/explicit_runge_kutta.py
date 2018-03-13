from __future__ import division

import numpy as np

from dymos.glm.ozone.methods.runge_kutta.runge_kutta import RungeKutta, RungeKuttaST


class ForwardEuler(RungeKutta):

    def __init__(self):
        self.order = 1

        super(ForwardEuler, self).__init__(A=0., B=1.)


def get_ExplicitMidpoint():
    ExplicitMidpoint_A = np.array([
        [ 0., 0.],
        [1/2, 0.],
    ])

    ExplicitMidpoint_B = np.array([
        [0., 1.],
    ])

    return ExplicitMidpoint_A, ExplicitMidpoint_B


class ExplicitMidpoint(RungeKutta):

    def __init__(self):
        self.order = 2

        ExplicitMidpoint_A, ExplicitMidpoint_B = get_ExplicitMidpoint()

        A = np.array(ExplicitMidpoint_A)
        B = np.array(ExplicitMidpoint_B)

        super(ExplicitMidpoint, self).__init__(A=A, B=B)


class ExplicitMidpointST(RungeKuttaST):

    def __init__(self):
        self.order = 2

        ExplicitMidpoint_A, ExplicitMidpoint_B = get_ExplicitMidpoint()

        A = np.zeros((3, 3))
        B = np.zeros((2, 3))

        A[:2, :2] = ExplicitMidpoint_A
        A[2, :2] = ExplicitMidpoint_B
        B[0, :2] = ExplicitMidpoint_B
        B[1, 2] = 1.

        super(ExplicitMidpointST, self).__init__(A=A, B=B)


class HeunsMethod(RungeKutta):

    def __init__(self):
        self.order = 2

        super(HeunsMethod, self).__init__(
            A=np.array([
                [0., 0.],
                [1., 0.],
            ]),
            B=np.array([
                [1/2, 1/2],
            ])
        )


class RalstonsMethod(RungeKutta):

    def __init__(self):
        self.order = 2

        super(RalstonsMethod, self).__init__(
            A=np.array([
                [0., 0.],
                [2 / 3, 0.],
            ]),
            B=np.array([
                [1 / 4, 3 / 4],
            ])
        )


def get_KuttaThirdOrder():
    KuttaThirdOrder_A = np.array([
        [0., 0., 0.],
        [1 / 2, 0., 0.],
        [-1., 2., 0.],
    ])

    KuttaThirdOrder_B = np.array([
        [1 / 6, 4 / 6, 1 / 6],
    ])

    return KuttaThirdOrder_A, KuttaThirdOrder_B


class KuttaThirdOrder(RungeKutta):

    def __init__(self):
        self.order = 3

        KuttaThirdOrder_A, KuttaThirdOrder_B = get_KuttaThirdOrder()

        A = np.array(KuttaThirdOrder_A)
        B = np.array(KuttaThirdOrder_B)

        super(KuttaThirdOrder, self).__init__(A=A, B=B)


class KuttaThirdOrderST(RungeKuttaST):

    def __init__(self):
        self.order = 3

        KuttaThirdOrder_A, KuttaThirdOrder_B = get_KuttaThirdOrder()

        A = np.zeros((4, 4))
        B = np.zeros((2, 4))

        A[:3, :3] = KuttaThirdOrder_A
        A[3, :3] = KuttaThirdOrder_B
        B[0, :3] = KuttaThirdOrder_B
        B[1, 3] = 1.

        super(KuttaThirdOrderST, self).__init__(A=A, B=B)


def get_RK4():
    RK4_A = np.array([
        [0., 0., 0., 0.],
        [1 / 2, 0., 0., 0.],
        [0., 1 / 2, 0., 0.],
        [0., 0., 1., 0.],
    ])

    RK4_B = np.array([
        [1 / 6, 1 / 3, 1 / 3, 1 / 6],
    ])

    return RK4_A, RK4_B


class RK4(RungeKutta):

    def __init__(self):
        self.order = 4

        RK4_A, RK4_B = get_RK4()

        A = np.array(RK4_A)
        B = np.array(RK4_B)

        super(RK4, self).__init__(A=A, B=B)


class RK4ST(RungeKuttaST):

    def __init__(self):
        self.order = 4

        RK4_A, RK4_B = get_RK4()

        A = np.zeros((5, 5))
        B = np.zeros((2, 5))

        A[:4, :4] = RK4_A
        A[4, :4] = RK4_B
        B[0, :4] = RK4_B
        B[1, 4] = 1.

        super(RK4ST, self).__init__(A=A, B=B)


def get_RK6(s):
    r = s * np.sqrt(5)

    RK6_A = np.zeros((7, 7))
    RK6_A[1, :1] = (5 - r) / 10
    RK6_A[2, :2] = [ (-r) / 10 , (5 + 2 * r) / 10]
    RK6_A[3, :3] = [ (-15 + 7 * r) / 20 , (-1 + r) / 4, (15 - 7 * r) / 10]
    RK6_A[4, 0] = (5 - r) / 60
    RK6_A[4, 2:4] = [ 1 / 6 , (15 + 7 * r) / 60 ]
    RK6_A[5, 0] = (5 + r) / 60
    RK6_A[5, 2:5] = [ (9 - 5 * r) / 12 , 1 / 6 , (-5 + 3 * r) / 10 ]
    RK6_A[6, 0] = 1 / 6
    RK6_A[6, 2:6] = [ (-55 + 25 * r) / 12 , (-25 - 7 * r) / 12 , 5 - 2 * r , (5 + r) / 2 ]

    RK6_B = np.zeros((1, 7))
    RK6_B[0, 0] = 1 / 12
    RK6_B[0, 4:7] = [ 5 / 12 , 5 / 12 , 1 / 12 ]

    return RK6_A, RK6_B


class RK6(RungeKutta):

    def __init__(self, s=1.):
        self.order = 6

        RK6_A, RK6_B = get_RK6(s)

        A = np.array(RK6_A)
        B = np.array(RK6_B)

        super(RK6, self).__init__(A=A, B=B)


class RK6ST(RungeKuttaST):

    def __init__(self, s=1.):
        self.order = 6

        RK6_A, RK6_B = get_RK6(s)

        A = np.zeros((8, 8))
        B = np.zeros((2, 8))

        A[:7, :7] = RK6_A
        A[7, :7] = RK6_B
        B[0, :7] = RK6_B
        B[1, 7] = 1.

        super(RK6ST, self).__init__(A=A, B=B)
