from __future__ import division

import numpy as np

from dymos.glm.ozone.methods.runge_kutta.runge_kutta import RungeKutta


class BackwardEuler(RungeKutta):

    def __init__(self):
        self.order = 1

        super(BackwardEuler, self).__init__(A=1., B=1.)


class ImplicitMidpoint(RungeKutta):

    def __init__(self):
        self.order = 2

        super(ImplicitMidpoint, self).__init__(A=1. / 2., B=1.)


class TrapezoidalRule(RungeKutta):

    def __init__(self):
        self.order = 2

        super(TrapezoidalRule, self).__init__(A=np.array([[0., 0.], [1 / 2, 1 / 2]]),
                                              B=np.array([1 / 2, 1 / 2]))
