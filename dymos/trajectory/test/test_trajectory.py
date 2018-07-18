from __future__ import print_function, division, absolute_import

import unittest

from dymos import Trajectory


class TestTrajectory(unittest.TestCase):

    def test_trajectory(self):
        t = Trajectory()
        t.link_phases(['a', 'b', 'c', 'd', 'e', 'f'])
