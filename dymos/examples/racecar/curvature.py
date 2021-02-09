import openmdao.api as om
import numpy as np

from .tracks import ovaltrack
from .spline import get_track_points, get_spline


class Curvature(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        track = ovaltrack  # remember to change here and in problemSolver.py

        points = get_track_points(track)
        finespline, gates, gatesd, curv, slope = get_spline(points)

        self.curv = curv
        self.track_length = track.get_total_length()

    def setup(self):
        nn = self.options['num_nodes']

        # constants
        self.add_input('s', val=np.zeros(nn), desc='distance along track', units='m')

        # outputs
        self.add_output('kappa', val=np.zeros(nn), desc='track centerline Curvature', units='1/m')

        # no partials needed

    def compute(self, inputs, outputs):
        s = inputs['s']

        num_curv_points = len(self.curv)

        kappa = np.zeros(len(s))

        for i in range(len(s)):
            index = np.floor((s[i]/self.track_length)*num_curv_points)
            index = np.minimum(index, num_curv_points-1)
            kappa[i] = self.curv[index.astype(int)]

        outputs['kappa'] = kappa
