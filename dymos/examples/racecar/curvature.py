import openmdao.api as om
import numpy as np

#track curvature imports
from scipy import interpolate
from scipy import signal
from Track import Track
import tracks
from spline import getTrackPoints,getSpline

class Curvature(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        track = tracks.ovaltrack #remember to change here and in problemSolver.py

        points = getTrackPoints(track)
        print(track.getTotalLength())
        finespline,gates,gatesd,curv,slope = getSpline(points)

        self.curv = curv
        self.trackLength = track.getTotalLength()

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('s', val=np.zeros(nn), desc='distance along track', units='m')

        #outputs
        self.add_output('kappa', val=np.zeros(nn), desc='track centerline Curvature', units='1/m')

        #no partials needed

    def compute(self, inputs, outputs):
        s = inputs['s']

        num_curv_points = len(self.curv)

        kappa = np.zeros(len(s))

        for i in range(len(s)):
            index = np.floor((s[i]/self.trackLength)*num_curv_points)
            index = np.minimum(index,num_curv_points-1)
            kappa[i] = self.curv[index.astype(int)]


        outputs['kappa'] = kappa

    def compute_partials(self, inputs, jacobian):
        pass

        








