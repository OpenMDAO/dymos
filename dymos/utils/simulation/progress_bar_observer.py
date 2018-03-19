from __future__ import print_function, division, absolute_import

import sys


class ProgressBarObserver(object):
    """
    A simple observer that outputs a progress bar based on the current and
    final simulation time.

    Parameters
    ----------
    ode_integrator : ScipyODEIntegrator
        The ODEIntegrator instance that will be calling this observer.
    out_stream : file-like
        The stream to which output will be written.
    """
    def __init__(self, ode_integrator, t0, tf, out_stream=sys.stdout):
        self._prob = ode_integrator.prob
        self.t0 = t0
        self.tf = tf
        self.out_stream = out_stream

    def __call__(self, t, y, prob):
        t0 = self.t0
        tf = self.tf
        print('Simulation time: {0:6.3f} of {1:6.3f} ({2:6.3f}%)'.format(
            t, tf, 100 * (t - t0) / (tf - t0)), file=self.out_stream)
