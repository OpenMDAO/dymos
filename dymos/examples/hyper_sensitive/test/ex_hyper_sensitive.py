from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import openmdao.api as om

import dymos as dm
from dymos.examples.hyper_sensitive.hyper_sensitive_ode import HyperSensitiveODE

SHOW_PLOTS = True


def hyper_sensitive(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                             compressed=True, optimizer='SLSQP'):
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['options'] = optimizer
    p.driver.declare_coloring()
