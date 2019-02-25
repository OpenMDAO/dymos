from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials

from dymos.phases.runge_kutta.components import RungeKuttaPathConstraintComp
from dymos.phases.grid_data import GridData
from dymos.phases.options import ControlOptionsDictionary


class TestPathConstraintCompExplicit(unittest.TestCase):

    def setUp(self):

        transcription = 'runge-kutta'

        self.gd = gd = GridData(num_segments=2,
                                transcription_order='rk4',
                                segment_ends=[0.0, 3.0, 10.0],
                                transcription=transcription)

        nn = 4

        self.p = Problem(model=Group())

        controls = {'a': ControlOptionsDictionary(),
                    'b': ControlOptionsDictionary(),
                    'c': ControlOptionsDictionary(),
                    'd': ControlOptionsDictionary()}

        controls['a'].update({'units': 'm', 'shape': (1,), 'opt': False})
        controls['b'].update({'units': 's', 'shape': (3,), 'opt': False})
        controls['c'].update({'units': 'kg', 'shape': (3, 3), 'opt': False})

        ivc = IndepVarComp()
        self.p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('a_disc', val=np.zeros((nn, 1)), units='m')
        ivc.add_output('b_disc', val=np.zeros((nn, 3)), units='s')
        ivc.add_output('c_disc', val=np.zeros((nn, 3, 3)), units='kg')

        path_comp = RungeKuttaPathConstraintComp(grid_data=gd)

        self.p.model.add_subsystem('path_constraints', subsys=path_comp)

        path_comp._add_path_constraint('a', var_class='ode', shape=(1,),
                                       lower=0, upper=10, units='m')
        path_comp._add_path_constraint('b', var_class='input_control', shape=(3,),
                                       lower=0, upper=10, units='s')
        path_comp._add_path_constraint('c', var_class='control_rate2', shape=(3, 3),
                                       lower=0, upper=10, units='kg')

        self.p.model.connect('a_disc', 'path_constraints.all_values:a')
        self.p.model.connect('b_disc', 'path_constraints.all_values:b')
        self.p.model.connect('c_disc', 'path_constraints.all_values:c')

        self.p.setup()

        self.p.run_model()

    def test_results(self):
        p = self.p
        assert_almost_equal(p['a_disc'],
                            p['path_constraints.path:a'][...])

        assert_almost_equal(p['b_disc'],
                            p['path_constraints.path:b'][...])

        assert_almost_equal(p['c_disc'],
                            p['path_constraints.path:c'][...])

    def test_partials(self):
        np.set_printoptions(linewidth=1024, edgeitems=1000)
        cpd = self.p.check_partials(out_stream=None)
        assert_check_partials(cpd)
