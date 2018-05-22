from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials

from dymos.phases.components import ControlInputComp
from dymos.phases.grid_data import GridData
from dymos.phases.options import ControlOptionsDictionary


class TestControlInputComp(unittest.TestCase):

    def setUp(self):

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=[0.0, 3.0, 10.0],
                      transcription='gauss-lobatto')

        nn = gd.subset_num_nodes['all']

        self.p = Problem(model=Group())

        controls = {'a': ControlOptionsDictionary(),
                    'b': ControlOptionsDictionary(),
                    'c': ControlOptionsDictionary(),
                    'd': ControlOptionsDictionary()}

        controls['a'].update({'units': 'm', 'shape': (1,), 'dynamic': True, 'opt': False})
        controls['b'].update({'units': 'N', 'shape': (1,), 'dynamic': True, 'opt': False})
        controls['c'].update({'units': 's', 'shape': (3,), 'dynamic': True, 'opt': False})
        controls['d'].update({'units': 'kg', 'shape': (3, 3), 'dynamic': True, 'opt': False})

        ivc = IndepVarComp()
        self.p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('controls:a', val=np.zeros((nn, 1)), units='m')
        ivc.add_output('controls:b', val=np.zeros((nn, 1)), units='N')
        ivc.add_output('controls:c', val=np.zeros((nn, 3)), units='s')
        ivc.add_output('controls:d', val=np.zeros((nn, 3, 3)), units='kg')

        self.p.model.add_subsystem('input_controls',
                                   subsys=ControlInputComp(num_nodes=nn, control_options=controls))

        self.p.model.connect('controls:a', 'input_controls.controls:a')
        self.p.model.connect('controls:b', 'input_controls.controls:b')
        self.p.model.connect('controls:c', 'input_controls.controls:c')
        self.p.model.connect('controls:d', 'input_controls.controls:d')

        self.p.setup()

        self.p.run_model()

    def test_results(self):
        assert_almost_equal(self.p['input_controls.controls:a_out'], self.p['controls:a'])
        assert_almost_equal(self.p['input_controls.controls:b_out'], self.p['controls:b'])
        assert_almost_equal(self.p['input_controls.controls:c_out'], self.p['controls:c'])
        assert_almost_equal(self.p['input_controls.controls:d_out'], self.p['controls:d'])

    def test_partials(self):
        cpd = self.p.check_partials(out_stream=None)
        assert_check_partials(cpd)


if __name__ == '__main__':
    unittest.main()
