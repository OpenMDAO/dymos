from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials

from dymos.phases.grid_data import GridData
from dymos.phases.components import ContinuityComp
from dymos.phases.options import StateOptionsDictionary, ControlOptionsDictionary


class TestContinuityComp(unittest.TestCase):

    def setUp(self):

        gd = GridData(num_segments=3,
                      transcription_order=[5, 3, 3],
                      segment_ends=[0.0, 3.0, 10.0, 20],
                      transcription='gauss-lobatto')

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        ndn = gd.subset_num_nodes['disc']
        nn = gd.subset_num_nodes['all']

        ivp.add_output('x', val=np.arange(ndn), units='m')
        ivp.add_output('y', val=np.arange(ndn), units='m/s')
        ivp.add_output('u', val=np.zeros((nn, 3)), units='deg')
        ivp.add_output('v', val=np.arange(nn), units='N')

        self.p.model.add_design_var('x', lower=0, upper=100)

        state_options = {'x': StateOptionsDictionary(),
                         'y': StateOptionsDictionary()}
        control_options = {'u': ControlOptionsDictionary(),
                           'v': ControlOptionsDictionary()}

        state_options['x']['units'] = 'm'
        state_options['y']['units'] = 'm/s'

        control_options['u']['units'] = 'deg'
        control_options['u']['shape'] = (3,)
        control_options['u']['continuity'] = True

        control_options['v']['units'] = 'N'

        cnty_comp = ContinuityComp(grid_data=gd, time_units='s',
                                   state_options=state_options, control_options=control_options)

        self.p.model.add_subsystem('cnty_comp', subsys=cnty_comp)

        self.p.model.connect('x', 'cnty_comp.states:x')
        self.p.model.connect('y', 'cnty_comp.states:y')

        size_u = nn * np.prod(control_options['u']['shape'])
        src_idxs_u = np.arange(size_u).reshape((nn,) + control_options['u']['shape'])
        src_idxs_u = src_idxs_u[gd.subset_node_indices['disc'], ...]

        size_v = nn * np.prod(control_options['v']['shape'])
        src_idxs_v = np.arange(size_v).reshape((nn,) + control_options['v']['shape'])
        src_idxs_v = src_idxs_v[gd.subset_node_indices['disc'], ...]

        self.p.model.connect('u', 'cnty_comp.controls:u', src_indices=src_idxs_u,
                             flat_src_indices=True)

        self.p.model.connect('v', 'cnty_comp.controls:v', src_indices=src_idxs_v,
                             flat_src_indices=True)

        self.p.setup(mode='fwd')

        self.p['x'] = np.random.rand(*self.p['x'].shape)
        self.p['y'] = np.random.rand(*self.p['y'].shape)
        self.p['u'] = np.random.rand(*self.p['u'].shape)
        self.p['v'] = np.random.rand(*self.p['v'].shape)

        self.p.run_model()

    def test_results(self):

        for state in ('x', 'y'):
            assert_almost_equal(self.p['cnty_comp.defect_states:{0}'.format(state)][0, ...],
                                self.p[state][2, ...] - self.p[state][3, ...])
            assert_almost_equal(self.p['cnty_comp.defect_states:{0}'.format(state)][1, ...],
                                self.p[state][4, ...] - self.p[state][5, ...])

        for state in ('u', 'v'):
            assert_almost_equal(self.p['cnty_comp.defect_controls:{0}'.format(state)][0, ...],
                                self.p[state][4, ...] - self.p[state][5, ...])
            assert_almost_equal(self.p['cnty_comp.defect_controls:{0}'.format(state)][1, ...],
                                self.p[state][7, ...] - self.p[state][8, ...])

    def test_partials(self):
        cpd = self.p.check_partials(suppress_output=True)
        assert_check_partials(cpd)


if __name__ == '__main__':
    unittest.main()
