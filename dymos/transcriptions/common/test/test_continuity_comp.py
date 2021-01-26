import unittest
import itertools

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.transcriptions.grid_data import GridData
from dymos.transcriptions.common import GaussLobattoContinuityComp, RadauPSContinuityComp
from dymos.phase.options import StateOptionsDictionary, ControlOptionsDictionary
from dymos.utils.testing_utils import assert_check_partials

# Modify class so we can run it standalone.
from dymos.utils.misc import CompWrapperConfig
GaussLobattoContinuityComp = CompWrapperConfig(GaussLobattoContinuityComp)
RadauPSContinuityComp = CompWrapperConfig(RadauPSContinuityComp)

params_list = itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                                ['compressed', 'uncompressed']  # compressed
                                )


@use_tempdirs
class TestContinuityComp(unittest.TestCase):

    def test_continuity_comp(self):
        for transcription, compressed in params_list:
            with self.subTest():

                dm.options['include_check_partials'] = True
                num_seg = 3

                gd = GridData(num_segments=num_seg,
                              transcription_order=[5, 3, 3],
                              segment_ends=[0.0, 3.0, 10.0, 20],
                              transcription=transcription,
                              compressed=compressed == 'compressed')

                self.p = om.Problem(model=om.Group())

                ivp = self.p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

                nn = gd.subset_num_nodes['all']

                ivp.add_output('x', val=np.arange(nn), units='m')
                ivp.add_output('y', val=np.arange(nn), units='m/s')
                ivp.add_output('u', val=np.zeros((nn, 3)), units='deg')
                ivp.add_output('v', val=np.arange(nn), units='N')
                ivp.add_output('u_rate', val=np.zeros((nn, 3)), units='deg/s')
                ivp.add_output('v_rate', val=np.arange(nn), units='N/s')
                ivp.add_output('u_rate2', val=np.zeros((nn, 3)), units='deg/s**2')
                ivp.add_output('v_rate2', val=np.arange(nn), units='N/s**2')
                ivp.add_output('t_duration', val=121.0, units='s')

                self.p.model.add_design_var('x', lower=0, upper=100)

                state_options = {'x': StateOptionsDictionary(),
                                 'y': StateOptionsDictionary()}
                control_options = {'u': ControlOptionsDictionary(),
                                   'v': ControlOptionsDictionary()}

                state_options['x']['units'] = 'm'
                state_options['y']['units'] = 'm/s'

                # Need these for later in the test.
                state_options['x']['shape'] = (1,)
                state_options['y']['shape'] = (1,)

                control_options['u']['units'] = 'deg'
                control_options['u']['shape'] = (3,)
                control_options['u']['continuity'] = True

                control_options['v']['units'] = 'N'

                # Need these for later in the test.
                state_options['x']['shape'] = (1,)
                state_options['y']['shape'] = (1,)
                control_options['v']['shape'] = (1,)

                if transcription == 'gauss-lobatto':
                    cnty_comp = GaussLobattoContinuityComp(grid_data=gd, time_units='s',
                                                           state_options=state_options,
                                                           control_options=control_options)
                elif transcription == 'radau-ps':
                    cnty_comp = RadauPSContinuityComp(grid_data=gd, time_units='s',
                                                      state_options=state_options,
                                                      control_options=control_options)
                else:
                    raise ValueError('unrecognized transcription')

                self.p.model.add_subsystem('cnty_comp', subsys=cnty_comp)
                # The sub-indices of state_disc indices that are segment ends
                num_seg_ends = gd.subset_num_nodes['segment_ends']
                segment_end_idxs = np.reshape(gd.subset_node_indices['segment_ends'],
                                              newshape=(num_seg_ends, 1))

                if compressed != 'compressed':
                    self.p.model.connect('x', 'cnty_comp.states:x', src_indices=segment_end_idxs)
                    self.p.model.connect('y', 'cnty_comp.states:y', src_indices=segment_end_idxs)

                self.p.model.connect('t_duration', 'cnty_comp.t_duration')

                size_u = nn * np.prod(control_options['u']['shape'])
                src_idxs_u = np.arange(size_u).reshape((nn,) + control_options['u']['shape'])
                src_idxs_u = src_idxs_u[gd.subset_node_indices['segment_ends'], ...]

                size_v = nn * np.prod(control_options['v']['shape'])
                src_idxs_v = np.arange(size_v).reshape((nn,) + control_options['v']['shape'])
                src_idxs_v = src_idxs_v[gd.subset_node_indices['segment_ends'], ...]

                # if transcription =='radau-ps' or compressed != 'compressed':
                self.p.model.connect('u', 'cnty_comp.controls:u', src_indices=src_idxs_u,
                                     flat_src_indices=True)

                self.p.model.connect('u_rate', 'cnty_comp.control_rates:u_rate', src_indices=src_idxs_u,
                                     flat_src_indices=True)

                self.p.model.connect('u_rate2', 'cnty_comp.control_rates:u_rate2', src_indices=src_idxs_u,
                                     flat_src_indices=True)

                # if transcription =='radau-ps' or compressed != 'compressed':
                self.p.model.connect('v', 'cnty_comp.controls:v', src_indices=src_idxs_v,
                                     flat_src_indices=True)

                self.p.model.connect('v_rate', 'cnty_comp.control_rates:v_rate', src_indices=src_idxs_v,
                                     flat_src_indices=True)

                self.p.model.connect('v_rate2', 'cnty_comp.control_rates:v_rate2', src_indices=src_idxs_v,
                                     flat_src_indices=True)

                self.p.setup(check=True, force_alloc_complex=True)

                self.p['x'] = np.random.rand(*self.p['x'].shape)
                self.p['y'] = np.random.rand(*self.p['y'].shape)
                self.p['u'] = np.random.rand(*self.p['u'].shape)
                self.p['v'] = np.random.rand(*self.p['v'].shape)
                self.p['u_rate'] = np.random.rand(*self.p['u'].shape)
                self.p['v_rate'] = np.random.rand(*self.p['v'].shape)
                self.p['u_rate2'] = np.random.rand(*self.p['u'].shape)
                self.p['v_rate2'] = np.random.rand(*self.p['v'].shape)

                self.p.run_model()

                if compressed != 'compressed':
                    for state in ('x', 'y'):
                        xpectd = self.p[state][segment_end_idxs, ...][2::2, ...] - \
                            self.p[state][segment_end_idxs, ...][1:-1:2, ...]

                        assert_near_equal(self.p['cnty_comp.defect_states:{0}'.format(state)],
                                          xpectd.reshape((num_seg - 1,) + state_options[state]['shape']))

                for ctrl in ('u', 'v'):

                    xpectd = self.p[ctrl][segment_end_idxs, ...][2::2, ...] - \
                        self.p[ctrl][segment_end_idxs, ...][1:-1:2, ...]

                    if compressed != 'compressed':
                        assert_near_equal(self.p['cnty_comp.defect_controls:{0}'.format(ctrl)],
                                          xpectd.reshape((num_seg - 1,) + control_options[ctrl]['shape']))

                np.set_printoptions(linewidth=1024)
                cpd = self.p.check_partials(method='cs', out_stream=None)
                assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
