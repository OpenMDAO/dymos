from copy import deepcopy
import importlib
import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, set_env_vars_context

import numpy as np
import openmdao.api as om
import dymos
from dymos.examples.brachistochrone.doc.brachistochrone_ode import BrachistochroneODE

@use_tempdirs
class TestSetVal(unittest.TestCase):

    def test_set_state_val(self):

        for dymos_2 in ['1', '0']:

            with set_env_vars_context(DYMOS_2=dymos_2):

                dm = importlib.reload(dymos)

                for tx in (dm.Radau(num_segments=5, order=3),
                           dm.GaussLobatto(num_segments=5, order=3),
                           dm.Birkhoff(num_nodes=30),
                           dm.PicardShooting(num_segments=3, nodes_per_seg=11, solve_segments='forward'),
                           dm.PicardShooting(num_segments=3, nodes_per_seg=11, solve_segments='backward')
                        ):

                    with self.subTest(f'{tx.__class__.__name__}'):

                        p = om.Problem()

                        traj = dm.Trajectory()
                        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
                        p.model.add_subsystem('phase', phase)
                        p.model.add_subsystem('traj', traj)

                        phase.set_state_options('x')
                        phase.set_state_options('y')
                        phase.set_state_options('v')

                        p.setup()

                        phase.set_time_val(initial=0., duration=1.8016)
                        phase.set_state_val('x', [0.0, 10.0], units='m')
                        phase.set_state_val('y', np.array([10.0, 5.0]) / 0.3048, units='ft')
                        phase.set_state_val('v', np.array([0.1, 100.0]), time_vals=[0, 1.8016], units='m/s')

                        p.final_setup()

                        x = phase.get_val('states:x')
                        y = phase.get_val('states:y')
                        v = phase.get_val('states:v')

                        # Test initial_states and final_states in those transcriptions which support them
                        if dymos_2 == '1' and isinstance(tx, dm.Radau) or isinstance(tx, (dm.Birkhoff, dm.PicardShooting)):
                            x0 = phase.get_val('initial_states:x')[0]
                            y0 = phase.get_val('initial_states:y')[0]
                            v0 = phase.get_val('initial_states:v')[0]

                            xf = phase.get_val('final_states:x')[0]
                            yf = phase.get_val('final_states:y')[0]
                            vf = phase.get_val('final_states:v')[0]

                            assert_near_equal(x0, 0.0)
                            assert_near_equal(x[0], 0.0)

                            assert_near_equal(y0, 10.0)
                            assert_near_equal(y[0], 10.0)

                            assert_near_equal(v0, 0.1)
                            assert_near_equal(v[0], 0.1)

                            assert_near_equal(xf, 10.0)
                            assert_near_equal(x[-1], 10.0)

                            assert_near_equal(yf, 5.0)
                            assert_near_equal(y[-1], 5.0)

                            assert_near_equal(vf, 100.0)
                            assert_near_equal(v[-1], 100.0)

                        if isinstance(tx, dm.Birkhoff):
                            # For birkhoff, guesses of state_rates:v should be applied since time values were provided.
                            x_rate = phase.get_val('state_rates:x')
                            y_rate = phase.get_val('state_rates:y')
                            v_rate = phase.get_val('state_rates:v')

                            assert_near_equal(x_rate, np.zeros_like(x_rate))
                            assert_near_equal(y_rate, np.zeros_like(y_rate))
                            assert_near_equal(v_rate, (100.0 - 0.1) / 1.8016 * np.ones_like(x_rate), tolerance=1.0E-12)

                        if isinstance(tx, dm.PicardShooting):
                            seg_end_idxs_in_all = phase.options['transcription'].grid_data.subset_node_indices['segment_ends']
                            seg_start_idxs = seg_end_idxs_in_all[::2]
                            seg_end_idxs = seg_end_idxs_in_all[1::2]
                            if phase.state_options['x']['solve_segments'] == 'forward':
                                x0_segs = phase.get_val(f'picard_update_comp.seg_initial_states:x')
                                y0_segs = phase.get_val(f'picard_update_comp.seg_initial_states:y')
                                v0_segs = phase.get_val(f'picard_update_comp.seg_initial_states:v')

                                assert_near_equal(x0_segs.ravel(), x[seg_start_idxs].ravel())
                                assert_near_equal(y0_segs.ravel(), y[seg_start_idxs].ravel())
                                assert_near_equal(v0_segs.ravel(), v[seg_start_idxs].ravel())
                            elif phase.state_options['x']['solve_segments'] == 'backward':
                                xf_segs = phase.get_val(f'picard_update_comp.seg_final_states:x')
                                yf_segs = phase.get_val(f'picard_update_comp.seg_final_states:y')
                                vf_segs = phase.get_val(f'picard_update_comp.seg_final_states:v')

                                assert_near_equal(xf_segs.ravel(), x[seg_end_idxs].ravel())
                                assert_near_equal(yf_segs.ravel(), y[seg_end_idxs].ravel())
                                assert_near_equal(vf_segs.ravel(), v[seg_end_idxs].ravel())

                        # Now set the values at all nodes.

                        rand_vals = np.random.random((tx.grid_data.subset_num_nodes['state_input'], 1))

                        x_new = x * rand_vals
                        y_new = y * rand_vals
                        v_new = v * rand_vals

                        phase.set_state_val('x', deepcopy(x_new))
                        phase.set_state_val('y', deepcopy(y_new))
                        phase.set_state_val('v', deepcopy(v_new))

                        x = phase.get_val('states:x')
                        y = phase.get_val('states:y')
                        v = phase.get_val('states:v')

                        assert_near_equal(x, x_new)
                        assert_near_equal(y, y_new)
                        assert_near_equal(v, v_new)



if __name__ == '__main__':  # pragma: no cover
    unittest.main()
