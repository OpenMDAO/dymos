import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om

from dymos.transcriptions.grid_data import GridData
from dymos.transcriptions.common import TimeComp

from dymos.utils.misc import CompWrapperConfig, GroupWrapperConfig
TimeComp = CompWrapperConfig(TimeComp)

_segends = np.array([0.0, 3.0, 10.0, 20])

# LGL node locations per http://mathworld.wolfram.com/LobattoQuadrature.html
_lgl_nodes = {3: np.array([-1.0, 0.0, 1.0]),
              5: np.array([-1.0, -np.sqrt(21) / 7.0, 0.0, np.sqrt(21) / 7.0, 1.0])}

# LGR node locations per http://mathworld.wolfram.com/RadauQuadrature.html with endpoint added
_lgr_nodes = {3: np.array([-1.0, (1 - np.sqrt(6)) / 5.0, (1 + np.sqrt(6)) / 5.0, 1.0]),
              5: np.array([-1.0, -0.72048, -0.167181, 0.446314, 0.885792, 1.0])}


class TestTimeComp(unittest.TestCase):

    def test_results_gauss_lobatto(self):

        gd = GridData(num_segments=3,
                      transcription_order=[5, 3, 3],
                      segment_ends=_segends,
                      transcription='gauss-lobatto')

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem(name='ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=100.0, units='s')

        p.model.add_subsystem('time',
                              TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                       node_dptau_dstau=gd.node_dptau_dstau, units='s'))

        p.model.connect('t_initial', 'time.t_initial')
        p.model.connect('t_duration', 'time.t_duration')

        p.setup(check=True, force_alloc_complex=True)

        p.run_model()

        # Affine transform the nodes [0, 100]
        dt_dptau = (p['t_duration'] - p['t_initial']) / 2.0

        nodes = []

        dt_dstau_per_node = []

        for i in range(gd.num_segments):
            a, b = gd.segment_ends[i], gd.segment_ends[i + 1]  # segment ends in phase tau space

            # ratio of phase tau to segment tau within the segment
            dptau_dstau = (b - a) / 2.0

            dt_dstau = dt_dptau * dptau_dstau * np.ones(gd.subset_num_nodes_per_segment['all'][i])
            dt_dstau_per_node.extend(dt_dstau.tolist())

            # nodes of current segment in segment tau space
            nodes_segi_stau = _lgl_nodes[gd.transcription_order[i]]

            # nodes of current segment in phase tau space
            nodes_segi_ptau = a + (nodes_segi_stau + 1) * dptau_dstau

            # nodes in time
            nodes_time = (nodes_segi_ptau + 1) * dt_dptau

            nodes.extend(nodes_time.tolist())

        assert_almost_equal(p['time.time'], nodes)
        assert_almost_equal(p['time.dt_dstau'], dt_dstau_per_node)

    def test_results_radau(self):

        gd = GridData(num_segments=3,
                      transcription_order=[5, 3, 3],
                      segment_ends=_segends,
                      transcription='radau-ps')

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem(name='ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=100.0, units='s')

        p.model.add_subsystem('time',
                              TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                       node_dptau_dstau=gd.node_dptau_dstau, units='s'))

        p.model.connect('t_initial', 'time.t_initial')
        p.model.connect('t_duration', 'time.t_duration')

        p.setup(check=True, force_alloc_complex=True)

        p.run_model()

        # Affine transform the nodes [0, 100]
        dt_dptau = (p['t_duration'] - p['t_initial']) / 2.0

        nodes = []

        dt_dstau_per_node = []

        for i in range(gd.num_segments):
            a, b = gd.segment_ends[i], gd.segment_ends[i + 1]  # segment ends in phase tau space

            # ratio of phase tau to segment tau within the segment
            dptau_dstau = (b - a) / 2.0

            dt_dstau = dt_dptau * dptau_dstau * np.ones(gd.subset_num_nodes_per_segment['all'][i])
            dt_dstau_per_node.extend(dt_dstau.tolist())

            # nodes of current segment in segment tau space
            nodes_segi_stau = _lgr_nodes[gd.transcription_order[i]]

            # nodes of current segment in phase tau space
            nodes_segi_ptau = a + (nodes_segi_stau + 1) * dptau_dstau

            # nodes in time
            nodes_time = (nodes_segi_ptau + 1) * dt_dptau

            nodes.extend(nodes_time.tolist())

        assert_almost_equal(p['time.time'], nodes, decimal=4)
        assert_almost_equal(p['time.dt_dstau'], dt_dstau_per_node)


if __name__ == "__main__":
    unittest.main()
