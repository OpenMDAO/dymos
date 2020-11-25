import unittest

import numpy as np
from numpy.testing import assert_almost_equal
import openmdao.api as om
from dymos.utils.testing_utils import assert_check_partials

from dymos.transcriptions.pseudospectral.components import StateInterpComp
from dymos.transcriptions.grid_data import GridData
from dymos.utils.lgr import lgr

# Modify class so we can run it standalone.
import dymos as dm
from dymos.utils.misc import CompWrapperConfig
StateInterpComp = CompWrapperConfig(StateInterpComp)

SHOW_PLOTS = False

if SHOW_PLOTS:
    import matplotlib.pyplot as plt


# Test 1:  Let x = t**2, f = 2*t
def x(t):
    return t ** 2


def f_x(t):
    return 2 * t


# Test 1:  Let v = t**3-10*t**2, f = 3*t**2 - 20*t
def v(t):
    return t ** 3 - 10 * t ** 2


def f_v(t):
    return 3 * t ** 2 - 20 * t


class TestStateInterpComp(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_state_interp_comp_lobatto(self):

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=3,
                      segment_ends=segends,
                      transcription='gauss-lobatto')

        p = om.Problem(model=om.Group())

        states = {'x': {'units': 'm', 'shape': (1,)},
                  'v': {'units': 'm/s', 'shape': (1,)}}

        X_ivc = om.IndepVarComp()
        p.model.add_subsystem('X_ivc', X_ivc, promotes=['state_disc:x', 'state_disc:v'])

        X_ivc.add_output('state_disc:x', val=np.zeros(gd.subset_num_nodes['state_disc']),
                         units='m')

        X_ivc.add_output('state_disc:v', val=np.zeros(gd.subset_num_nodes['state_disc']),
                         units='m/s')

        F_ivc = om.IndepVarComp()
        p.model.add_subsystem('F_ivc', F_ivc, promotes=['staterate_disc:x', 'staterate_disc:v'])

        F_ivc.add_output('staterate_disc:x',
                         val=np.zeros(gd.subset_num_nodes['state_disc']),
                         units='m/s')

        F_ivc.add_output('staterate_disc:v',
                         val=np.zeros(gd.subset_num_nodes['state_disc']),
                         units='m/s**2')

        dt_dtau_ivc = om.IndepVarComp()
        p.model.add_subsystem('dt_dstau_ivc', dt_dtau_ivc, promotes=['dt_dstau'])

        dt_dtau_ivc.add_output('dt_dstau', val=0.0*np.zeros(gd.subset_num_nodes['col']), units='s')

        p.model.add_subsystem('state_interp_comp',
                              subsys=StateInterpComp(transcription='gauss-lobatto',
                                                     grid_data=gd,
                                                     state_options=states,
                                                     time_units='s'))

        p.model.connect('state_disc:x', 'state_interp_comp.state_disc:x')
        p.model.connect('state_disc:v', 'state_interp_comp.state_disc:v')
        p.model.connect('staterate_disc:x', 'state_interp_comp.staterate_disc:x')
        p.model.connect('staterate_disc:v', 'state_interp_comp.staterate_disc:v')
        p.model.connect('dt_dstau', 'state_interp_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        segends_disc = segends[np.array((0, 1, 1, 2), dtype=int)]

        p['state_disc:x'] = [x(t) for t in segends_disc]
        p['staterate_disc:x'] = [f_x(t) for t in segends_disc]

        p['state_disc:v'] = [v(t) for t in segends_disc]
        p['staterate_disc:v'] = [f_v(t) for t in segends_disc]

        p['dt_dstau'] = (segends[1:] - segends[:-1]) / 2.0

        p.run_model()

        t_disc = segends_disc
        t_col = (segends[1:] + segends[:-1]) / 2.0

        if SHOW_PLOTS:  # pragma: no cover
            f, ax = plt.subplots(2, 1)

            t = np.linspace(0, 10, 100)
            x1 = x(t)
            xdot1 = f_x(t)

            x2 = v(t)
            xdot2 = f_v(t)

            ax[0].plot(t, x1, 'b-', label='$x$')
            ax[0].plot(t, xdot1, 'b--', label=r'$\dot{x}$')
            ax[0].plot(t_disc, p['state_disc:x'], 'bo', label='$X_d:x$')
            ax[0].plot(t_col, p['state_interp_comp.state_col:x'], 'bv', label='$X_c:x$')
            ax[0].plot(t_col, p['state_interp_comp.staterate_col:x'], marker='v', color='None',
                       mec='b', label='$Xdot_c:x$')

            ax[1].plot(t, x2, 'r-', label='$v$')
            ax[1].plot(t, xdot2, 'r--', label=r'$\dot{v}$')
            ax[1].plot(t_disc, p['state_disc:v'], 'ro', label='$X_d:v$')
            ax[1].plot(t_col, p['state_interp_comp.state_col:v'], 'rv', label='$X_c:v$')
            ax[1].plot(t_col, p['state_interp_comp.staterate_col:v'], marker='v', color='None',
                       mec='r', label='$Xdot_c:v$')

            ax[0].legend(loc='upper left', ncol=3)
            ax[1].legend(loc='upper left', ncol=3)

            plt.show()

        # Test 1
        assert_almost_equal(
            p['state_interp_comp.state_col:x'][:, 0], x(t_col))
        assert_almost_equal(
            p['state_interp_comp.staterate_col:x'][:, 0], f_x(t_col))

        # Test 2
        assert_almost_equal(
            p['state_interp_comp.state_col:v'][:, 0], v(t_col))
        assert_almost_equal(
            p['state_interp_comp.staterate_col:v'][:, 0], f_v(t_col))

        cpd = p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd, atol=5.0E-5)

    def test_state_interp_comp_lobatto_vectorized(self):

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=3,
                      segment_ends=segends,
                      transcription='gauss-lobatto')

        p = om.Problem(model=om.Group())

        states = {'pos': {'units': 'm', 'shape': (2,)}}

        X_ivc = om.IndepVarComp()
        p.model.add_subsystem('X_ivc', X_ivc, promotes=['state_disc:pos'])

        X_ivc.add_output('state_disc:pos',
                         val=np.zeros((gd.subset_num_nodes['state_disc'], 2)), units='m')

        F_ivc = om.IndepVarComp()
        p.model.add_subsystem('F_ivc', F_ivc, promotes=['staterate_disc:pos'])

        F_ivc.add_output('staterate_disc:pos',
                         val=np.zeros((gd.subset_num_nodes['state_disc'], 2)),
                         units='m/s')

        dt_dtau_ivc = om.IndepVarComp()
        p.model.add_subsystem('dt_dstau_ivc', dt_dtau_ivc, promotes=['dt_dstau'])

        dt_dtau_ivc.add_output('dt_dstau', val=0.0*np.zeros(gd.subset_num_nodes['col']), units='s')

        p.model.add_subsystem('state_interp_comp',
                              subsys=StateInterpComp(transcription='gauss-lobatto',
                                                     grid_data=gd,
                                                     state_options=states,
                                                     time_units='s'))

        p.model.connect('state_disc:pos', 'state_interp_comp.state_disc:pos')
        p.model.connect('staterate_disc:pos', 'state_interp_comp.staterate_disc:pos')
        p.model.connect('dt_dstau', 'state_interp_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        segends_disc = segends[np.array((0, 1, 1, 2), dtype=int)]

        p['state_disc:pos'][:, 0] = [x(t) for t in segends_disc]  # [0.0, 25.0, 25.0, 100.0]
        p['staterate_disc:pos'][:, 0] = [f_x(t) for t in segends_disc]

        p['state_disc:pos'][:, 1] = [v(t) for t in segends_disc]
        p['staterate_disc:pos'][:, 1] = [f_v(t) for t in segends_disc]

        p['dt_dstau'] = (segends[1:] - segends[:-1]) / 2.0

        p.run_model()

        t_disc = segends_disc
        t_col = (segends[1:] + segends[:-1]) / 2.0

        if SHOW_PLOTS:  # pragma: no cover
            f, ax = plt.subplots(2, 1)

            print(t_disc)
            print(t_col)
            print(p['dt_dstau'])
            print(p['state_disc:pos'][:, 0])
            print(p['staterate_disc:pos'][:, 0])
            print(p['state_disc:pos'][:, 0])
            print(p['staterate_disc:pos'][:, 1])

            t = np.linspace(0, 10, 100)
            x1 = x(t)
            xdot1 = f_x(t)

            x2 = v(t)
            xdot2 = f_v(t)

            ax[0].plot(t, x1, 'b-', label='$x$')
            ax[0].plot(t, xdot1, 'b--', label='r$\dot{x}$')
            ax[0].plot(t_disc, p['state_disc:pos'][:, 0], 'bo', label='$X_d:pos$')
            ax[0].plot(t_col, p['state_interp_comp.state_col:pos'][:, 0], 'bv', label='$X_c:pos$')
            ax[0].plot(t_col, p['state_interp_comp.staterate_col:pos'][:, 0], marker='v',
                       color='None', mec='b', label='$Xdot_c:pos$')

            ax[1].plot(t, x2, 'r-', label='$v$')
            ax[1].plot(t, xdot2, 'r--', label='r$\dot{v}$')
            ax[1].plot(t_disc, p['state_disc:pos'][:, 1], 'ro', label='$X_d:vel$')
            ax[1].plot(t_col, p['state_interp_comp.state_col:pos'][:, 1], 'rv', label='$X_c:vel$')
            ax[1].plot(t_col, p['state_interp_comp.staterate_col:pos'][:, 1], marker='v',
                       color='None', mec='r', label='$Xdot_c:vel$')

            ax[0].legend(loc='upper left', ncol=3)
            ax[1].legend(loc='upper left', ncol=3)

            plt.show()

        # Test 1
        assert_almost_equal(
            p['state_interp_comp.state_col:pos'][:, 0], x(t_col))
        assert_almost_equal(
            p['state_interp_comp.staterate_col:pos'][:, 0], f_x(t_col))

        # Test 2
        assert_almost_equal(
            p['state_interp_comp.state_col:pos'][:, 1], v(t_col))
        assert_almost_equal(
            p['state_interp_comp.staterate_col:pos'][:, 1], f_v(t_col))

        cpd = p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd, atol=5.0E-5)

    def test_state_interp_comp_lobatto_vectorized_different_orders(self):

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=[3, 5],
                      segment_ends=segends,
                      transcription='gauss-lobatto')

        p = om.Problem(model=om.Group())

        states = {'pos': {'units': 'm', 'shape': (2,)}}

        X_ivc = om.IndepVarComp()
        p.model.add_subsystem('X_ivc', X_ivc, promotes=['state_disc:pos'])

        X_ivc.add_output('state_disc:pos',
                         val=np.zeros((gd.subset_num_nodes['state_disc'], 2)), units='m')

        F_ivc = om.IndepVarComp()
        p.model.add_subsystem('F_ivc', F_ivc, promotes=['staterate_disc:pos'])

        F_ivc.add_output('staterate_disc:pos',
                         val=np.zeros((gd.subset_num_nodes['state_disc'], 2)),
                         units='m/s')

        dt_dtau_ivc = om.IndepVarComp()
        p.model.add_subsystem('dt_dstau_ivc', dt_dtau_ivc, promotes=['dt_dstau'])

        dt_dtau_ivc.add_output('dt_dstau', val=0.0*np.zeros(gd.subset_num_nodes['col']), units='s')

        p.model.add_subsystem('state_interp_comp',
                              subsys=StateInterpComp(transcription='gauss-lobatto',
                                                     grid_data=gd,
                                                     state_options=states,
                                                     time_units='s'))

        p.model.connect('state_disc:pos', 'state_interp_comp.state_disc:pos')
        p.model.connect('staterate_disc:pos', 'state_interp_comp.staterate_disc:pos')
        p.model.connect('dt_dstau', 'state_interp_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        segends_disc = np.array((0, 3, 3, 6.5, 10))

        p['state_disc:pos'][:, 0] = [x(t) for t in segends_disc]  # [0.0, 25.0, 25.0, 100.0]
        p['staterate_disc:pos'][:, 0] = [f_x(t) for t in segends_disc]

        p['state_disc:pos'][:, 1] = [v(t) for t in segends_disc]
        p['staterate_disc:pos'][:, 1] = [f_v(t) for t in segends_disc]

        p['dt_dstau'] = [3.0/2., 7.0/2, 7.0/2]

        p.run_model()

        cpd = p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd, atol=5.0E-5)

    def test_state_interp_comp_radau(self):

        gd = GridData(num_segments=1,
                      transcription_order=3,
                      segment_ends=np.array([0, 10]),
                      transcription='radau-ps')

        p = om.Problem(model=om.Group())

        states = {'x': {'units': 'm', 'shape': (1,)},
                  'v': {'units': 'm/s', 'shape': (1,)}}

        X_ivc = om.IndepVarComp()
        p.model.add_subsystem('X_ivc', X_ivc, promotes=['state_disc:x', 'state_disc:v'])

        X_ivc.add_output('state_disc:x', val=np.zeros(gd.subset_num_nodes['state_disc']),
                         units='m')

        X_ivc.add_output('state_disc:v', val=np.zeros(gd.subset_num_nodes['state_disc']),
                         units='m/s')

        dt_dtau_ivc = om.IndepVarComp()
        dt_dtau_ivc.add_output('dt_dstau', val=0.0*np.zeros(gd.subset_num_nodes['col']), units='s')

        p.model.add_subsystem('dt_dstau_ivc', dt_dtau_ivc, promotes=['dt_dstau'])

        p.model.add_subsystem('state_interp_comp',
                              subsys=StateInterpComp(transcription='radau-ps',
                                                     grid_data=gd,
                                                     state_options=states,
                                                     time_units='s'))

        p.model.connect('state_disc:x', 'state_interp_comp.state_disc:x')
        p.model.connect('state_disc:v', 'state_interp_comp.state_disc:v')
        p.model.connect('dt_dstau', 'state_interp_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        lgr_nodes, lgr_weights = lgr(3, include_endpoint=True)
        t_disc = (lgr_nodes + 1.0) * 5.0
        t_col = t_disc[:-1]

        # Test 1:  Let x = t**2, f = 2*t
        p['state_disc:x'] = t_disc**2

        # Test 1:  Let v = t**3-10*t**2, f = 3*t**2 - 20*t
        p['state_disc:v'] = t_disc**3-10*t_disc**2

        p['dt_dstau'] = 10/2.0

        p.run_model()

        if SHOW_PLOTS:  # pragma: no cover
            f, ax = plt.subplots(2, 1)

            t_disc = np.array([0, 5, 10])
            t_col = np.array([2.5, 7.5])

            t = np.linspace(0, 10, 100)
            x1 = t**2
            xdot1 = 2*t

            x2 = t**3 - 10*t**2
            xdot2 = 3*t**2 - 20*t

            ax[0].plot(t, x1, 'b-', label='$x$')
            ax[0].plot(t, xdot1, 'b--', label='$\dot{x}$')
            ax[0].plot(t_disc, p['state_disc:x'], 'bo', label='$X_d:x$')
            ax[0].plot(t_col, p['state_interp_comp.state_col:x'], 'bv', label='$X_c:x$')
            ax[0].plot(t_col, p['state_interp_comp.staterate_col:x'], marker='v', color='None',
                       mec='b', label='$Xdot_c:x$')

            ax[1].plot(t, x2, 'r-', label='$v$')
            ax[1].plot(t, xdot2, 'r--', label='$\dot{v}$')
            ax[1].plot(t_disc, p['state_disc:v'], 'ro', label='$X_d:v$')
            ax[1].plot(t_col, p['state_interp_comp.state_col:v'], 'rv', label='$X_c:v$')
            ax[1].plot(t_col, p['state_interp_comp.staterate_col:v'], marker='v', color='None',
                       mec='r', label='$Xdot_c:v$')

            ax[0].legend(loc='upper left', ncol=3)
            ax[1].legend(loc='upper left', ncol=3)

            plt.show()

        # Test 1
        assert_almost_equal(
            p['state_interp_comp.staterate_col:x'][:, 0], 2*t_col)

        # Test 2
        assert_almost_equal(
            p['state_interp_comp.staterate_col:v'][:, 0], 3*t_col**2 - 20*t_col)

        cpd = p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd, atol=1.0E-5)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
