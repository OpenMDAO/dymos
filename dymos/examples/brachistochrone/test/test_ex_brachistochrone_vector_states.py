import os
import unittest
from numpy.testing import assert_almost_equal

import dymos.examples.brachistochrone.test.ex_brachistochrone_vector_states as ex_brachistochrone_vs
from openmdao.utils.testing_utils import use_tempdirs

from openmdao.utils.general_utils import set_pyoptsparse_opt, printoptions
from openmdao.utils.assert_utils import assert_check_partials

OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')


class TestBrachistochroneVectorStatesExample(unittest.TestCase):

    def assert_results(self, p):
        t_initial = p.get_val('phase0.time')[0]
        t_final = p.get_val('phase0.time')[-1]

        x0 = p.get_val('phase0.timeseries.states:pos')[0, 0]
        xf = p.get_val('phase0.timeseries.states:pos')[0, -1]

        y0 = p.get_val('phase0.timeseries.states:pos')[0, 1]
        yf = p.get_val('phase0.timeseries.states:pos')[-1, 1]

        v0 = p.get_val('phase0.timeseries.states:v')[0, 0]
        vf = p.get_val('phase0.timeseries.states:v')[-1, 0]

        g = p.get_val('phase0.timeseries.design_parameters:g')

        thetaf = p.get_val('phase0.timeseries.controls:theta')[-1, 0]

        assert_almost_equal(t_initial, 0.0)
        assert_almost_equal(x0, 0.0)
        assert_almost_equal(y0, 10.0)
        assert_almost_equal(v0, 0.0)

        assert_almost_equal(t_final, 1.8016, decimal=4)
        assert_almost_equal(xf, 10.0, decimal=3)
        assert_almost_equal(yf, 5.0, decimal=3)
        assert_almost_equal(vf, 9.902, decimal=3)
        assert_almost_equal(g, 9.80665, decimal=3)

        assert_almost_equal(thetaf, 100.12, decimal=0)

    def assert_partials(self, p):
        with printoptions(linewidth=1024, edgeitems=100):
            cpd = p.check_partials(method='cs')
        assert_check_partials(cpd)

    @use_tempdirs
    def test_ex_brachistochrone_vs_radau_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           run_driver=True)
        self.assert_results(p)
        self.assert_partials(p)
        self.tearDown()
        if os.path.exists('ex_brachvs_radau_compressed.db'):
            os.remove('ex_brachvs_radau_compressed.db')

    @use_tempdirs
    def test_ex_brachistochrone_vs_radau_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=False,
                                                           force_alloc_complex=True,
                                                           run_driver=True)
        self.assert_results(p)
        self.assert_partials(p)
        self.tearDown()
        if os.path.exists('ex_brachvs_radau_uncompressed.db'):
            os.remove('ex_brachvs_radau_uncompressed.db')

    @use_tempdirs
    def test_ex_brachistochrone_vs_gl_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           run_driver=True)

        self.assert_results(p)
        self.assert_partials(p)
        self.tearDown()
        if os.path.exists('ex_brachvs_gl_compressed.db'):
            os.remove('ex_brachvs_gl_compressed.db')

    @use_tempdirs
    def test_ex_brachistochrone_vs_gl_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           transcription_order=5,
                                                           compressed=False,
                                                           force_alloc_complex=True,
                                                           run_driver=True)
        self.assert_results(p)
        self.assert_partials(p)
        self.tearDown()
        if os.path.exists('ex_brachvs_gl_compressed.db'):
            os.remove('ex_brachvs_gl_compressed.db')

    @use_tempdirs
    def test_ex_brachistochrone_vs_rungekutta_compressed(self):
        import openmdao.api as om
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_vector_states_ode import \
            BrachistochroneVectorStatesODE

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.RungeKutta(num_segments=20, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos',
                        shape=(2,),
                        rate_source=BrachistochroneVectorStatesODE.states['pos']['rate_source'],
                        units=BrachistochroneVectorStatesODE.states['pos']['units'],
                        fix_initial=True, fix_final=False)
        phase.add_state('v',
                        rate_source=BrachistochroneVectorStatesODE.states['v']['rate_source'],
                        targets=BrachistochroneVectorStatesODE.states['v']['targets'],
                        units=BrachistochroneVectorStatesODE.states['v']['units'],
                        fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg',
                          targets=BrachistochroneVectorStatesODE.parameters['theta']['targets'],
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_design_parameter('g',
                                   targets=BrachistochroneVectorStatesODE.parameters['g']['targets'],
                                   units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('pos', loc='final', lower=[10, 5])

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.80162174

        pos0 = [0, 10]
        posf = [10, 5]

        p['phase0.states:pos'] = phase.interpolate(ys=[pos0, posf], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[0.46, 100.22900215],
                                                       nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_driver()

        self.assert_results(p)
        self.tearDown()


if __name__ == "__main__":
    unittest.main()
