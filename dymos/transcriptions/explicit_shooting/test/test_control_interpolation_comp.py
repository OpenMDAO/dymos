import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from sympy import Poly, poly
import dymos as dm

from dymos.utils.lgl import lgl
from dymos.transcriptions.explicit_shooting.vandermonde_control_interp_comp import VandermondeControlInterpComp
from dymos.transcriptions.explicit_shooting.barycentric_control_interp_comp import BarycentricControlInterpComp

from dymos.phase.options import TimeOptionsDictionary, ControlOptionsDictionary, \
    PolynomialControlOptionsDictionary

_TOL = 1.0E-8


class TestVandermondeControlInterpolationComp(unittest.TestCase):

    def test_eval_control_gl_compressed(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
                                                         transcription_order=[3, 5], compressed=True)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'rad'

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         control_options=control_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1

        p.set_val('interp.controls:u1', [[0.0, 3.0, 0.0, 4.0, 3.0, 4.0, 3.0]])

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), np.zeros((1, 1)), tolerance=_TOL)

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        p.set_val('interp.stau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=True, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        p.set_val('interp.stau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=True, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        interp_comp.options['segment_index'] = 0

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=True, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        p.set_val('interp.stau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=True, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        p.set_val('interp.stau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        p.set_val('interp.stau', 0.54262)
        p.run_model()
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

    def test_eval_control_radau_compressed(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='radau-ps',
                                                         transcription_order=[3, 5], compressed=True)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'rad'

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         control_options=control_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1
        p.set_val('interp.controls:u1', [0.0, 3.0, 1.5, 0.0, 4.0, 3.0, 4.0, 3.0])

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.stau', -0.72048)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[4.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', -0.167181)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.446314)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[4.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.885792)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)

        interp_comp.options['segment_index'] = 0

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', -0.28989795)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.68989795)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[1.5]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.54262)
        p.run_model()

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd)

    def test_eval_control_gl_uncompressed(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
                                                         transcription_order=[3, 5], compressed=False)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'rad'

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         control_options=control_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1
        p.set_val('interp.controls:u1', [0.0, 3.0, 0.0, 0.0, 4.0, 3.0, 4.0, 3.0])

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.stau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)

        p.set_val('interp.stau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)

        interp_comp.options['segment_index'] = 0

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.stau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)

        p.set_val('interp.stau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.stau', 0.54262)
        p.run_model()

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

    def test_eval_control_radau_uncompressed(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='radau-ps',
                                                         transcription_order=[3, 5], compressed=False)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'rad'

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         control_options=control_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1
        p.set_val('interp.controls:u1', [0.0, 3.0, 1.5, 0.0, 4.0, 3.0, 4.0, 3.0])

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.stau', -0.72048)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[4.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', -0.167181)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.446314)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[4.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.885792)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)

        interp_comp.options['segment_index'] = 0

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', -0.28989795)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.68989795)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[1.5]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.54262)
        p.run_model()

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

    def test_eval_control_radau_uncompressed_vectorized(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='radau-ps',
                                                         transcription_order=[3, 5], compressed=False)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'rad'

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         control_options=control_options,
                                                                         vec_size=5,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1
        p.set_val('interp.controls:u1', [0.0, 3.0, 1.5, 0.0, 4.0, 3.0, 4.0, 3.0])

        p.set_val('interp.stau', [-1.0, -0.72048, -0.167181, 0.446314, 0.885792])

        p.run_model()

        expected = np.array([[0.0, 4.0, 3.0, 4.0, 3.0]]).T

        assert_near_equal(p.get_val('interp.control_values:u1'), expected, tolerance=1.0E-6)

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)


class TestPolynomialVandermondeControlInterpolation(unittest.TestCase):

    def test_eval_polycontrol_gl_compressed(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
                                                         transcription_order=[3, 5], compressed=True)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        pc_options = {'u1': dm.phase.options.PolynomialControlOptionsDictionary()}

        pc_options['u1']['shape'] = (1,)
        pc_options['u1']['units'] = 'rad'
        pc_options['u1']['order'] = 6

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         polynomial_control_options=pc_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1
        p.set_val('interp.t_duration', 12.2352)
        p.set_val('interp.dstau_dt', .3526)
        p.set_val('interp.polynomial_controls:u1', [0.0, 3.0, 0.0, 1.5, 4.0, 3.0, 4.0])

        p.set_val('interp.ptau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.ptau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[1.5]], tolerance=_TOL)

        p.set_val('interp.ptau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[4.0]], tolerance=_TOL)

        interp_comp.options['segment_index'] = 0

        p.set_val('interp.ptau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.ptau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[1.5]], tolerance=_TOL)

        p.set_val('interp.ptau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[4.0]], tolerance=_TOL)

        p.set_val('interp.ptau', 0.54262)
        p.run_model()

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

    def test_eval_polycontrol_gl_compressed_vectorized(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
                                                         transcription_order=[3, 5], compressed=True)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        pc_options = {'u1': dm.phase.options.PolynomialControlOptionsDictionary()}

        pc_options['u1']['shape'] = (1,)
        pc_options['u1']['units'] = 'rad'
        pc_options['u1']['order'] = 6

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         vec_size=6,
                                                                         polynomial_control_options=pc_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1
        p.set_val('interp.t_duration', 12.2352)
        p.set_val('interp.dstau_dt', .3526)
        p.set_val('interp.polynomial_controls:u1', [0.0, 3.0, 0.0, 1.5, 4.0, 3.0, 4.0])

        p.set_val('interp.ptau', -1.0)
        p.run_model()

        ptau = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])

        p.set_val('interp.ptau', ptau)
        p.run_model()

        expected = np.array([[0.0, 1.5, 4.0, 0.0, 1.5, 4.0]]).T

        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), expected, tolerance=_TOL)

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

    def test_eval_polycontrol_radau_compressed(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='radau-ps',
                                                         transcription_order=[3, 5], compressed=True)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        pc_options = {'u1': dm.phase.options.PolyomialControlOptionsDictionary()}

        pc_options['u1']['shape'] = (1,)
        pc_options['u1']['units'] = 'rad'
        pc_options['u1']['order'] = 6

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         polynomial_control_options=pc_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        p.set_val('interp.t_duration', 12.252)
        interp_comp.options['segment_index'] = 1
        p.set_val('interp.polynomial_controls:u1', [0.0, 3.0, 0.0, 1.5, 4.0, 3.0, 4.0])

        p.set_val('interp.ptau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.ptau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[1.5]], tolerance=_TOL)

        p.set_val('interp.ptau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[4.0]], tolerance=_TOL)

        interp_comp.options['segment_index'] = 0

        p.set_val('interp.ptau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.ptau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[1.5]], tolerance=_TOL)

        p.set_val('interp.ptau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[4.0]], tolerance=_TOL)

        p.set_val('interp.ptau', 0.54262)
        p.run_model()

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)


class TestBarycentricControlInterpolationComp(unittest.TestCase):

    def test_single_segment(self):
        grid_data = dm.transcriptions.grid_data.BirkhoffGrid(num_nodes=21, grid_type='cgl')

        time_options = TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'rad'

        polynomial_control_options = {'p1': PolynomialControlOptionsDictionary()}

        polynomial_control_options['p1']['shape'] = (1,)
        polynomial_control_options['p1']['order'] = 5
        polynomial_control_options['p1']['units'] = 'kW'

        p = om.Problem()

        interp_comp = p.model.add_subsystem('interp',
                                            BarycentricControlInterpComp(grid_data=grid_data,
                                                                         control_options=control_options,
                                                                         polynomial_control_options=polynomial_control_options,
                                                                         standalone_mode=True,
                                                                         compute_derivs=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.set_segment_index(0)

        x_sample = (grid_data.node_stau + 1) * np.pi
        p.set_val('interp.controls:u1', np.sin(x_sample))
        p_sample = lgl(6)[0]
        p.set_val('interp.polynomial_controls:p1', p_sample ** 2)
        p.set_val('interp.t_duration', 2 * np.pi)
        p.set_val('interp.dstau_dt', 1 / np.pi)
        x_truth = np.linspace(0, 2 * np.pi, 101)
        truth = np.sin(x_truth)
        truth_rate = np.cos(x_truth)
        truth_rate2 = -np.sin(x_truth)

        results = []
        rate_results = []
        rate2_results = []

        p_results = []
        prate_results = []
        prate2_results = []
        for stau in np.linspace(-1, 1, 101):
            p.set_val('interp.stau', stau)
            p.set_val('interp.ptau', stau)

            p.run_model()

            results.append(p.get_val('interp.control_values:u1').ravel()[0])
            rate_results.append(p.get_val('interp.control_rates:u1_rate').ravel()[0])
            rate2_results.append(p.get_val('interp.control_rates:u1_rate2').ravel()[0])

            p_results.append(p.get_val('interp.polynomial_control_values:p1').ravel()[0])
            prate_results.append(p.get_val('interp.polynomial_control_rates:p1_rate').ravel()[0])
            prate2_results.append(p.get_val('interp.polynomial_control_rates:p1_rate2').ravel()[0])

        # assert_near_equal(results, truth, tolerance=1.0E-6)

        p.set_val('interp.stau', 0.66666666666667)
        p.set_val('interp.ptau', 0.66666666666667)
        p.run_model()

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=True, method='cs')#, out_stream=None)
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('MacOSX')
        plt.plot(x_sample, np.sin(x_sample), 'o')
        plt.plot(np.linspace(0, 2*np.pi, 101), results, '.')
        plt.plot(x_truth, truth, '-')
        # Now the rates
        plt.plot(np.linspace(0, 2*np.pi, 101), rate_results, '.')
        plt.plot(x_truth, truth_rate, '-')
        # Now the second derivatives
        plt.plot(np.linspace(0, 2*np.pi, 101), rate2_results, '.')
        plt.plot(x_truth, truth_rate2, '-')

        plt.figure()
        plt.plot(p_sample, p_sample**2, 'o')
        plt.plot(np.linspace(-1, 1, 101), p_results, '.')
        # plt.plot(x_truth, truth, '-')

        plt.show()

        # return

    def test_eval_control_gl_compressed(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
                                                         transcription_order=[3, 5], compressed=True)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'rad'

        p = om.Problem()
        # interp_comp = p.model.add_subsystem('interp',
        #                                     VandermondeControlInterpComp(grid_data=grid_data,
        #                                                                  control_options=control_options,
        #                                                                  standalone_mode=True,
        #                                                                  time_units='s'))
        interp_comp = p.model.add_subsystem('interp',
                                            BarycentricControlInterpComp(grid_data=grid_data,
                                                                         control_options=control_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.set_segment_index(1)

        p.set_val('interp.controls:u1', [[0.0, 3.0, 0.0, 4.0, 3.0, 4.0, 3.0]])

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), np.zeros((1, 1)), tolerance=_TOL)

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        p.set_val('interp.stau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=True, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        p.set_val('interp.stau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=True, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        interp_comp.options['segment_index'] = 0

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=True, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        p.set_val('interp.stau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=True, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        p.set_val('interp.stau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

        p.set_val('interp.stau', 0.54262)
        p.run_model()
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

    def test_eval_control_radau_compressed(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='radau-ps',
                                                         transcription_order=[3, 5], compressed=True)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'rad'

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         control_options=control_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1
        p.set_val('interp.controls:u1', [0.0, 3.0, 1.5, 0.0, 4.0, 3.0, 4.0, 3.0])

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.stau', -0.72048)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[4.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', -0.167181)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.446314)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[4.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.885792)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)

        interp_comp.options['segment_index'] = 0

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', -0.28989795)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.68989795)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[1.5]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.54262)
        p.run_model()

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd)

    def test_eval_control_gl_uncompressed(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
                                                         transcription_order=[3, 5], compressed=False)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'rad'

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         control_options=control_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1
        p.set_val('interp.controls:u1', [0.0, 3.0, 0.0, 0.0, 4.0, 3.0, 4.0, 3.0])

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.stau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)

        p.set_val('interp.stau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)

        interp_comp.options['segment_index'] = 0

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.stau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=_TOL)

        p.set_val('interp.stau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.stau', 0.54262)
        p.run_model()

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

    def test_eval_control_radau_uncompressed(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='radau-ps',
                                                         transcription_order=[3, 5], compressed=False)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'rad'

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         control_options=control_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1
        p.set_val('interp.controls:u1', [0.0, 3.0, 1.5, 0.0, 4.0, 3.0, 4.0, 3.0])

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.stau', -0.72048)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[4.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', -0.167181)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.446314)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[4.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.885792)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)

        interp_comp.options['segment_index'] = 0

        p.set_val('interp.stau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[0.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', -0.28989795)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[3.0]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.68989795)
        p.run_model()
        assert_near_equal(p.get_val('interp.control_values:u1'), [[1.5]], tolerance=1.0E-5)

        p.set_val('interp.stau', 0.54262)
        p.run_model()

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

    def test_eval_control_radau_uncompressed_vectorized(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='radau-ps',
                                                         transcription_order=[3, 5], compressed=False)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

        control_options['u1']['shape'] = (1,)
        control_options['u1']['units'] = 'rad'

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         control_options=control_options,
                                                                         vec_size=5,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1
        p.set_val('interp.controls:u1', [0.0, 3.0, 1.5, 0.0, 4.0, 3.0, 4.0, 3.0])

        p.set_val('interp.stau', [-1.0, -0.72048, -0.167181, 0.446314, 0.885792])

        p.run_model()

        expected = np.array([[0.0, 4.0, 3.0, 4.0, 3.0]]).T

        assert_near_equal(p.get_val('interp.control_values:u1'), expected, tolerance=1.0E-6)

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)


class TestPolynomialBarycentricControlInterpolation(unittest.TestCase):



    def test_eval_polycontrol_gl_compressed(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
                                                         transcription_order=[3, 5], compressed=True)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        pc_options = {'u1': dm.phase.options.PolynomialControlOptionsDictionary()}

        pc_options['u1']['shape'] = (1,)
        pc_options['u1']['units'] = 'rad'
        pc_options['u1']['order'] = 6

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         polynomial_control_options=pc_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1
        p.set_val('interp.t_duration', 12.2352)
        p.set_val('interp.dstau_dt', .3526)
        p.set_val('interp.polynomial_controls:u1', [0.0, 3.0, 0.0, 1.5, 4.0, 3.0, 4.0])

        p.set_val('interp.ptau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.ptau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[1.5]], tolerance=_TOL)

        p.set_val('interp.ptau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[4.0]], tolerance=_TOL)

        interp_comp.options['segment_index'] = 0

        p.set_val('interp.ptau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.ptau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[1.5]], tolerance=_TOL)

        p.set_val('interp.ptau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[4.0]], tolerance=_TOL)

        p.set_val('interp.ptau', 0.54262)
        p.run_model()

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

    def test_eval_polycontrol_gl_compressed_vectorized(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='gauss-lobatto',
                                                         transcription_order=[3, 5], compressed=True)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        pc_options = {'u1': dm.phase.options.PolynomialControlOptionsDictionary()}

        pc_options['u1']['shape'] = (1,)
        pc_options['u1']['units'] = 'rad'
        pc_options['u1']['order'] = 6

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         vec_size=6,
                                                                         polynomial_control_options=pc_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        interp_comp.options['segment_index'] = 1
        p.set_val('interp.t_duration', 12.2352)
        p.set_val('interp.dstau_dt', .3526)
        p.set_val('interp.polynomial_controls:u1', [0.0, 3.0, 0.0, 1.5, 4.0, 3.0, 4.0])

        p.set_val('interp.ptau', -1.0)
        p.run_model()

        ptau = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])

        p.set_val('interp.ptau', ptau)
        p.run_model()

        expected = np.array([[0.0, 1.5, 4.0, 0.0, 1.5, 4.0]]).T

        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), expected, tolerance=_TOL)

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)

    def test_eval_polycontrol_radau_compressed(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=2, transcription='radau-ps',
                                                         transcription_order=[3, 5], compressed=True)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        pc_options = {'u1': dm.phase.options.PolynomialControlOptionsDictionary()}

        pc_options['u1']['shape'] = (1,)
        pc_options['u1']['units'] = 'rad'
        pc_options['u1']['order'] = 6

        p = om.Problem()
        interp_comp = p.model.add_subsystem('interp',
                                            VandermondeControlInterpComp(grid_data=grid_data,
                                                                         polynomial_control_options=pc_options,
                                                                         standalone_mode=True,
                                                                         time_units='s'))
        p.setup(force_alloc_complex=True)

        p.set_val('interp.t_duration', 12.252)
        interp_comp.options['segment_index'] = 1
        p.set_val('interp.polynomial_controls:u1', [0.0, 3.0, 0.0, 1.5, 4.0, 3.0, 4.0])

        p.set_val('interp.ptau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.ptau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[1.5]], tolerance=_TOL)

        p.set_val('interp.ptau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[4.0]], tolerance=_TOL)

        interp_comp.options['segment_index'] = 0

        p.set_val('interp.ptau', -1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[0.0]], tolerance=_TOL)

        p.set_val('interp.ptau', 0.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[1.5]], tolerance=_TOL)

        p.set_val('interp.ptau', 1.0)
        p.run_model()
        assert_near_equal(p.get_val('interp.polynomial_control_values:u1'), [[4.0]], tolerance=_TOL)

        p.set_val('interp.ptau', 0.54262)
        p.run_model()

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='cs')
            assert_check_partials(cpd, atol=_TOL, rtol=_TOL)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
