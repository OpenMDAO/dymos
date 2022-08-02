import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
import dymos as dm

from dymos.transcriptions.explicit_shooting.vandermonde_control_interp_comp import VandermondeControlInterpComp

_TOL = 1.0E-8


class TestControlInterpolationComp(unittest.TestCase):

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


class TestPolynomialControlInterpolation(unittest.TestCase):

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
