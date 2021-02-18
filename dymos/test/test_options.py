import unittest

from openmdao.utils.testing_utils import use_tempdirs
import dymos as dm
from dymos.examples.brachistochrone.test.ex_brachistochrone import brachistochrone_min_time


@use_tempdirs
class TestOptions(unittest.TestCase):

    def test_include_check_partials_false_radau(self):
        dm.options['include_check_partials'] = False
        p = brachistochrone_min_time(transcription='radau-ps', compressed=False,
                                     run_driver=False, force_alloc_complex=True)
        cpd = p.check_partials(out_stream=None)
        self.assertSetEqual(set(cpd.keys()), {'traj0.phases.phase0.rhs_all'})

    def test_include_check_partials_false_gl(self):
        dm.options['include_check_partials'] = False
        p = brachistochrone_min_time(transcription='gauss-lobatto', compressed=False,
                                     run_driver=False, force_alloc_complex=True)
        cpd = p.check_partials(out_stream=None, method='fd')
        self.assertSetEqual(set(cpd.keys()), {'traj0.phases.phase0.rhs_disc',
                                              'traj0.phases.phase0.rhs_col'})

    def test_include_check_partials_true_radau(self):
        dm.options['include_check_partials'] = True
        p = brachistochrone_min_time(transcription='radau-ps', compressed=False,
                                     run_driver=False, force_alloc_complex=True)
        cpd = p.check_partials(out_stream=None, method='fd')
        self.assertTrue(len(list(cpd.keys())) > 1)

    def test_include_check_partials_true_gl(self):
        dm.options['include_check_partials'] = True
        p = brachistochrone_min_time(transcription='gauss-lobatto', compressed=False,
                                     run_driver=False, force_alloc_complex=True)
        cpd = p.check_partials(out_stream=None, method='fd')
        self.assertTrue(len(list(cpd.keys())) > 1)
