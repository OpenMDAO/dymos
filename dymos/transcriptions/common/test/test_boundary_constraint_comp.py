import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from dymos.utils.testing_utils import assert_check_partials

import dymos as dm
from dymos.transcriptions.common.boundary_constraint_comp import BoundaryConstraintComp

# Modify class so we can run it standalone.
from dymos.utils.misc import CompWrapperConfig
BoundaryConstraintComp = CompWrapperConfig(BoundaryConstraintComp)


class TestInitialScalarBoundaryValue(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

        self.p = om.Problem(model=om.Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('x', val=np.arange(100))
        self.p.model.add_design_var('x', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp(loc='initial'))
        bv_comp._add_constraint(name='x', shape=(1,))

        self.p.model.connect('x', 'bv_comp.initial_value_in:x', src_indices=[0])

        self.p.setup(force_alloc_complex=True)
        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results(self):

        self.assertAlmostEqual(self.p['bv_comp.initial_value_in:x'][0], self.p['x'][0])
        self.assertAlmostEqual(self.p['bv_comp.initial_value:x'], self.p['x'][0])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestFinalScalarBoundaryValue(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

        self.p = om.Problem(model=om.Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('x', val=np.arange(100))
        self.p.model.add_design_var('x', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp(loc='final'))
        bv_comp._add_constraint(name='x', shape=(1,))

        self.p.model.connect('x', 'bv_comp.final_value_in:x', src_indices=[-1])

        self.p.setup(force_alloc_complex=True)
        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results(self):

        self.assertAlmostEqual(self.p['bv_comp.final_value_in:x'][0], self.p['x'][-1])
        self.assertAlmostEqual(self.p['bv_comp.final_value:x'], self.p['x'][-1])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestVectorInitialBoundaryValue(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

        self.p = om.Problem(model=om.Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('pos', val=np.zeros((100, 3)))
        self.p.model.add_design_var('pos', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp(loc='initial'))
        bv_comp._add_constraint(name='pos', shape=(3,))

        src_idxs = np.array([0, 1, 2], dtype=int)

        self.p.model.connect('pos', 'bv_comp.initial_value_in:pos', src_indices=src_idxs,
                             flat_src_indices=True)

        self.p.setup(force_alloc_complex=True)

        self.p['pos'][:, 0] = np.arange(100)
        self.p['pos'][:, 1] = 100 + np.arange(100)
        self.p['pos'][:, 2] = 200 + np.arange(100)

        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results(self):
        assert_almost_equal(self.p['bv_comp.initial_value_in:pos'], self.p['pos'][0, :])
        assert_almost_equal(self.p['bv_comp.initial_value:pos'], self.p['pos'][0, :])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestVectorFinalBoundaryValue(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

        self.p = om.Problem(model=om.Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('pos', val=np.zeros((100, 3)))
        self.p.model.add_design_var('pos', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp(loc='final'))
        bv_comp._add_constraint(name='pos', shape=(3,))

        src_idxs = np.array([-3, -2, -1])

        self.p.model.connect('pos', 'bv_comp.final_value_in:pos', src_indices=src_idxs,
                             flat_src_indices=True)

        self.p.setup(force_alloc_complex=True)

        self.p['pos'][:, 0] = np.arange(100)
        self.p['pos'][:, 1] = 100 + np.arange(100)
        self.p['pos'][:, 2] = 200 + np.arange(100)

        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results(self):

        assert_almost_equal(self.p['bv_comp.final_value_in:pos'], self.p['pos'][-1, :])
        assert_almost_equal(self.p['bv_comp.final_value:pos'], self.p['pos'][-1, :])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestMatrixInitialBoundaryValue(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

        self.p = om.Problem(model=om.Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('M', val=np.zeros((100, 3, 3)))
        self.p.model.add_design_var('M', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp(loc='initial'))
        bv_comp._add_constraint(name='M', shape=(3, 3))

        src_idxs = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        self.p.model.connect('M', 'bv_comp.initial_value_in:M', src_indices=src_idxs,
                             flat_src_indices=True)

        self.p.setup(force_alloc_complex=True)

        self.p['M'][:, 0, 0] = np.arange(100)
        self.p['M'][:, 0, 1] = 100 + np.arange(100)
        self.p['M'][:, 0, 2] = 200 + np.arange(100)

        self.p['M'][:, 1, 0] = 1000 + np.arange(100)
        self.p['M'][:, 1, 1] = 1000 + 100 + np.arange(100)
        self.p['M'][:, 1, 2] = 1000 + 200 + np.arange(100)

        self.p['M'][:, 2, 0] = 2000 + np.arange(100)
        self.p['M'][:, 2, 1] = 2000 + 100 + np.arange(100)
        self.p['M'][:, 2, 2] = 2000 + 200 + np.arange(100)

        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results(self):

        assert_almost_equal(self.p['bv_comp.initial_value_in:M'], self.p['M'][0, ...])
        assert_almost_equal(self.p['bv_comp.initial_value:M'], self.p['M'][0, ...])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestMatrixFinalBoundaryValue(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

        self.p = om.Problem(model=om.Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('M', val=np.zeros((100, 3, 3)))
        self.p.model.add_design_var('M', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp(loc='final'))
        bv_comp._add_constraint(name='M', shape=(3, 3))

        src_idxs = np.array([[-9, -8, -7], [-6, -5, -4], [-3, -2, -1]])

        self.p.model.connect('M', 'bv_comp.final_value_in:M', src_indices=src_idxs,
                             flat_src_indices=True)

        self.p.setup(force_alloc_complex=True)

        self.p['M'][:, 0, 0] = np.arange(100)
        self.p['M'][:, 0, 1] = 100 + np.arange(100)
        self.p['M'][:, 0, 2] = 200 + np.arange(100)

        self.p['M'][:, 1, 0] = 1000 + np.arange(100)
        self.p['M'][:, 1, 1] = 1000 + 100 + np.arange(100)
        self.p['M'][:, 1, 2] = 1000 + 200 + np.arange(100)

        self.p['M'][:, 2, 0] = 2000 + np.arange(100)
        self.p['M'][:, 2, 1] = 2000 + 100 + np.arange(100)
        self.p['M'][:, 2, 2] = 2000 + 200 + np.arange(100)

        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results(self):

        assert_almost_equal(self.p['bv_comp.final_value_in:M'], self.p['M'][-1, ...])
        assert_almost_equal(self.p['bv_comp.final_value:M'], self.p['M'][-1, ...])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestMultipleConstraints(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

        self.p = om.Problem(model=om.Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('M', val=np.zeros((100, 3, 3)))
        ivp.add_output('pos', val=np.zeros((100, 3)))
        ivp.add_output('x', val=np.zeros((100,)))

        self.p.model.add_design_var('M', lower=0, upper=100)
        self.p.model.add_design_var('pos', lower=0, upper=100)
        self.p.model.add_design_var('x', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp(loc='initial'))

        M_shape = (3, 3)
        M_size = np.prod(M_shape)
        bv_comp._add_constraint(name='M', shape=M_shape)
        src_idxs = np.arange(M_size, dtype=int).reshape(M_shape)
        self.p.model.connect('M', 'bv_comp.initial_value_in:M', src_indices=src_idxs,
                             flat_src_indices=True)

        pos_shape = (3,)
        pos_size = np.prod(pos_shape)
        bv_comp._add_constraint(name='pos', shape=pos_shape)
        src_idxs = np.arange(pos_size, dtype=int).reshape(pos_shape)
        self.p.model.connect('pos', 'bv_comp.initial_value_in:pos', src_indices=src_idxs,
                             flat_src_indices=True)

        x_shape = (1,)
        x_size = np.prod(x_shape)
        bv_comp._add_constraint(name='x', shape=x_shape)
        src_idxs = np.arange(x_size, dtype=int).reshape(x_shape)
        self.p.model.connect('x', 'bv_comp.initial_value_in:x', src_indices=src_idxs,
                             flat_src_indices=True)

        self.p.setup(force_alloc_complex=True)

        self.p['M'][:, 0, 0] = np.arange(100)
        self.p['M'][:, 0, 1] = 100 + np.arange(100)
        self.p['M'][:, 0, 2] = 200 + np.arange(100)

        self.p['M'][:, 1, 0] = 1000 + np.arange(100)
        self.p['M'][:, 1, 1] = 1000 + 100 + np.arange(100)
        self.p['M'][:, 1, 2] = 1000 + 200 + np.arange(100)

        self.p['M'][:, 2, 0] = 2000 + np.arange(100)
        self.p['M'][:, 2, 1] = 2000 + 100 + np.arange(100)
        self.p['M'][:, 2, 2] = 2000 + 200 + np.arange(100)

        self.p['pos'][:, 0] = 100000 + np.arange(100)
        self.p['pos'][:, 1] = 100000 + 100 + np.arange(100)
        self.p['pos'][:, 2] = 100000 + 200 + np.arange(100)

        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results(self):

        assert_almost_equal(self.p['bv_comp.initial_value_in:M'], self.p['M'][0, ...])
        assert_almost_equal(self.p['bv_comp.initial_value:M'], self.p['M'][0, ...])

        assert_almost_equal(self.p['bv_comp.initial_value_in:pos'], self.p['pos'][0, ...])
        assert_almost_equal(self.p['bv_comp.initial_value:pos'], self.p['pos'][0, ...])

        assert_almost_equal(self.p['bv_comp.initial_value_in:x'], self.p['x'][0, ...])
        assert_almost_equal(self.p['bv_comp.initial_value:x'], self.p['x'][0, ...])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
