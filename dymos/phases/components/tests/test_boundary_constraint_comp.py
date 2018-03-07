from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials

from dymos.phases.components import BoundaryConstraintComp


class TestInitialScalarBoundaryValue(unittest.TestCase):

    def setUp(self):

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('x', val=np.arange(100))
        self.p.model.add_design_var('x', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp())
        bv_comp._add_initial_constraint(name='x')

        self.p.model.connect('x', 'bv_comp.boundary_values:x', src_indices=[0, -1])

        self.p.setup(force_alloc_complex=True)
        self.p.run_model()

    def test_results(self):

        self.assertAlmostEqual(self.p['bv_comp.boundary_values:x'][0], self.p['x'][0])
        self.assertAlmostEqual(self.p['bv_comp.boundary_values:x'][1], self.p['x'][-1])
        self.assertAlmostEqual(self.p['bv_comp.initial_value:x'], self.p['x'][0])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestFinalScalarBoundaryValue(unittest.TestCase):

    def setUp(self):

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('x', val=np.arange(100))
        self.p.model.add_design_var('x', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp())
        bv_comp._add_final_constraint(name='x')

        self.p.model.connect('x', 'bv_comp.boundary_values:x', src_indices=[0, -1])

        self.p.setup(force_alloc_complex=True)
        self.p.run_model()

    def test_results(self):

        self.assertAlmostEqual(self.p['bv_comp.boundary_values:x'][0], self.p['x'][0])
        self.assertAlmostEqual(self.p['bv_comp.boundary_values:x'][1], self.p['x'][-1])
        self.assertAlmostEqual(self.p['bv_comp.final_value:x'], self.p['x'][-1])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestInitialAndFinalScalarBoundaryValue(unittest.TestCase):

    def setUp(self):

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('x', val=np.arange(100))
        self.p.model.add_design_var('x', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp())
        bv_comp._add_initial_constraint(name='x')
        bv_comp._add_final_constraint(name='x')

        self.p.model.connect('x', 'bv_comp.boundary_values:x', src_indices=[0, -1])

        self.p.setup(force_alloc_complex=True)
        self.p.run_model()

    def test_results(self):

        self.assertAlmostEqual(self.p['bv_comp.boundary_values:x'][0], self.p['x'][0])
        self.assertAlmostEqual(self.p['bv_comp.boundary_values:x'][1], self.p['x'][-1])
        self.assertAlmostEqual(self.p['bv_comp.initial_value:x'], self.p['x'][0])

        self.assertAlmostEqual(self.p['bv_comp.boundary_values:x'][0], self.p['x'][0])
        self.assertAlmostEqual(self.p['bv_comp.boundary_values:x'][1], self.p['x'][-1])
        self.assertAlmostEqual(self.p['bv_comp.final_value:x'], self.p['x'][-1])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestVectorInitialBoundaryValue(unittest.TestCase):

    def setUp(self):

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('pos', val=np.zeros((100, 3)))
        self.p.model.add_design_var('pos', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp())
        bv_comp._add_initial_constraint(name='pos', shape=(3,))

        src_idxs = np.array([[0, 1, 2],
                             [-3, -2, -1]])

        self.p.model.connect('pos', 'bv_comp.boundary_values:pos', src_indices=src_idxs,
                             flat_src_indices=True)

        self.p.setup(force_alloc_complex=True)

        self.p['pos'][:, 0] = np.arange(100)
        self.p['pos'][:, 1] = 100 + np.arange(100)
        self.p['pos'][:, 2] = 200 + np.arange(100)

        self.p.run_model()

    def test_results(self):

        assert_almost_equal(self.p['bv_comp.boundary_values:pos'][0, :], self.p['pos'][0, :])
        assert_almost_equal(self.p['bv_comp.boundary_values:pos'][1, :], self.p['pos'][-1, :])
        assert_almost_equal(self.p['bv_comp.initial_value:pos'], self.p['pos'][0, :])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestVectorFinalBoundaryValue(unittest.TestCase):

    def setUp(self):

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('pos', val=np.zeros((100, 3)))
        self.p.model.add_design_var('pos', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp())
        bv_comp._add_final_constraint(name='pos', shape=(3,))

        src_idxs = np.array([[0, 1, 2],
                             [-3, -2, -1]])

        self.p.model.connect('pos', 'bv_comp.boundary_values:pos', src_indices=src_idxs,
                             flat_src_indices=True)

        self.p.setup(force_alloc_complex=True)

        self.p['pos'][:, 0] = np.arange(100)
        self.p['pos'][:, 1] = 100 + np.arange(100)
        self.p['pos'][:, 2] = 200 + np.arange(100)

        self.p.run_model()

    def test_results(self):

        assert_almost_equal(self.p['bv_comp.boundary_values:pos'][0, :], self.p['pos'][0, :])
        assert_almost_equal(self.p['bv_comp.boundary_values:pos'][1, :], self.p['pos'][-1, :])
        assert_almost_equal(self.p['bv_comp.final_value:pos'], self.p['pos'][-1, :])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestVectorInitialAndFinalBoundaryValue(unittest.TestCase):

    def setUp(self):

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('pos', val=np.zeros((100, 3)))
        self.p.model.add_design_var('pos', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp())
        bv_comp._add_initial_constraint(name='pos', shape=(3,))
        bv_comp._add_final_constraint(name='pos', shape=(3,))

        src_idxs = np.array([[0, 1, 2],
                             [-3, -2, -1]])

        self.p.model.connect('pos', 'bv_comp.boundary_values:pos', src_indices=src_idxs,
                             flat_src_indices=True)

        self.p.setup(force_alloc_complex=True)

        self.p['pos'][:, 0] = np.arange(100)
        self.p['pos'][:, 1] = 100 + np.arange(100)
        self.p['pos'][:, 2] = 200 + np.arange(100)

        self.p.run_model()

    def test_results(self):

        assert_almost_equal(self.p['bv_comp.boundary_values:pos'][0, :], self.p['pos'][0, :])
        assert_almost_equal(self.p['bv_comp.boundary_values:pos'][1, :], self.p['pos'][-1, :])
        assert_almost_equal(self.p['bv_comp.initial_value:pos'], self.p['pos'][0, :])
        assert_almost_equal(self.p['bv_comp.final_value:pos'], self.p['pos'][-1, :])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestMatrixInitialBoundaryValue(unittest.TestCase):

    def setUp(self):

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('M', val=np.zeros((100, 3, 3)))
        self.p.model.add_design_var('M', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp())
        bv_comp._add_initial_constraint(name='M', shape=(3, 3))

        src_idxs = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                             [[-9, -8, -7], [-6, -5, -4], [-3, -2, -1]]])

        self.p.model.connect('M', 'bv_comp.boundary_values:M', src_indices=src_idxs,
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

    def test_results(self):

        assert_almost_equal(self.p['bv_comp.boundary_values:M'][0, ...], self.p['M'][0, ...])
        assert_almost_equal(self.p['bv_comp.boundary_values:M'][1, ...], self.p['M'][-1, ...])
        assert_almost_equal(self.p['bv_comp.initial_value:M'], self.p['M'][0, ...])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestMatrixFinalBoundaryValue(unittest.TestCase):

    def setUp(self):

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('M', val=np.zeros((100, 3, 3)))
        self.p.model.add_design_var('M', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp())
        bv_comp._add_final_constraint(name='M', shape=(3, 3))

        src_idxs = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                             [[-9, -8, -7], [-6, -5, -4], [-3, -2, -1]]])

        self.p.model.connect('M', 'bv_comp.boundary_values:M', src_indices=src_idxs,
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

    def test_results(self):

        assert_almost_equal(self.p['bv_comp.boundary_values:M'][0, ...], self.p['M'][0, ...])
        assert_almost_equal(self.p['bv_comp.boundary_values:M'][1, ...], self.p['M'][-1, ...])
        assert_almost_equal(self.p['bv_comp.final_value:M'], self.p['M'][-1, ...])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestMatrixInitialAndFinalBoundaryValue(unittest.TestCase):

    def setUp(self):

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('M', val=np.zeros((100, 3, 3)))
        self.p.model.add_design_var('M', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp())
        shape = (3, 3)
        size = np.prod(shape)
        bv_comp._add_initial_constraint(name='M', shape=shape)
        bv_comp._add_final_constraint(name='M', shape=shape)

        src_idxs_initial = np.arange(size, dtype=int).reshape(shape)
        src_idxs_final = np.arange(-9, 0, dtype=int).reshape(shape)

        src_idxs = np.stack((src_idxs_initial, src_idxs_final))

        self.p.model.connect('M', 'bv_comp.boundary_values:M', src_indices=src_idxs,
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

    def test_results(self):

        assert_almost_equal(self.p['bv_comp.boundary_values:M'][0, ...], self.p['M'][0, ...])
        assert_almost_equal(self.p['bv_comp.boundary_values:M'][1, ...], self.p['M'][-1, ...])
        assert_almost_equal(self.p['bv_comp.final_value:M'], self.p['M'][-1, ...])
        assert_almost_equal(self.p['bv_comp.initial_value:M'], self.p['M'][0, ...])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


class TestMultipleConstraints(unittest.TestCase):

    def setUp(self):

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        ivp.add_output('M', val=np.zeros((100, 3, 3)))
        ivp.add_output('pos', val=np.zeros((100, 3)))
        ivp.add_output('x', val=np.zeros((100,)))

        self.p.model.add_design_var('M', lower=0, upper=100)
        self.p.model.add_design_var('pos', lower=0, upper=100)
        self.p.model.add_design_var('x', lower=0, upper=100)

        bv_comp = self.p.model.add_subsystem('bv_comp', BoundaryConstraintComp())

        M_shape = (3, 3)
        M_size = np.prod(M_shape)
        bv_comp._add_initial_constraint(name='M', shape=M_shape)
        bv_comp._add_final_constraint(name='M', shape=M_shape)
        src_idxs_initial = np.arange(M_size, dtype=int).reshape(M_shape)
        src_idxs_final = np.arange(-M_size, 0, dtype=int).reshape(M_shape)
        src_idxs = np.stack((src_idxs_initial, src_idxs_final))
        self.p.model.connect('M', 'bv_comp.boundary_values:M', src_indices=src_idxs,
                             flat_src_indices=True)

        pos_shape = (3,)
        pos_size = np.prod(pos_shape)
        bv_comp._add_initial_constraint(name='pos', shape=pos_shape)
        bv_comp._add_final_constraint(name='pos', shape=pos_shape)
        src_idxs_initial = np.arange(pos_size, dtype=int).reshape(pos_shape)
        src_idxs_final = np.arange(-pos_size, 0, dtype=int).reshape(pos_shape)
        src_idxs = np.stack((src_idxs_initial, src_idxs_final))
        self.p.model.connect('pos', 'bv_comp.boundary_values:pos', src_indices=src_idxs,
                             flat_src_indices=True)

        x_shape = (1,)
        x_size = np.prod(x_shape)
        bv_comp._add_initial_constraint(name='x', shape=x_shape)
        bv_comp._add_final_constraint(name='x', shape=x_shape)
        src_idxs_initial = np.arange(x_size, dtype=int).reshape(x_shape)
        src_idxs_final = np.arange(-x_size, 0, dtype=int).reshape(x_shape)
        src_idxs = np.stack((src_idxs_initial, src_idxs_final))
        self.p.model.connect('x', 'bv_comp.boundary_values:x', src_indices=src_idxs,
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

    def test_results(self):

        assert_almost_equal(self.p['bv_comp.boundary_values:M'][0, ...], self.p['M'][0, ...])
        assert_almost_equal(self.p['bv_comp.boundary_values:M'][1, ...], self.p['M'][-1, ...])
        assert_almost_equal(self.p['bv_comp.final_value:M'], self.p['M'][-1, ...])
        assert_almost_equal(self.p['bv_comp.initial_value:M'], self.p['M'][0, ...])

        assert_almost_equal(self.p['bv_comp.boundary_values:pos'][0, ...], self.p['pos'][0, ...])
        assert_almost_equal(self.p['bv_comp.boundary_values:pos'][1, ...], self.p['pos'][-1, ...])
        assert_almost_equal(self.p['bv_comp.final_value:pos'], self.p['pos'][-1, ...])
        assert_almost_equal(self.p['bv_comp.initial_value:pos'], self.p['pos'][0, ...])

        assert_almost_equal(self.p['bv_comp.boundary_values:x'][0, ...], self.p['x'][0, ...])
        assert_almost_equal(self.p['bv_comp.boundary_values:x'][1, ...], self.p['x'][-1, ...])
        assert_almost_equal(self.p['bv_comp.final_value:x'], self.p['x'][-1, ...])
        assert_almost_equal(self.p['bv_comp.initial_value:x'], self.p['x'][0, ...])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


if __name__ == '__main__':
    unittest.main()
