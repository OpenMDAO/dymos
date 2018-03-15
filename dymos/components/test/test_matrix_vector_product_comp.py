from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from dymos.components import MatrixVectorProductComp


class TestMatrixVectorProductComp3x3(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='A', shape=(self.nn, 3, 3))
        ivc.add_output(name='x', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['A', 'x'])

        self.p.model.add_subsystem(name='mat_vec_product_comp',
                                   subsys=MatrixVectorProductComp(num_nodes=self.nn))

        self.p.model.connect('A', 'mat_vec_product_comp.A')
        self.p.model.connect('x', 'mat_vec_product_comp.x')

        self.p.setup(force_alloc_complex=True)

        self.p['A'] = np.random.rand(self.nn, 3, 3)
        self.p['x'] = np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            A_i = self.p['A'][i, :, :]
            x_i = self.p['x'][i, :]
            b_i = self.p['mat_vec_product_comp.b'][i, :]

            expected_i = np.dot(A_i, x_i)
            np.testing.assert_almost_equal(b_i, expected_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMatrixVectorProductComp6x4(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='A', shape=(self.nn, 6, 4))
        ivc.add_output(name='x', shape=(self.nn, 4))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['A', 'x'])

        self.p.model.add_subsystem(name='mat_vec_product_comp',
                                   subsys=MatrixVectorProductComp(num_nodes=self.nn,
                                                                  A_shape=(6, 4)))

        self.p.model.connect('A', 'mat_vec_product_comp.A')
        self.p.model.connect('x', 'mat_vec_product_comp.x')

        self.p.setup(force_alloc_complex=True)

        self.p['A'] = np.random.rand(self.nn, 6, 4)
        self.p['x'] = np.random.rand(self.nn, 4)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            A_i = self.p['A'][i, :, :]
            x_i = self.p['x'][i, :]
            b_i = self.p['mat_vec_product_comp.b'][i, :]

            expected_i = np.dot(A_i, x_i)
            np.testing.assert_almost_equal(b_i, expected_i)

    def test_partials(self):
        cpd = self.p.check_partials(out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestForDocs(unittest.TestCase):

    def test(self):
        import numpy as np
        from openmdao.api import Problem, Group, IndepVarComp
        from dymos.components import MatrixVectorProductComp

        nn = 100

        p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='A', shape=(nn, 3, 3))
        ivc.add_output(name='x', shape=(nn, 3))

        p.model.add_subsystem(name='ivc',
                              subsys=ivc,
                              promotes_outputs=['A', 'x'])

        p.model.add_subsystem(name='mat_vec_product_comp',
                              subsys=MatrixVectorProductComp(A_name='M', num_nodes=nn))

        p.model.connect('A', 'mat_vec_product_comp.M')
        p.model.connect('x', 'mat_vec_product_comp.x')

        p.setup()

        p['A'] = np.random.rand(nn, 3, 3)
        p['x'] = np.random.rand(nn, 3)

        p.run_model()

        for i in range(nn):
            A_i = p['A'][i, :, :]
            x_i = p['x'][i, :]
            b_i = p['mat_vec_product_comp.b'][i, :]

            expected_i = np.dot(A_i, x_i)
            # np.testing.assert_almost_equal(b_i, expected_i)
            assert_rel_error(self, b_i, expected_i, 1e-5)

if __name__ == "__main__": 
    unittest.main()
