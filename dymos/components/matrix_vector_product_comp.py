from six import string_types

import numpy as np
import scipy.linalg as spla

from openmdao.api import ExplicitComponent


class MatrixVectorProductComp(ExplicitComponent):
    """
    Computes a vectorized matrix-vector product

    math::
        b = np.dot(A, x)

    where A is of shape (num_nodes, n, m)
          x is of shape (num_nodes, m)
          b is of shape (num_nodes, m)
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int, desc='The number of nodes at which the matrix'
                                                           '-vector product is to be computed')
        self.metadata.declare('A_name', types=string_types, default='A',
                              desc='The variable name for the matrix.')
        self.metadata.declare('A_shape', types=tuple, default=(3, 3),
                              desc='The shape of the input matrix.')
        self.metadata.declare('A_units', types=string_types, allow_none=True, default=None,
                              desc='The units of the input matrix.')
        self.metadata.declare('x_name', types=string_types, default='x',
                              desc='The name of the input vector.')
        self.metadata.declare('x_units', types=string_types, default=None, allow_none=True,
                              desc='The units of the input vector.')
        self.metadata.declare('b_name', types=string_types, default='b',
                              desc='The variable name of the output vector.')
        self.metadata.declare('b_units', types=string_types, allow_none=True, default=None,
                              desc='The units of the output vector.')

    def setup(self):
        meta = self.metadata
        nn = meta['num_nodes']

        vec_size = meta['A_shape'][1]

        self.add_input(name=meta['A_name'],
                       shape=(nn,) + meta['A_shape'],
                       units=meta['A_units'])

        self.add_input(name=meta['x_name'],
                       shape=(nn,) + (vec_size,),
                       units=meta['x_units'])

        self.add_output(name=meta['b_name'],
                        shape=(nn,) + (meta['A_shape'][0],),
                        units=meta['b_units'])

        # Make a dummy version of A so we can figure out the nonzero indices
        A = np.ones(shape=(nn,) + meta['A_shape'])
        x = np.ones(shape=(nn,) + (vec_size,))
        bd_A = spla.block_diag(*A)
        x_repeat = np.repeat(x, A.shape[1], axis=0)
        bd_x_repeat = spla.block_diag(*x_repeat)
        db_dx_rows, db_dx_cols = np.nonzero(bd_A)
        db_dA_rows, db_dA_cols = np.nonzero(bd_x_repeat)

        self.declare_partials(of=meta['b_name'], wrt=meta['A_name'],
                              rows=db_dA_rows, cols=db_dA_cols)
        self.declare_partials(of=meta['b_name'], wrt=meta['x_name'],
                              rows=db_dx_rows, cols=db_dx_cols)

    def compute(self, inputs, outputs):
        meta = self.metadata
        A = inputs[meta['A_name']]
        x = inputs[meta['x_name']]
        outputs[meta['b_name']] = np.einsum('nij,nj->ni', A, x)

    def compute_partials(self, inputs, partials):
        meta = self.metadata
        A_name = meta['A_name']
        x_name = meta['x_name']
        b_name = meta['b_name']
        A = inputs[A_name]
        x = inputs[x_name]

        # This works for dense partials
        # partials[meta['b_name'], meta['x_name']] = spla.block_diag(*A)
        # x_repeat = np.repeat(x, A.shape[1], axis=0)
        # partials[meta['b_name'], meta['A_name']] = spla.block_diag(*x_repeat)

        # Use the following for sparse partials
        partials[b_name, A_name] = np.repeat(x, A.shape[1], axis=0).ravel()
        partials[b_name, x_name] = A.ravel()


def _for_docs():  # pragma: no cover
    return MatrixVectorProductComp()
