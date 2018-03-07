from six import string_types

import numpy as np
import scipy.linalg as spla

from openmdao.api import ExplicitComponent


class CrossProductComp(ExplicitComponent):
    """
    Computes a vectorized dot product

    math::
        c = np.cross(a, b)

    where a is of shape (num_nodes, 3)
          b is of shape (num_nodes, 3)
          c is of shape (num_nodes, 3)
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int,
                              desc='The number of nodes at which the dot product is computed')
        self.metadata.declare('a_name', types=string_types, default='a',
                              desc='The variable name for vector a.')
        self.metadata.declare('b_name', types=string_types, default='b',
                              desc='The variable name for vector b.')
        self.metadata.declare('c_name', types=string_types, default='c',
                              desc='The variable name for vector c.')
        self.metadata.declare('a_units', types=string_types, default=None, allow_none=True,
                              desc='The units for vector a.')
        self.metadata.declare('b_units', types=string_types, default=None, allow_none=True,
                              desc='The units for vector b.')
        self.metadata.declare('c_units', types=string_types, default=None, allow_none=True,
                              desc='The units for vector c.')

        self._k = np.array([[0, 0, 0, -1, 0, 1],
                            [0, 1, 0, 0, -1, 0],
                            [-1, 0, 1, 0, 0, 0]], dtype=np.float64)

    def setup(self):
        meta = self.metadata
        nn = meta['num_nodes']

        self.add_input(name=meta['a_name'],
                       shape=(nn, 3),
                       units=meta['a_units'])

        self.add_input(name=meta['b_name'],
                       shape=(nn, 3),
                       units=meta['b_units'])

        self.add_output(name=meta['c_name'],
                        val=np.zeros(shape=(nn, 3)),
                        units=meta['c_units'])

        row_idxs = np.repeat(np.arange(nn*3, dtype=int), 2)
        col_idxs = np.empty((0,), dtype=int)
        M = np.array([1, 2, 0, 2, 0, 1], dtype=int)
        for i in range(nn):
            col_idxs = np.concatenate((col_idxs, M + i * 3))

        self.declare_partials(of=meta['c_name'], wrt=meta['a_name'],
                              rows=row_idxs, cols=col_idxs, val=0)

        self.declare_partials(of=meta['c_name'], wrt=meta['b_name'],
                              rows=row_idxs, cols=col_idxs, val=0)

    def compute(self, inputs, outputs):
        meta = self.metadata
        a = inputs[meta['a_name']]
        b = inputs[meta['b_name']]
        outputs[meta['c_name']] = np.cross(a, b)

    def compute_partials(self, inputs, partials):
        meta = self.metadata
        a = inputs[meta['a_name']]
        b = inputs[meta['b_name']]

        # Use the following for sparse partials
        partials[meta['c_name'], meta['a_name']] = np.einsum('nj,ji->ni', b, self._k*-1).ravel()
        partials[meta['c_name'], meta['b_name']] = np.einsum('nj,ji->ni', a, self._k).ravel()


def _for_docs():  # pragma: no cover
    return CrossProductComp()
