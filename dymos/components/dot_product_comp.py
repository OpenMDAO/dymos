from six import string_types

import numpy as np
import scipy.linalg as spla

from openmdao.api import ExplicitComponent


class DotProductComp(ExplicitComponent):
    """
    Computes a vectorized dot product

    math::
        c = np.dot(a, b)

    where a is of shape (num_nodes, n)
          b is of shape (num_nodes, n)
          c is of shape (num_nodes)
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int,
                              desc='The number of nodes at which the dot product is computed')
        self.metadata.declare('vec_size', types=int, default=3,
                              desc='The size of vectors a and b')
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

    def setup(self):
        meta = self.metadata
        nn = meta['num_nodes']
        m = meta['vec_size']

        self.add_input(name=meta['a_name'],
                       shape=(nn, m),
                       units=meta['a_units'])

        self.add_input(name=meta['b_name'],
                       shape=(nn, m),
                       units=meta['b_units'])

        self.add_output(name=meta['c_name'],
                        val=np.zeros(shape=(nn,)),
                        units=meta['c_units'])

        row_idxs = np.repeat(np.arange(nn), m)
        col_idxs = np.arange(nn * m)
        self.declare_partials(of=meta['c_name'], wrt=meta['a_name'], rows=row_idxs, cols=col_idxs)
        self.declare_partials(of=meta['c_name'], wrt=meta['b_name'], rows=row_idxs, cols=col_idxs)

    def compute(self, inputs, outputs):
        meta = self.metadata
        a = inputs[meta['a_name']]
        b = inputs[meta['b_name']]
        outputs[meta['c_name']] = np.einsum('ni,ni->n', a, b)

    def compute_partials(self, inputs, partials):
        meta = self.metadata
        a = inputs[meta['a_name']]
        b = inputs[meta['b_name']]

        # Use the following for sparse partials
        partials[meta['c_name'], meta['a_name']] = b.ravel()
        partials[meta['c_name'], meta['b_name']] = a.ravel()


def _for_docs():  # pragma: no cover
    return DotProductComp()
