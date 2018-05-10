from __future__ import division, print_function, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent
from six import string_types, iteritems

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units


class StateInterpComp(ExplicitComponent):
    """
    StateInterpComp is designed to operate in either *gauss-lobatto* transcription or in *radau-ps*
    transcription.

    When the transcription is *gauss-lobatto* it accepts the state values and derivatives
    at discretization nodes and computes the interpolated state values and derivatives
    at the collocation nodes, using a Hermite interpolation scheme:

    .. math:: x_c = \left[ A_i \right] x_d + \frac{dt}{d\tau} \left[ B_i \right] f_d
    .. math:: \dot{x}_c = \frac{d\tau}{dt} \left[ A_d \right] x_d + \left[ B_d \right] f_d

    When the transcription is *radau-ps* it accepts the state values at the discretization nodes
    and computes the interpolated state derivatives at the collocation nodes, using a Lagrange
    interpolation scheme:

    .. math:: \dot{x}_c = \frac{d\tau}{dt} \left[ A_d \right] x_d

    """

    def initialize(self):

        self.options.declare(
            'transcription', values=['gauss-lobatto', 'radau-ps'],
            desc='Transcription technique of the optimal control problem.')

        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of state names/options for the phase')

        self.options.declare(
            'time_units', default=None, allow_none=True, types=string_types,
            desc='Units of the integration variable')

    def setup(self):
        time_units = self.options['time_units']

        num_disc_nodes = self.options['grid_data'].subset_num_nodes['state_disc']
        num_col_nodes = self.options['grid_data'].subset_num_nodes['col']

        state_options = self.options['state_options']

        transcription = self.options['transcription']

        self.add_input(name='dt_dstau', shape=(num_col_nodes,), units=time_units,
                       desc='For each node, the duration of its '
                            'segment in the integration variable')

        self.xd_str = {}
        self.fd_str = {}
        self.xc_str = {}
        self.xdotc_str = {}

        for state_name, options in state_options.items():
            shape = options['shape']
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            self.add_input(
                name='state_disc:{0}'.format(state_name),
                shape=(num_disc_nodes,) + shape,
                desc='Values of state {0} at discretization nodes'.format(state_name),
                units=units)

            if transcription == 'gauss-lobatto':
                self.add_input(
                    name='staterate_disc:{0}'.format(state_name),
                    shape=(num_disc_nodes,) + shape,
                    units=rate_units,
                    desc='EOM time derivative of state {0} at '
                         'discretization nodes'.format(state_name))

                self.add_output(
                    name='state_col:{0}'.format(state_name),
                    shape=(num_col_nodes,) + shape, units=units,
                    desc='Interpolated values of state {0} at '
                         'collocation nodes'.format(state_name))

            self.add_output(
                name='staterate_col:{0}'.format(state_name),
                shape=(num_col_nodes,) + shape,
                units=rate_units,
                desc='Interpolated rate of state {0} at collocation nodes'.format(state_name))

            self.xd_str[state_name] = 'state_disc:{0}'.format(state_name)
            self.fd_str[state_name] = 'staterate_disc:{0}'.format(state_name)
            self.xc_str[state_name] = 'state_col:{0}'.format(state_name)
            self.xdotc_str[state_name] = 'staterate_col:{0}'.format(state_name)

        if transcription == 'gauss-lobatto':
            Ai, Bi, Ad, Bd = self.options['grid_data'].phase_hermite_matrices('state_disc', 'col')
        elif transcription == 'radau-ps':
            Ai, Ad = self.options['grid_data'].phase_lagrange_matrices('state_disc', 'col')
            Bi = Bd = np.zeros(shape=(num_col_nodes, num_disc_nodes))
        else:
            raise ValueError('unhandled transcription type: '
                             '{0}'.format(self.options['transcription']))
        self.matrices = {'Ai': Ai, 'Bi': Bi, 'Ad': Ad, 'Bd': Bd}

        # Setup partials

        self.jacs = {'Ai': {}, 'Bi': {}, 'Ad': {}, 'Bd': {}}
        self.Bi_rows = {}
        self.Bi_cols = {}
        self.Ad_rows = {}
        self.Ad_cols = {}
        self.sizes = {}
        self.num_col_nodes = num_col_nodes
        self.num_disc_nodes = num_disc_nodes
        for name, options in iteritems(state_options):
            shape = options['shape']

            size = np.prod(shape)

            for key in self.jacs:
                jac = np.zeros((num_col_nodes, size, num_disc_nodes, size))
                for i in range(size):
                    jac[:, i, :, i] = self.matrices[key]
                jac = jac.reshape((num_col_nodes * size, num_disc_nodes * size), order='C')
                self.jacs[key][name] = jac

            self.sizes[name] = size

            #
            # Partial of xdotc wrt dt_dstau
            #

            # The partial of xdotc (state rate at collocation nodes) is a matrix of m bands
            # on and below the diagonal where m is the size of the variable.
            rs_dtdstau = np.zeros(num_col_nodes * size, dtype=int)
            cs_dtdstau = np.tile(np.arange(0, num_col_nodes, dtype=int), reps=size)
            r_band = np.arange(0, num_col_nodes, dtype=int) * size

            r0 = 0
            for i in range(size):
                rs_dtdstau[r0:r0 + num_col_nodes] = r_band + i
                r0 += num_col_nodes

            self.declare_partials(of=self.xdotc_str[name], wrt='dt_dstau',
                                  rows=rs_dtdstau, cols=cs_dtdstau)

            if transcription == 'gauss-lobatto':
                self.declare_partials(
                    of=self.xc_str[name], wrt='dt_dstau',
                    rows=rs_dtdstau, cols=cs_dtdstau)

                Ai_rows, Ai_cols = np.where(self.jacs['Ai'][name] != 0)
                self.declare_partials(of=self.xc_str[name], wrt=self.xd_str[name],
                                      rows=Ai_rows, cols=Ai_cols,
                                      val=self.jacs['Ai'][name][Ai_rows, Ai_cols])

                self.Bi_rows[name], self.Bi_cols[name] = np.where(self.jacs['Bi'][name] != 0)
                self.declare_partials(of=self.xc_str[name], wrt=self.fd_str[name],
                                      rows=self.Bi_rows[name], cols=self.Bi_cols[name])

                Bd_rows, Bd_cols = np.where(self.jacs['Bd'][name] != 0)
                self.declare_partials(of=self.xdotc_str[name], wrt=self.fd_str[name],
                                      rows=Bd_rows, cols=Bd_cols,
                                      val=self.jacs['Bd'][name][Bd_rows, Bd_cols])

            self.Ad_rows[name], self.Ad_cols[name] = np.where(self.jacs['Ad'][name] != 0)
            self.declare_partials(of=self.xdotc_str[name], wrt=self.xd_str[name],
                                  rows=self.Ad_rows[name], cols=self.Ad_cols[name])

    def _compute_radau(self, inputs, outputs):
        state_options = self.options['state_options']

        dt_dstau = np.atleast_1d(inputs['dt_dstau'])

        num_col_nodes = self.num_col_nodes

        for name in state_options:
            xdotc_str = self.xdotc_str[name]
            xd_str = self.xd_str[name]

            xd = np.atleast_2d(inputs[xd_str])

            # Use transpose to divide each "row" of a by dt_dstau
            a = np.tensordot(self.matrices['Ad'], xd, axes=(1, 0)).T
            outputs[xdotc_str] = (a / dt_dstau).T

    def _compute_gauss_lobatto(self, inputs, outputs):
        state_options = self.options['state_options']

        dt_dstau = np.atleast_1d(inputs['dt_dstau'])

        num_col_nodes = self.num_col_nodes

        for name in state_options:
            xc_str = self.xc_str[name]
            xdotc_str = self.xdotc_str[name]
            xd_str = self.xd_str[name]
            fd_str = self.fd_str[name]

            xd = np.atleast_2d(inputs[xd_str])

            # TODO: einsum magic
            # outputs[xc_str] = np.einsum('i,ij...,jk...->ik...',
            #                             dt_dstau,
            #                             self.matrices['Bi'],
            #                             fd)

            a = np.tensordot(self.matrices['Bi'], inputs[fd_str], axes=(1, 0)).T
            outputs[xc_str] = (a * dt_dstau).T

            outputs[xc_str] += np.tensordot(
                self.matrices['Ai'], xd, axes=(1, 0))

            outputs[xdotc_str] = np.tensordot(self.matrices['Ad'], xd, axes=(1, 0))

            if len(outputs[xdotc_str].shape) == 1:
                outputs[xdotc_str] /= dt_dstau
            elif len(outputs[xdotc_str].shape) == 2:
                outputs[xdotc_str] /= dt_dstau[:, np.newaxis]
            elif len(outputs[xdotc_str].shape) == 3:
                outputs[xdotc_str] /= dt_dstau[:, np.newaxis, np.newaxis]
            else:
                for i in range(num_col_nodes):
                    outputs[xdotc_str][i, ...] /= dt_dstau[i]

            outputs[xdotc_str] += np.tensordot(
                self.matrices['Bd'], inputs[fd_str], axes=(1, 0))

    def _compute_partials_radau(self, inputs, partials):
        state_options = self.options['state_options']

        ndn = self.num_disc_nodes

        for name in state_options:
            size = self.sizes[name]

            xdotc_name = self.xdotc_str[name]
            xd_name = self.xd_str[name]

            # Unroll matrix-shaped states into an array at each node
            xd = np.reshape(inputs[xd_name], (ndn, size))

            partials[xdotc_name, 'dt_dstau'] = -np.dot(self.matrices['Ad'], xd).ravel(order='F') \
                / np.tile(inputs['dt_dstau'], size) ** 2

            dt_dstau_x_size = np.repeat(inputs['dt_dstau'], size)[:, np.newaxis]

            r_nz, c_nz = self.Ad_rows[name], self.Ad_cols[name]
            partials[xdotc_name, xd_name] = (self.jacs['Ad'][name] / dt_dstau_x_size)[r_nz, c_nz]

    def _compute_partials_gauss_lobatto(self, inputs, partials):
        ndn = self.num_disc_nodes

        dt_dstau = inputs['dt_dstau']

        for name, options in iteritems(self.options['state_options']):
            size = self.sizes[name]

            xdotc_name = self.xdotc_str[name]
            xd_name = self.xd_str[name]

            xc_name = self.xc_str[name]
            fd_name = self.fd_str[name]

            # Unroll matrix-shaped states into an array at each node
            xd = np.reshape(inputs[xd_name], (ndn, size))
            fd = np.reshape(inputs[fd_name], (ndn, size))

            dt_dstau_x_size = np.repeat(inputs['dt_dstau'], size)[:, np.newaxis]

            partials[xc_name, 'dt_dstau'] = np.dot(self.matrices['Bi'], fd).ravel(order='F')

            partials[xdotc_name, 'dt_dstau'] = -np.dot(self.matrices['Ad'], xd).ravel(order='F') \
                / np.tile(dt_dstau, size) ** 2

            r_nz, c_nz = self.Bi_rows[name], self.Bi_cols[name]
            partials[xc_name, fd_name] = (self.jacs['Bi'][name] * dt_dstau_x_size)[r_nz, c_nz]

            r_nz, c_nz = self.Ad_rows[name], self.Ad_cols[name]
            partials[xdotc_name, xd_name] = (self.jacs['Ad'][name] / dt_dstau_x_size)[r_nz, c_nz]

    def compute(self, inputs, outputs):
        transcription = self.options['transcription']

        if transcription == 'gauss-lobatto':
            self._compute_gauss_lobatto(inputs, outputs)
        elif transcription == 'radau-ps':
            self._compute_radau(inputs, outputs)
        else:
            raise ValueError('Invalid transcription: {0}'.format(transcription))

    def compute_partials(self, inputs, partials):
        transcription = self.options['transcription']

        if transcription == 'gauss-lobatto':
            self._compute_partials_gauss_lobatto(inputs, partials)
        elif transcription == 'radau-ps':
            self._compute_partials_radau(inputs, partials)
        else:
            raise ValueError('Invalid transcription: {0}'.format(transcription))
