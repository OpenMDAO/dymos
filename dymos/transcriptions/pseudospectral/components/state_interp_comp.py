import numpy as np
import scipy.sparse as sp
import openmdao.api as om
from ...grid_data import GridData
from ....utils.misc import get_rate_units
from ....options import options as dymos_options


class StateInterpComp(om.ExplicitComponent):
    r"""
    Provide interpolated state values and/or rates for pseudospectral transcriptions.

    When the transcription is *gauss-lobatto* it accepts the state values and derivatives
    at discretization nodes and computes the interpolated state values and derivatives
    at the collocation nodes, using a Hermite interpolation scheme.

    .. math:: x_c = \left[ A_i \right] x_d + \frac{dt}{d\tau_s} \left[ B_i \right] f_d
    .. math:: \dot{x}_c = \frac{d\tau_s}{dt} \left[ A_d \right] x_d + \left[ B_d \right] f_d

    When the transcription is *radau-ps* it accepts the state values at the discretization nodes
    and computes the interpolated state derivatives at the collocation nodes, using a Lagrange
    interpolation scheme.

    .. math:: \dot{x}_c = \frac{d\tau_s}{dt} \left[ A_d \right] x_d

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
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
            'time_units', default=None, allow_none=True, types=str,
            desc='Units of the integration variable')

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine shape and units.
        """
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
            Ai, Bi, Ad, Bd = self.options['grid_data'].phase_hermite_matrices('state_disc', 'col', sparse=True)

        elif transcription == 'radau-ps':
            Ai, Ad = self.options['grid_data'].phase_lagrange_matrices('state_disc', 'col', sparse=True)
            Bi = Bd = sp.csr_matrix(np.zeros(shape=(num_col_nodes, num_disc_nodes)))

        else:
            raise ValueError('unhandled transcription type: '
                             '{0}'.format(self.options['transcription']))

        self.matrices = {'Ai': Ai, 'Bi': Bi, 'Ad': Ad, 'Bd': Bd}
        self.jacs = {'Ai': {}, 'Bi': {}, 'Ad': {}, 'Bd': {}}
        self.sizes = {}
        # self.num_col_nodes = num_col_nodes
        # self.num_disc_nodes = num_disc_nodes

        for name, options in state_options.items():
            shape = options['shape']

            size = np.prod(shape)
            self.sizes[name] = size

            for key in self.jacs:
                # Each jacobian matrix has a form that is defined by the Kronecker product
                # of the interpolation matrix and np.eye(size). Make sure to specify csc format
                # here to avoid spurious zeros.
                self.jacs[key][name] = sp.kron(sp.csr_matrix(self.matrices[key]),
                                               sp.eye(size),
                                               format='csc')

            self.sizes[name] = size

            #
            # Partial of xdotc wrt dt_dstau
            #
            rs = np.arange(num_col_nodes * size, dtype=int)
            cs = np.repeat(np.arange(num_col_nodes, dtype=int), size)

            self.declare_partials(of=self.xdotc_str[name], wrt='dt_dstau',
                                  rows=rs, cols=cs)

            if transcription == 'gauss-lobatto':
                self.declare_partials(
                    of=self.xc_str[name], wrt='dt_dstau',
                    rows=rs, cols=cs)

                Ai_rows, Ai_cols, data = sp.find(self.jacs['Ai'][name])
                self.declare_partials(of=self.xc_str[name], wrt=self.xd_str[name],
                                      rows=Ai_rows, cols=Ai_cols, val=data)

                Bi_rows, Bi_cols, _ = sp.find(self.jacs['Bi'][name])
                self.declare_partials(of=self.xc_str[name], wrt=self.fd_str[name],
                                      rows=Bi_rows, cols=Bi_cols)

                Bd_rows, Bd_cols, data = sp.find(self.jacs['Bd'][name])
                self.declare_partials(of=self.xdotc_str[name], wrt=self.fd_str[name],
                                      rows=Bd_rows, cols=Bd_cols, val=data)

            Ad_rows, Ad_cols, _ = sp.find(self.jacs['Ad'][name])
            self.declare_partials(of=self.xdotc_str[name], wrt=self.xd_str[name],
                                  rows=Ad_rows, cols=Ad_cols)

    def _compute_radau(self, inputs, outputs):
        num_disc_nodes = self.options['grid_data'].subset_num_nodes['disc']
        num_col_nodes = self.options['grid_data'].subset_num_nodes['col']
        state_options = self.options['state_options']
        dt_dstau = inputs['dt_dstau'][:, np.newaxis]

        for name in state_options:
            xdotc_str = self.xdotc_str[name]
            xd_str = self.xd_str[name]
            shape = state_options[name]['shape']
            size = np.prod(shape)

            xd_flat = np.reshape(inputs[xd_str],
                                 newshape=(num_disc_nodes, size))

            outputs[xdotc_str] = np.reshape(self.matrices['Ad'].dot(xd_flat) / dt_dstau,
                                            newshape=(num_col_nodes,) + shape)

    def _compute_gauss_lobatto(self, inputs, outputs):
        state_options = self.options['state_options']

        dt_dstau = inputs['dt_dstau'][:, np.newaxis]

        num_disc_nodes = self.options['grid_data'].subset_num_nodes['disc']
        num_col_nodes = self.options['grid_data'].subset_num_nodes['col']

        Ai = self.matrices['Ai']
        Bi = self.matrices['Bi']
        Ad = self.matrices['Ad']
        Bd = self.matrices['Bd']

        for name in state_options:
            shape = state_options[name]['shape']
            size = np.prod(shape)
            xc_str = self.xc_str[name]
            xdotc_str = self.xdotc_str[name]
            xd_str = self.xd_str[name]
            fd_str = self.fd_str[name]

            xd_flat = np.reshape(inputs[xd_str], newshape=(num_disc_nodes, size))
            fd_flat = np.reshape(inputs[fd_str], newshape=(num_disc_nodes, size))

            col_val = Bi.dot(fd_flat) * dt_dstau + Ai.dot(xd_flat)

            outputs[xc_str] = np.reshape(col_val, (num_col_nodes,) + shape)

            col_rate = Ad.dot(xd_flat) / dt_dstau + Bd.dot(fd_flat)

            outputs[xdotc_str] = np.reshape(col_rate, (num_col_nodes,) + shape)

    def _compute_partials_radau(self, inputs, partials):
        state_options = self.options['state_options']

        ndn = self.options['grid_data'].subset_num_nodes['disc']

        Ad = self.matrices['Ad']

        dstau_dt = np.reciprocal(inputs['dt_dstau'])
        dstau_dt2 = (dstau_dt ** 2)

        for name in state_options:
            size = self.sizes[name]

            xdotc_name = self.xdotc_str[name]
            xd_name = self.xd_str[name]

            # Unroll matrix-shaped states into an array at each node
            xd_flat = np.reshape(inputs[xd_name], (ndn, size))

            partials[xdotc_name, 'dt_dstau'] = (-Ad.dot(xd_flat) * dstau_dt2[:, np.newaxis]).ravel()

            dstau_dt_x_size = np.repeat(dstau_dt, size)[:, np.newaxis]

            partials[xdotc_name, xd_name] = self.jacs['Ad'][name].multiply(dstau_dt_x_size).data

    def _compute_partials_gauss_lobatto(self, inputs, partials):
        ndn = self.options['grid_data'].subset_num_nodes['disc']

        Ad = self.matrices['Ad']
        Bi = self.matrices['Bi']

        dstau_dt = np.reciprocal(inputs['dt_dstau'])
        dstau_dt2 = dstau_dt ** 2

        for name, options in self.options['state_options'].items():
            size = self.sizes[name]

            xdotc_name = self.xdotc_str[name]
            xd_name = self.xd_str[name]

            xc_name = self.xc_str[name]
            fd_name = self.fd_str[name]

            # Unroll matrix-shaped states into an array at each node
            xd = np.reshape(inputs[xd_name], (ndn, size))
            fd = np.reshape(inputs[fd_name], (ndn, size))

            dt_dstau_x_size = np.repeat(inputs['dt_dstau'], size)[:, np.newaxis]

            partials[xc_name, 'dt_dstau'] = Bi.dot(fd).ravel()

            partials[xdotc_name, 'dt_dstau'] = (-Ad.dot(xd) * dstau_dt2[:, np.newaxis]).ravel()

            dxc_dfd = self.jacs['Bi'][name].multiply(dt_dstau_x_size)
            partials[xc_name, fd_name] = dxc_dfd.data

            dxdotc_dxd = self.jacs['Ad'][name].multiply(np.reciprocal(dt_dstau_x_size))
            partials[xdotc_name, xd_name] = dxdotc_dxd.data

    def compute(self, inputs, outputs):
        """
        Interpolate state values.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        transcription = self.options['transcription']

        if transcription == 'gauss-lobatto':
            self._compute_gauss_lobatto(inputs, outputs)
        elif transcription == 'radau-ps':
            self._compute_radau(inputs, outputs)
        else:
            raise ValueError('Invalid transcription: {0}'.format(transcription))

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Subjac components written to partials[output_name, input_name].
        """
        transcription = self.options['transcription']
        if transcription == 'gauss-lobatto':
            self._compute_partials_gauss_lobatto(inputs, partials)
        elif transcription == 'radau-ps':
            self._compute_partials_radau(inputs, partials)
        else:
            raise ValueError('Invalid transcription: {0}'.format(transcription))
