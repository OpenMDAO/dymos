import numpy as np
import scipy.sparse as sp

import openmdao.api as om

from dymos.transcriptions.grid_data import GridData
from dymos.utils.misc import get_rate_units
from dymos._options import options as dymos_options
from dymos.utils.lgl import lgl
from dymos.utils.lgr import lgr
from dymos.utils.cgl import cgl
from dymos.utils.birkhoff import birkhoff_matrix


class BirkhoffCollocationComp(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of state names/options for the phase')

        self.options.declare(
            'time_units', default=None, allow_none=True, types=str,
            desc='Units of time')

    def configure_io(self, phase):
        """
        I/O creation is delayed until configure so we can determine shape and units.
        """
        gd = self.options['grid_data']
        num_nodes = gd.subset_num_nodes['col']
        num_segs = gd.num_segments
        time_units = self.options['time_units']
        state_options = self.options['state_options']

        self.add_input('dt_dstau', units=time_units, shape=(gd.subset_num_nodes['col'],))
        self.var_names = var_names = {}
        for state_name, options in state_options.items():
            var_names[state_name] = {
                'f_value': f'state_rates:{state_name}',
                'f_computed': f'f_computed:{state_name}',
                'state_value': f'states:{state_name}',
                'state_segment_ends': f'state_segment_ends:{state_name}',
                'state_initial_value': f'initial_states:{state_name}',
                'state_final_value': f'final_states:{state_name}',
                'state_defect': f'state_defects:{state_name}',
                'state_rate_defect': f'state_rate_defects:{state_name}',
                'initial_state_defect': f'initial_state_defects:{state_name}',
                'final_state_defect': f'final_state_defects:{state_name}'
            }

        for state_name, options in state_options.items():
            shape = options['shape']
            units = options['units']

            rate_units = get_rate_units(units, time_units)
            var_names = self.var_names[state_name]

            rate_source = options['rate_source']
            rate_source_type = phase.classify_var(rate_source)

            self.add_input(
                name=var_names['f_value'],
                shape=(num_nodes,) + shape,
                desc=f'Estimated derivative of state {state_name} at the polynomial nodes',
                units=units)

            if rate_source_type == 'state':
                var_names['f_computed'] = f'states:{rate_source}'
            else:
                self.add_input(
                    name=var_names['f_computed'],
                    shape=(num_nodes,) + shape,
                    desc=f'Computed derivative of state {state_name} at the polynomial nodes',
                    units=rate_units)

            self.add_input(
                name=var_names['state_value'],
                shape=(num_nodes,) + shape,
                desc=f'Value of the state {state_name} at the polynomial nodes',
                units=units
            )

            self.add_input(
                name=var_names['state_segment_ends'],
                shape=(2,) + shape,
                desc=f'Initial and final value of the state {state_name} at each segment end.',
                units=units
            )

            self.add_input(
                name=var_names['state_initial_value'],
                shape=(1,) + shape,
                desc=f'Desired initial value of state {state_name}',
                units=units
            )

            self.add_input(
                name=var_names['state_final_value'],
                shape=(1,) + shape,
                desc=f'Desired final value of state {state_name}',
                units=units
            )

            self.add_output(
                name=var_names['state_defect'],
                shape=(num_nodes + 1,) + shape,
                units=units
            )

            self.add_output(
                name=var_names['state_rate_defect'],
                shape=(num_nodes,) + shape,
                units=units
            )

            self.add_output(
                name=var_names['initial_state_defect'],
                shape=shape,
                units=units
            )

            self.add_output(
                name=var_names['final_state_defect'],
                shape=shape,
                units=units
            )

            if 'defect_ref' in options and options['defect_ref'] is not None:
                defect_ref = options['defect_ref']
            elif 'defect_scaler' in options and options['defect_scaler'] is not None:
                defect_ref = np.divide(1.0, options['defect_scaler'])
            else:
                if 'ref' in options and options['ref'] is not None:
                    defect_ref = options['ref']
                elif 'scaler' in options and options['scaler'] is not None:
                    defect_ref = np.divide(1.0, options['scaler'])
                else:
                    defect_ref = 1.0

            if not np.isscalar(defect_ref):
                defect_ref = np.asarray(defect_ref)
                if defect_ref.shape == shape:
                    defect_ref = np.tile(defect_ref.flatten(), num_nodes)
                else:
                    raise ValueError('array-valued scaler/ref must length equal to state-size')

            if not options['solve_segments']:
                self.add_constraint(name=var_names['state_defect'],
                                    equals=0.0,
                                    ref=defect_ref)

                self.add_constraint(name=var_names['state_rate_defect'],
                                    equals=0.0,
                                    ref=defect_ref)

                self.add_constraint(name=var_names['initial_state_defect'],
                                    equals=0.0,
                                    ref=defect_ref)

                self.add_constraint(name=var_names['final_state_defect'],
                                    equals=0.0,
                                    ref=defect_ref)

        if gd.grid_type == 'lgl':
            tau, w = lgl(num_nodes)
        elif gd.grid_type == 'lgr':
            tau, w = lgr(num_nodes, include_endpoint=False)
        elif gd.grid_type == 'cgl':
            tau, w = cgl(num_nodes)
        else:
            raise ValueError('invalid grid type')

        B = birkhoff_matrix(tau, w, grid_type=gd.grid_type)

        self._A = np.zeros((num_nodes + 1, 2 * num_nodes))
        self._A[:num_nodes, :num_nodes] = np.eye(num_nodes)
        self._A[:num_nodes, num_nodes:] = -B
        self._A[-1, num_nodes:] = B[-1, :]

        self._C = np.zeros((num_nodes + 1, 2))
        self._C[:-1, 0] = 1.
        self._C[-1, :] = [-1, 1]

        # Setup partials

        for state_name, options in state_options.items():
            shape = options['shape']
            size = np.prod(shape)

            ar1 = np.arange(num_nodes * size)
            c1 = np.repeat(np.arange(num_nodes), size)

            var_names = self.var_names[state_name]

            # The derivative of the state defect wrt x_ab is [-C].
            # Take the Kronecker product of that and an identity matrix
            # of the state's size to get the pattern that includes each
            # individual element in the state.
            c_sparse = sp.kron(self._C, sp.eye(size))
            c_rows, c_cols = c_sparse.nonzero()
            c_data = c_sparse.data

            self.declare_partials(of=var_names['state_defect'],
                                  wrt=var_names['state_segment_ends'],
                                  rows=c_rows, cols=c_cols, val=-c_data)

            # The derivative of the state defect wrt [X;V] is [A].
            # Since X comprises the first n elements in [X;V], we only
            # need the first `n` columns of [A], which are an identity matrix
            # with a row of zeros beneath for each segment.
            # Take the Kronecker product of that and an identity matrix
            # of the state's size to get the pattern that includes each
            # individual element in the state.
            self.declare_partials(of=var_names['state_defect'],
                                  wrt=var_names['state_value'],
                                  rows=ar1, cols=ar1, val=1)

            # Similar to the states wrt X, since V is the final n elements in
            # [X;V] we take the last n columns of A and take the kronecker product
            # of it and an identity matrix of the state's size to get the
            # overall sparsity pattern.
            b_sparse = sp.kron(self._A[:, num_nodes:], sp.eye(size))
            b_rows, b_cols = b_sparse.nonzero()
            b_data = b_sparse.data

            self.declare_partials(of=var_names['state_defect'],
                                  wrt=var_names['f_value'],
                                  rows=b_rows, cols=b_cols, val=b_data)

            self.declare_partials(of=var_names['state_rate_defect'],
                                  wrt=var_names['f_value'],
                                  rows=ar1, cols=ar1, val=1.0)

            self.declare_partials(of=var_names['state_rate_defect'],
                                  wrt=var_names['f_computed'],
                                  rows=ar1, cols=ar1)

            self.declare_partials(of=var_names['state_rate_defect'],
                                  wrt='dt_dstau',
                                  rows=ar1, cols=c1)

            self.declare_partials(of=var_names['initial_state_defect'],
                                  wrt=var_names['state_initial_value'],
                                  rows=np.arange(size, dtype=int),
                                  cols=np.arange(size, dtype=int),
                                  val=-1.0)

            self.declare_partials(of=var_names['initial_state_defect'],
                                  wrt=var_names['state_segment_ends'],
                                  rows=np.arange(size, dtype=int),
                                  cols=np.arange(size, dtype=int),
                                  val=1.0)

            self.declare_partials(of=var_names['final_state_defect'],
                                  wrt=var_names['state_final_value'],
                                  rows=np.arange(size, dtype=int),
                                  cols=np.arange(size, dtype=int),
                                  val=-1.0)

            self.declare_partials(of=var_names['final_state_defect'],
                                  wrt=var_names['state_segment_ends'],
                                  rows=np.arange(size, dtype=int),
                                  cols=size + np.arange(size, dtype=int),
                                  val=1.0)

            # self.declare_partials(of=var_names['final_state_defect'],
            #                       wrt=var_names['state_segment_ends'],
            #                       method='fd')

            # self.declare_partials(of=var_names['final_state_defect'],
            #                       wrt=var_names['state_final_value'],
            #                       rows=ar2, cols=ar2,
            #                       val=-1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dt_dstau = np.atleast_2d(inputs['dt_dstau']).T

        for state_name, options in self.options['state_options'].items():
            var_names = self.var_names[state_name]

            x_a = inputs[var_names['state_initial_value']]
            x_b = inputs[var_names['state_final_value']]
            x_ab = inputs[var_names['state_segment_ends']]
            X = inputs[var_names['state_value']]
            V = inputs[var_names['f_value']]
            f = inputs[var_names['f_computed']]

            # X_AB = np.vstack((x_a[np.newaxis, :], x_b[np.newaxis, :]))
            XV = np.vstack((X, V))
            state_defect = np.einsum('ij,jk...->ik...', self._A, XV) - np.einsum('ij, j...->i...', self._C, x_ab)

            outputs[var_names['state_defect']] = state_defect
            outputs[var_names['state_rate_defect']] = (V - np.einsum('i...,i...->i...', f, dt_dstau))
            outputs[var_names['initial_state_defect']] = x_ab[0, ...] - x_a
            outputs[var_names['final_state_defect']] = x_ab[1, ...] - x_b

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
        dt_dstau = inputs['dt_dstau']

        for state_name, options in self.options['state_options'].items():
            var_names = self.var_names[state_name]
            shape = options['shape']
            size = np.prod(shape)
            f = inputs[var_names['f_computed']]

            partials[var_names['state_rate_defect'], var_names['f_computed']] = np.repeat(-dt_dstau, size)
            partials[var_names['state_rate_defect'], 'dt_dstau'] = -f.ravel()
