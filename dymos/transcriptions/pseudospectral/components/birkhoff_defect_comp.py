import numpy as np
import scipy
import scipy.sparse as sp

import openmdao.api as om

from dymos.transcriptions.grid_data import GridData
from dymos.utils.misc import get_rate_units
from dymos._options import options as dymos_options
from dymos.utils.lgl import lgl
from dymos.utils.cgl import cgl
from dymos.utils.birkhoff import birkhoff_matrix


class BirkhoffDefectComp(om.ExplicitComponent):
    """
    Class definition for the BirkhoffDefectComp.

    BirkhoffDefectComp computes the generalized defects of a segment for implicit collocation.
    There are four defects to be evaluated; state, state rate, initial state, and final state.
    The state defect is the difference between the state value design variables and the state rate
    design variables.
    The state rate defect is the difference between the state rate design variables and the computed
    state derivative at the collocation nodes.
    The initial and final state defects are the differences between the initial and final state
    design variables and the first and last index of the state design variable array.

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

        Parameters
        ----------
        phase : Phase
            The phase object that contains this collocation comp.
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
                'state_initial_value': f'initial_states:{state_name}',
                'state_final_value': f'final_states:{state_name}',
                'state_defect': f'state_defects:{state_name}',
                'state_rate_defect': f'state_rate_defects:{state_name}',
                'initial_state_defect': f'initial_state_defects:{state_name}',
                'final_state_defect': f'final_state_defects:{state_name}',
                'state_continuity_defect': f'state_cnty_defects:{state_name}'
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
                shape=(num_nodes + num_segs,) + shape,
                units=units
            )

            self.add_output(
                name=var_names['state_rate_defect'],
                shape=(num_nodes,) + shape,
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
                    defect_ref_state = np.tile(defect_ref.flatten(), num_nodes+num_segs)
                    defect_ref_v = np.tile(defect_ref.flatten(), num_nodes)
                else:
                    raise ValueError('array-valued scaler/ref must length equal to state-size')
            else:
                defect_ref_state = defect_ref
                defect_ref_v = defect_ref

            if not options['solve_segments']:
                self.add_constraint(name=var_names['state_defect'],
                                    equals=0.0,
                                    ref=defect_ref_state,
                                    linear=True)

                self.add_constraint(name=var_names['state_rate_defect'],
                                    equals=0.0,
                                    ref=defect_ref_v)

        A_blocks = []
        B_blocks = []
        C_blocks = []

        # _xv_idxs is a set of indices that arranges the stacked
        # [[X^T],[V^T]] arrays into a segment-by-segment ordering
        # instead of all of the state values followed by all of the state rates.
        self._xv_idxs = []

        idx0 = 0

        for i in range(num_segs):
            nnps_i = gd.subset_num_nodes_per_segment['all'][0]
            if gd.grid_type == 'lgl':
                tau_i, w_i = lgl(nnps_i)
            elif gd.grid_type == 'cgl':
                tau_i, w_i = cgl(nnps_i)
            else:
                raise ValueError('invalid grid type')

            B_i = birkhoff_matrix(tau_i, w_i, grid_type=gd.grid_type)

            A_i = np.zeros((nnps_i + 1, 2 * nnps_i))
            A_i[:nnps_i, :nnps_i] = np.eye(nnps_i)
            A_i[:nnps_i, nnps_i:] = -B_i
            A_i[-1, nnps_i:] = B_i[-1, :]

            C_i = np.zeros((nnps_i + 1, 2))
            C_i[:-1, 0] = 1.
            C_i[-1, :] = [-1, 1]

            A_blocks.append(A_i)
            B_blocks.append(B_i)
            C_blocks.append(C_i)

            ar_nnps = np.arange(nnps_i, dtype=int)
            self._xv_idxs.extend(idx0 + ar_nnps)
            self._xv_idxs.extend(idx0 + ar_nnps + num_nodes)
            idx0 += nnps_i

        self._A = scipy.linalg.block_diag(*A_blocks)
        self._B = scipy.linalg.block_diag(*B_blocks)
        self._C = scipy.linalg.block_diag(*C_blocks)

        # Setup partials
        for state_name, options in state_options.items():
            shape = options['shape']
            units = options['units']
            size = np.prod(shape)

            ar1 = np.arange(num_nodes * size)
            c1 = np.repeat(np.arange(num_nodes), size)

            var_names = self.var_names[state_name]

            # The derivative of the state defect wrt x_ab is [-C].
            # The derivative of x_ab wrt x_a is eye(size).
            # Take the Kronecker product of that and an identity matrix
            # of the state's size to get the pattern that includes each
            # individual element in the state.

            # The derivative of x_ab wrt x_a is [I(size), 0(size)]^T
            # The derivative of x_ab wrt x_b is [0(size), I(size)]^T
            d_state_defect_dxa = sp.kron(np.ones((self._C.shape[0], 1)), -sp.eye(size), format='coo')
            d_state_defect_dxa.data[-size:] = 1.0
            d_dxa_r, d_dxa_c = d_state_defect_dxa.nonzero()
            d_dxa_data = d_state_defect_dxa.data.ravel()

            self.declare_partials(of=var_names['state_defect'],
                                  wrt=var_names['state_initial_value'],
                                  rows=d_dxa_r, cols=d_dxa_c, val=d_dxa_data)

            self.declare_partials(of=var_names['state_defect'],
                                  wrt=var_names['state_final_value'],
                                  rows=d_dxa_r[-size:], cols=d_dxa_c[-size:], val=-1.0)

            d_state_defect_dXV = self._A
            dXV_dX = np.vstack((np.eye(num_nodes), np.zeros((num_nodes, num_nodes))))[self._xv_idxs]
            dXV_dV = np.vstack((np.zeros((num_nodes, num_nodes)), np.eye(num_nodes)))[self._xv_idxs]
            d_state_defect_dX = sp.csr_matrix(np.kron(np.dot(d_state_defect_dXV, dXV_dX), np.eye(size)))
            d_state_defect_dV = sp.csr_matrix(np.kron(np.dot(d_state_defect_dXV, dXV_dV), np.eye(size)))

            rs_dX, cs_dX = d_state_defect_dX.nonzero()
            rs_dV, cs_dV = d_state_defect_dV.nonzero()

            self.declare_partials(of=var_names['state_defect'],
                                  wrt=var_names['state_value'],
                                  rows=rs_dX, cols=cs_dX, val=d_state_defect_dX.data.ravel())

            self.declare_partials(of=var_names['state_defect'],
                                  wrt=var_names['f_value'],
                                  rows=rs_dV, cols=cs_dV, val=d_state_defect_dV.data.ravel())

            self.declare_partials(of=var_names['state_rate_defect'],
                                  wrt=var_names['f_value'],
                                  rows=ar1, cols=ar1, val=1.0)

            self.declare_partials(of=var_names['state_rate_defect'],
                                  wrt=var_names['f_computed'],
                                  rows=ar1, cols=ar1)

            self.declare_partials(of=var_names['state_rate_defect'],
                                  wrt='dt_dstau',
                                  rows=ar1, cols=c1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute component outputs.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        discrete_outputs : dict or None
            If not None, dict containing discrete output values.
        """
        dt_dstau = np.atleast_2d(inputs['dt_dstau']).T
        num_segs = self.options['grid_data'].num_segments

        for state_name, options in self.options['state_options'].items():
            var_names = self.var_names[state_name]

            x_a = inputs[var_names['state_initial_value']]
            x_b = inputs[var_names['state_final_value']]
            # x_ab = inputs[var_names['state_segment_ends']]
            X = inputs[var_names['state_value']]
            V = inputs[var_names['f_value']]
            f = inputs[var_names['f_computed']]
            shape = x_a.shape[1:]
            size = x_a.size

            # Get stacked XV, but sorted in segment-by-segment order
            XV = np.vstack((X, V))[self._xv_idxs]

            x_ab = np.stack([x_a, x_b], axis=0).reshape((2,) + shape)

            state_defect = np.einsum('ij,jk...->ik...', self._A, XV) - \
                np.einsum('ij, j...->i...', self._C, x_ab)

            outputs[var_names['state_defect']] = state_defect
            outputs[var_names['state_rate_defect']] = (V - np.einsum('i...,i...->i...', f, dt_dstau))

            if num_segs > 1:
                (num_segs - 1) * size
                outputs[var_names['state_continuity_defect']] = x_ab[:-1, 1, ...] - x_ab[1:, 0, ...]

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
