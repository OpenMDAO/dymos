import numpy as np
import openmdao.api as om

from dymos._options import options as dymos_options
from dymos.transcriptions.grid_data import GridData
from dymos.utils.misc import get_rate_units


class RadauDefectComp(om.ExplicitComponent):
    """
    Class definiton for the Collocationcomp.

    CollocationComp computes the generalized defect of a segment for implicit collocation.
    The defect is the interpolated state derivative at the collocation nodes minus
    the computed state derivative at the collocation nodes.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """Declare component options."""
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
            The phase in which this component exists.
        """
        gd: GridData = self.options['grid_data']
        num_segs: int = gd.num_segments
        num_nodes: int = gd.subset_num_nodes['all']
        num_col_nodes: int = gd.subset_num_nodes['col']
        time_units: str = self.options['time_units']
        state_options = self.options['state_options']

        # The radau differentiation matrix
        _, self._D = self.options['grid_data'].phase_lagrange_matrices('state_disc',
                                                                       'col',
                                                                       sparse=True)

        self.add_input('dt_dstau', units=time_units, shape=(num_col_nodes,))

        self.var_names = var_names = {}
        for state_name in state_options:
            var_names[state_name] = {
                'initial_val': f'initial_states:{state_name}',
                'final_val': f'final_states:{state_name}',
                'val': f'states:{state_name}',
                'f_ode': f'f_ode:{state_name}',
                'rate_defect': f'state_rate_defects:{state_name}',
                'cnty_defect': f'state_cnty_defects:{state_name}',
                'initial_defect': f'initial_state_defects:{state_name}',
                'final_defect': f'final_state_defects:{state_name}'
            }

        for state_name, options in state_options.items():
            shape = options['shape']
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            var_names = self.var_names[state_name]

            self.add_input(name=var_names['initial_val'],
                           shape=(1,) + shape,
                           units=units,
                           desc='Initial value of the state at the start of the phase.')

            self.add_input(name=var_names['final_val'],
                           shape=(1,) + shape,
                           units=units,
                           desc='Final value of the state at the end of the phase.')

            self.add_input(var_names['val'],
                           shape=(num_nodes,) + shape,
                           units=units,
                           desc='state value at all nodes within the phase')

            self.add_input(
                name=var_names['f_ode'],
                shape=(num_col_nodes,) + shape,
                desc=f'Computed derivative of state {state_name} at the collocation nodes',
                units=rate_units)

            self.add_output(
                name=var_names['initial_defect'],
                shape=(1,) + shape,
                desc=f'Initial value defect of state {state_name}',
                units=units)

            self.add_output(
                name=var_names['final_defect'],
                shape=(1,) + shape,
                desc=f'Final value defect of state {state_name}',
                units=units)

            self.add_output(
                name=var_names['rate_defect'],
                shape=(num_col_nodes,) + shape,
                desc=f'Interior defects of state {state_name}',
                units=units)

            if gd.num_segments > 1 and not gd.compressed:
                self.add_output(
                    name=var_names['cnty_defect'],
                    shape=(num_segs - 1,) + shape,
                    desc=f'Segment boundary defect of state {state_name}',
                    units=units)

            if 'defect_ref' in options and options['defect_ref'] is not None:
                defect_ref = options['defect_ref']
            elif 'defect_scaler' in options and options['defect_scaler'] is not None:
                defect_ref = np.divide(1.0, options['defect_scaler'])
            elif 'ref' in options and options['ref'] is not None:
                defect_ref = options['ref']
            elif 'scaler' in options and options['scaler'] is not None:
                defect_ref = np.divide(1.0, options['scaler'])
            else:
                defect_ref = 1.0

            if not np.isscalar(defect_ref):
                defect_ref = np.asarray(defect_ref)
                if defect_ref.shape == shape:
                    defect_ref = np.tile(defect_ref.flatten(), num_col_nodes)
                else:
                    raise ValueError('array-valued scaler/ref must length equal to state-size')

            if not options['solve_segments']:
                self.add_constraint(name=var_names['rate_defect'],
                                    equals=0.0,
                                    ref=defect_ref)

                self.add_constraint(name=var_names['initial_defect'],
                                    equals=0.0,
                                    ref=defect_ref)

                self.add_constraint(name=var_names['final_defect'],
                                    equals=0.0,
                                    ref=defect_ref)

                if gd.num_segments > 1 and not gd.compressed:
                    self.add_constraint(name=var_names['cnty_defect'],
                                        equals=0.0,
                                        ref=defect_ref)

        # Setup partials
        num_col_nodes = self.options['grid_data'].subset_num_nodes['col']
        state_options = self.options['state_options']

        for state_name, options in state_options.items():
            shape = options['shape']
            size = np.prod(shape)

            r = np.arange(num_col_nodes * size)

            var_names = self.var_names[state_name]

            self.declare_partials(of=var_names['rate_defect'],
                                  wrt=var_names['f_ode'],
                                  rows=r, cols=r, val=-1.0)

            c = np.repeat(np.arange(num_col_nodes), size)
            self.declare_partials(of=var_names['rate_defect'],
                                  wrt='dt_dstau',
                                  rows=r, cols=c)

            # The state rate defects wrt the state values at the discretization nodes
            # are given by the differentiation matrix.
            r, c = self._D.nonzero()
            self.declare_partials(of=var_names['rate_defect'],
                                  wrt=var_names['val'],
                                  rows=r, cols=c, val=self._D.data)

            # The initial value defect is just an identity matrix at the "top left" corner of the jacobian.
            ar_size = np.arange(size, dtype=int)
            self.declare_partials(of=var_names['initial_defect'],
                                  wrt=var_names['val'],
                                  rows=ar_size, cols=ar_size, val=-1.0)

            self.declare_partials(of=var_names['initial_defect'],
                                  wrt=var_names['initial_val'],
                                  rows=ar_size, cols=ar_size, val=1.0)

            # The final value defect is an identity matrix at the "bottom right" corner of the jacobian.
            r = np.arange(size, dtype=int)
            c = np.arange(num_nodes - size, num_nodes, dtype=int)
            self.declare_partials(of=var_names['final_defect'],
                                  wrt=var_names['val'],
                                  rows=r, cols=c, val=-1.0)

            self.declare_partials(of=var_names['final_defect'],
                                  wrt=var_names['final_val'],
                                  rows=ar_size, cols=ar_size, val=1.0)

            if gd.num_segments > 1 and not gd.compressed:
                rs = np.repeat(np.arange(num_segs - 1, dtype=int), 2)
                cs = gd.subset_node_indices['segment_ends'][1:-1]
                val = np.tile([-1., 1.], num_segs-1)
                self.declare_partials(of=var_names['cnty_defect'],
                                      wrt=var_names['val'], rows=rs, cols=cs, val=val)

    def compute(self, inputs, outputs):
        """
        Compute collocation defects.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        gd: GridData = self.options['grid_data']
        num_disc_nodes: int = gd.subset_num_nodes['state_disc']
        num_col_nodes: int = gd.subset_num_nodes['col']
        idxs_se: int = gd.subset_node_indices['segment_ends']

        state_options = self.options['state_options']
        dt_dstau: np.ndarray = inputs['dt_dstau']
        D = self._D

        for state_name, state_options in state_options.items():
            shape = state_options['shape']
            size = np.prod(shape)
            var_names = self.var_names[state_name]

            f_ode = inputs[var_names['f_ode']]
            x = inputs[var_names['val']]
            x_0 = inputs[var_names['initial_val']]
            x_f = inputs[var_names['final_val']]

            # The defect is computed as
            # defect = D @ x - f_ode * dt_dstau  # noqa: ERA001
            # But scipy.sparse only handles 2D matrices, so we need to force x to be 2D
            # and then change the product back to the proper shape.

            x_flat = np.reshape(x, newshape=(num_disc_nodes, size))
            f_approx = np.reshape(D.dot(x_flat), newshape=(num_col_nodes,) + shape)

            outputs[var_names['rate_defect']] = f_approx - (f_ode.T * dt_dstau).T
            outputs[var_names['initial_defect']] = x_0 - x[0, ...]
            outputs[var_names['final_defect']] = x_f - x[-1, ...]

            if gd.num_segments > 1 and not gd.compressed:
                outputs[var_names['cnty_defect']] = x[idxs_se[2::2], ...] - x[idxs_se[1:-2:2], ...]

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
            size = np.prod(options['shape'])
            var_names = self.var_names[state_name]
            f_ode = inputs[var_names['f_ode']]

            partials[var_names['rate_defect'], var_names['f_ode']] = -np.repeat(dt_dstau, size)
            partials[var_names['rate_defect'], 'dt_dstau'] = -f_ode.ravel()
