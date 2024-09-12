import numpy as np
import openmdao.api as om
from openmdao.utils.units import unit_conversion
from scipy import sparse as sp

from ...transcriptions.grid_data import GridData
from ..._options import options as dymos_options
from ...utils.lagrange import lagrange_matrices


class TimeseriesOutputComp(om.ExplicitComponent):
    """
    Class definition of the TimeseriesOutputComp.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._no_check_partials = not dymos_options['include_check_partials']

        # _vars keeps track of the name of each output and maps to its metadata;
        # a tuple of (input_name, name, shape, rate)
        self._vars = {}

        # _sources is used internally to map the source of a connection to the timeseries to
        # the corresponding input variable.  This is used to ensure that we don't need to connect
        # the same source to this timeseries multiple times.
        self._sources = {}

        # the number of nodes from the source (this is generally all nodes)
        self.input_num_nodes = 0

        # the number of nodes in the output
        self.output_num_nodes = 0

        # Used to track conversion factors for instances when one output that relies on an input
        # from another variable has potentially different units
        self._units = {}
        self._conversion_factors = {}

        # Flag to set if no multiplication by the interpolation matrix is necessary
        self._no_interp = False

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare('input_grid_data',
                             types=GridData,
                             desc='Container object for grid on which inputs are provided.')

        self.options.declare('output_grid_data',
                             types=GridData,
                             allow_none=True,
                             default=None,
                             desc='Container object for grid on which outputs are interpolated.')

        self.options.declare('output_subset',
                             types=str,
                             default='all',
                             desc='Name of the node subset at which outputs are desired.')

        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of time')

    def setup(self):
        """
        Define the independent variables as output variables.
        """

        # igd = self.options['input_grid_data']
        # ogd = self.options['output_grid_data']
        # output_subset = self.options['output_subset']

        # if ogd is None:
        #     ogd = igd

        # if ogd == igd and output_subset == 'all':
        #     self._no_interp = True

        # self.input_num_nodes = igd.num_nodes
        # self.output_num_nodes = ogd.subset_num_nodes[output_subset]

        # # Build the interpolation matrix which maps from the input grid to the output grid.
        # # Rather than a single phase-wide interpolating polynomial, map each segment.
        # # To do this, find the nodes in the output grid which fall in each segment of the input
        # # grid.  Then build a Lagrange interpolating polynomial for that segment
        # L_blocks = []
        # D_blocks = []
        # output_nodes_ptau = ogd.node_ptau[ogd.subset_node_indices[output_subset]]

        # for iseg in range(igd.num_segments):
        #     i1, i2 = igd.segment_indices[iseg]
        #     iptau_segi = igd.node_ptau[i1:i2]
        #     istau_segi = igd.node_stau[i1:i2]

        #     # The indices of the output grid that fall within this segment of the input grid
        #     if ogd is igd and output_subset == 'all':
        #         optau_segi = iptau_segi
        #     else:
        #         ptau_hi = igd.segment_ends[iseg+1]
        #         if iseg < igd.num_segments - 1:
        #             optau_segi = output_nodes_ptau[output_nodes_ptau <= ptau_hi]
        #         else:
        #             optau_segi = output_nodes_ptau

        #         # Remove the captured nodes so we don't accidentally include them again
        #         output_nodes_ptau = output_nodes_ptau[len(optau_segi):]

        #     # # Now get the output nodes which fall in iseg in iseg's segment tau space.
        #     ostau_segi = 2.0 * (optau_segi - iptau_segi[0]) / (iptau_segi[-1] - iptau_segi[0]) - 1

        #     # Create the interpolation matrix and add it to the blocks
        #     L, D = lagrange_matrices(istau_segi, ostau_segi)
        #     L_blocks.append(L)
        #     D_blocks.append(D)

        # self.interpolation_matrix = sp.block_diag(L_blocks, format='csr')
        # self.differentiation_matrix = sp.block_diag(D_blocks, format='csr')

    def _add_output_configure(self, name, units, shape, desc='', src=None, rate=False):
        """
        Add a single timeseries output.

        Can be called by parent groups in configure.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array.
            Default is None.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
        desc : str
            description of the timeseries output variable.
        src : str
            The src path of the variables input, used to prevent redundant inputs.
        rate : bool
            If True, timeseries output is a rate.

        Returns
        -------
        bool
            True if a new input was added for the output, or False if it reuses an existing input.
        """
        input_num_nodes = self.input_num_nodes
        output_num_nodes = self.output_num_nodes
        added_source = False

        if name in self._vars:
            return False

        if src in self._sources:
            # If we're already pulling the source into this timeseries, use that as the
            # input for this output.
            input_name = self._sources[src]
            input_units = self._units[input_name]
        else:
            input_name = f'input_values:{name}'
            self.add_input(input_name,
                           shape=(input_num_nodes,) + shape,
                           units=units, desc=desc)
            self._sources[src] = input_name
            input_units = self._units[input_name] = units
            added_source = True

        output_name = name
        self.add_output(output_name, shape=(output_num_nodes,) + shape, units=units, desc=desc)

        self._vars[name] = (input_name, output_name, shape, rate)

        size = np.prod(shape)

        # Get any scaling that needs to be performed for the derivative
        if input_units is None or units is None:
            scale = 1.0
        else:
            scale, offset = unit_conversion(input_units, units)
            self._conversion_factors[output_name] = scale, offset

        if not rate and self._no_interp:
            # Case 1: No outputs just echo inputs.
            # Jacobian values are just the scale factor
            jac_rows = jac_cols = np.arange(input_num_nodes * size, dtype=int)
            jac_val = scale
        else:
            if rate:
                # Case 2: The output is a rate
                # Jacobian values depend on the differentiation matrix
                # This is not well conditioned for large numbers of nodes.
                mat = self.differentiation_matrix
            else:
                # Case 3: the output is an interpolation
                # Jacobian values depend on the interpolation matrix.
                # This is subject to noise with large numbers of nodes.
                mat = self.interpolation_matrix

            # Preallocate lists to hold data for jac
            jac_data = []
            jac_indices = []
            jac_indptr = [0]

            # Iterate over the dense dimension 'size'
            for s in range(size):
                # Extend the data and indices using the CSR attributes of mat
                jac_data.extend(mat.data)
                jac_indices.extend(mat.indices + s * input_num_nodes)

                # For every non-zero row in mat, update jac's indptr
                new_indptr = mat.indptr[1:] + s * len(mat.data)
                jac_indptr.extend(new_indptr)

            # Correct the last entry of jac_indptr
            jac_indptr[-1] = len(jac_data)

            # Construct the sparse jac matrix in CSR format
            jac = sp.csr_matrix((jac_data, jac_indices, jac_indptr),
                                shape=(output_num_nodes * size, input_num_nodes * size))

            # Now, if you want to get the row and column indices of the non-zero entries in the jac matrix:
            jac_rows, jac_cols = jac.nonzero()

            # There's a chance that the input for this output was pulled from another variable with
            # different units, so account for that with a conversion.
            jac_val = np.squeeze(np.array(jac[jac_rows, jac_cols])) * scale

        self.declare_partials(of=output_name, wrt=input_name,
                              rows=jac_rows, cols=jac_cols,
                              val=jac_val)

        return added_source

    def _build_matrices(self, compute_L=True, compute_D=True):
        """
        Build the interpolation and differentiation matrices for the component, if necessary.

        Parameters
        ----------
        compute_L : bool, optional
            If True, compute and store the interpolation matrix. Otherwise return None in its place. Default is True.
        compute_D : bool, optional
            If True, compute and store the diferentiation matrix. Otherwise return None in its place. Default is True.
        """
        igd = self.options['input_grid_data']
        ogd = self.options['output_grid_data']
        output_subset = self.options['output_subset']

        if ogd is None:
            ogd = igd

        self.input_num_nodes = igd.num_nodes
        self.output_num_nodes = ogd.subset_num_nodes[output_subset]

        self.add_input('dt_dstau', shape=(self.input_num_nodes,), units=self.options['time_units'])

        # Build the interpolation matrix which maps from the input grid to the output grid.
        # Rather than a single phase-wide interpolating polynomial, map each segment.
        # To do this, find the nodes in the output grid which fall in each segment of the input
        # grid.  Then build a Lagrange interpolating polynomial for that segment
        L_blocks = []
        D_blocks = []
        output_nodes_ptau = ogd.node_ptau[ogd.subset_node_indices[output_subset]]

        for iseg in range(igd.num_segments):
            i1, i2 = igd.segment_indices[iseg]
            iptau_segi = igd.node_ptau[i1:i2]
            istau_segi = igd.node_stau[i1:i2]

            # The indices of the output grid that fall within this segment of the input grid
            if ogd is igd and output_subset == 'all':
                optau_segi = iptau_segi
            else:
                ptau_hi = igd.segment_ends[iseg+1]
                if iseg < igd.num_segments - 1:
                    optau_segi = output_nodes_ptau[output_nodes_ptau <= ptau_hi]
                else:
                    optau_segi = output_nodes_ptau

                # Remove the captured nodes so we don't accidentally include them again
                output_nodes_ptau = output_nodes_ptau[len(optau_segi):]

            # # Now get the output nodes which fall in iseg in iseg's segment tau space.
            ostau_segi = 2.0 * (optau_segi - iptau_segi[0]) / (iptau_segi[-1] - iptau_segi[0]) - 1

            # Create the interpolation matrix and add it to the blocks
            L, D = lagrange_matrices(istau_segi, ostau_segi,
                                     compute_interp_matrix=compute_L,
                                     compute_diff_matrix=compute_D)
            L_blocks.append(L)
            D_blocks.append(D)

        if compute_L:
            self.interpolation_matrix = sp.block_diag(L_blocks, format='csr')
        if compute_D:
            self.differentiation_matrix = sp.block_diag(D_blocks, format='csr')

    def _configure_io(self, timeseries_options):
        """
        Configure the component IO for the given timeseries.

        Parameters
        ----------
        phase : Phase
            The phase to which this component belongs.
        timeseries_name : str
            The name of the timeseries being configured.

        Returns
        -------
        inputs : list
            The timeseries inputs to this component.
        """
        igd = self.options['input_grid_data']
        ogd = self.options['output_grid_data']
        output_subset = self.options['output_subset']

        if ogd is None:
            ogd = igd

        if ogd == igd and output_subset == 'all':
            self._no_interp = True

        self.input_num_nodes = igd.num_nodes
        self.output_num_nodes = ogd.subset_num_nodes[output_subset]

        # Check if we ever use rates, if not, don't bother computing/storing [D]
        compute_D = any([ts_options['is_rate'] for ts_options in timeseries_options['outputs'].values()])
        compute_L = not self._no_interp

        # Compute and store the interpolation and differentiation matrices, if necessary.
        self._build_matrices(compute_L, compute_D)

        # Add the IO for the given timeseries
        ts_inputs = []
        for ts_output_name, ts_output in timeseries_options['outputs'].items():
            name = ts_output['output_name'] if ts_output['output_name'] is not None else ts_output['name']
            units = ts_output['units']
            shape = ts_output['shape']
            src = ts_output['src']
            is_rate = ts_output['is_rate']

            added_src = self._add_output_configure(name, shape=shape,
                                                   units=units, desc='',
                                                   src=src, rate=is_rate)

            if added_src:
                # If we hadn't already used this input, add it now.
                ts_inputs.append((f'input_values:{name}', src, ts_output['src_idxs']))
        return ts_inputs

    def compute(self, inputs, outputs):
        """
        Compute component outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        # convert dt_dstau to a column vector
        dt_dstau = inputs['dt_dstau'][:, np.newaxis]

        for (input_name, output_name, _, is_rate) in self._vars.values():
            if self._no_interp:
                interp_vals = inputs[input_name]
            else:
                inp = inputs[input_name]
                if len(inp.shape) > 2:
                    # Dot product always performs the sum product over axis 2.
                    inp = inp.swapaxes(0, 1)
                interp_vals = self.interpolation_matrix.dot(inp)

            if is_rate:
                interp_vals = self.differentiation_matrix.dot(interp_vals) / dt_dstau

            if output_name in self._conversion_factors:
                scale, offset = self._conversion_factors[output_name]
                outputs[output_name] = scale * (interp_vals + offset)
            else:
                outputs[output_name] = interp_vals
