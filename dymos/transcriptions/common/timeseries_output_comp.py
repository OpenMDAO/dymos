import numpy as np
import openmdao.api as om
from scipy.linalg import block_diag

from ...transcriptions.grid_data import GridData
from ...utils.lagrange import lagrange_matrices
from ...options import options as dymos_options


class TimeseriesOutputCompBase(om.ExplicitComponent):
    """
    TimeseriesOutputComp collects variable values from the phase and provides them in chronological
    order as outputs.  Some phase types don't internally have access to a contiguous array of all
    values of a given variable in the phase.  For instance, the GaussLobatto pseudospectral has
    separate arrays of variable values at discretization and collocation nodes.  These values
    need to be interleaved to provide a time series.  Pseudospectral techniques provide timeseries
    data at 'all' nodes, while ExplicitPhase provides values at the step boundaries.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):

        self._timeseries_outputs = []

        self._vars = []

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


class PseudospectralTimeseriesOutputComp(TimeseriesOutputCompBase):

    def __init__(self, **kwargs):
        super(PseudospectralTimeseriesOutputComp, self).__init__(**kwargs)

        self.input_num_nodes = 0
        self.output_num_nodes = 0

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        igd = self.options['input_grid_data']
        ogd = self.options['output_grid_data']
        output_subset = self.options['output_subset']

        if ogd is None:
            ogd = igd

        self.input_num_nodes = igd.num_nodes
        self.output_num_nodes = ogd.subset_num_nodes[output_subset]

        # Build the interpolation matrix which maps from the input grid to the output grid.
        # Rather than a single phase-wide interpolating polynomial, map each segment.
        # To do this, find the nodes in the output grid which fall in each segment of the input
        # grid.  Then build a Lagrange interpolating polynomial for that segment
        L_blocks = []
        output_nodes_ptau = ogd.node_ptau[ogd.subset_node_indices[output_subset]].tolist()

        for iseg in range(igd.num_segments):
            i1, i2 = igd.segment_indices[iseg]
            iptau_segi = igd.node_ptau[i1:i2]
            istau_segi = igd.node_stau[i1:i2]

            # The indices of the output grid that fall within this segment of the input grid
            if ogd is igd:
                optau_segi = iptau_segi
            else:
                ptau_hi = igd.segment_ends[iseg+1]
                if iseg < igd.num_segments - 1:
                    idxs_in_iseg = np.where(output_nodes_ptau <= ptau_hi)[0]
                else:
                    idxs_in_iseg = np.arange(len(output_nodes_ptau))
                optau_segi = np.asarray(output_nodes_ptau)[idxs_in_iseg]
                # Remove the captured nodes so we don't accidentally include them again
                output_nodes_ptau = output_nodes_ptau[len(idxs_in_iseg):]

            # # Now get the output nodes which fall in iseg in iseg's segment tau space.
            ostau_segi = 2.0 * (optau_segi - iptau_segi[0]) / (iptau_segi[-1] - iptau_segi[0]) - 1

            # Create the interpolation matrix and add it to the blocks
            L, _ = lagrange_matrices(istau_segi, ostau_segi)
            L_blocks.append(L)

        self.interpolation_matrix = block_diag(*L_blocks)

        for (name, kwargs) in self._timeseries_outputs:
            units = kwargs['units']
            desc = kwargs['units']
            shape = kwargs['shape']
            self._add_output_configure(name, units, shape, desc)

    def _add_output_configure(self, name, units, shape, desc=None):
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
        """
        input_num_nodes = self.input_num_nodes
        output_num_nodes = self.output_num_nodes

        input_name = 'input_values:{0}'.format(name)
        self.add_input(input_name,
                       shape=(input_num_nodes,) + shape,
                       units=units, desc=desc)

        output_name = name
        self.add_output(output_name,
                        shape=(output_num_nodes,) + shape,
                        units=units, desc=desc)

        self._vars.append((input_name, output_name, shape))

        size = np.prod(shape)
        val_jac = np.zeros((output_num_nodes, size, input_num_nodes, size))

        for i in range(size):
            val_jac[:, i, :, i] = self.interpolation_matrix

        val_jac = val_jac.reshape((output_num_nodes * size, input_num_nodes * size),
                                  order='C')

        val_jac_rows, val_jac_cols = np.where(val_jac != 0)

        rs, cs = val_jac_rows, val_jac_cols
        self.declare_partials(of=output_name,
                              wrt=input_name,
                              rows=rs, cols=cs, val=val_jac[rs, cs])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for (input_name, output_name, _) in self._vars:
            outputs[output_name] = np.tensordot(self.interpolation_matrix, inputs[input_name],
                                                axes=(1, 0))
