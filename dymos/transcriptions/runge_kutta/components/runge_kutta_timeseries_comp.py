import numpy as np
from scipy.linalg import block_diag

from ...common.timeseries_output_comp import TimeseriesOutputCompBase
from dymos.utils.lagrange import lagrange_matrices


class RungeKuttaTimeseriesOutputComp(TimeseriesOutputCompBase):
    """
    Class definition for the RungeKuttaTimeseriesOutputComp.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def setup(self):
        """
        Define the independent variables as output variables.
        """
        igd = self.options['input_grid_data']
        ogd = self.options['output_grid_data']
        output_subset = self.options['output_subset']

        if ogd is None:
            ogd = igd

        # Build the interpolation matrix which maps from the input grid to the output grid.
        # Rather than a single phase-wide interpolating polynomial, map each segment.
        # To do this, find the nodes in the output grid which fall in each segment of the input
        # grid.  Then build a Lagrange interpolating polynomial for that segment
        L_blocks = []
        output_nodes_ptau = list(ogd.node_ptau[ogd.subset_node_indices[output_subset]])

        for iseg in range(igd.num_segments):
            i1, i2 = igd.segment_indices[iseg]
            iptau_segi = np.take(igd.node_ptau, (i1, i2-1))
            istau_segi = np.take(igd.node_stau, (i1, i2-1))

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

            # Now get the output nodes which fall in iseg in iseg's segment tau space.
            ostau_segi = 2.0 * (optau_segi - iptau_segi[0]) / (iptau_segi[-1] - iptau_segi[0]) - 1

            # Create the interpolation matrix and add it to the blocks
            L, _ = lagrange_matrices(istau_segi, ostau_segi)
            L_blocks.append(L)

        self.interpolation_matrix = block_diag(*L_blocks)

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
        input_name = 'input_values:{0}'.format(name)
        output_num_nodes, input_num_nodes = self.interpolation_matrix.shape

        self.add_input(input_name,
                       shape=(input_num_nodes,) + shape,
                       units=units,
                       desc=desc)

        self.add_output(name, shape=(output_num_nodes,) + shape, units=units, desc=desc)

        self._vars[name] = (input_name, name, shape)

        size = np.prod(shape)
        val_jac = np.zeros((output_num_nodes, size, input_num_nodes, size))

        for i in range(size):
            val_jac[:, i, :, i] = self.interpolation_matrix

        val_jac = val_jac.reshape((output_num_nodes * size, input_num_nodes * size),
                                  order='C')

        val_jac_rows, val_jac_cols = np.where(val_jac != 0)

        rs, cs = val_jac_rows, val_jac_cols
        self.declare_partials(of=name,
                              wrt=input_name,
                              rows=rs, cols=cs, val=val_jac[rs, cs])

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
        for (input_name, output_name, _) in self._vars.values():
            outputs[output_name] = np.tensordot(self.interpolation_matrix, inputs[input_name],
                                                axes=(1, 0))
