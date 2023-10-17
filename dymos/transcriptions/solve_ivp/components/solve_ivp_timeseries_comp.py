import numpy as np
import openmdao.api as om
import scipy.sparse as sp

from openmdao.utils.units import unit_conversion

from ....utils.lagrange import lagrange_matrices
from ...common import TimeseriesOutputComp


class SolveIVPTimeseriesOutputComp(TimeseriesOutputComp):
    """
    Class definition for SolveIVPTimeseriesOutputComp.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def initialize(self):
        """
        Declare component options.
        """
        super().initialize()

        self.options.declare('output_nodes_per_seg', default=None, types=(int,), allow_none=True,
                             desc='If None, results are provided at the all nodes within each'
                                  'segment.  If an int (n) then results are provided at n '
                                  'equally distributed points in time within each segment.')

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        igd = self.options['input_grid_data']

        if self.options['output_nodes_per_seg'] is None:
            self.output_num_nodes = self.input_num_nodes = igd.num_nodes
            ogd = igd
        else:
            self.output_num_nodes = self.input_num_nodes = igd.num_segments * self.options['output_nodes_per_seg']
            output_nodes_stau = np.linspace(-1, 1, self.options['output_nodes_per_seg'])
            output_nodes_ptau = igd.node_ptau[igd.subset_node_indices['all']]
            ogd = None

        # Build the interpolation matrix which maps from the input grid to the output grid.
        # Rather than a single phase-wide interpolating polynomial, map each segment.
        # To do this, find the nodes in the output grid which fall in each segment of the input
        # grid.  Then build a Lagrange interpolating polynomial for that segment
        # L_blocks = []
        D_blocks = []

        for iseg in range(igd.num_segments):
            # Create the interpolation matrix and add it to the blocks
            if self.options['output_nodes_per_seg'] is None:
                i1, i2 = igd.segment_indices[iseg]
                iptau_segi = igd.node_ptau[i1:i2]
                istau_segi = igd.node_stau[i1:i2]

                # The indices of the output grid that fall within this segment of the input grid
                if ogd is igd:
                    optau_segi = iptau_segi
                else:
                    ptau_hi = igd.segment_ends[iseg + 1]
                    if iseg < igd.num_segments - 1:
                        optau_segi = output_nodes_ptau[output_nodes_ptau <= ptau_hi]
                    else:
                        optau_segi = output_nodes_ptau

                    # Remove the captured nodes so we don't accidentally include them again
                    output_nodes_ptau = output_nodes_ptau[len(optau_segi):]

                # # Now get the output nodes which fall in iseg in iseg's segment tau space.
                ostau_segi = 2.0 * (optau_segi - iptau_segi[0]) / (iptau_segi[-1] - iptau_segi[0]) - 1

                _, D = lagrange_matrices(istau_segi, ostau_segi)
            else:
                _, D = lagrange_matrices(output_nodes_stau, output_nodes_stau)
            D_blocks.append(D)

        self.differentiation_matrix = sp.block_diag(D_blocks, format='csr')

        for (name, kwargs) in self._vars:
            units = kwargs['units']
            desc = kwargs['desc']
            shape = kwargs['shape']
            src = kwargs['src']
            rate = kwargs['rate']
            self._add_output_configure(name, units, shape, desc, src, rate)

        self.add_input('dt_dstau', shape=(self.output_num_nodes,), units=self.options['time_units'])

    def _add_output_configure(self, name, units, shape, desc, src=None, rate=False):
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
        src : str or None
            The source of the timeseries output.
        rate : bool
            If True, timeseries output is a rate.
        """
        if rate:
            om.issue_warning(f'Timeseries rate outputs not currently supported in simulate: {name} being skipped.')

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

        # There's a chance that the input for this output was pulled from another variable with
        # different units, so account for that with a conversion.
        if input_units is None or units is None:
            pass
        else:
            scale, offset = unit_conversion(input_units, units)
            self._conversion_factors[output_name] = scale, offset

        return added_source

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
        dt_dstau = inputs['dt_dstau'][:, np.newaxis]
        for (input_name, output_name, _, is_rate) in self._vars.values():
            interp_vals = inputs[input_name]

            if is_rate:
                interp_vals = self.differentiation_matrix.dot(interp_vals) / dt_dstau

            if output_name in self._conversion_factors:
                scale, offset = self._conversion_factors[output_name]
                outputs[output_name] = scale * (interp_vals + offset)
            else:
                outputs[output_name] = interp_vals
