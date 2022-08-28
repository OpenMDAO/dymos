from openmdao.utils.units import unit_conversion
from dymos.transcriptions.common.timeseries_output_comp import TimeseriesOutputCompBase


class SolveIVPTimeseriesOutputComp(TimeseriesOutputCompBase):
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
        super(SolveIVPTimeseriesOutputComp, self).initialize()

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
        else:
            self.output_num_nodes = self.input_num_nodes = igd.num_segments * self.options['output_nodes_per_seg']

        for (name, kwargs) in self._vars:
            units = kwargs['units']
            desc = kwargs['units']
            shape = kwargs['shape']
            self._add_output_configure(name, units, shape, desc)

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
            raise NotImplementedError("Timeseries output rates are not currently supported for "
                                      "SolveIVP transcriptions.")
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
        for (input_name, output_name, _, is_rate) in self._vars.values():
            input_vals = inputs[input_name]

            # if is_rate:
            #     interp_vals = self.differentiation_matrix.dot(interp_vals) / dt_dstau

            if output_name in self._conversion_factors:
                scale, offset = self._conversion_factors[output_name]
                outputs[output_name] = scale * (input_vals + offset)
            else:
                outputs[output_name] = input_vals
