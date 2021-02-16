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
        grid_data = self.options['input_grid_data']
        if self.options['output_nodes_per_seg'] is None:
            self.num_nodes = grid_data.num_nodes
        else:
            self.num_nodes = grid_data.num_segments * self.options['output_nodes_per_seg']

        for (name, kwargs) in self._timeseries_outputs:
            units = kwargs['units']
            desc = kwargs['units']
            shape = kwargs['shape']
            self._add_output_configure(name, units, shape, desc)

    def _add_output_configure(self, name, units, shape, desc):
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
        num_nodes = self.num_nodes

        input_name = f'all_values:{name}'
        self.add_input(input_name,
                       shape=(num_nodes,) + shape,
                       units=units, desc=desc)

        output_name = name
        self.add_output(output_name,
                        shape=(num_nodes,) + shape,
                        units=units, desc=desc)

        self._vars[name] = (input_name, output_name, shape)

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
            outputs[output_name] = inputs[input_name]
