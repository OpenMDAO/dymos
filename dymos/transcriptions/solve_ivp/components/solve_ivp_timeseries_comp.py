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

    def _add_output_configure(self, name, units, shape, desc, rate=False):
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
        rate : bool
            If True, timeseries output is a rate.
        """
        if rate:
            raise NotImplementedError("Timeseries output rates are not currently supported for "
                                      "SolveIVP transcriptions.")

        nodeshape = (self.num_nodes,)
        input_name = f'all_values:{name}'

        self.add_input(input_name, shape=nodeshape + shape,  units=units, desc=desc)
        self.add_output(name, shape=nodeshape + shape, units=units, desc=desc)

        self._vars[name] = (input_name, name, shape, rate)

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
        outputs.set_val(inputs.asarray())
