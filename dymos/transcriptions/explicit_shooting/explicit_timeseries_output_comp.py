import numpy as np
from dymos.transcriptions.common.timeseries_output_comp import TimeseriesOutputCompBase


class ExplicitTimeseriesOutputComp(TimeseriesOutputCompBase):
    """
    Class definition for SolveIVPTimeseriesOutputComp.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        grid_data = self.options['input_grid_data']
        self.num_nodes = 2 * grid_data.num_segments

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

        size = np.prod(shape, dtype=int)
        ar = np.arange(size * num_nodes, dtype=int)
        self.declare_partials(of=output_name, wrt=input_name, rows=ar, cols=ar)

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
