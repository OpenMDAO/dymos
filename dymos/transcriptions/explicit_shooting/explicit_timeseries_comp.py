import numpy as np

from openmdao.utils.units import unit_conversion
from ...options import options as dymos_options

from ..common.timeseries_output_comp import TimeseriesOutputCompBase


class ExplicitTimeseriesComp(TimeseriesOutputCompBase):

    """
    Class definition of the ExplicitTimeseriesComp.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super(ExplicitTimeseriesComp, self).__init__(**kwargs)

        self.input_num_nodes = 0
        self.output_num_nodes = 0

        # Sources is used internally to map the source of a connection to the timeseries to
        # the corresponding input variable.  This is used to ensure that we don't need to connect
        # the same source to this timeseries multiple times.
        self._sources = {}

        # Used to track conversion factors for instances when one output that relies on an input
        # from another variable has potentially different units
        self._units = {}
        self._conversion_factors = {}
        self._vars = {}

        self._no_check_partials = not dymos_options['include_check_partials']

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        igd = self.options['input_grid_data']
        ogd = self.options['output_grid_data']

        if ogd is not None:
            raise ValueError('Currently ExplicitTimeseriesComp does not accept a separate '
                             'GridData for output.  Leave output_grid_data as None')

        # Make the assumption here that we
        self.input_num_nodes = igd.subset_num_nodes['segment_ends']
        self.output_num_nodes = self.input_num_nodes

    def _add_output_configure(self, name, units, shape, desc='', src=None):
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
        self.add_output(output_name,
                        shape=(output_num_nodes,) + shape,
                        units=units, desc=desc)

        self._vars[name] = (input_name, output_name, shape)

        size = np.prod(shape)
        rs = cs = np.arange(output_num_nodes * size, dtype=int)

        # There's a chance that the input for this output was pulled from another variable with
        # different units, so account for that with a conversion.
        if None in {input_units, units}:
            scale = 1.0
            offset = 0
        else:
            scale, offset = unit_conversion(input_units, units)
        self._conversion_factors[output_name] = scale, offset

        self.declare_partials(of=output_name,
                              wrt=input_name,
                              rows=rs, cols=cs, val=scale)

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
        for (input_name, output_name, _) in self._vars.values():
            scale, offset = self._conversion_factors[output_name]
            outputs[output_name] = scale * (inputs[input_name] + offset)
