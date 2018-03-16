from __future__ import print_function, division, absolute_import

from openmdao.utils.units import valid_units, convert_units


class SimulationResults(object):
    """
    SimulationResults is returned by phase.simulate.  It's primary
    purpose is to hold the dictionary of results from the integration
    and to provide a `get_values` interface that is equivalent to that
    in Phase (except that it has no knowledge of nodes).
    """
    def __init__(self, state_options, control_options):
        """

        Parameters
        ----------
        phase : dymos.Phase object
            The phase being simulated.  Phase is passed on initialization of
            SimulationResults so that it can gather knowledge of time units,
            state options, control options, and ODE outputs.
        """
        self.state_options = state_options
        self.control_options = control_options
        self.outputs = {}
        self.units = {}

    def get_values(self, var, units=None, nodes=None):

        if units is not None and not valid_units(units):
            raise ValueError('{0} is not a valid set of units.'.format(units))

        if var == 'time':
            output_path = 'time'

        elif var in self.state_options:
            output_path = 'states:{0}'.format(var)

        elif var in self.control_options and self.control_options[var]['opt']:
            output_path = 'controls:{0}'.format(var)

        elif var in self.control_options and not self.control_options[var]['opt']:
            # TODO: make a test for this, haven't experimented with this yet.
            output_path = 'controls:{0}'.format(var)

        elif var.endswith('_rate') and var[:-5] in self.control_options:
            output_path = 'control_rates:{0}'.format(var)

        elif var.endswith('_rate2') and var[:-6] in self.control_options:
            output_path = 'control_rates:{0}'.format(var)

        else:
            output_path = 'ode.{0}'.format(var)

        output = convert_units(self.outputs[output_path]['value'],
                               self.outputs[output_path]['units'],
                               units)

        return output
