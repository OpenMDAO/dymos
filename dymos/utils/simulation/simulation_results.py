from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, SqliteRecorder
from openmdao.utils.units import valid_units, convert_units

from dymos.utils.misc import get_rate_units


class SimulationResults(object):
    """
    SimulationResults is returned by phase.simulate.  It's primary
    purpose is to hold the dictionary of results from the integration
    and to provide a `get_values` interface that is equivalent to that
    in Phase (except that it has no knowledge of nodes).
    """
    def __init__(self, time_options, state_options, control_options):
        """

        Parameters
        ----------
        phase : dymos.Phase object
            The phase being simulated.  Phase is passed on initialization of
            SimulationResults so that it can gather knowledge of time units,
            state options, control options, and ODE outputs.
        """
        self.time_options = time_options
        self.state_options = state_options
        self.control_options = control_options
        self.outputs = {}
        self.units = {}

    def record_results(self, filename, ode_class, ode_init_kwargs,
                       time_options, state_options, control_options):
        p = Problem(model=Group())
        time = self.get_values('time')
        nn = len(time)
        ode_sys = ode_class(num_nodes=nn, **ode_init_kwargs)
        ivc = p.model.add_subsystem('inputs', subsys=IndepVarComp(), promotes_outputs=['*'])
        p.model.add_subsystem('ode', subsys=ode_sys)

        # Connect times
        ivc.add_output('time', val=np.zeros(nn), units=self.time_options['units'])
        p.model.connect('time', ['ode.{0}'.format(t) for t in self.time_options['targets']])

        # Connect states
        for name, options in iteritems(self.state_options):
            ivc.add_output('states:{0}'.format(name),
                           val=np.zeros((nn,) + options['shape']), units=options['units'])
            p.model.connect('states:{0}'.format(name),
                            ['ode.{0}'.format(t) for t in options['targets']])

        # Connect controls
        sys_param_options = ode_sys.ode_options._dynamic_parameters
        for name, options in iteritems(self.control_options):
            units = options['units']
            ivc.add_output('controls:{0}'.format(name),
                           val=np.zeros((nn,) + options['shape']), units=units)
            p.model.connect('controls:{0}'.format(name),
                            ['ode.{0}'.format(t) for t in sys_param_options[name]['targets']],
                            src_indices=np.arange(nn, dtype=int))
            if options['rate_param']:
                rate_targets = sys_param_options[options['rate_param']]['targets']
                rate_units = get_rate_units(units, self.time_options['units'], deriv=1)
                ivc.add_output('control_rates:{0}_rate'.format(name),
                               val=np.zeros((nn,) + options['shape']),
                               units=rate_units)
                p.model.connect('control_rates:{0}_rate'.format(name),
                                ['ode.{0}'.format(t) for t in rate_targets],
                                src_indices=np.arange(nn, dtype=int))
            if options['rate2_param']:
                rate2_targets = sys_param_options[options['rate2_param']]['targets']
                rate2_units = get_rate_units(units, self.time_options['units'], deriv=2)
                ivc.add_output('control_rates:{0}_rate2'.format(name),
                               val=np.zeros((nn,) + options['shape']),
                               units=rate2_units)
                p.model.connect('control_rates:{0}_rate2'.format(name),
                                ['ode.{0}'.format(t) for t in rate2_targets],
                                src_indices=np.arange(nn, dtype=int))

        p.setup(check=True)

        p.model.add_recorder(SqliteRecorder(filename))

        # Assign times
        p['time'] = time[:, 0]

        # Assign states
        for name in self.state_options:
            p['states:{0}'.format(name)] = self.get_values(name)

        # Assign controls
        for name, options in iteritems(self.control_options):
            shape = p['controls:{0}'.format(name)].shape
            p['controls:{0}'.format(name)] = np.reshape(self.get_values(name), shape)
            if options['rate_param']:
                p['control_rates:{0}_rate'.format(name)] = \
                    np.reshape(self.get_values('{0}_rate'.format(name)), shape)
            if options['rate2_param']:
                p['control_rates:{0}_rate2'.format(name)] = \
                    np.reshape(self.get_values('{0}_rate2'.format(name)), shape)

        # Populate outputs of ODE
        prom2abs_ode_outputs = p.model.ode._var_allprocs_prom2abs_list['output']
        for prom_name, abs_name in iteritems(prom2abs_ode_outputs):
            p[abs_name[0]] = np.reshape(self.get_values(prom_name), p[abs_name[0]].shape)

        # Run model to record file
        p.run_model()

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
