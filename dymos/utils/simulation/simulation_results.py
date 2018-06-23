from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, SqliteRecorder, CaseReader
from openmdao.utils.units import valid_units, convert_units

from dymos.utils.misc import get_rate_units
from dymos.phases.options import TimeOptionsDictionary, StateOptionsDictionary, \
    ControlOptionsDictionary


class SimulationResults(object):
    """
    SimulationResults is returned by phase.simulate.  It's primary
    purpose is to hold the dictionary of results from the integration
    and to provide a `get_values` interface that is equivalent to that
    in Phase (except that it has no knowledge of nodes).

    Parameters
    ----------
    filepath : str or None
        A filepath from which SimulationResults are to be loaded.
    time_options : dymos.TimeOptionsDictionary or None
        The options dictionary for the phase tied to this instance of simulation results.  If
        being loaded from a file, this is not needed at instantiation.
    state_options : dict of dymos.StateOptionsDictionary or None
        The options dictionary for the phase tied to this instance of simulation results.  If
        being loaded from a file, this is not needed at instantiation.
    control_options : dict of dymos.ControlOptionsDictionary or None
        The options dictionary for the phase tied to this instance of simulation results.  If
        being loaded from a file, this is not needed at instantiation.
    design_parameter_options : dict of dymos.DesignParameterOptionsDictionary or None
        The options dictionary for the phase tied to this instance of simulation results.  If
        being loaded from a file, this is not needed at instantiation.
    """
    def __init__(self, filepath=None, time_options=None, state_options=None, control_options=None,
                 design_parameter_options=None):
        self.time_options = time_options
        self.state_options = state_options
        self.control_options = control_options
        self.design_parameter_options = design_parameter_options
        self.outputs = {}
        self.units = {}

        if isinstance(filepath, str):
            self._load_results(filepath)

    def record_results(self, filename, ode_class, ode_init_kwargs=None):
        """
        Record the outputs to the given filename.  This is done by instantiating a new
        problem with an IndepVarComp for the time, states, controls, and control rates, as
        well as an instance of the given ODE class (instantiated with the number of nodes equal
        to the number of times in the outputs).

        The system is populated with data from outputs, has a recorder attached, and is executed
        via problem.run_model.  This data can then be retrieved from the `system_cases` attribute
        of CaseReader.

        Parameters
        ----------
        filename : str
            The filename to which the recording should be saved.
        ode_class : openmdao.System
            A system class with the appropriate ODE metadata attached via the dymos declare_time,
            declare_state, and declare_parameter decorators.
        ode_init_kwargs : dict or None
            A dictionary of keyword arguments with which ode_class should be instantiated.
        """
        init_kwargs = {} if ode_init_kwargs is None else ode_init_kwargs

        p = Problem(model=Group())
        time = self.get_values('time')
        nn = len(time)
        ode_sys = ode_class(num_nodes=nn, **init_kwargs)
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
        sys_param_options = ode_sys.ode_options._parameters
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
        # Connect design parameters
        for name, options in iteritems(self.design_parameter_options):
            units = options['units']
            ivc.add_output('design_parameters:{0}'.format(name),
                           val=np.zeros((nn,) + options['shape']), units=units)
            p.model.connect('design_parameters:{0}'.format(name),
                            ['ode.{0}'.format(t) for t in sys_param_options[name]['targets']],
                            src_indices=np.arange(nn, dtype=int))

        p.setup(check=True)

        p.model.add_recorder(SqliteRecorder(filename))
        p.model.recording_options['record_metadata'] = True
        p.model.recording_options['record_outputs'] = True

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

        # Assign design parameters
        for name, options in iteritems(self.design_parameter_options):
            shape = p['design_parameters:{0}'.format(name)].shape
            p['design_parameters:{0}'.format(name)] = np.reshape(self.get_values(name), shape)

        # Populate outputs of ODE
        prom2abs_ode_outputs = p.model.ode._var_allprocs_prom2abs_list['output']
        for prom_name, abs_name in iteritems(prom2abs_ode_outputs):
            if p[abs_name[0]].shape[0] == 1:
                p[abs_name[0]] = self.get_values(prom_name)[0]
            else:
                p[abs_name[0]] = np.reshape(self.get_values(prom_name), p[abs_name[0]].shape)

        # Run model to record file
        p.run_model()

    def _load_results(self, filename):
        """
        Load SimulationResults from the given file.

        Parameters
        ----------
        filename : str
            The path of the file from which to load the simulation results.
        """
        cr = CaseReader(filename)
        case = cr.system_cases.get_case(-1)

        loaded_outputs = case.outputs._prom2abs['output']
        for name in loaded_outputs:
            self.outputs[name] = {}
            self.outputs[name]['value'] = case.outputs[name]
            self.outputs[name]['units'] = None
            self.outputs[name]['shape'] = case.outputs[name].shape[1:]

        # TODO: Get time, state, and control options from the case metadata
        self.time_options = TimeOptionsDictionary()
        self.state_options = {}
        self.control_options = {}

        states = [s.split(':')[-1] for s in loaded_outputs if s.startswith('states:')]
        controls = [s.split(':')[-1] for s in loaded_outputs if s.startswith('controls:')]

        for s in states:
            self.state_options[s] = StateOptionsDictionary()

        for c in controls:
            self.control_options[c] = ControlOptionsDictionary()

    def get_values(self, var, units=None, nodes=None):

        if units is not None and not valid_units(units):
            raise ValueError('{0} is not a valid set of units.'.format(units))

        if nodes is not None:
            raise RuntimeWarning('Argument nodes has no meaning for SimulationResults.get_values '
                                 'and is included for compatibility with Phase.get_values')

        if var == 'time':
            output_path = 'time'

        elif var in self.state_options:
            output_path = 'states:{0}'.format(var)

        elif var in self.control_options:  # and self.control_options[var]['opt']:
            output_path = 'controls:{0}'.format(var)

        elif var in self.design_parameter_options:  # and self.design_parameter_options[var]['opt']:
            output_path = 'design_parameters:{0}'.format(var)

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
