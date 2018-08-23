from __future__ import print_function, division, absolute_import

from six import iteritems
import sys

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, SqliteRecorder, CaseReader
from openmdao.utils.units import valid_units, convert_units

from dymos.utils.misc import get_rate_units, convert_to_ascii


class PhaseSimulationResults(object):
    """
    PhaseSimulationResults is returned by phase.simulate.  It's primary
    purpose is to hold the dictionary of results from the integration
    and to provide a `get_values` interface that is equivalent to that
    in Phase (except that it has no knowledge of nodes).

    Parameters
    ----------
    filepath : str or None
        A filepath from which PhaseSimulationResults are to be loaded.
    time_options : {str: TimeOptionsDictionary} or None
        The options dictionary for the phase tied to this instance of simulation results.  If
        being loaded from a file, this is not needed at instantiation.
    state_options : dict of {str: StateOptionsDictionary} or None
        The options dictionary for the phase tied to this instance of simulation results.  If
        being loaded from a file, this is not needed at instantiation.
    control_options : dict of {str: ControlOptionsDictionary} or None
        The options dictionary for the phase tied to this instance of simulation results.  If
        being loaded from a file, this is not needed at instantiation.
    design_parameter_options : dict of {str: DesignParameterOptionsDictionary} or None
        The options dictionary for the phase tied to this instance of simulation results.  If
        being loaded from a file, this is not needed at instantiation.
    design_parameter_options : dict of {str: DesignParameterOptionsDictionary} or None
        The options dictionary for the phase tied to this instance of simulation results.  If
        being loaded from a file, this is not needed at instantiation.
    """
    def __init__(self, filepath=None, time_options=None, state_options=None,
                 control_options=None, design_parameter_options=None, input_parameter_options=None,
                 traj_design_parameter_options=None, traj_input_parameter_options=None):
        self.time_options = {} if time_options is None else time_options
        self.state_options = {} if state_options is None else state_options
        self.control_options = {} if control_options is None else control_options
        self.design_parameter_options = {} if design_parameter_options is None else \
            design_parameter_options
        self.input_parameter_options = {} if input_parameter_options is None else \
            input_parameter_options
        self.traj_design_parameter_options = {} if traj_design_parameter_options is None else \
            traj_design_parameter_options
        self.traj_input_parameter_options = {} if traj_input_parameter_options is None else \
            traj_input_parameter_options
        self.outputs = {'indep': {}, 'states': {}, 'controls': {}, 'control_rates': {},
                        'design_parameters': {}, 'input_parameters': {},
                        'traj_design_parameters': {}, 'traj_input_parameters': {}, 'ode': {}}
        self.units = {}

        if isinstance(filepath, str):
            self._load_results(filepath)

    def record_results(self, phase_name, filename, ode_class, ode_init_kwargs=None):
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
        phase_name : str
            The name of the phase whose simulation results are being recorded.
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
        ivc.add_output('time', val=np.zeros((nn, 1)), units=self.time_options['units'])
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

            rate_units = get_rate_units(units, self.time_options['units'], deriv=1)
            ivc.add_output('control_rates:{0}_rate'.format(name),
                           val=np.zeros((nn,) + options['shape']),
                           units=rate_units)
            if options['rate_param']:
                rate_targets = sys_param_options[options['rate_param']]['targets']
                p.model.connect('control_rates:{0}_rate'.format(name),
                                ['ode.{0}'.format(t) for t in rate_targets],
                                src_indices=np.arange(nn, dtype=int))

            rate2_units = get_rate_units(units, self.time_options['units'], deriv=2)
            ivc.add_output('control_rates:{0}_rate2'.format(name),
                           val=np.zeros((nn,) + options['shape']),
                           units=rate2_units)
            if options['rate2_param']:
                rate2_targets = sys_param_options[options['rate2_param']]['targets']
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

        # Connect input parameters
        for name, options in iteritems(self.input_parameter_options):
            units = options['units']
            ivc.add_output('input_parameters:{0}'.format(name),
                           val=np.zeros((nn,) + options['shape']), units=units)
            p.model.connect('input_parameters:{0}'.format(name),
                            ['ode.{0}'.format(t) for t in sys_param_options[name]['targets']],
                            src_indices=np.arange(nn, dtype=int))

        # Connect trajectory design parameters
        for name, options in iteritems(self.traj_design_parameter_options):
            units = options['units']
            ivc.add_output('traj_design_parameters:{0}'.format(name),
                           val=np.zeros((nn,) + options['shape']), units=units)
            param_name = name if options['targets'] is None else \
                options['targets'].get(phase_name, None)
            p.model.connect('traj_design_parameters:{0}'.format(name),
                            ['ode.{0}'.format(t) for t in sys_param_options[param_name]['targets']],
                            src_indices=np.arange(nn, dtype=int))

        # Connect trajectory input parameters
        for name, options in iteritems(self.traj_design_parameter_options):
            units = options['units']
            ivc.add_output('traj_input_parameters:{0}'.format(name),
                           val=np.zeros((nn,) + options['shape']), units=units)
            param_name = name if options['targets'] is None else \
                options['targets'].get(phase_name, None)
            p.model.connect('traj_input_parameters:{0}'.format(name),
                            ['ode.{0}'.format(t) for t in sys_param_options[param_name]['targets']],
                            src_indices=np.arange(nn, dtype=int))

        p.setup(check=False)

        p.model.add_recorder(SqliteRecorder(filename))
        p.model.recording_options['record_metadata'] = True
        p.model.recording_options['record_outputs'] = True

        # Assign times
        p['time'] = time

        # Assign states
        for name in self.state_options:
            p['states:{0}'.format(name)] = self.get_values(name)

        # Assign controls
        for name, options in iteritems(self.control_options):
            shape = p['controls:{0}'.format(name)].shape
            p['controls:{0}'.format(name)] = np.reshape(self.get_values(name), shape)

            p['control_rates:{0}_rate'.format(name)] = \
                np.reshape(self.get_values('{0}_rate'.format(name)), shape)

            p['control_rates:{0}_rate2'.format(name)] = \
                np.reshape(self.get_values('{0}_rate2'.format(name)), shape)

        # Assign design parameters
        for name, options in iteritems(self.design_parameter_options):
            shape = p['design_parameters:{0}'.format(name)].shape
            p['design_parameters:{0}'.format(name)] = np.reshape(self.get_values(name), shape)

        # Assign input parameters
        for name, options in iteritems(self.input_parameter_options):
            shape = p['input_parameters:{0}'.format(name)].shape
            p['input_parameters:{0}'.format(name)] = np.reshape(self.get_values(name), shape)

        # Assign trajectory design parameters
        for name, options in iteritems(self.traj_design_parameter_options):
            shape = p['traj_design_parameters:{0}'.format(name)].shape
            p['traj_design_parameters:{0}'.format(name)] = np.reshape(self.get_values(name), shape)

        # Assign trajectory design parameters
        for name, options in iteritems(self.traj_input_parameter_options):
            shape = p['traj_input_parameters:{0}'.format(name)].shape
            p['traj_input_parameters:{0}'.format(name)] = np.reshape(self.get_values(name), shape)

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
        Load PhaseSimulationResults from the given file.

        Parameters
        ----------
        filename : str
            The path of the file from which to load the simulation results.
        """
        cr = CaseReader(filename)
        case = cr.system_cases.get_case(-1)
        loaded_outputs = cr.list_outputs(case=case, explicit=True, implicit=True, values=True,
                                         units=True, shape=True, out_stream=None)

        self.outputs = {'indep': {}, 'states': {}, 'controls': {}, 'control_rates': {},
                        'design_parameters': {}, 'traj_design_parameters': {}, 'ode': {}}

        for output_name, options in loaded_outputs:

            if output_name.startswith('inputs.'):
                output_name = output_name.replace('inputs.', '')

                if output_name == 'time':
                    var_type = 'indep'
                    var_name = 'time'
                if output_name.startswith('states:'):
                    var_type = 'states'
                    var_name = output_name.replace('states:', '', 1)
                elif output_name.startswith('controls:'):
                    var_type = 'controls'
                    var_name = output_name.replace('controls:', '', 1)
                elif output_name.startswith('control_rates:'):
                    var_type = 'control_rates'
                    var_name = output_name.replace('control_rates:', '', 1)
                elif output_name.startswith('design_parameters:'):
                    var_type = 'design_parameters'
                    var_name = output_name.replace('design_parameters:', '', 1)
                elif output_name.startswith('traj_design_parameters:'):
                    var_type = 'traj_design_parameters'
                    var_name = output_name.replace('traj_design_parameters:', '', 1)

                val = options['value']

            elif output_name.startswith('ode.'):
                var_type = 'ode'
                var_name = output_name.replace('ode.', '')

                if len(options['value'].shape) == 1:
                    val = options['value'][:, np.newaxis]
                else:
                    val = options['value']
            else:
                raise RuntimeError('unexpected output in file {1}: {0}'.format(output_name,
                                                                               filename))

            self.outputs[var_type][var_name] = {}
            self.outputs[var_type][var_name]['value'] = val
            self.outputs[var_type][var_name]['units'] = convert_to_ascii(options['units'])
            self.outputs[var_type][var_name]['shape'] = tuple(val.shape[1:])

    def get_values(self, var, units=None):

        if units is not None and not valid_units(units):
            raise ValueError('{0} is not a valid set of units.'.format(units))

        var_in_phase = True

        if var == 'time':
            var_type = 'indep'
        elif var in self.outputs['states']:
            var_type = 'states'
        elif var in self.outputs['controls']:
            var_type = 'controls'
        elif var in self.outputs['design_parameters']:
            var_type = 'design_parameters'
        elif var in self.outputs['input_parameters']:
            var_type = 'input_parameters'
        elif var in self.outputs['traj_design_parameters']:
            var_type = 'traj_design_parameters'
        elif var in self.outputs['traj_input_parameters']:
            var_type = 'traj_input_parameters'
        elif var in self.outputs['control_rates']:
            var_type = 'control_rates'
        elif var in self.outputs['ode']:
            var_type = 'ode'
        else:
            var_in_phase = False

        if not var_in_phase:
            raise ValueError('Variable "{0}" not found in phase '
                             'simulation results.'.format(var))

        output = convert_units(self.outputs[var_type][var]['value'],
                               self.outputs[var_type][var]['units'],
                               units)

        return output
