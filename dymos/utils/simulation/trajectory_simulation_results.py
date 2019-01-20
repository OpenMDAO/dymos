from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from openmdao.api import Problem, Group, ParallelGroup, IndepVarComp, SqliteRecorder, CaseReader
from openmdao.utils.units import valid_units, convert_units

from dymos.utils.misc import get_rate_units, convert_to_ascii


class TrajectorySimulationResults(object):
    """
    TrajectorySimulationResults is returned by trajectory.simulate.  It's primary
    purpose is to hold the dictionary of results from the integration
    and to provide a `get_values` interface that is equivalent to that
    in trajectory (except that it has no knowledge of nodes).

    TrajectorySimulationResults is created by combining the PhaseSimulationResults from
    the phases contained within it.

    Parameters
    ----------
    filepath : str or None
        A filepath from which PhaseSimulationResults are to be loaded.
    exp_outs : dict of {str: PhaseSimulationResults}
        A dictionary of PhaseSimulationResults for each phase within the trajectory, keyed by
        phase name.
    """
    def __init__(self, filepath=None, exp_outs=None):

        self.outputs = {'phases': {}}

        if filepath is not None and exp_outs is not None:
            raise RuntimeError('TrajectorySimulationResults should be instantiated with either '
                               'filepath or exp_outs, but not both.')

        if filepath:
            self._load_results(filepath)
        elif exp_outs:
            for phase_name, phase_results in iteritems(exp_outs):
                self.outputs['phases'][phase_name] = {}
                self.outputs['phases'][phase_name]['indep'] = {}
                self.outputs['phases'][phase_name]['states'] = {}
                self.outputs['phases'][phase_name]['controls'] = {}
                self.outputs['phases'][phase_name]['control_rates'] = {}
                self.outputs['phases'][phase_name]['design_parameters'] = {}
                self.outputs['phases'][phase_name]['input_parameters'] = {}
                self.outputs['phases'][phase_name]['ode'] = {}

                for var_type in ('indep', 'states', 'controls', 'control_rates',
                                 'design_parameters', 'input_parameters', 'ode'):
                    pdict = phase_results.outputs[var_type]
                    self.outputs['phases'][phase_name][var_type].update(pdict)

    def record_results(self, traj, exp_outs, filename):
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
        traj : Trajectory
            The Trajectory whose simulated results are to be recorded.
        exp_outs : dict of {str : PhaseSimulationResults}.
            A dictionary of {phase_name : PhaseSimulationResults} for each phase to be recorded
        filename : str
            The filename to which the recording should be saved.
        """
        # Make sure we have the same keys/phases in both the trajectory and exp_outs
        if set(traj._phases) != set(exp_outs):
            raise ValueError('TrajectorySimulationResults trajectory._phases and '
                             'exp_outs must contain the same keys.')

        p = Problem(model=Group())

        traj_group = p.model.add_subsystem('phases', ParallelGroup())

        phase_groups = {}

        for phase_name, phase in iteritems(traj._phases):
            ode_class = phase.options['ode_class']
            init_kwargs = phase.options['ode_init_kwargs']
            exp_out = exp_outs[phase_name]

            phase_group = traj_group.add_subsystem(phase_name, subsys=Group())
            phase_groups[phase_name] = phase_group

            time = exp_out.get_values('time')
            nn = len(time)
            ode_sys = ode_class(num_nodes=nn, **init_kwargs)
            ivc = phase_group.add_subsystem('inputs', subsys=IndepVarComp(), promotes_outputs=['*'])
            phase_group.add_subsystem('ode', subsys=ode_sys)

            # Connect times
            ivc.add_output('time', val=np.zeros((nn, 1)), units=phase.time_options['units'])
            phase_group.connect('time',
                                ['ode.{0}'.format(t) for t in phase.time_options['targets']])

            # Connect states
            for name, options in iteritems(phase.state_options):
                ivc.add_output('states:{0}'.format(name),
                               val=np.zeros((nn,) + options['shape']), units=options['units'])
                phase_group.connect('states:{0}'.format(name),
                                    ['ode.{0}'.format(t) for t in options['targets']])

            # Connect controls
            sys_param_options = ode_sys.ode_options._parameters
            for name, options in iteritems(phase.control_options):
                units = options['units']
                ivc.add_output('controls:{0}'.format(name),
                               val=np.zeros((nn,) + options['shape']), units=units)
                phase_group.connect('controls:{0}'.format(name),
                                    ['ode.{0}'.format(t) for
                                     t in sys_param_options[name]['targets']],
                                    src_indices=np.arange(nn, dtype=int))

                rate_units = get_rate_units(units, phase.time_options['units'], deriv=1)
                ivc.add_output('control_rates:{0}_rate'.format(name),
                               val=np.zeros((nn,) + options['shape']),
                               units=rate_units)
                if options['rate_param']:
                    rate_targets = sys_param_options[options['rate_param']]['targets']
                    phase_group.connect('control_rates:{0}_rate'.format(name),
                                        ['ode.{0}'.format(t) for t in rate_targets],
                                        src_indices=np.arange(nn, dtype=int))

                rate2_units = get_rate_units(units, phase.time_options['units'], deriv=2)
                ivc.add_output('control_rates:{0}_rate2'.format(name),
                               val=np.zeros((nn,) + options['shape']),
                               units=rate2_units)
                if options['rate2_param']:
                    rate2_targets = sys_param_options[options['rate2_param']]['targets']
                    phase_group.connect('control_rates:{0}_rate2'.format(name),
                                        ['ode.{0}'.format(t) for t in rate2_targets],
                                        src_indices=np.arange(nn, dtype=int))

            # Connect design parameters
            for name, options in iteritems(phase.design_parameter_options):
                units = options['units']
                ivc.add_output('design_parameters:{0}'.format(name),
                               val=np.zeros((nn,) + options['shape']), units=units)
                phase_group.connect('design_parameters:{0}'.format(name),
                                    ['ode.{0}'.format(t) for t in
                                     sys_param_options[name]['targets']],
                                    src_indices=np.arange(nn, dtype=int))

            # Connect input parameters
            for name, options in iteritems(phase.input_parameter_options):
                units = options['units']
                ivc.add_output('input_parameters:{0}'.format(name),
                               val=np.zeros((nn,) + options['shape']), units=units)
                phase_group.connect('input_parameters:{0}'.format(name),
                                    ['ode.{0}'.format(t) for t in
                                    sys_param_options[options['target_param']]['targets']],
                                    src_indices=np.arange(nn, dtype=int))

        p.setup(check=True)

        p.model.add_recorder(SqliteRecorder(filename))
        p.model.recording_options['record_metadata'] = True
        p.model.recording_options['record_outputs'] = True

        # Assign values

        for phase_name, phase in iteritems(traj._phases):
            exp_out = exp_outs[phase_name]

            # Assign times
            p['phases.{0}.time'.format(phase_name)] = exp_out.get_values('time')

            # Assign states
            for name in phase.state_options:
                p['phases.{0}.states:{1}'.format(phase_name, name)] = exp_out.get_values(name)

            # Assign controls
            for name, options in iteritems(phase.control_options):
                shape = p['phases.{0}.controls:{1}'.format(phase_name, name)].shape

                p['phases.{0}.controls:{1}'.format(phase_name, name)] = \
                    np.reshape(exp_out.get_values(name), shape)

                p['phases.{0}.control_rates:{1}_rate'.format(phase_name, name)] = \
                    np.reshape(exp_out.get_values('{0}_rate'.format(name)), shape)

                p['phases.{0}.control_rates:{1}_rate2'.format(phase_name, name)] = \
                    np.reshape(exp_out.get_values('{0}_rate2'.format(name)), shape)

            # Assign design parameters
            for name, options in iteritems(phase.design_parameter_options):
                shape = p['phases.{0}.design_parameters:{1}'.format(phase_name, name)].shape
                p['phases.{0}.design_parameters:{1}'.format(phase_name, name)] = \
                    np.reshape(exp_out.get_values(name), shape)

            # Assign input parameters
            for name, options in iteritems(phase.input_parameter_options):
                shape = p['phases.{0}.input_parameters:{1}'.format(phase_name, name)].shape
                p['phases.{0}.input_parameters:{1}'.format(phase_name, name)] = \
                    np.reshape(exp_out.get_values(name), shape)

            # Populate outputs of ODE
            prom2abs_ode_outputs = \
                phase_groups[phase_name].ode._var_allprocs_prom2abs_list['output']
            for prom_name, abs_name in iteritems(prom2abs_ode_outputs):
                if p[abs_name[0]].shape[0] == 1:
                    p[abs_name[0]] = exp_out.get_values(prom_name)[0]
                else:
                    p[abs_name[0]] = np.reshape(exp_out.get_values(prom_name), p[abs_name[0]].shape)

        # Run model to record file
        p.run_model()

    def get_phase_names(self):
        """
        Retrieve the names of the phases stored in this TrajectorySimulationResults object.

        Returns
        -------
        list
            The names of the phases in the TrajectorySimulationResults
        """
        return list(self.outputs['phases'].keys())

    def _load_results(self, filename):
        """
        Load PhaseSimulationResults from the given file.

        Parameters
        ----------
        filename : str
            The path of the file from which to load the simulation results.
        """
        cr = CaseReader(filename)

        system_cases = cr.list_cases('root')

        case = cr.get_case(system_cases[-1])

        loaded_outputs = case.list_outputs(explicit=True, implicit=True, values=True,
                                           units=True, shape=True, out_stream=None)

        phase_names = set([s[0].split('.')[1] for s in loaded_outputs
                           if s[0].startswith('phases.')])

        for phase_name in phase_names:
            self.outputs['phases'][phase_name] = {}
            self.outputs['phases'][phase_name]['indep'] = {}
            self.outputs['phases'][phase_name]['states'] = {}
            self.outputs['phases'][phase_name]['controls'] = {}
            self.outputs['phases'][phase_name]['control_rates'] = {}
            self.outputs['phases'][phase_name]['design_parameters'] = {}
            self.outputs['phases'][phase_name]['input_parameters'] = {}
            self.outputs['phases'][phase_name]['ode'] = {}

        for name, options in loaded_outputs:
            if name.startswith('phases'):
                phase_name = name.split('.')[1]
                output_name = name.replace('phases.{0}.'.format(phase_name), '')

                if output_name.startswith('inputs.'):
                    output_name = output_name.replace('inputs.', '')

                    if output_name == 'time':
                        var_type = 'indep'
                        var_name = 'time'
                    elif output_name == 'time_phase':
                        var_type = 'time_phase'
                        var_name = 'time_phase'
                    elif output_name.startswith('states:'):
                        var_type = 'states'
                        var_name = output_name.split(':')[-1]
                    elif output_name.startswith('controls:'):
                        var_type = 'controls'
                        var_name = output_name.split(':')[-1]
                    elif output_name.startswith('control_rates:'):
                        var_type = 'control_rates'
                        var_name = output_name.split(':')[-1]
                    elif output_name.startswith('design_parameters:'):
                        var_type = 'design_parameters'
                        var_name = output_name.split(':')[-1]
                    elif output_name.startswith('input_parameters:'):
                        var_type = 'input_parameters'
                        var_name = output_name.split(':')[-1]

                elif output_name.startswith('ode.'):
                    var_type = 'ode'
                    var_name = output_name.replace('ode.', '')

                else:
                    raise RuntimeError('unexpected output in phase {1}: {0}'.format(name,
                                                                                    phase_name))

                self.outputs['phases'][phase_name][var_type][var_name] = {}
                self.outputs['phases'][phase_name][var_type][var_name]['value'] = options['value']
                self.outputs['phases'][phase_name][var_type][var_name]['units'] = \
                    convert_to_ascii(options['units'])
                self.outputs['phases'][phase_name][var_type][var_name]['shape'] = options['shape']

    def get_values(self, var, phases=None, units=None, flat=False):
        """
        Returns the values of the given variable from the given phases, if provided.
        If the variable is not present in one ore more phases, it will be returned as
        numpy.nan at each time step.

        Parameters
        ----------
        var : str
            The variable whose values are to be returned.
        phases : Sequence, None
            The phases from which the values are desired.  If None, included all Phases.
        units : str, None
            The units in which the values are desired.
        flat : bool
            If False return the values in a dictionary keyed by phase name.  If True,
            return a single array incorporating values from all phases.

        Returns
        -------
        dict or np.array
            If flat=False, a dictionary of the values of the variable in each phase will be
            returned, keyed by Phase name.  If the values are not present in a subset of the phases,
            return numpy.nan at each time point in those phases.

        Raises
        ------
        KeyError
            If the given variable is not found in any phase, a KeyError is raised.

        """

        if units is not None and not valid_units(units):
            raise ValueError('{0} is not a valid set of units.'.format(units))

        if isinstance(phases, str): # allow strings if you just want one phase 
            phases = [phases]
            
        phases = self.get_phase_names() if phases is None else phases

        return_vals = dict([(phase_name, {}) for phase_name in phases])

        var_in_traj = False

        times = {}
        time_units = None

        for phase_name in phases:
            var_in_phase = True

            # Gather times for the purposes of flattening the returned values
            # Note the adjustment to the last time, for the purposes of sorting only
            if time_units is None:
                time_units = self.outputs['phases'][phase_name]['indep']['time']['units']
            times[phase_name] = convert_units(
                self.outputs['phases'][phase_name]['indep']['time']['value'],
                self.outputs['phases'][phase_name]['indep']['time']['units'],
                time_units)
            times[phase_name][-1, ...] -= 1.0E-15

            if var == 'time':
                var_type = 'indep'
            elif var == 'time_phase':
                var_type = 'time_phase'
            elif var in self.outputs['phases'][phase_name]['states']:
                var_type = 'states'
            elif var in self.outputs['phases'][phase_name]['controls']:
                var_type = 'controls'
            elif var in self.outputs['phases'][phase_name]['design_parameters']:
                var_type = 'design_parameters'
            elif var in self.outputs['phases'][phase_name]['input_parameters']:
                var_type = 'input_parameters'
            elif var.endswith('_rate') \
                    and var[:-5] in self.outputs['phases'][phase_name]['controls']:
                var_type = 'control_rates'
            elif var.endswith('_rate2') \
                    and var[:-6] in self.outputs['phases'][phase_name]['controls']:
                var_type = 'control_rates'
            elif var in self.outputs['phases'][phase_name]['ode']:
                var_type = 'ode'
            else:
                var_in_phase = False

            if var_in_phase:
                var_in_traj = True
                output = convert_units(self.outputs['phases'][phase_name][var_type][var]['value'],
                                       self.outputs['phases'][phase_name][var_type][var]['units'],
                                       units)
            else:
                indep_var = list(self.outputs['phases'][phase_name]['indep'].keys())[0]
                n = len(self.outputs['phases'][phase_name]['indep'][indep_var]['value'])
                output = np.empty(n)
                output[:] = np.nan

            if not var_in_traj:
                raise KeyError('Variable "{0}" not found in trajectory '
                               'simulation results.'.format(var))

            return_vals[phase_name] = output

        if flat:
            time_array = np.concatenate([times[pname] for pname in phases])
            sort_idxs = np.argsort(time_array, axis=0).ravel()
            return_vals = np.concatenate([return_vals[pname] for pname in phases])[sort_idxs, ...]

        return return_vals
