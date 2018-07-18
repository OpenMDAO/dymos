from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from openmdao.api import Problem, Group, ParallelGroup, IndepVarComp, SqliteRecorder, CaseReader
from openmdao.utils.units import valid_units, convert_units

from dymos.utils.misc import get_rate_units
from dymos.phases.options import TimeOptionsDictionary, StateOptionsDictionary, \
    ControlOptionsDictionary


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
    traj : dict of {str : Phase}
        The trajectory object whose results are to be recorded/loaded.
    filepath : str or None
        A filepath from which PhaseSimulationResults are to be loaded.
    """
    def __init__(self, filepath=None, traj=None):

        self.traj = traj

        if isinstance(filepath, str):
            self._load_results(filepath)

    def record_results(self, exp_outs, filename=None):
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
        exp_outs : dict of {str : PhaseSimulationResults}.
            A dictionary of {phase_name : PhaseSimulationResults} for each phase to be recorded
        filename : str, None
            The filename to which the recording should be saved.
            If None, save to '<traj_name>_sim.db'
        """
        # Make sure we have the same keys/phases in both the trajectory and exp_outs
        if set(self.traj._phases) != set(exp_outs):
            raise ValueError('TrajectorySimulationResults trajectory._phases and '
                             'exp_outs must contain the same keys.')

        traj = self.traj

        p = Problem(model=Group())
        traj_group = p.model.add_subsystem('phases', ParallelGroup(), promotes_inputs=['*'],
                                           promotes_outputs=['*'])

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
            ivc.add_output('time', val=np.zeros(nn), units=phase.time_options['units'])
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
                if options['rate_param']:
                    rate_targets = sys_param_options[options['rate_param']]['targets']
                    rate_units = get_rate_units(units, phase.time_options['units'], deriv=1)
                    ivc.add_output('control_rates:{0}_rate'.format(name),
                                   val=np.zeros((nn,) + options['shape']),
                                   units=rate_units)
                    phase_group.connect('control_rates:{0}_rate'.format(name),
                                        ['ode.{0}'.format(t) for t in rate_targets],
                                        src_indices=np.arange(nn, dtype=int))
                if options['rate2_param']:
                    rate2_targets = sys_param_options[options['rate2_param']]['targets']
                    rate2_units = get_rate_units(units, phase.time_options['units'], deriv=2)
                    ivc.add_output('control_rates:{0}_rate2'.format(name),
                                   val=np.zeros((nn,) + options['shape']),
                                   units=rate2_units)
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

        p.setup(check=True)

        filename = filename if filename else '{0}_sim.db'.format(traj.name)
        p.model.add_recorder(SqliteRecorder(filename))
        p.model.recording_options['record_metadata'] = True
        p.model.recording_options['record_outputs'] = True

        # Assign values
        for phase_name, phase in iteritems(traj._phases):
            exp_out = exp_outs[phase_name]

            # Assign times
            p['{0}.time'.format(phase_name)] = exp_out.get_values('time')[:, 0]
    
            # Assign states
            for name in phase.state_options:
                p['{0}.states:{1}'.format(phase_name, name)] = exp_out.get_values(name)
    
            # Assign controls
            for name, options in iteritems(phase.control_options):
                shape = p['{0}.controls:{1}'.format(phase_name, name)].shape
                p['{0}.controls:{1}'.format(phase_name, name)] = \
                    np.reshape(exp_out.get_values(name), shape)
                if options['rate_param']:
                    p['{0}.control_rates:{1}_rate'.format(phase_name, name)] = \
                        np.reshape(exp_out.get_values('{0}_rate'.format(name)), shape)
                if options['rate2_param']:
                    p['{0}.control_rates:{1}_rate2'.format(phase_name, name)] = \
                        np.reshape(exp_out.get_values('{0}_rate2'.format(name)), shape)
    
            # Assign design parameters
            for name, options in iteritems(phase.design_parameter_options):
                shape = p['{0}.design_parameters:{1}'.format(phase_name, name)].shape
                p['{0}.design_parameters:{1}'.format(phase_name, name)] = \
                    np.reshape(exp_out.get_values(name), shape)
    
            # Populate outputs of ODE
            prom2abs_ode_outputs = phase_groups[phase_name].ode._var_allprocs_prom2abs_list['output']
            for prom_name, abs_name in iteritems(prom2abs_ode_outputs):
                if p[abs_name[0]].shape[0] == 1:
                    p[abs_name[0]] = exp_out.get_values(prom_name)[0]
                else:
                    p[abs_name[0]] = np.reshape(exp_out.get_values(prom_name), p[abs_name[0]].shape)

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
        raise NotImplementedError('_load_results has not yet been implemented')
        # cr = CaseReader(filename)
        # case = cr.system_cases.get_case(-1)
        #
        # loaded_outputs = case.outputs._prom2abs['output']
        # for name in loaded_outputs:
        #     self.outputs[name] = {}
        #     self.outputs[name]['value'] = case.outputs[name]
        #     self.outputs[name]['units'] = None
        #     self.outputs[name]['shape'] = case.outputs[name].shape[1:]
        #
        # # TODO: Get time, state, and control options from the case metadata
        # self.time_options = TimeOptionsDictionary()
        # self.state_options = {}
        # self.control_options = {}
        #
        # states = [s.split(':')[-1] for s in loaded_outputs if s.startswith('states:')]
        # controls = [s.split(':')[-1] for s in loaded_outputs if s.startswith('controls:')]
        #
        # for s in states:
        #     self.state_options[s] = StateOptionsDictionary()
        #
        # for c in controls:
        #     self.control_options[c] = ControlOptionsDictionary()

    def get_values(self, var, units=None, nodes=None):

        if units is not None and not valid_units(units):
            raise ValueError('{0} is not a valid set of units.'.format(units))

        if nodes is not None:
            raise RuntimeWarning('Argument nodes has no meaning for'
                                 'TrajectorySimulationResults.get_values and is included for '
                                 'compatibility with Trajectory.get_values')

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
