
from __future__ import print_function, division, absolute_import

from collections import Sequence

import numpy as np
from six import iteritems

from openmdao.api import Group, ParallelGroup, IndepVarComp, DirectSolver


class SimulationTrajectory(Group):
    """
    SimulationPhase is a Group that resembles a Trajectory in structure but is intended for
    use with scipy.solve_ivp to verify the accuracy of the implicit solutions of Dymos.

    This Trajectory is not currently a fully-fledged Trajectory object.  It does not support
    constraints or objectives (or anything used by run_driver in general).  It does not accurately
    compute derivatives across the model and should only be used via run_model to verify the
    accuracy of solutions achieved via the other Phase classes.
    """
    def __init__(self, phases, **kwargs):
        super(SimulationTrajectory, self).__init__(**kwargs)

        # Phases cannot be pickled, so we won't declare them as options.
        self._phases = phases

        self.design_parameter_options = {}
        self.input_parameter_options = {}

    def initialize(self):
        self.options.declare('times', types=(Sequence, np.ndarray, int, str),
                             desc='number of times to include in timeseries output or values of'
                                  'time for timeseries output')
        self.options.declare('time_units', types=(Sequence, np.ndarray, int, str),
                             desc='units of specified times, if numeric.')

    # def _setup_design_parameters(self, ivc):
    #     gd = self.options['grid_data']
    #     num_seg = gd.num_segments
    #     num_points = sum([len(a) for a in list(self.t_eval_per_seg.values())])
    #
    #     for name, options in iteritems(self.design_parameter_options):
    #         ivc.add_output('design_parameters:{0}'.format(name),
    #                        val=np.ones((1,) + options['shape']),
    #                        units=options['units'])
    #
    #         for i in range(num_seg):
    #             self.connect(src_name='design_parameters:{0}'.format(name),
    #                          tgt_name='segment_{0}.design_parameters:{1}'.format(i, name))
    #
    #         if options['targets']:
    #             self.connect(src_name='design_parameters:{0}'.format(name),
    #                          tgt_name=['ode.{0}'.format(tgt) for tgt in options['targets']],
    #                          src_indices=np.zeros(num_points, dtype=int))
    #
    # def _setup_input_parameters(self, ivc):
    #     gd = self.options['grid_data']
    #     num_seg = gd.num_segments
    #     num_points = sum([len(a) for a in list(self.t_eval_per_seg.values())])
    #
    #     for name, options in iteritems(self.input_parameter_options):
    #         ivc.add_output('input_parameters:{0}'.format(name),
    #                        val=np.ones((1,) + options['shape']),
    #                        units=options['units'])
    #
    #         for i in range(num_seg):
    #             self.connect(src_name='input_parameters:{0}'.format(name),
    #                          tgt_name='segment_{0}.input_parameters:{1}'.format(i, name))
    #
    #         if options['targets']:
    #             self.connect(src_name='input_parameters:{0}'.format(name),
    #                          tgt_name=['ode.{0}'.format(tgt) for tgt in options['targets']],
    #                          src_indices=np.zeros(num_points, dtype=int))

    def _setup_input_parameters(self, ivc):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        for name, options in iteritems(self.input_parameter_options):
            ivc.add_output(name='input_parameters:{0}'.format(name),
                           val=options['val'],
                           shape=(1, np.prod(options['shape'])),
                           units=options['units'])

            # Connect the input parameter to its target in each phase
            src_name = 'input_parameters:{0}'.format(name)

            target_params = options['target_params']
            for phase_name, phs in iteritems(self._sim_phases):
                tgt_param_name = target_params.get(phase_name, None) \
                    if isinstance(target_params, dict) else name
                if tgt_param_name:
                    # phs.add_input_parameter(tgt_param_name, val=options['val'],
                    #                         units=options['units'], alias=tgt_param_name)
                    if tgt_param_name not in phs.traj_parameter_options:
                        phs._add_traj_parameter(tgt_param_name, val=options['val'],
                                                units=options['units'])
                    tgt = '{0}.traj_parameters:{1}'.format(phase_name, tgt_param_name)
                    self.connect(src_name=src_name, tgt_name=tgt)

    def _setup_design_parameters(self, ivc):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        for name, options in iteritems(self.design_parameter_options):
            ivc.add_output(name='design_parameters:{0}'.format(name),
                           val=options['val'],
                           shape=(1, np.prod(options['shape'])),
                           units=options['units'])

            # Connect the design parameter to its target in each phase
            src_name = 'design_parameters:{0}'.format(name)

            target_params = options['target_params']
            for phase_name, phs in iteritems(self._sim_phases):
                if isinstance(target_params, dict):
                    tgt_param_name = target_params.get(phase_name, None)
                else:
                    tgt_param_name = name

                if tgt_param_name:
                    if tgt_param_name not in phs.traj_parameter_options:
                        phs._add_traj_parameter(tgt_param_name, val=options['val'],
                                                units=options['units'])
                    tgt = '{0}.traj_parameters:{1}'.format(phase_name, tgt_param_name)
                    self.connect(src_name=src_name, tgt_name=tgt)

    def _setup_phases(self, times_dict):
        phases_group = self.add_subsystem('phases',
                                          subsys=ParallelGroup(),
                                          promotes_inputs=['*'],
                                          promotes_outputs=['*'])

        for name, phs in iteritems(self._phases):
            self._sim_phases[name] = phs._init_simulation_phase(times_dict[name])
            # DirectSolvers were moved down into the phases for use with MPI
            self._sim_phases[name].linear_solver = DirectSolver()
            phases_group.add_subsystem(name, self._sim_phases[name])

    def setup(self):

        self._sim_phases = {}

        ivc = self.add_subsystem(name='ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        # Get a dictionary that maps times to each phase name
        times = self.options['times']
        phases = self._phases
        if isinstance(times, dict):
            times_dict = times
        else:
            if isinstance(times, str):
                times_dict = dict([(phase_name, times) for phase_name in phases])
            elif isinstance(times, int):
                times_dict = dict([(phase_name, times) for phase_name in phases])
            else:
                times_dict = {}
                for name, phs in iteritems(phases):
                    # Find the times that are within the given phase
                    times_dict[name] = times

                    op = phs.list_outputs(units=True, out_stream=None)
                    op_dict = dict([(name, options) for (name, options) in op])
                    phase_times = op_dict['{0}.time.time'.format(phs.pathname)]['value']

                    t_initial = phase_times[0]
                    t_final = phase_times[-1]

                    times_dict[name] = times[np.where(times >= t_initial and times <= t_final)[0]]

        self._setup_phases(times_dict)

        self._setup_design_parameters(ivc)

        self._setup_input_parameters(ivc)
