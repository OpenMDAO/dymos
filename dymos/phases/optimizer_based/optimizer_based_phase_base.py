from __future__ import division, print_function, absolute_import

from collections import Iterable

import numpy as np
from dymos.phases.optimizer_based.components import CollocationComp, StateInterpComp
from dymos.phases.components import EndpointConditionsComp, ContinuityComp
from dymos.phases.phase_base import PhaseBase
from dymos.utils.interpolate import LagrangeBarycentricInterpolant
from dymos.utils.misc import CoerceDesvar
from dymos.utils.simulation import ScipyODEIntegrator, SimulationResults, \
    StdOutObserver, ProgressBarObserver
from openmdao.api import IndepVarComp
from six import string_types, iteritems


class OptimizerBasedPhaseBase(PhaseBase):
    """
    OptimizerBasedPhaseBase serves as the base class for GaussLobattoPhase and RadauPSPhase.

    Attributes
    ----------
    self.time_options : dict of TimeOptionsDictionary
        A dictionary of options for time (integration variable) in the phase.

    self.state_options : dict of StateOptionsDictionary
        A dictionary of options for the RHS states in the Phase.

    self.control_options : dict of ControlOptionsDictionary
        A dictionary of options for the controls in the Phase.

    self._ode_controls : dict of ControlOptionsDictionary
        A dictionary of the default options for controllable inputs of the Phase RHS

    """
    def simulate(self, times='all', integrator='vode', integrator_params=None,
                 observer=None, direction='forward', record_file=None, record=True,
                 check_setup=False):
        """
        Integrate the current phase using the current values of time, states, and controls.

        Parameters
        ----------
        times : str or sequence
            The times at which the observing function will be called, and outputs will be saved.
            If given as a string, it must be a valid node subset name.
            If given as a sequence, it directly provides the times at which output is provided,
            *in addition to the segment boundaries*.
        integrator : str
            The integrator to be used by scipy.ode.  This is one of:
            'vode', 'lsoda', 'dopri5', or 'dopri853'.
        integrator_params : dict
            Parameters specific to the chosen integrator.  See the scipy.integrate.ode
            documentation for details.
        observer : callable, str, or None
            A callable function to be called at the specified timesteps in
            `integrate_times`.  This can be used to record the integrated trajectory.
            If 'progress-bar', a ProgressBarObserver will be used, which outputs the simulation
            process to the screen as a ProgressBar.
            If 'stdout', a StdOutObserver will be used, which outputs all variables
            in the model to standard output by default.
            If None, no observer will be called.
        direction : str
            The direction of the integration.  If 'forward' (the default) then the integration
            begins with the initial conditions at the start of the phase and propagates forward
            to the end of the phase.  If 'reverse', the integration begins with the final values
            of time, states, and controls in the phase and propagates backwards to the start of
            the phase.
        record_file : str or None
            A string given the name of the recorded file to which the results of the explicit
            simulation should be saved.  If None, automatically save to '<phase_name>_sim.db'.
        record : bool
            If True (default), save the explicit simulation results to the file specified
            by record_file.

        Returns
        -------
        results : SimulationResults object


        """
        if not self.state_options:
            msg = 'Phase has no states, nothing to simulate. \n' \
                  'Call run_model() on the containing problem instead.'
            raise RuntimeError(msg)

        if self._outputs is None:
            msg = 'Unable to obtain initial state values.\n' \
                  'Call run_model() or run_driver() on the containing problem to populate outputs' \
                  ' before simulating the phase.'
            raise RuntimeError(msg)

        rhs_integrator = ScipyODEIntegrator(ode_class=self.options['ode_class'],
                                            ode_init_kwargs=self.options['ode_init_kwargs'],
                                            time_options=self.time_options,
                                            state_options=self.state_options,
                                            control_options=self.control_options,
                                            design_parameter_options=self.design_parameter_options)

        if observer == 'default':
            observer = StdOutObserver(rhs_integrator)
        elif observer == 'progress-bar':
            observer = ProgressBarObserver(rhs_integrator, t0=self._outputs['time.time'][0],
                                           tf=self._outputs['time.time'][-1])

        gd = self.grid_data

        x0 = {}

        x0_idx = 0 if direction == 'forward' else -1

        for state_name, options in iteritems(self.state_options):
            x0[state_name] = self._outputs['states:{0}'.format(state_name)][x0_idx, ...]

        rhs_integrator.setup(check=check_setup)

        exp_out = SimulationResults(time_options=self.time_options,
                                    state_options=self.state_options,
                                    control_options=self.control_options,
                                    design_parameter_options=self.design_parameter_options)

        seg_sequence = range(gd.num_segments)
        if direction == 'reverse':
            seg_sequence = reversed(seg_sequence)

        for param_name, options in iteritems(self.design_parameter_options):
            if options['opt']:
                val = self._outputs['design_parameters:{0}'.format(param_name)]
            else:
                val = self._outputs['design_parameters:{0}_out'.format(param_name)]
            rhs_integrator.set_design_param_value(param_name, val[0, ...], options['units'])

        first_seg = True
        for seg_i in seg_sequence:
            seg_idxs = gd.segment_indices[seg_i, :]

            seg_times = self._outputs['time.time'][seg_idxs[0]:seg_idxs[1]]

            for control_name, options in iteritems(self.control_options):

                if options['opt']:
                    control_vals = self._outputs['controls:{0}'.format(control_name)]
                else:
                    control_vals = self._outputs['controls:{0}_out'.format(control_name)]

                # if options['dynamic']:
                map_input_idxs_to_all = self.grid_data.input_maps['dynamic_control_input_to_disc']
                interp = LagrangeBarycentricInterpolant(gd.node_stau[seg_idxs[0]:seg_idxs[1]])
                ctrl_vals = control_vals[map_input_idxs_to_all][seg_idxs[0]:seg_idxs[1]].ravel()
                interp.setup(x0=seg_times[0], xf=seg_times[-1], f_j=ctrl_vals)
                rhs_integrator.set_interpolant(control_name, interp)

            if not first_seg:
                for state_name, options in iteritems(self.state_options):
                    x0[state_name] = seg_out.outputs['states:{0}'.format(state_name)]['value'][-1,
                                                                                               ...]

            if not isinstance(times, string_types) and isinstance(times, Iterable):
                idxs_times_in_seg = np.where(np.logical_and(times > seg_times[0],
                                                            times < seg_times[-1]))[0]
                t_out = np.zeros(len(idxs_times_in_seg) + 2, dtype=float)
                t_out[1:-1] = times[idxs_times_in_seg]
                t_out[0] = seg_times[0]
                t_out[-1] = seg_times[-1]
            elif times in ('disc', 'state_disc'):
                t_out = seg_times[::2]
            elif times == 'all':
                t_out = seg_times
            elif times == 'col':
                t_out = seg_times[1::2]
            else:
                raise ValueError('Invalid value for option times. '
                                 'Must be \'disc\', \'all\', \'col\', or Iterable')

            if direction == 'reverse':
                t_out = t_out[::-1]

            seg_out = rhs_integrator.integrate_times(x0, t_out,
                                                     integrator=integrator,
                                                     integrator_params=integrator_params,
                                                     observer=observer)

            if first_seg:
                exp_out.outputs.update(seg_out.outputs)
            else:
                for var in seg_out.outputs:
                    exp_out.outputs[var]['value'] = np.concatenate((exp_out.outputs[var]['value'],
                                                                    seg_out.outputs[var]['value']),
                                                                   axis=0)

            first_seg = False

        # Save
        if record:
            phase_name = self.pathname.split('.')[0]
            filepath = record_file if record_file else '{0}_sim.db'.format(phase_name)

            exp_out.record_results(filepath, self.options['ode_class'],
                                   self.options['ode_init_kwargs'])

        return exp_out

    def setup(self):
        super(OptimizerBasedPhaseBase, self).setup()

        transcription = self.options['transcription']
        grid_data = self.grid_data

        num_opt_controls = len([name for (name, options) in iteritems(self.control_options)
                                if options['opt']])

        num_input_controls = len([name for (name, options) in iteritems(self.control_options)
                                  if not options['opt']])

        num_opt_design_params = len([name for (name, options) in
                                     iteritems(self.design_parameter_options) if options['opt']])

        num_input_design_params = len([name for (name, options) in
                                       iteritems(self.design_parameter_options)
                                       if not options['opt']])

        num_controls = len(self.control_options)

        indep_controls = ['indep_controls'] if num_opt_controls > 0 else []
        input_controls = ['input_controls'] if num_input_controls > 0 else []
        indep_design_params = ['indep_design_params'] if num_opt_design_params > 0 else []
        input_design_params = ['input_design_params'] if num_input_design_params > 0 else []
        control_rate_comp = ['control_rate_comp'] if num_controls > 0 else []
        control_defect_comp = ['control_defect_comp'] if num_opt_controls > 0 else []

        order = self._time_extents + input_controls + indep_controls + \
            indep_design_params + input_design_params + \
            ['indep_states', 'time'] + control_rate_comp + ['indep_jumps', 'endpoint_conditions']

        if transcription == 'gauss-lobatto':
            order = order + ['rhs_disc', 'state_interp', 'rhs_col', 'collocation_constraint']
        elif transcription == 'radau-ps':
            order = order + control_defect_comp + \
                ['state_interp', 'rhs_all', 'collocation_constraint']
        else:
            raise ValueError('Invalid transcription: {0}'.format(transcription))

        if self._requires_continuity_comp():
            order.append('continuity_constraint')
        if getattr(self, 'boundary_constraints', None) is not None:
            order.append('boundary_constraints')
        if getattr(self, 'path_constraints', None) is not None:
            order.append('path_constraints')
        self.set_order(order)

    def _setup_rhs(self):
        grid_data = self.grid_data
        time_units = self.time_options['units']
        map_input_indices_to_disc = self.grid_data.input_maps['state_input_to_disc']
        num_input_nodes = self.grid_data.num_state_input_nodes

        self.add_subsystem('state_interp',
                           subsys=StateInterpComp(grid_data=grid_data,
                                                  state_options=self.state_options,
                                                  time_units=time_units,
                                                  transcription=self.options['transcription']))

        self.connect(
            'time.dt_dstau', 'state_interp.dt_dstau',
            src_indices=grid_data.subset_node_indices['col'])

        for name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])

            src_idxs_mat = np.reshape(np.arange(size * num_input_nodes, dtype=int),
                                      (num_input_nodes, size), order='C')

            src_idxs = src_idxs_mat[map_input_indices_to_disc, :]

            self.connect('states:{0}'.format(name),
                         'state_interp.state_disc:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

    def _setup_states(self):
        """
        Add an IndepVarComp for the states and setup the states as design variables.
        """
        grid_data = self.grid_data
        num_state_input_nodes = grid_data.num_state_input_nodes

        indep = IndepVarComp()
        for name, options in iteritems(self.state_options):
            indep.add_output(name='states:{0}'.format(name),
                             shape=(num_state_input_nodes, np.prod(options['shape'])),
                             units=options['units'])
        self.add_subsystem('indep_states', indep, promotes_outputs=['*'])

        for name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])
            if options['opt']:
                desvar_indices = list(range(size * num_state_input_nodes))

                if options['fix_initial']:
                    if options['initial_bounds'] is not None:
                        raise ValueError('Cannot specify \'fix_initial=True\' and specify '
                                         'initial_bounds for state {0}'.format(name))
                    if isinstance(options['fix_initial'], Iterable):
                        idxs_to_fix = np.where(np.asarray(options['fix_initial']))[0]
                        for idx_to_fix in reversed(sorted(idxs_to_fix)):
                            del desvar_indices[idx_to_fix]
                    else:
                        del desvar_indices[:size]
                if options['fix_final']:
                    if options['final_bounds'] is not None:
                        raise ValueError('Cannot specify \'fix_final=True\' and specify '
                                         'final_bounds for state {0}'.format(name))
                    if isinstance(options['fix_final'], Iterable):
                        idxs_to_fix = np.where(np.asarray(options['fix_final']))[0]
                        for idx_to_fix in reversed(sorted(idxs_to_fix)):
                            del desvar_indices[-size + idx_to_fix]
                    else:
                        del desvar_indices[-size:]

                if len(desvar_indices) > 0:
                    coerce_desvar_option = CoerceDesvar(num_state_input_nodes, desvar_indices,
                                                        options)

                    lb = np.zeros_like(desvar_indices, dtype=float)
                    lb[:] = coerce_desvar_option('lower') \
                        if coerce_desvar_option('lower') is not None else np.finfo(float).min

                    ub = np.zeros_like(desvar_indices, dtype=float)
                    ub[:] = coerce_desvar_option('upper') \
                        if coerce_desvar_option('upper') is not None else np.finfo(float).max

                    if options['initial_bounds'] is not None:
                        lb[0] = options['initial_bounds'][0]
                        ub[0] = options['initial_bounds'][-1]

                    if options['final_bounds'] is not None:
                        lb[-1] = options['final_bounds'][0]
                        ub[-1] = options['final_bounds'][-1]

                    self.add_design_var(name='states:{0}'.format(name),
                                        lower=lb,
                                        upper=ub,
                                        scaler=coerce_desvar_option('scaler'),
                                        adder=coerce_desvar_option('adder'),
                                        ref0=coerce_desvar_option('ref0'),
                                        ref=coerce_desvar_option('ref'),
                                        indices=desvar_indices)

    def _requires_continuity_comp(self):
        """
        Returns True if the phase requires a continuity enforcement component.
        """
        grid_data = self.grid_data
        compressed = self.options['compressed']
        num_seg = grid_data.num_segments

        continuous_states = \
            [name for (name, opts) in iteritems(self.state_options) if opts['continuity']]

        continuous_controls = \
            [name for (name, opts) in iteritems(self.control_options) if opts['continuity']]

        rate_continuous_controls = \
            [name for (name, opts) in iteritems(self.control_options) if opts['rate_continuity']]

        rate2_continuous_controls = \
            [name for (name, opts) in iteritems(self.control_options) if opts['rate2_continuity']]

        state_val_cont_required = not compressed and num_seg > 1 and continuous_states
        control_val_cont_required = not compressed and num_seg > 1 and continuous_controls
        control_rate_cont_required = num_seg > 1 and rate_continuous_controls
        control_rate2_cont_required = num_seg > 1 and rate2_continuous_controls

        return (state_val_cont_required or control_val_cont_required or
                control_rate_cont_required or control_rate2_cont_required)

    def _setup_defects(self):
        """
        Setup the Collocation and Continuity components as necessary.
        """
        grid_data = self.grid_data
        compressed = self.options['compressed']
        num_seg = grid_data.num_segments

        time_units = self.time_options['units']

        self.add_subsystem('collocation_constraint',
                           CollocationComp(grid_data=grid_data,
                                           state_options=self.state_options,
                                           time_units=time_units))

        self.connect(
            'time.dt_dstau', ('collocation_constraint.dt_dstau'),
            src_indices=grid_data.subset_node_indices['col'])

        # Add the continuity constraint component if necessary
        if self._requires_continuity_comp():
            self.add_subsystem('continuity_constraint',
                               ContinuityComp(grid_data=grid_data,
                                              state_options=self.state_options,
                                              control_options=self.control_options,
                                              time_units=time_units))
            self.connect('t_duration', 'continuity_constraint.t_duration')

        if num_seg > 1:

            for name, options in iteritems(self.state_options):
                if not compressed and options['continuity']:
                    # The sub-indices of state_disc indices that are segment ends
                    state_disc_idxs = grid_data.subset_node_indices['state_disc']
                    segment_end_idxs = grid_data.subset_node_indices['segment_ends']
                    disc_subidxs = np.where(np.in1d(state_disc_idxs, segment_end_idxs))[0]
                    self.connect('states:{0}'.format(name),
                                 'continuity_constraint.states:{}'.format(name),
                                 src_indices=disc_subidxs)

            for name, options in iteritems(self.control_options):
                control_src_name = 'controls:{0}'.format(name) if options['opt'] \
                                   else 'controls:{0}_out'.format(name)

                if options['dynamic'] and options['continuity'] and not compressed:
                    self.connect(control_src_name,
                                 'continuity_constraint.controls:{0}'.format(name),
                                 src_indices=grid_data.subset_node_indices['segment_ends'])

                if options['opt'] and options['dynamic']:
                    if options['rate_continuity']:
                        self.connect('control_rates:{0}_rate'.format(name),
                                     'continuity_constraint.control_rates:{}_rate'.format(name),
                                     src_indices=grid_data.subset_node_indices['segment_ends'])

                    if options['rate2_continuity']:
                        self.connect('control_rates:{0}_rate2'.format(name),
                                     'continuity_constraint.control_rates:{}_rate2'.format(name),
                                     src_indices=grid_data.subset_node_indices['segment_ends'])

    def _setup_endpoint_conditions(self):

        jump_comp = self.add_subsystem('indep_jumps', subsys=IndepVarComp(),
                                       promotes_outputs=['*'])

        jump_comp.add_output('initial_jump:time', val=0.0, units=self.time_options['units'],
                             desc='discontinuity in time at the start of the phase')

        jump_comp.add_output('final_jump:time', val=0.0, units=self.time_options['units'],
                             desc='discontinuity in time at the end of the phase')

        endpoint_comp = EndpointConditionsComp(time_options=self.time_options,
                                               state_options=self.state_options,
                                               control_options=self.control_options)

        self.connect('time', 'endpoint_conditions.values:time')

        self.connect('initial_jump:time',
                     'endpoint_conditions.initial_jump:time')

        self.connect('final_jump:time',
                     'endpoint_conditions.final_jump:time')

        promoted_list = ['time--', 'time-+', 'time+-', 'time++']

        for state_name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])
            ar = np.arange(size)

            jump_comp.add_output('initial_jump:{0}'.format(state_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'start of the phase'.format(state_name))

            jump_comp.add_output('final_jump:{0}'.format(state_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'end of the phase'.format(state_name))

            self.connect('states:{0}'.format(state_name),
                         'endpoint_conditions.values:{0}'.format(state_name))

            self.connect('initial_jump:{0}'.format(state_name),
                         'endpoint_conditions.initial_jump:{0}'.format(state_name),
                         src_indices=ar, flat_src_indices=True)

            self.connect('final_jump:{0}'.format(state_name),
                         'endpoint_conditions.final_jump:{0}'.format(state_name),
                         src_indices=ar, flat_src_indices=True)

            promoted_list += ['states:{0}--'.format(state_name),
                              'states:{0}-+'.format(state_name),
                              'states:{0}+-'.format(state_name),
                              'states:{0}++'.format(state_name)]

        for control_name, options in iteritems(self.control_options):
            size = np.prod(options['shape'])
            ar = np.arange(size)

            if options['opt']:
                suffix = ''
            else:
                suffix = '_out'

            jump_comp.add_output('initial_jump:{0}'.format(control_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'start of the phase'.format(control_name))

            jump_comp.add_output('final_jump:{0}'.format(control_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'end of the phase'.format(control_name))

            self.connect('controls:{0}{1}'.format(control_name, suffix),
                         'endpoint_conditions.values:{0}'.format(control_name))

            self.connect('initial_jump:{0}'.format(control_name),
                         'endpoint_conditions.initial_jump:{0}'.format(control_name),
                         src_indices=ar, flat_src_indices=True)

            self.connect('final_jump:{0}'.format(control_name),
                         'endpoint_conditions.final_jump:{0}'.format(control_name),
                         src_indices=ar, flat_src_indices=True)

            promoted_list += ['controls:{0}--'.format(control_name),
                              'controls:{0}-+'.format(control_name),
                              'controls:{0}+-'.format(control_name),
                              'controls:{0}++'.format(control_name)]

        self.add_subsystem(name='endpoint_conditions', subsys=endpoint_comp,
                           promotes_outputs=promoted_list)
