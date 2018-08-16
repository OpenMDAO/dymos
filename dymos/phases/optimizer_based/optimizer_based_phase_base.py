from __future__ import division, print_function, absolute_import

from collections import Iterable

import numpy as np
from dymos.phases.optimizer_based.components import CollocationComp, StateInterpComp
from dymos.phases.components import EndpointConditionsComp
from dymos.phases.phase_base import PhaseBase
from dymos.utils.interpolate import LagrangeBarycentricInterpolant
from dymos.utils.constants import INF_BOUND
from dymos.utils.misc import CoerceDesvar
from dymos.utils.simulation import ScipyODEIntegrator, PhaseSimulationResults, \
    StdOutObserver, ProgressBarObserver, simulate_phase
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
                 observer=None, record_file=None, record=True):
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
        record_file : str or None
            A string given the name of the recorded file to which the results of the explicit
            simulation should be saved.  If None, automatically save to '<phase_name>_sim.db'.
        record : bool
            If True (default), save the explicit simulation results to the file specified
            by record_file.

        Returns
        -------
        results : PhaseSimulationResults object
        """

        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']
        time_values = self.get_values('time').ravel()
        state_values = {}
        control_values = {}
        design_parameter_values = {}
        for state_name, options in iteritems(self.state_options):
            state_values[state_name] = self.get_values(state_name, nodes='all')
        for control_name, options in iteritems(self.control_options):
            control_values[control_name] = self.get_values(control_name, nodes='all')
        for dp_name, options in iteritems(self.design_parameter_options):
            design_parameter_values[dp_name] = self.get_values(dp_name, nodes='all')

        exp_out = simulate_phase(self.name,
                                 ode_class=ode_class,
                                 time_options=self.time_options,
                                 state_options=self.state_options,
                                 control_options=self.control_options,
                                 design_parameter_options=self.design_parameter_options,
                                 time_values=time_values,
                                 state_values=state_values,
                                 control_values=control_values,
                                 design_parameter_values=design_parameter_values,
                                 ode_init_kwargs=ode_init_kwargs,
                                 grid_data=self.grid_data,
                                 times=times,
                                 record=record,
                                 record_file=record_file,
                                 observer=observer,
                                 integrator=integrator,
                                 integrator_params=integrator_params)

        return exp_out

    def setup(self):
        super(OptimizerBasedPhaseBase, self).setup()

        transcription = self.options['transcription']

        num_opt_controls = len([name for (name, options) in iteritems(self.control_options)
                                if options['opt']])

        num_design_params = len(self.design_parameter_options)

        num_opt_design_params = len([name for (name, options) in
                                     iteritems(self.design_parameter_options) if options['opt']])

        num_input_design_params = len([name for (name, options) in
                                       iteritems(self.design_parameter_options)
                                       if options['input_value']])

        num_controls = len(self.control_options)

        indep_controls = ['indep_controls'] \
            if num_opt_controls > 0 else []
        indep_design_params = ['indep_design_params'] \
            if num_input_design_params < num_design_params else []
        input_design_params = ['input_design_params'] \
            if num_input_design_params > 0 else []
        control_interp_comp = ['control_interp_comp'] \
            if num_controls > 0 else []

        order = self._time_extents + indep_controls + \
            indep_design_params + input_design_params + \
            ['indep_states', 'time'] + control_interp_comp + ['indep_jumps', 'endpoint_conditions']

        if transcription == 'gauss-lobatto':
            order = order + ['rhs_disc', 'state_interp', 'rhs_col', 'collocation_constraint']
        elif transcription == 'radau-ps':
            order = order + ['state_interp', 'rhs_all', 'collocation_constraint']
        else:
            raise ValueError('Invalid transcription: {0}'.format(transcription))

        if self.grid_data.num_segments > 1:
            order.append('continuity_comp')
        if getattr(self, 'boundary_constraints', None) is not None:
            order.append('boundary_constraints')
        if getattr(self, 'path_constraints', None) is not None:
            order.append('path_constraints')
        self.set_order(order)

    def _setup_rhs(self):
        grid_data = self.grid_data
        time_units = self.time_options['units']
        map_input_indices_to_disc = self.grid_data.input_maps['state_input_to_disc']
        num_input_nodes = self.grid_data.subset_num_nodes['state_input']

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
        num_state_input_nodes = grid_data.subset_num_nodes['state_input']

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
                    lb[:] = -INF_BOUND if coerce_desvar_option('lower') is None else \
                        coerce_desvar_option('lower')

                    ub = np.zeros_like(desvar_indices, dtype=float)
                    ub[:] = INF_BOUND if coerce_desvar_option('upper') is None else \
                        coerce_desvar_option('upper')

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

    def _setup_defects(self):
        """
        Setup the Collocation and Continuity components as necessary.
        """
        grid_data = self.grid_data
        num_seg = grid_data.num_segments

        time_units = self.time_options['units']

        self.add_subsystem('collocation_constraint',
                           CollocationComp(grid_data=grid_data,
                                           state_options=self.state_options,
                                           time_units=time_units))

        self.connect('time.dt_dstau', ('collocation_constraint.dt_dstau'),
                     src_indices=grid_data.subset_node_indices['col'])

        # Add the continuity constraint component if necessary
        if num_seg > 1:

            for name, options in iteritems(self.state_options):
                # The sub-indices of state_disc indices that are segment ends
                state_disc_idxs = grid_data.subset_node_indices['state_disc']
                segment_end_idxs = grid_data.subset_node_indices['segment_ends']
                disc_subidxs = np.where(np.in1d(state_disc_idxs, segment_end_idxs))[0]
                self.connect('states:{0}'.format(name),
                             'continuity_comp.states:{}'.format(name),
                             src_indices=disc_subidxs)

            for name, options in iteritems(self.control_options):
                control_src_name = 'control_interp_comp.control_values:{0}'.format(name)
                self.connect(control_src_name,
                             'continuity_comp.controls:{0}'.format(name),
                             src_indices=grid_data.subset_node_indices['segment_ends'])

                self.connect('control_rates:{0}_rate'.format(name),
                             'continuity_comp.control_rates:{}_rate'.format(name),
                             src_indices=grid_data.subset_node_indices['segment_ends'])

                self.connect('control_rates:{0}_rate2'.format(name),
                             'continuity_comp.control_rates:{}_rate2'.format(name),
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

            self.connect('control_interp_comp.control_values:{0}'.format(control_name),
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
