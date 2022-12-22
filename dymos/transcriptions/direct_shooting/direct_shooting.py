import numpy as np

import openmdao.api as om

from ..pseudospectral.components import PseudospectralTimeseriesOutputComp
from .direct_shooting_continuity_comp import DirectShootingContinuityComp
from ..transcription_base import TranscriptionBase
from ..grid_data import GridData, GaussLobattoGrid, RadauGrid, UniformGrid
from .ode_integration_comp import ODEIntegrationComp
from ...utils.misc import get_rate_units, CoerceDesvar
from ...utils.indexing import get_src_indices_by_row
from ...utils.introspection import get_promoted_vars, get_source_metadata, get_targets
from ...utils.constants import INF_BOUND
from ..common import TimeComp, TimeseriesOutputGroup, ControlGroup


class DirectShooting(TranscriptionBase):
    """
    The Transcription class for single explicit shooting.

    This transcription uses an external explicit integrator to propagate the states, and optionally their
    sensitivities wrt the 'inputs' to the integration.

    If we view integration as a function

    .. math::

    \bar{x}_{f} = \mathcal{I}(\bar{x}_0, t_0, t_d, \bar{\theta}) = \bar{x}_{0} + \int_{t_0}^{t_0+t_d} \left( f_{ode}(\bar{x}, t, \bar{\theta}) \right) dt

    then the inputs are the initial states ($\bar{x}$), the initial time and duration ($t_0$ and $t_d$), and some set
    of parameters that impact the ODE ($\theta$). For Dymos, $\theta$ may include the phase parameters, or the node values
    that govern the shape of the controls and polynomial controls.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of arguments.
    """
    def __init__(self, **kwargs):
        super(DirectShooting, self).__init__(**kwargs)
        self._rhs_source = 'ode'

    def initialize(self):
        """
        Declare transcription options.
        """
        self.options.declare('method', default='DOP853', desc='The integration method used.')
        self.options.declare('atol', types=float, default=1.0E-6)
        self.options.declare('rtol', types=float, default=1.0E-9)
        self.options.declare('first_step', types=float, allow_none=True, default=None)
        self.options.declare('max_step', types=float, default=np.inf)
        self.options.declare('propagate_derivs', types=bool, default=True,
                             desc='If True, propagate the state and derivatives of the state and time with respect to '
                                  'the integration parameters. If False, only propagate the primal states. If only '
                                  'using this transcription to propagate an ODE and derivatives are needed, setting '
                                  'this option to False should result in faster execution.')
        self.options.declare('subprob_reports', default=False,
                             desc='Controls the reports made when running the subproblems for DirectShooting')
        self.options.declare('input_grid', types=(GaussLobattoGrid, RadauGrid),
                             desc='The grid distribution used to layout the control inputs.')
        self.options.declare('output_grid', types=(GaussLobattoGrid, RadauGrid, UniformGrid), allow_none=True,
                             default=None,
                             desc='The grid distribution determining the location of the output nodes. The default '
                                  'value of None will result in the use of the input_grid for outputs. This is useful '
                                  'for the implementation of path constraints but can result in highly nonlinear '
                                  'dynamics being smoothed over in the outputs. When used for validation through '
                                  'simulation, it is generally wise to choose an output grid that is more dense '
                                  'than the input grid to capture this nonlinearity.')

    def init_grid(self):
        """
        Setup the GridData object for the Transcription.
        """
        """
        Setup the GridData object for the Transcription.
        """
        self._input_grid_data = self.options['input_grid']

        self.grid_data = self._input_grid_data

        if self.options['output_grid']:
            self._output_grid_data = self.options['output_grid']
        else:
            self._output_grid_data = self._input_grid_data

    def setup_time(self, phase):
        """
        Setup the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        time_options = phase.time_options
        t_name = time_options['name']
        t_phase_name = f'{t_name}_phase'
        time_units = time_options['units']
        num_seg = self._output_grid_data.num_segments

        # Warn about invalid options
        phase.check_time_options()

        for ts_name, ts_options in phase._timeseries.items():
            if t_name not in ts_options['outputs']:
                phase.add_timeseries_output(t_name, timeseries=ts_name)
            if t_phase_name not in ts_options['outputs']:
                phase.add_timeseries_output(t_phase_name, timeseries=ts_name)

        # if times_per_seg is None:
        # Case 1:  Compute times at 'all' node set.
        num_nodes = self._output_grid_data.num_nodes
        node_ptau = self._output_grid_data.node_ptau
        node_dptau_dstau = self._output_grid_data.node_dptau_dstau
        # else:
        #     # Case 2:  Compute times at n equally distributed points per segment.
        #     num_nodes = num_seg * times_per_seg
        #     node_stau = np.linspace(-1, 1, times_per_seg)
        #     node_ptau = np.empty(0, )
        #     node_dptau_dstau = np.empty(0, )
        #     # Append our nodes in phase tau space
        #     for iseg in range(num_seg):
        #         v0 = self._output_grid_data.segment_ends[iseg]
        #         v1 = self._output_grid_data.segment_ends[iseg + 1]
        #         node_ptau = np.concatenate((node_ptau, v0 + 0.5 * (node_stau + 1) * (v1 - v0)))
        #         node_dptau_dstau = np.concatenate((node_dptau_dstau,
        #                                            0.5 * (v1 - v0) * np.ones_like(node_stau)))

        time_comp = TimeComp(num_nodes=num_nodes, node_ptau=node_ptau,
                             node_dptau_dstau=node_dptau_dstau, units=time_units)

        phase.add_subsystem('time', time_comp, promotes=['*'])

    def configure_time(self, phase):
        """
        Configure the inputs/outputs on the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        time_options = phase.time_options
        t_name = time_options['name']
        tphase_name = f'{t_name}_phase'
        t_units = time_options['units']

        integ = phase._get_subsystem('integrator')
        integ._configure_time()

        time_comp = phase._get_subsystem('time')
        time_comp.configure_io()

        ode = phase._get_subsystem('ode')
        ode_inputs = get_promoted_vars(ode, 'input')

        phase.set_input_defaults('t_initial', val=0.0)
        phase.set_input_defaults('t_duration', val=1.0)

        phase.promotes('integrator', inputs=['t_initial', 't_duration'])

        if not time_options['fix_initial']:
            lb, ub = time_options['initial_bounds']
            lb = -INF_BOUND if lb is None else lb
            ub = INF_BOUND if ub is None else ub

            phase.add_design_var('t_initial',
                                 lower=lb,
                                 upper=ub,
                                 scaler=time_options['initial_scaler'],
                                 adder=time_options['initial_adder'],
                                 ref0=time_options['initial_ref0'],
                                 ref=time_options['initial_ref'])

        if not (time_options['input_duration'] or time_options['fix_duration']):
            lb, ub = time_options['duration_bounds']
            lb = -INF_BOUND if lb is None else lb
            ub = INF_BOUND if ub is None else ub

            phase.add_design_var('t_duration',
                                 lower=lb,
                                 upper=ub,
                                 scaler=time_options['duration_scaler'],
                                 adder=time_options['duration_adder'],
                                 ref0=time_options['duration_ref0'],
                                 ref=time_options['duration_ref'])

        # The tuples here are (name, user_specified_targets, dynamic)
        for name, targets, dynamic in [(t_name, time_options['targets'], True),
                                       (tphase_name, time_options['time_phase_targets'], True)]:
            if targets:
                src_idxs = self._output_grid_data.subset_node_indices['all'] if dynamic else None
                phase.connect(f'integrator.{name}', [f'ode.{t}' for t in targets], src_indices=src_idxs,
                              flat_src_indices=True if dynamic else None)

        for name, targets in [('t_initial', time_options['t_initial_targets']),
                              ('t_duration', time_options['t_duration_targets'])]:
            for t in targets:
                shape = ode_inputs[t]['shape']

                if shape == (1,):
                    src_idxs = None
                    flat_src_idxs = None
                    src_shape = None
                else:
                    src_idxs = np.zeros(self._output_grid_data.subset_num_nodes['all'])
                    flat_src_idxs = True
                    src_shape = (1,)

                phase.promotes('ode', inputs=[(t, name)], src_indices=src_idxs,
                               flat_src_indices=flat_src_idxs, src_shape=src_shape)
            if targets:
                phase.set_input_defaults(name=name,
                                         val=np.ones((1,)),
                                         units=t_units)

    def setup_states(self, phase):
        """
        Setup the states for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_states(self, phase):
        """
        Configure state connections post-introspection.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        integ = phase._get_subsystem('integrator')
        integ._configure_states()

        for name, options in phase.state_options.items():
            phase.promotes('integrator', inputs=[f'states:{name}'])
            for ts_name, ts_options in phase._timeseries.items():
                if f'states:{name}' not in ts_options['outputs']:
                    phase.add_timeseries_output(name, output_name=f'states:{name}',
                                                timeseries=ts_name)

        # Add the appropriate design parameters
        for state_name, options in phase.state_options.items():
            if options['fix_final']:
                raise ValueError('fix_final is not a valid option for states when using the '
                                 'ExplicitShooting transcription.')
            if options['opt'] and not options['fix_initial']:
                phase.add_design_var(name=f'states:{state_name}',
                                     lower=options['lower'],
                                     upper=options['upper'],
                                     scaler=options['scaler'],
                                     adder=options['adder'],
                                     ref0=options['ref0'],
                                     ref=options['ref'])

    def _get_ode(self, phase):
        ode = phase._get_subsystem('ode')
        return ode

    def setup_ode(self, phase):
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        integ = ODEIntegrationComp(ode_class=phase.options['ode_class'],
                                   time_options=phase.time_options,
                                   state_options=phase.state_options,
                                   parameter_options=phase.parameter_options,
                                   control_options=phase.control_options,
                                   polynomial_control_options=phase.polynomial_control_options,
                                   method=self.options['method'],
                                   atol=self.options['atol'],
                                   rtol=self.options['rtol'],
                                   first_step=self.options['first_step'],
                                   max_step=self.options['max_step'],
                                   propagate_derivs=self.options['propagate_derivs'],
                                   input_grid_data=self._input_grid_data,
                                   output_grid_data=self._output_grid_data,
                                   ode_init_kwargs=phase.options['ode_init_kwargs'],
                                   standalone_mode=False,
                                   reports=self.options['subprob_reports'])
        phase.add_subsystem('integrator', integ)
        phase.add_subsystem('ode', phase.options['ode_class'](num_nodes=self._output_grid_data.num_nodes,
                                                              **phase.options['ode_init_kwargs']))

    def configure_ode(self, phase):
        """
        Create connections to the introspected states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        integ = phase._get_subsystem('integrator')
        integ._configure_storage()

        ode = phase._get_subsystem('ode')

        ode_inputs = get_promoted_vars(ode, 'input')

        for name, options in phase.state_options.items():

            targets = get_targets(ode_inputs, name=name, user_targets=options['targets'])
            if targets:
                phase.connect(f'integrator.states_out:{name}',
                              [f'ode.{tgt}' for tgt in targets])

    def setup_controls(self, phase):
        """
        Setup the control group.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase._check_control_options()

        if phase.control_options:
            control_group = ControlGroup(control_options=phase.control_options,
                                         time_units=phase.time_options['units'],
                                         grid_data=self._input_grid_data,
                                         output_grid_data=self._output_grid_data)

            phase.add_subsystem('control_group',
                                subsys=control_group,
                                promotes_inputs=['controls:*', 'dt_dstau'],
                                promotes_outputs=['control_values:*', 'control_rates:*'])

            for name, options in phase.control_options.items():
                for ts_name, ts_options in phase._timeseries.items():
                    if f'controls:{name}' not in ts_options['outputs']:
                        phase.add_timeseries_output(name, output_name=f'controls:{name}',
                                                    timeseries=ts_name)
                    if f'control_rates:{name}_rate' not in ts_options['outputs']:
                        phase.add_timeseries_output(f'{name}_rate', output_name=f'control_rates:{name}_rate',
                                                    timeseries=ts_name)
                    if f'control_rates:{name}_rate2' not in ts_options['outputs']:
                        phase.add_timeseries_output(f'{name}_rate2', output_name=f'control_rates:{name}_rate2',
                                                    timeseries=ts_name)

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        if not phase.control_options:
            return

        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp._configure_controls()
        control_group = phase._get_subsystem('control_group')
        control_group.configure_io()

        ode = phase._get_subsystem('ode')
        ode_inputs = get_promoted_vars(ode, 'input')

        # Add the appropriate design parameters
        ncin = self._input_grid_data.subset_num_nodes['control_input']
        for control_name, options in phase.control_options.items():

            phase.promotes('integrator', inputs=[f'controls:{control_name}'])

            if options['opt']:
                coerce_desvar_option = CoerceDesvar(num_input_nodes=ncin, options=options)

                phase.add_design_var(name=f'controls:{control_name}',
                                     lower=coerce_desvar_option('lower'),
                                     upper=coerce_desvar_option('upper'),
                                     scaler=coerce_desvar_option('scaler'),
                                     adder=coerce_desvar_option('adder'),
                                     ref0=coerce_desvar_option('ref0'),
                                     ref=coerce_desvar_option('ref'),
                                     indices=om.slicer[coerce_desvar_option.desvar_indices, ...])

            # Control targets are detected automatically
            targets = get_targets(ode_inputs, control_name, options['targets'])

            if targets:
                phase.connect(f'control_values:{control_name}',
                              [f'ode.{t}' for t in targets])

            # Rate targets
            rate_targets = get_targets(ode_inputs, control_name, options['rate_targets'], control_rates=1)

            if rate_targets:
                phase.connect(f'control_rates:{control_name}_rate',
                              [f'ode.{t}' for t in rate_targets])

            # Second time derivative targets must be specified explicitly
            rate2_targets = get_targets(ode_inputs, control_name, options['rate2_targets'], control_rates=2)

            if rate2_targets:
                phase.connect(f'control_rates:{control_name}_rate2',
                              [f'ode.{t}' for t in targets])


    # def setup_polynomial_controls(self, phase):
    #     """
    #     Adds the polynomial control group to the model if any polynomial controls are present.
    #
    #     Parameters
    #     ----------
    #     phase : dymos.Phase
    #         The phase object to which this transcription instance applies.
    #     """
    #     phase._check_polynomial_control_options()
    #     if phase.polynomial_control_options:
    #         for name, options in phase.polynomial_control_options.items():
    #             for ts_name, ts_options in phase._timeseries.items():
    #                 if f'polynomial_controls:{name}' not in ts_options['outputs']:
    #                     phase.add_timeseries_output(name, output_name=f'polynomial_controls:{name}',
    #                                                 timeseries=ts_name)
    #                 if f'polynomial_control_rates:{name}_rate' not in ts_options['outputs']:
    #                     phase.add_timeseries_output(f'{name}_rate', output_name=f'polynomial_control_rates:{name}_rate',
    #                                                 timeseries=ts_name)
    #                 if f'polynomial_control_rates:{name}_rate2' not in ts_options['outputs']:
    #                     phase.add_timeseries_output(f'{name}_rate2',
    #                                                 output_name=f'polynomial_control_rates:{name}_rate2',
    #                                                 timeseries=ts_name)

    def configure_polynomial_controls(self, phase):
        """
        Configure the inputs/outputs for the polynomial controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_controls(phase)

        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp._configure_polynomial_controls()

        # Add the appropriate design parameters
        for name, options in phase.polynomial_control_options.items():
            if options['opt']:
                ncin = options['order'] + 1
                coerce_desvar_option = CoerceDesvar(num_input_nodes=ncin, options=options)

                phase.add_design_var(name=f'polynomial_controls:{name}',
                                     lower=coerce_desvar_option('lower'),
                                     upper=coerce_desvar_option('upper'),
                                     scaler=coerce_desvar_option('scaler'),
                                     adder=coerce_desvar_option('adder'),
                                     ref0=coerce_desvar_option('ref0'),
                                     ref=coerce_desvar_option('ref'),
                                     indices=coerce_desvar_option.desvar_indices)

    def configure_parameters(self, phase):
        """
        Configure parameter promotion.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_parameters(phase)

        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp._configure_parameters()
        #
        # for param, options in phase.parameter_options.items():
        #     phase.connect(f'parameter_vals:{param}', f'integrator.parameters:{param}')
        #     phase.connect(f'parameter_vals:{param}', [f'ode.{target}' for target in options['targets']])

    def setup_defects(self, phase):
        """
        Create the continuity_comp to house the defects.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        state_cont, control_cont, rate_cont = self._requires_continuity_constraints(phase)

        if state_cont or control_cont or rate_cont:
            phase.add_subsystem('continuity_comp',
                                DirectShootingContinuityComp(grid_data=self._input_grid_data,
                                                             state_options=phase.state_options,
                                                             control_options=phase.control_options,
                                                             time_units=phase.time_options['units']))

    def configure_defects(self, phase):
        """
        Not used in ExplicitShooting.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        igd = self._input_grid_data
        ogd = self._output_grid_data
        any_state_cnty, any_control_cnty, any_rate_cnty = self._requires_continuity_constraints(phase)
        src_idxs = om.slicer[ogd.subset_node_indices['segment_ends'], ...]

        controls_to_enforce = set()
        control_rates_to_enforce = set()
        control_rates2_to_enforce = set()

        # if any((any_state_cnty, any_control_cnty, any_rate_cnty)):
        #     phase.continuity_comp.configure_io()

        for control_name, options in phase.control_options.items():

            if options['continuity'] and any_control_cnty:
                controls_to_enforce.add(control_name)
                phase.connect(f'timeseries.controls:{control_name}',
                              f'continuity_comp.controls:{control_name}',
                              src_indices=src_idxs)
            if options['rate_continuity'] and any_rate_cnty:
                control_rates_to_enforce.add(control_name)
                phase.connect(f'timeseries.control_rates:{control_name}_rate',
                              f'continuity_comp.control_rates:{control_name}_rate',
                              src_indices=src_idxs)
            if options['rate2_continuity'] and any_rate_cnty:
                control_rates2_to_enforce.add(control_name)
                phase.connect(f'timeseries.control_rates:{control_name}_rate2',
                              f'continuity_comp.control_rates:{control_name}_rate2',
                              src_indices=src_idxs)

        phase.continuity_comp.configure_io(controls_to_enforce=controls_to_enforce,
                                           control_rates_to_enforce=control_rates_to_enforce,
                                           control_rates2_to_enforce=control_rates2_to_enforce)

        if any_rate_cnty:
            phase.promotes('continuity_comp', inputs=['t_duration'])

    def setup_timeseries_outputs(self, phase):
        """
        Setup the timeseries for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        ogd = self._output_grid_data

        for name, options in phase._timeseries.items():
            has_expr = False
            for _, output_options in options['outputs'].items():
                if output_options['is_expr']:
                    has_expr = True
                    break

            if options['transcription'] is None:
                ogd = None
            else:
                ogd = options['transcription'].grid_data

            timeseries_comp = PseudospectralTimeseriesOutputComp(input_grid_data=self._input_grid_data,
                                                                 output_grid_data=self._output_grid_data,
                                                                 output_subset=options['subset'],
                                                                 time_units=phase.time_options['units'])
            timeseries_group = TimeseriesOutputGroup(has_expr=has_expr, timeseries_output_comp=timeseries_comp)
            phase.add_subsystem(name, subsys=timeseries_group)

            phase.connect('dt_dstau', f'{name}.dt_dstau', flat_src_indices=True)

    def configure_timeseries_outputs(self, phase):
        """
        Create connections from time series to all post-introspection sources.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_timeseries_outputs(phase)

    def setup_solvers(self, phase):
        """
        Setup the solvers for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_solvers(self, phase):
        """
        Setup the solvers for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def get_parameter_connections(self, name, phase):
        """
        Returns info about a parameter's target connections in the phase.

        Parameters
        ----------
        name : str
            Parameter name.
        phase : dymos.Phase
            The phase object to which this transcription instance applies.

        Returns
        -------
        list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design variable is to be connected.
        """
        connection_info = []

        if name in phase.parameter_options:
            options = phase.parameter_options[name]
            if not options['static_target']:
                src_idxs_raw = np.zeros(self._output_grid_data.subset_num_nodes['all'], dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
                if options['shape'] == (1,):
                    src_idxs = src_idxs.ravel()
            else:
                src_idxs_raw = np.zeros(1, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
                src_idxs = np.squeeze(src_idxs, axis=0)

            connection_info.append(([f'integrator.parameters:{name}'], None))
            connection_info.append(([f'ode.{tgt}' for tgt in options['targets']], (src_idxs,)))

        return connection_info

    def _get_objective_src(self, var, loc, phase, ode_outputs=None):
        """
        Return the path to the variable that will be used as the objective.

        Parameters
        ----------
        var : str
            Name of the variable to be used as the objective.
        loc : str
            The location of the objective in the phase ['initial', 'final'].
        phase : dymos.Phase
            Phase object containing in which the objective resides.
        ode_outputs : dict or None
            A dictionary of ODE outputs as returned by get_promoted_vars.

        Returns
        -------
        obj_path : str
            Path to the source.
        shape : tuple
            Source shape.
        units : str
            Source units.
        linear : bool
            True if the objective quantity1 is linear.
        """
        time_units = phase.time_options['units']
        var_type = phase.classify_var(var)

        if var_type == 't':
            shape = (1,)
            units = time_units
            linear = True
            if loc == 'initial':
                obj_path = 't_initial'
            else:
                obj_path = 'integrator.t_final'
        elif var_type == 't_phase':
            shape = (1,)
            units = time_units
            linear = True
            obj_path = 'integrator.t_phase'
        elif var_type == 'state':
            shape = phase.state_options[var]['shape']
            units = phase.state_options[var]['units']
            linear = loc == 'initial'
            obj_path = f'integrator.states_out:{var}'
        elif var_type == 'indep_control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = True
            obj_path = f'control_values:{var}'
        elif var_type == 'input_control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = False
            obj_path = f'control_values:{var}'
        elif var_type == 'indep_polynomial_control':
            shape = phase.polynomial_control_options[var]['shape']
            units = phase.polynomial_control_options[var]['units']
            linear = True
            obj_path = f'polynomial_control_values:{var}'
        elif var_type == 'input_polynomial_control':
            shape = phase.polynomial_control_options[var]['shape']
            units = phase.polynomial_control_options[var]['units']
            linear = False
            obj_path = f'polynomial_control_values:{var}'
        elif var_type == 'parameter':
            shape = phase.parameter_options[var]['shape']
            units = phase.parameter_options[var]['units']
            linear = True
            obj_path = f'parameter_vals:{var}'
        elif var_type in ('control_rate', 'control_rate2'):
            control_var = var[:-5] if var_type == 'control_rate' else var[:-6]
            shape = phase.control_options[control_var]['shape']
            control_units = phase.control_options[control_var]['units']
            d = 2 if var_type == 'control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            units = control_rate_units
            linear = False
            obj_path = f'control_rates:{var}'
        elif var_type in ('polynomial_control_rate', 'polynomial_control_rate2'):
            control_var = var[:-5]
            shape = phase.polynomial_control_options[control_var]['shape']
            control_units = phase.polynomial_control_options[control_var]['units']
            d = 2 if var_type == 'polynomial_control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            units = control_rate_units
            linear = False
            obj_path = f'polynomial_control_rates:{var}'
        else:
            # Failed to find variable, assume it is in the ODE. This requires introspection.
            raise NotImplementedError('cannot yet constrain/optimize an ODE output using explicit shooting')
            obj_path = f'{self._rhs_source}.{var}'
            if ode_outputs is None:
                ode = self._get_ode(phase)
            else:
                ode = ode_outputs
            meta = get_source_metadata(ode, var, user_units=None, user_shape=None)
            shape = meta['shape']
            units = meta['units']
            linear = False

        return obj_path, shape, units, linear

    def _requires_continuity_constraints(self, phase):
        """
        Tests whether state and/or control and/or control rate continuity are required.

        Parameters
        ----------
        phase : dymos.Phase
            The phase to which this transcription applies.

        Returns
        -------
        state_continuity : bool
            True if any state continuity is required to be enforced.
        control_continuity : bool
            True if any control value continuity is required to be enforced.
        control_rate_continuity : bool
            True if any control rate continuity is required to be enforced.
        """
        num_seg = self._input_grid_data.num_segments
        compressed = self._input_grid_data.compressed
        transcription = self._input_grid_data.transcription

        state_continuity = False
        any_control_continuity = any([opts['continuity'] for opts in phase.control_options.values()])
        any_control_continuity = any_control_continuity and num_seg > 1 and not (compressed or transcription == 'radau-ps')
        any_rate_continuity = any([opts['rate_continuity'] or opts['rate2_continuity']
                                   for opts in phase.control_options.values()])
        any_rate_continuity = any_rate_continuity and num_seg > 1

        return state_continuity, any_control_continuity, any_rate_continuity

    def _get_num_timeseries_nodes(self):
        """
        Returns the number of nodes in the default timeseries for this transcription.

        Returns
        -------
        int
            The number of nodes in the default timeseries for this transcription.
        """
        return self._output_grid_data.subset_num_nodes['all']

    def _get_timeseries_var_source(self, var, output_name, phase):
        """
        Return the source path and indices for a given variable to be connected to a timeseries.

        Parameters
        ----------
        var : str
            Name of the timeseries variable whose source is desired.
        output_name : str
            Name of the timeseries output whose source is desired.
        phase : dymos.Phase
            Phase object containing the variable, either as state, time, control, etc., or as an ODE output.

        Returns
        -------
        meta : dict
            Metadata pertaining to the variable at the given path. This dict contains 'src' (the path to the
            timeseries source), 'src_idxs' (an array of the
            source indices), 'units' (the units of the source variable), and 'shape' (the shape of the variable at
            a given node).
        """
        var_type = phase.classify_var(var)
        time_units = phase.time_options['units']
        time_name = phase.time_options['name']

        transcription = phase.options['transcription']
        ode = transcription._get_ode(phase)
        ode_outputs = get_promoted_vars(ode, 'output')

        # The default for node_idxs, applies to everything except states and parameters.
        node_idxs = None
        meta = {}

        # Determine the path to the variable
        if var_type == 't':
            path = f'integrator.{time_name}'
            src_units = time_units
            src_shape = (1,)
        elif var_type == 't_phase':
            path = f'integrator.{time_name}_phase'
            src_units = time_units
            src_shape = (1,)
        elif var_type == 'state':
            path = f'integrator.states_out:{var}'
            src_units = phase.state_options[var]['units']
            src_shape = phase.state_options[var]['shape']
        elif var_type in ['indep_control', 'input_control']:
            path = f'control_values:{var}'
            src_units = phase.control_options[var]['units']
            src_shape = phase.control_options[var]['shape']
        elif var_type == 'control_rate':
            control_name = var[:-5]
            path = f'control_rates:{control_name}_rate'
            control_name = var[:-5]
            src_units = get_rate_units(phase.control_options[control_name]['units'], time_units, deriv=1)
            src_shape = phase.control_options[control_name]['shape']
        elif var_type == 'control_rate2':
            control_name = var[:-6]
            path = f'control_rates:{control_name}_rate2'
            src_units = get_rate_units(phase.control_options[control_name]['units'], time_units, deriv=2)
            src_shape = phase.control_options[control_name]['shape']
        elif var_type in ['indep_polynomial_control', 'input_polynomial_control']:
            path = f'control_group.polynomial_control_values:{var}'
            src_units = phase.polynomial_control_options[var]['units']
            src_shape = phase.polynomial_control_options[var]['shape']
        elif var_type == 'polynomial_control_rate':
            control_name = var[:-5]
            path = f'control_group.polynomial_control_rates:{control_name}_rate'
            control = phase.polynomial_control_options[control_name]
            src_units = get_rate_units(control['units'], time_units, deriv=1)
            src_shape = control['shape']
        elif var_type == 'polynomial_control_rate2':
            control_name = var[:-6]
            path = f'control_group.polynomial_control_rates:{control_name}_rate2'
            control = phase.polynomial_control_options[control_name]
            src_units = get_rate_units(control['units'], time_units, deriv=2)
            src_shape = control['shape']
        elif var_type == 'parameter':
            path = f'parameter_vals:{var}'
            num_seg = self._input_grid_data.num_segments
            node_idxs = np.zeros(self._output_grid_data.subset_num_nodes['all'], dtype=int)
            src_units = phase.parameter_options[var]['units']
            src_shape = phase.parameter_options[var]['shape']
        else:
            # Failed to find variable, assume it is in the ODE
            meta = get_source_metadata(ode_outputs, src=var)
            src_shape = meta['shape']
            src_units = meta['units']
            src_tags = meta['tags']
            if src_tags:
                for tag in src_tags:
                    if 'dymos.state_rate_source' in tag:
                        path = f"integrator.timeseries:state_rates:{tag.split(':')[-1]}"
                        break
                    else:
                        path = f'integrator.timeseries:{var}'
            else:
                path = f'integrator.timeseries:{var}'
            if 'dymos.static_output' in src_tags:
                raise RuntimeError(f'ODE output {var} is tagged with "dymos.static_output" and cannot be a '
                                   f'timeseries output.')

        src_idxs = None if node_idxs is None else om.slicer[node_idxs, ...]

        meta['src'] = path
        meta['src_idxs'] = src_idxs
        meta['units'] = src_units
        meta['shape'] = src_shape

        return meta
