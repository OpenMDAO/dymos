from collections.abc import Sequence

import numpy as np

import openmdao.api as om
from openmdao.utils.om_warnings import warn_deprecation

from .explicit_shooting_continuity_comp import ExplicitShootingContinuityComp
from ..transcription_base import TranscriptionBase
from ..grid_data import BirkhoffGrid, GaussLobattoGrid, RadauGrid, UniformGrid, ChebyshevGaussLobattoGrid
from .ode_integration_comp import ODEIntegrationComp
from ...utils.misc import get_rate_units, CoerceDesvar, reshape_val
from ...utils.indexing import get_src_indices_by_row
from ...utils.introspection import get_promoted_vars, get_source_metadata, get_targets, _get_targets_metadata
from ...utils.constants import INF_BOUND

from ...utils.ode_utils import _make_ode_system
from ..common import TimeComp, TimeseriesOutputComp, ControlInterpComp, ParameterComp


class ExplicitShooting(TranscriptionBase):
    r"""
    The Transcription class for single explicit shooting.

    This transcription uses an external explicit integrator to propagate the states, and optionally their
    sensitivities wrt the 'inputs' to the integration.

    If we view integration as a function

    .. math::
        \bar{x}_{f} = \mathcal{I}(\bar{x}_0, t_0, t_d, \bar{\theta}) \\
        \bar{x}_{f} = \bar{x}_{0} + \int_{t_0}^{t_0+t_d} \left( f_{ode}(\bar{x}, t, \bar{\theta}) \right) dt

    then the inputs are the initial states ($\bar{x}$), the initial time and duration ($t_0$ and $t_d$), and some set
    of parameters that impact the ODE ($\theta$). For Dymos, $\theta$ may include the phase parameters, or the node values
    that govern the shape of the controls.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of arguments.
    """  # nopep8: E501, W605
    def __init__(self, **kwargs):
        super(ExplicitShooting, self).__init__(**kwargs)
        self._rhs_source = 'ode'

    def initialize(self):
        """
        Declare transcription options.
        """
        self.options.declare('method', types=str, default='DOP853',
                             desc='The integration method used.')
        self.options.declare('atol', types=float, default=1.0E-6)
        self.options.declare('rtol', types=float, default=1.0E-9)
        self.options.declare('first_step', types=float, allow_none=True, default=None)
        self.options.declare('max_step', types=float, default=np.inf)
        self.options.declare('propagate_derivs', types=bool, default=True,
                             desc='If True, propagate the state and derivatives of the state and time with respect to '
                                  'the integration parameters. If False, only propagate the primal states. If only '
                                  'using this transcription to propagate an ODE and derivatives are nof needed, '
                                  'setting this option to False should result in faster execution.')
        self.options.declare('subprob_reports', default=False,
                             desc='Controls the reports made when running the subproblems for ExplicitShooting')
        self.options.declare('grid', types=(GaussLobattoGrid, ChebyshevGaussLobattoGrid,
                                            RadauGrid, BirkhoffGrid, str), allow_none=True, default=None,
                             desc='The grid distribution used to layout the control inputs and provide the default '
                                  'output nodes.')
        self.options.declare('output_grid', types=(GaussLobattoGrid, ChebyshevGaussLobattoGrid, RadauGrid,
                                                   UniformGrid, BirkhoffGrid), allow_none=True,
                             default=None,
                             desc='The grid distribution determining the location of the output nodes. The default '
                                  'value of None will result in the use of the grid for outputs. This is useful '
                                  'for the implementation of path constraints but can result in highly nonlinear '
                                  'dynamics being smoothed over in the outputs. When used for validation through '
                                  'simulation, it is generally wise to choose an output grid that is more dense '
                                  'than the input grid to capture this nonlinearity.')
        self.options.declare('num_steps_per_segment', types=int, allow_none=True,
                             default=None, desc='Number of integration steps in each segment',
                             deprecation='Option `num_steps_per_segment is deprecated. ExplicitShooting now uses '
                                         'adaptive-step methods.')
        self.options.declare('control_interp', values=['vandermonde', 'barycentric', 'cubic'], default='vandermonde',
                             desc='Control interpolation algorithm, one of either "vandermonde", "barycentric",'
                             ' or "cubic". In general, Vandermonde is faster but Barycentric is necessary for the'
                             ' Birkhoff transcription where the number of nodes per segment can exceed 20 to 30.'
                             ' Cubic uses the scipy interpolate cubic spline method and is typically the fastest'
                             ' of the three')

        # Deprecated options previously inherited from transcription base.
        self.options.declare('num_segments', types=int, desc='Number of segments')
        self.options.declare('segment_ends', default=None, types=(Sequence, np.ndarray),
                             allow_none=True, desc='Locations of segment ends or None for equally '
                             'spaced segments')
        self.options.declare('order', default=3, types=(int, Sequence, np.ndarray),
                             desc='Order of the state transcription. The order of the control '
                                  'transcription is `order - 1`.')
        self.options.declare('compressed', default=True, types=bool,
                             desc='Use compressed transcription, meaning state and control values'
                                  'at segment boundaries are not duplicated on input.  This '
                                  'implicitly enforces value continuity between segments but in '
                                  'some cases may make the problem more difficult to solve.')

    def init_grid(self):
        """
        Setup the GridData object for the Transcription.
        """
        if self.options['grid'] in ('gauss-lobatto', None):
            self.options['grid'] = GaussLobattoGrid(num_segments=self.options['num_segments'],
                                                    nodes_per_seg=self.options['order'],
                                                    segment_ends=self.options['segment_ends'],
                                                    compressed=self.options['compressed'])
        elif self.options['grid'] == 'radau-ps':
            self.options['grid'] = RadauGrid(num_segments=self.options['num_segments'],
                                             nodes_per_seg=self.options['order'] + 1,
                                             segment_ends=self.options['segment_ends'],
                                             compressed=self.options['compressed'])

        dep_methods = {'rk4', '3/8', 'euler', 'ralston', 'rkf', 'rkck', 'dopri'}
        if self.options['method'] in dep_methods:
            warn_deprecation(f'Integration method {self.options["method"]} is no longer a valid option. Please use one '
                             f'of \'DOP853\', \'RK45\', \'RK23\' instead. Falling back to the default \'DOP853\'.')
            self.options['method'] = 'DOP853'
        # End deprecation-handling

        self.grid_data = self.options['grid']

        if self.options['output_grid']:
            self._output_grid_data = self.options['output_grid']
        else:
            self._output_grid_data = self.grid_data

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

        # Warn about invalid options
        phase.check_time_options()

        for ts_name, ts_options in phase._timeseries.items():
            if t_name not in ts_options['outputs']:
                phase.add_timeseries_output(t_name, timeseries=ts_name)
            if t_phase_name not in ts_options['outputs'] and phase.timeseries_options['include_t_phase']:
                phase.add_timeseries_output(t_phase_name, timeseries=ts_name)

        # Case 1:  Compute times at 'all' node set.
        num_nodes = self._output_grid_data.num_nodes
        node_ptau = self._output_grid_data.node_ptau
        node_dptau_dstau = self._output_grid_data.node_dptau_dstau

        time_comp = TimeComp(num_nodes=num_nodes, node_ptau=node_ptau,
                             node_dptau_dstau=node_dptau_dstau, units=time_units)

        phase.add_subsystem('param_comp', subsys=ParameterComp(time_options=time_options),
                            promotes_inputs=['*'], promotes_outputs=['*'])

        phase.add_subsystem('time', time_comp, promotes_outputs=['*'])

        phase.connect('t_initial_val', ['time.t_initial', 'integrator.t_initial'])
        phase.connect('t_duration_val', ['time.t_duration', 'integrator.t_duration'])

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

        integ = phase._get_subsystem('integrator')
        integ._configure_time()

        time_comp = phase._get_subsystem('time')
        time_comp.configure_io()

        ode = phase._get_subsystem('ode')

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
                                       (tphase_name, time_options['time_phase_targets'], True),
                                       ('dt_dstau', time_options['dt_dstau_targets'], True)]:
            if targets:
                if name == 'dt_dstau':
                    raise ValueError('dt_dstau_targets in ExplicitShooting are not supported at this time.')
                src_idxs = self._output_grid_data.subset_node_indices['all'] if dynamic else None
                phase.connect(f'integrator.{name}', [f'ode.{t}' for t in targets], src_indices=src_idxs,
                              flat_src_indices=True if dynamic else None)

        for name, tgts in [('t_initial', time_options['t_initial_targets']),
                           ('t_duration', time_options['t_duration_targets'])]:

            targets = _get_targets_metadata(ode, name, user_targets=tgts)
            for t, meta in targets.items():
                tgt_shape = meta['shape']

                if tgt_shape == (1,):
                    src_idxs = None
                    flat_src_idxs = None
                else:
                    src_idxs = np.zeros(self._output_grid_data.subset_num_nodes['all'])
                    flat_src_idxs = True

                phase.connect(f'{name}_val', f'ode.{name}',
                              src_indices=src_idxs,
                              flat_src_indices=flat_src_idxs)

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

        state_prefix = 'states:' if phase.timeseries_options['use_prefix'] else ''

        for name, options in phase.state_options.items():
            phase.promotes('integrator', inputs=[(f'states:{name}', f'initial_states:{name}')])
            for ts_name, ts_options in phase._timeseries.items():
                if f'{state_prefix}{name}' not in ts_options['outputs']:
                    phase.add_timeseries_output(name, output_name=f'{state_prefix}{name}',
                                                timeseries=ts_name)

        # Add the appropriate design variables
        for state_name, options in phase.state_options.items():
            if options['fix_final']:
                raise ValueError('fix_final is not a valid option for states when using the '
                                 'ExplicitShooting transcription.')
            if options['opt'] and not options['fix_initial']:
                phase.add_design_var(name=f'initial_states:{state_name}',
                                     lower=options['lower'],
                                     upper=options['upper'],
                                     scaler=options['scaler'],
                                     adder=options['adder'],
                                     ref0=options['ref0'],
                                     ref=options['ref'])

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
                                   method=self.options['method'],
                                   atol=self.options['atol'],
                                   rtol=self.options['rtol'],
                                   first_step=self.options['first_step'],
                                   max_step=self.options['max_step'],
                                   propagate_derivs=self.options['propagate_derivs'],
                                   input_grid_data=self.grid_data,
                                   output_grid_data=self._output_grid_data,
                                   ode_init_kwargs=phase.options['ode_init_kwargs'],
                                   standalone_mode=False,
                                   reports=self.options['subprob_reports'],
                                   control_interp=self.options['control_interp'],
                                   calc_exprs=phase._calc_exprs)
        phase.add_subsystem('integrator', integ)

        ode = _make_ode_system(ode_class=phase.options['ode_class'],
                               num_nodes=self._output_grid_data.num_nodes,
                               calc_exprs=phase._calc_exprs,
                               ode_init_kwargs=phase.options['ode_init_kwargs'],
                               parameter_options=phase.parameter_options)

        phase.add_subsystem('ode', ode)

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
            control_comp = ControlInterpComp(control_options=phase.control_options,
                                             time_units=phase.time_options['units'],
                                             grid_data=self.options['grid'],
                                             output_grid_data=self._output_grid_data)

            phase.add_subsystem('control_comp',
                                subsys=control_comp,
                                promotes=[('t_duration', 't_duration_val'), 'dt_dstau',
                                          '*controls:*', '*control_values:*', '*control_rates:*'])

            control_prefix = 'controls:' if phase.timeseries_options['use_prefix'] else ''
            control_rate_prefix = 'control_rates:' if phase.timeseries_options['use_prefix'] else ''

            for name, options in phase.control_options.items():
                for ts_name, ts_options in phase._timeseries.items():

                    if f'{control_prefix}{name}' not in ts_options['outputs']:
                        phase.add_timeseries_output(name, output_name=f'{control_prefix}{name}',
                                                    timeseries=ts_name)
                    if f'{control_rate_prefix}{name}_rate' not in ts_options['outputs'] \
                            and phase.timeseries_options['include_control_rates']:
                        phase.add_timeseries_output(f'{name}_rate', output_name=f'{control_rate_prefix}{name}_rate',
                                                    timeseries=ts_name)
                    if f'{control_rate_prefix}{name}_rate2' not in ts_options['outputs'] \
                            and phase.timeseries_options['include_control_rates']:
                        phase.add_timeseries_output(f'{name}_rate2', output_name=f'{control_rate_prefix}{name}_rate2',
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
        control_comp = phase._get_subsystem('control_comp')
        control_comp.configure_io()

        ode = phase._get_subsystem('ode')
        ode_inputs = get_promoted_vars(ode, 'input')

        # Add the appropriate design parameters
        for control_name, options in phase.control_options.items():
            if options['control_type'] == 'polynomial':
                ncin = options['order'] + 1
            else:
                ncin = self.options['grid'].subset_num_nodes['control_input']

            phase.promotes('integrator', inputs=[f'controls:{control_name}'])
            default_val = reshape_val(options['val'], options['shape'], ncin)
            phase.set_input_defaults(f'controls:{control_name}', val=default_val)

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
            rate_targets = get_targets(ode_inputs, control_name, options['rate_targets'])

            if rate_targets:
                phase.connect(f'control_rates:{control_name}_rate',
                              [f'ode.{t}' for t in rate_targets])

            # Second time derivative targets must be specified explicitly
            rate2_targets = get_targets(ode_inputs, control_name, options['rate2_targets'])

            if rate2_targets:
                phase.connect(f'control_rates:{control_name}_rate2',
                              [f'ode.{t}' for t in targets])

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
                                ExplicitShootingContinuityComp(grid_data=self.options['grid'],
                                                               state_options=phase.state_options,
                                                               control_options=phase.control_options,
                                                               time_units=phase.time_options['units']))

            if rate_cont:
                phase.connect('t_duration_val', 'continuity_comp.t_duration')

    def configure_defects(self, phase):
        """
        Not used in ExplicitShooting.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        ogd = self._output_grid_data
        any_state_cnty, any_control_cnty, any_rate_cnty = self._requires_continuity_constraints(phase)
        src_idxs = om.slicer[ogd.subset_node_indices['segment_ends'], ...]

        controls_to_enforce = set()
        control_rates_to_enforce = set()
        control_rates2_to_enforce = set()

        for control_name, options in phase.control_options.items():

            if options['continuity'] and any_control_cnty:
                controls_to_enforce.add(control_name)
                phase.connect(f'control_values:{control_name}',
                              f'continuity_comp.controls:{control_name}',
                              src_indices=src_idxs)
            if options['rate_continuity'] and any_rate_cnty:
                control_rates_to_enforce.add(control_name)
                phase.connect(f'control_rates:{control_name}_rate',
                              f'continuity_comp.control_rates:{control_name}_rate',
                              src_indices=src_idxs)
            if options['rate2_continuity'] and any_rate_cnty:
                control_rates2_to_enforce.add(control_name)
                phase.connect(f'control_rates:{control_name}_rate2',
                              f'continuity_comp.control_rates:{control_name}_rate2',
                              src_indices=src_idxs)

        if any((controls_to_enforce, control_rates_to_enforce, control_rates2_to_enforce)):
            phase.continuity_comp.configure_io(controls_to_enforce=controls_to_enforce,
                                               control_rates_to_enforce=control_rates_to_enforce,
                                               control_rates2_to_enforce=control_rates2_to_enforce)

    def setup_timeseries_outputs(self, phase):
        """
        Setup the timeseries for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        for name, options in phase._timeseries.items():
            timeseries_comp = TimeseriesOutputComp(input_grid_data=self._output_grid_data,
                                                   output_grid_data=self._output_grid_data,
                                                   output_subset=options['subset'],
                                                   time_units=phase.time_options['units'])
            phase.add_subsystem(name, subsys=timeseries_comp)

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

    def configure_solvers(self, phase, requires_solvers=None):
        """
        Setup the solvers for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        requires_solvers : None
            Required to extend TranscriptionBase.configure_solvers but not used.
        """
        super().configure_solvers(phase, requires_solvers=None)

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
            for tgt in options['targets']:
                if tgt in options['static_targets']:
                    src_idxs_raw = np.zeros(1, dtype=int)
                    src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
                    src_idxs = np.squeeze(src_idxs, axis=0)
                else:
                    src_idxs_raw = np.zeros(self._output_grid_data.subset_num_nodes['all'], dtype=int)
                    src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
                    if options['shape'] == (1,):
                        src_idxs = src_idxs.ravel()

                connection_info.append((f'ode.{tgt}', (src_idxs,)))
            connection_info.append(([f'integrator.parameters:{name}'], None))

        return connection_info

    def _get_response_src(self, var, loc, phase, ode_outputs=None):
        """
        Return the path to the variable that will be used as a response..

        Parameters
        ----------
        var : str
            Name of the variable to be used as the response.
        loc : str
            The location of the response in the phase ['initial', 'final'].
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
            True if the objective quantity is linear.
        """
        time_units = phase.time_options['units']
        var_type = phase.classify_var(var)

        if var_type == 't':
            shape = (1,)
            units = time_units
            linear = True
            obj_path = 't'
        elif var_type == 't_phase':
            shape = (1,)
            units = time_units
            linear = True
            obj_path = 't_phase'
        elif var_type == 'state':
            shape = phase.state_options[var]['shape']
            units = phase.state_options[var]['units']
            linear = loc == 'initial'
            obj_path = f'integrator.states_out:{var}'
        elif var_type == 'control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = False
            obj_path = f'control_values:{var}'
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
        else:
            # Failed to find variable, assume it is in the ODE. This requires introspection.
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
        if not self.options['propagate_derivs']:
            # We're not propagating derivatives because we're just doing a simulation run_model.
            # No continuity is needed.
            return False, False, False

        num_seg = self.options['grid'].num_segments
        compressed = self.options['grid'].compressed
        transcription = self.options['grid'].transcription

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
        elif var_type in ['control']:
            path = f'control_values:{var}'
            src_units = phase.control_options[var]['units']
            src_shape = phase.control_options[var]['shape']
        elif var_type == 'control_rate':
            control_name = var[:-5]
            path = f'control_rates:{control_name}_rate'
            src_units = get_rate_units(phase.control_options[control_name]['units'], time_units, deriv=1)
            src_shape = phase.control_options[control_name]['shape']
        elif var_type == 'control_rate2':
            control_name = var[:-6]
            path = f'control_rates:{control_name}_rate2'
            src_units = get_rate_units(phase.control_options[control_name]['units'], time_units, deriv=2)
            src_shape = phase.control_options[control_name]['shape']
        elif var_type == 'parameter':
            path = f'parameter_vals:{var}'
            node_idxs = np.zeros(self._output_grid_data.subset_num_nodes['all'], dtype=int)
            src_units = phase.parameter_options[var]['units']
            src_shape = phase.parameter_options[var]['shape']
        else:
            # Failed to find variable, assume it is in the ODE
            meta = get_source_metadata(ode_outputs, src=var)
            src_shape = meta['shape']
            src_units = meta['units']
            src_tags = meta['tags']
            path = f'ode.{var}'
            if 'dymos.static_output' in src_tags:
                raise RuntimeError(f'ODE output {var} is tagged with "dymos.static_output" and cannot be a '
                                   f'timeseries output.')

        src_idxs = None if node_idxs is None else om.slicer[node_idxs, ...]

        meta['src'] = path
        meta['src_idxs'] = src_idxs
        meta['units'] = src_units
        meta['shape'] = src_shape

        return meta

    def _phase_set_state_val(self, phase, name, vals, time_vals=None, interpolation_kind=None):
        """
        Method to interpolate the provided input and return the variables that need to be set
        along with their appropriate value.

        Parameters
        ----------
        phase : dymos.Phase
            The phase to which this transcription applies.
        name : str
            The name of the phase variable to be set.
        vals : ndarray or Sequence or float
            Array of control/state/parameter values.
        time_vals : ndarray or Sequence or None
            Array of integration variable values.
        interpolation_kind : str or None
            Specifies the kind of interpolation, as per the scipy.interpolate package.
            One of ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
            where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of zeroth, first, second or third order) or as an
            integer specifying the order of the spline interpolator to use.
            Default is 'linear'.

        Returns
        -------
        input_data : dict
            Dict containing the values that need to be set in the phase

        """
        if np.isscalar(vals):
            val = vals
        else:
            val = vals[0]
        input_data = {f'initial_states:{name}': val}

        return input_data
