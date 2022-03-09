from collections.abc import Sequence

import numpy as np

import openmdao.api as om

from .common import BoundaryConstraintComp, ControlGroup, PolynomialControlGroup, PathConstraintComp
from ..utils.constants import INF_BOUND
from ..utils.misc import get_rate_units, _unspecified
from ..utils.introspection import get_promoted_vars, get_source_metadata, get_target_metadata


class TranscriptionBase(object):
    """
    Base class for all dymos transcriptions.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):

        self.grid_data = None

        self.options = om.OptionsDictionary()

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

        self._declare_options()
        self.initialize()
        self.options.update(kwargs)
        self.init_grid()

        # Where to query var info.
        self._rhs_source = None

    def _declare_options(self):
        pass

    def initialize(self):
        """
        Declare transcription options.
        """
        pass

    def init_grid(self):
        """
        Setup the GridData object for the Transcription.
        """
        raise NotImplementedError('Transcription {0} does not implement method'
                                  'init_grid.'.format(self.__class__.__name__))

    def setup_time(self, phase):
        """
        Setup up the time component and time extents for the phase.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        time_options = phase.time_options

        # Warn about invalid options
        phase.check_time_options()

        if not time_options['input_initial'] or not time_options['input_duration']:
            phase.add_subsystem('time_extents', om.IndepVarComp(),
                                promotes_outputs=['*'])

    def configure_time(self, phase):
        """
        Configure the inputs/outputs on the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        time_options = phase.time_options

        # Determine the time unit.
        if time_options['units'] in (None, _unspecified):
            if time_options['targets']:
                ode = phase._get_subsystem(self._rhs_source)

                _, time_options['units'] = get_target_metadata(ode, name='time',
                                                               user_targets=time_options['targets'],
                                                               user_units=time_options['units'],
                                                               user_shape='')

        time_units = time_options['units']
        indeps = []
        default_vals = {'t_initial': phase.time_options['initial_val'],
                        't_duration': phase.time_options['duration_val']}

        if not time_options['input_initial']:
            indeps.append('t_initial')

        if not time_options['input_duration']:
            indeps.append('t_duration')

        for var in indeps:
            phase.time_extents.add_output(var, val=default_vals[var], units=time_units)

        if not (time_options['input_initial'] or time_options['fix_initial']):
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
                                         grid_data=self.grid_data)

            phase.add_subsystem('control_group',
                                subsys=control_group)

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def setup_polynomial_controls(self, phase):
        """
        Adds the polynomial control group to the model if any polynomial controls are present.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        if phase.polynomial_control_options:
            sys = PolynomialControlGroup(grid_data=self.grid_data,
                                         polynomial_control_options=phase.polynomial_control_options,
                                         time_units=phase.time_options['units'])
            phase.add_subsystem('polynomial_control_group', subsys=sys,
                                promotes_inputs=['*'], promotes_outputs=['*'])

    def configure_polynomial_controls(self, phase):
        """
        Configure the inputs/outputs for the polynomial controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        if phase.polynomial_control_options:
            phase.polynomial_control_group.configure_io()

    def setup_parameters(self, phase):
        """
        Sets input defaults for parameters and optionally adds design variables.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase._check_parameter_options()

        if phase.parameter_options:
            for name, options in phase.parameter_options.items():
                src_name = 'parameters:{0}'.format(name)

                if options['opt']:
                    lb = -INF_BOUND if options['lower'] is None else options['lower']
                    ub = INF_BOUND if options['upper'] is None else options['upper']

                    phase.add_design_var(name=src_name,
                                         lower=lb,
                                         upper=ub,
                                         scaler=options['scaler'],
                                         adder=options['adder'],
                                         ref0=options['ref0'],
                                         ref=options['ref'])

    def configure_parameters(self, phase):
        """
        Configure parameter promotion.

        This method assumes that utils.introspection.configure_parameters_introspection has already populated
        the parameter options with the appropriate targets, units, shape, and static_target fields.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        if phase.parameter_options:
            ode = self._get_ode(phase)

            for name, options in phase.parameter_options.items():
                prom_name = f'parameters:{name}'

                for tgts, src_idxs in self.get_parameter_connections(name, phase):
                    for pathname in tgts:
                        parts = pathname.split('.')
                        sub_sys = parts[0]
                        tgt_var = '.'.join(parts[1:])
                        if not options['static_target']:
                            phase.promotes(sub_sys, inputs=[(tgt_var, prom_name)],
                                           src_indices=src_idxs, flat_src_indices=True)
                        else:
                            phase.promotes(sub_sys, inputs=[(tgt_var, prom_name)])

                val = options['val']
                _shape = options['shape']
                shaped_val = np.broadcast_to(val, _shape)
                phase.set_input_defaults(name=prom_name,
                                         val=shaped_val,
                                         units=options['units'])

    def setup_states(self, phase):
        """
        Setup the states for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        raise NotImplementedError('Transcription {0} does not implement method '
                                  'setup_states.'.format(self.__class__.__name__))

    def setup_ode(self, phase):
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        raise NotImplementedError('Transcription {0} does not implement method '
                                  'setup_ode.'.format(self.__class__.__name__))

    def setup_timeseries_outputs(self, phase):
        """
        Setup the timeseries for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        raise NotImplementedError('Transcription {0} does not implement method '
                                  'setup_timeseries_outputs.'.format(self.__class__.__name__))

    def setup_boundary_constraints(self, loc, phase):
        """
        Setup the boundary constraints.

        Adds BoundaryConstraintComp for initial and/or final boundary constraints if necessary
        and issues appropriate connections.

        Parameters
        ----------
        loc : str
            The kind of boundary constraints being setup.  Must be one of 'initial' or 'final'.
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        if loc not in ('initial', 'final'):
            raise ValueError('loc must be one of \'initial\' or \'final\'.')

        bc_dict = phase._initial_boundary_constraints \
            if loc == 'initial' else phase._final_boundary_constraints

        if bc_dict:
            phase.add_subsystem(f'{loc}_boundary_constraints',
                                subsys=BoundaryConstraintComp(loc=loc))

    def configure_boundary_constraints(self, loc, phase):
        """
        Configures the boundary constraints.

        Adds BoundaryConstraintComp for initial and/or final boundary constraints if necessary
        and issues appropriate connections.

        Parameters
        ----------
        loc : str
            The kind of boundary constraints being setup.  Must be one of 'initial' or 'final'.
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        bc_dict = phase._initial_boundary_constraints \
            if loc == 'initial' else phase._final_boundary_constraints

        sys_name = f'{loc}_boundary_constraints'
        bc_comp = phase._get_subsystem(sys_name)
        ode_outputs = get_promoted_vars(self._get_ode(phase), 'output')

        for var, options in bc_dict.items():
            con_name = options['constraint_name']

            _, shape, units, linear = self._get_boundary_constraint_src(var, loc, phase, ode_outputs=ode_outputs)

            if options['indices'] is not None:
                # Sliced shape.
                con_shape = (len(options['indices']), )
                # Indices provided, make sure lower/upper/equals have shape of the indices.
                if options['lower'] and not np.isscalar(options['lower']) and \
                        np.asarray(options['lower']).shape != con_shape:
                    raise ValueError('The lower bounds of boundary constraint on {0} are not '
                                     'compatible with its shape, and no indices were '
                                     'provided.'.format(var))

                if options['upper'] and not np.isscalar(options['upper']) and \
                        np.asarray(options['upper']).shape != con_shape:
                    raise ValueError('The upper bounds of boundary constraint on {0} are not '
                                     'compatible with its shape, and no indices were '
                                     'provided.'.format(var))

                if options['equals'] and not np.isscalar(options['equals']) and \
                        np.asarray(options['equals']).shape != con_shape:
                    raise ValueError('The equality boundary constraint value on {0} is not '
                                     'compatible the provided indices. Provide them as a '
                                     'flat array with the same size as indices.'.format(var))

            else:
                # Indices not provided, make sure lower/upper/equals have shape of source.
                if 'lower' in options and options['lower'] is not None and \
                        not np.isscalar(options['lower']) and np.asarray(options['lower']).shape != shape:
                    raise ValueError('The lower bounds of boundary constraint on {0} are not '
                                     'compatible with its shape, and no indices were '
                                     'provided. Expected a shape of {1} but given shape '
                                     'is {2}'.format(var, shape, np.asarray(options['lower']).shape))

                if 'upper' in options and options['upper'] is not None and \
                        not np.isscalar(options['upper']) and np.asarray(options['upper']).shape != shape:
                    raise ValueError('The upper bounds of boundary constraint on {0} are not '
                                     'compatible with its shape, and no indices were '
                                     'provided. Expected a shape of {1} but given shape '
                                     'is {2}'.format(var, shape, np.asarray(options['upper']).shape))

                if 'equals' in options and options['equals'] is not None and \
                        not np.isscalar(options['equals']) and np.asarray(options['equals']).shape != shape:
                    raise ValueError('The equality boundary constraint value on {0} is not '
                                     'compatible with its shape, and no indices were '
                                     'provided. Expected a shape of {1} but given shape '
                                     'is {2}'.format(var, shape, np.asarray(options['equals']).shape))

            # Constraint options are a copy of options with constraint_name key removed.
            con_options = options.copy()
            con_options.pop('constraint_name')

            # By now, all possible constraint target shapes should have been introspected.
            con_options['shape'] = options['shape'] = shape

            # If user overrides the introspected unit, then change the unit on the add_constraint call.
            con_units = options['units']
            con_options['units'] = units if con_units is None else con_units
            con_options['linear'] = linear

            # bc_comp._add_constraint(con_name, **con_options)
            con_options.pop('shape', None)

            # indices = om.slicer[0, ...] if loc == 'initial' else om.slicer[-1, ...]

            con_output, constraint_kwargs = self._get_constraint_kwargs(loc, var, options, phase)

            constraint_kwargs.pop('shape', None)
            # indices = constraint_kwargs.pop('indices', None)
            # print(constraint_kwargs['indices'])

            phase.add_constraint(con_output, **constraint_kwargs)

    def setup_path_constraints(self, phase):
        pass

    def _get_constraint_kwargs(self, constraint_type, var, options, phase):
        constraint_kwargs = options.copy()
        time_units = phase.time_options['units']
        con_units = constraint_kwargs['units'] = options.get('units', None)
        con_name = constraint_kwargs.pop('constraint_name')

        # Determine the path to the variable which we will be constraining
        var_type = phase.classify_var(var)

        if var_type == 'time':
            constraint_kwargs['shape'] = (1,)
            constraint_kwargs['units'] = time_units if con_units is None else con_units
            constraint_kwargs['linear'] = True
            con_output = 'timeseries.time'

        elif var_type == 'time_phase':
            constraint_kwargs['shape'] = (1,)
            constraint_kwargs['units'] = time_units if con_units is None else con_units
            constraint_kwargs['linear'] = True
            con_output = 'timeseries.time_phase'

        elif var_type == 'state':
            state_shape = phase.state_options[var]['shape']
            state_units = phase.state_options[var]['units']
            constraint_kwargs['shape'] = state_shape
            constraint_kwargs['units'] = state_units if con_units is None else con_units
            constraint_kwargs['linear'] = False
            con_output = f'timeseries.states:{var}'
            
        elif var_type == 'parameter:':
            con_output = f'parameter_values:{var}'

        elif var_type == 'indep_control':
            control_shape = phase.control_options[var]['shape']
            control_units = phase.control_options[var]['units']

            constraint_kwargs['shape'] = control_shape
            constraint_kwargs['units'] = control_units if con_units is None else con_units
            constraint_kwargs['linear'] = True
            con_output = f'timeseries.controls:{var}'

        elif var_type == 'input_control':
            control_shape = phase.control_options[var]['shape']
            control_units = phase.control_options[var]['units']

            constraint_kwargs['shape'] = control_shape
            constraint_kwargs['units'] = control_units if con_units is None else con_units
            constraint_kwargs['linear'] = True
            con_output = f'timeseries.controls:{var}'

        elif var_type == 'indep_polynomial_control':
            control_shape = phase.polynomial_control_options[var]['shape']
            control_units = phase.polynomial_control_options[var]['units']
            constraint_kwargs['shape'] = control_shape
            constraint_kwargs['units'] = control_units if con_units is None else con_units
            constraint_kwargs['linear'] = False
            con_output = f'timeseries.polynomial_controls:{var}'

        elif var_type == 'input_polynomial_control':
            control_shape = phase.polynomial_control_options[var]['shape']
            control_units = phase.polynomial_control_options[var]['units']
            constraint_kwargs['shape'] = control_shape
            constraint_kwargs['units'] = control_units if con_units is None else con_units
            constraint_kwargs['linear'] = False
            con_output = f'timeseries.polynomial_controls:{var}'

        elif var_type == 'control_rate':
            control_name = var[:-5]
            control_shape = phase.control_options[control_name]['shape']
            control_units = phase.control_options[control_name]['units']
            constraint_kwargs['shape'] = control_shape
            constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=1) \
                if con_units is None else con_units
            con_output = f'timeseries.control_rates:{var}'

        elif var_type == 'control_rate2':
            control_name = var[:-6]
            control_shape = phase.control_options[control_name]['shape']
            control_units = phase.control_options[control_name]['units']
            constraint_kwargs['shape'] = control_shape
            constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=2) \
                if con_units is None else con_units
            con_output = f'timeseries.control_rates:{var}'

        elif var_type == 'polynomial_control_rate':
            control_name = var[:-5]
            control_shape = phase.polynomial_control_options[control_name]['shape']
            control_units = phase.polynomial_control_options[control_name]['units']
            constraint_kwargs['shape'] = control_shape
            constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=1) \
                if con_units is None else con_units
            con_output = f'timeseries.polynomial_control_rates:{var}'

        elif var_type == 'polynomial_control_rate2':
            control_name = var[:-6]
            control_shape = phase.polynomial_control_options[control_name]['shape']
            control_units = phase.polynomial_control_options[control_name]['units']
            constraint_kwargs['shape'] = control_shape
            constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=2) \
                if con_units is None else con_units
            con_output = f'timeseries.polynomial_control_rates:{var}'

        else:
            # Failed to find variable, assume it is in the ODE. This requires introspection.
            ode = phase._get_subsystem(self._rhs_source)

            shape, units = get_source_metadata(ode, src=var,
                                               user_units=options['units'],
                                               user_shape=options['shape'])

            constraint_kwargs['linear'] = False
            constraint_kwargs['shape'] = shape
            constraint_kwargs['units'] = units
            con_output = f'timeseries.{con_name}'

        user_idxs = options['indices'] if options['indices'] is not None else om.slicer[...]

        if constraint_type == 'initial':
            constraint_kwargs['indices'] = om.slicer[0, user_idxs]
            constraint_kwargs['alias'] = f'{phase.pathname}.{con_output} [bc_initial:{con_name}]'
        elif constraint_type == 'final':
            constraint_kwargs['indices'] =  om.slicer[-1, user_idxs]
            constraint_kwargs['alias'] = f'{phase.pathname}.{con_output} [bc_final:{con_name}]'
        else:
            if var in phase._initial_boundary_constraints:
                constraint_kwargs['indices'] = om.slicer[1:, user_idxs]
            elif var in phase._final_boundary_constraints:
                constraint_kwargs['indices'] = om.slicer[:-1, user_idxs]
            else:
                constraint_kwargs['indices'] = om.slicer[:, user_idxs]

            constraint_kwargs['alias'] = f'{phase.pathname}.{con_output} [path:{con_name}]'

        return con_output, constraint_kwargs

    def configure_path_constraints(self, phase):
        """
        Handle the common operations for configuration of the path constraints.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        for var, options in phase._path_constraints.items():
            con_output, constraint_kwargs = self._get_constraint_kwargs('path', var, options, phase)

            # Propagate the introspected shape back into the options dict.
            # Some transcriptions use this later.
            options['shape'] = constraint_kwargs['shape']

            constraint_kwargs.pop('constraint_name', None)
            constraint_kwargs.pop('shape', None)

            phase.add_constraint(con_output, **constraint_kwargs)


    def configure_objective(self, phase):
        """
        Find the path of the objective(s) and add them.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        for name, options in phase._objectives.items():
            index = options['index']
            loc = options['loc']

            obj_path, shape, units, _ = self._get_boundary_constraint_src(name, loc, phase)

            shape = options['shape'] if shape is None else shape

            size = int(np.prod(shape))

            if size > 1 and index is None:
                raise ValueError('Objective variable is non-scaler {0} but no index specified '
                                 'for objective'.format(shape))

            idx = 0 if index is None else index
            if idx < 0:
                idx = size + idx

            if idx >= size or idx < -size:
                raise ValueError('Objective index={0}, but the shape of the objective '
                                 'variable is {1}'.format(index, shape))

            if loc == 'final':
                obj_index = -size + idx
            elif loc == 'initial':
                obj_index = idx
            else:
                raise ValueError('Invalid value for objective loc: {0}. Must be '
                                 'one of \'initial\' or \'final\'.'.format(loc))

            from ..phase import Phase
            super(Phase, phase).add_objective(obj_path, ref=options['ref'], ref0=options['ref0'],
                                              index=obj_index, flat_indices=True, adder=options['adder'],
                                              scaler=options['scaler'],
                                              parallel_deriv_color=options['parallel_deriv_color'])

    def _get_boundary_constraint_src(self, name, loc, phase, ode_outputs=None):
        raise NotImplementedError('Transcription {0} does not implement method'
                                  '_get_boundary_constraint_source.'.format(self.__class__.__name__))

    def _get_rate_source_path(self, name, loc, phase):
        raise NotImplementedError('Transcription {0} does not implement method'
                                  '_get_rate_source_path.'.format(self.__class__.__name__))

    def _get_ode(self, phase):
        """
        Returns an instance of the ODE used in the phase that can be interrogated for IO metadata.

        Parameters
        ----------
        phase : dm.Phase
            The Phase instance to which this transcription applies

        Returns
        -------
        ode : om.System
            The OpenMDAO system which serves as the ODE for the given Phase.

        """
        return phase._get_subsystem(self._rhs_source)

    def get_parameter_connections(self, name, phase):
        """
        Returns info about a parameter's target connections in the phase.

        Parameters
        ----------
        name : str
            The name of the parameter for which connection information is desired.
        phase : dymos.Phase
            The phase object to which this transcription applies.

        Returns
        -------
        list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design variable is to be connected.
        """
        raise NotImplementedError('Transcription {0} does not implement method '
                                  'get_parameter_connections.'.format(self.__class__.__name__))

    def is_static_ode_output(self, var, phase, num_nodes):
        """
        Test whether the given output is a static output of the ODE.

        A variable is considered static if it's first dimension is different than the
        number of nodes in the ODE.

        Parameters
        ----------
        var : str
            The ode-relative path of the variable of interest.
        phase : dymos.Phase or dict
            The phase to which this transcription applies or a dict of the ODE outputs as returned by get_promoted_vars.
        num_nodes : int
            The number of nodes in the ODE.

        Returns
        -------
        bool
            True if the given variable is a static output, otherwise False if it is dynamic.

        Raises
        ------
        KeyError
            KeyError is raised if the given variable isn't present in the ode outputs.
        """
        if isinstance(phase, dict):
            ode_outputs = phase
        else:
            ode_outputs = get_promoted_vars(self._get_ode(phase), 'output')
        ode_shape = ode_outputs[var]['shape']
        return ode_shape[0] != num_nodes

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
        raise NotImplementedError(f'The transcription {self.__class__} does not provide an '
                                  f'implementation of _requires_continuity_constraints')
