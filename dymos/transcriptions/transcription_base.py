from collections.abc import Sequence

import numpy as np

import openmdao.api as om

from .common import BoundaryConstraintComp, ControlGroup, PolynomialControlGroup, PathConstraintComp
from ..phase.options import StateOptionsDictionary
from ..utils.constants import INF_BOUND
from ..utils.misc import get_rate_units, _unspecified, get_target_metadata, get_source_metadata


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
                             desc='Order of the state transcription')
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

    def _configure_state_introspection(self, state_name, options, phase):
        """
        Modifies state options in-place, automatically determining 'targets', 'units', and 'shape'
        if necessary.

        The precedence rules for the state shape and units are as follows:
        1. If the user has specified units and shape in the state options, use those.
        2a. If the user has not specified shape, and targets exist, then pull the shape from the targets.
        2b. If the user has not specified shape and no targets exist, then pull the shape from the rate source.
        2c. If shape cannot be inferred, assume (1,)
        3a. If the user has not specified units, first try to pull units from a target
        3b. If there are no targets, pull units from the rate source and multiply by time units.

        Parameters
        ----------
        state_name : str
            The name of the state variable of interest.
        options : OptionsDictionary
            The options dictionary for the state variable of interest.
        phase : dymos.Phase
            The phase associated with the transcription.
        """
        time_units = phase.time_options['units']
        user_targets = options['targets']
        user_units = options['units']
        user_shape = options['shape']

        need_units = user_units is _unspecified
        need_shape = user_shape in {None, _unspecified}

        ode = phase._get_subsystem(self._rhs_source)

        # Automatically determine targets of state if left _unspecified
        if user_targets is _unspecified:
            from dymos.utils.introspection import get_targets
            options['targets'] = get_targets(ode, state_name, user_targets)

        # 1. No introspection necessary
        if not(need_shape or need_units):
            return

        # 2. Attempt target introspection
        if options['targets']:
            try:
                from dymos.utils.introspection import get_state_target_metadata
                tgt_shape, tgt_units = get_state_target_metadata(ode, state_name, options['targets'],
                                                                 options['units'], options['shape'])
                options['shape'] = tgt_shape
                options['units'] = tgt_units
                return
            except ValueError:
                pass

        # 3. Attempt rate-source introspection
        rate_src = options['rate_source']
        rate_src_type = phase.classify_var(rate_src)

        if rate_src_type in ['time', 'time_phase']:
            rate_src_units = phase.time_options['units']
            rate_src_shape = (1,)
        elif rate_src_type == 'state':
            rate_src_units = phase.state_options[rate_src]['units']
            rate_src_shape = phase.state_options[rate_src]['shape']
        elif rate_src_type in ['input_control', 'indep_control']:
            rate_src_units = phase.control_options[rate_src]['units']
            rate_src_shape = phase.control_options[rate_src]['shape']
        elif rate_src_type in ['input_polynomial_control', 'indep_polynomial_control']:
            rate_src_units = phase.polynomial_control_options[rate_src]['units']
            rate_src_shape = phase.polynomial_control_options[rate_src]['shape']
        elif rate_src_type == 'parameter':
            rate_src_units = phase.parameter_options[rate_src]['units']
            rate_src_shape = phase.parameter_options[rate_src]['shape']
        elif rate_src_type == 'control_rate':
            control_name = rate_src[:-5]
            control = phase.control_options[control_name]
            rate_src_units = get_rate_units(control['units'], time_units, deriv=1)
            rate_src_shape = control['shape']
        elif rate_src_type == 'control_rate2':
            control_name = rate_src[:-6]
            control = phase.control_options[control_name]
            rate_src_units = get_rate_units(control['units'], time_units, deriv=2)
            rate_src_shape = control['shape']
        elif rate_src_type == 'polynomial_control_rate':
            control_name = rate_src[:-5]
            control = phase.polynomial_control_options[control_name]
            rate_src_units = get_rate_units(control['units'], time_units, deriv=1)
            rate_src_shape = control['shape']
        elif rate_src_type == 'polynomial_control_rate2':
            control_name = rate_src[:-6]
            control = phase.polynomial_control_options[control_name]
            rate_src_units = get_rate_units(control['units'], time_units, deriv=2)
            rate_src_shape = control['shape']
        elif rate_src_type == 'ode':
            rate_src_shape, rate_src_units = get_source_metadata(ode,
                                                                 src=rate_src,
                                                                 user_units=options['units'],
                                                                 user_shape=options['shape'])
        else:
            rate_src_shape = (1,)
            rate_src_units = None

        if need_shape:
            options['shape'] = rate_src_shape

        if need_units:
            options['units'] = time_units if rate_src_units is None else f'{rate_src_units}*{time_units}'

        return

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        ode = phase._get_subsystem(self._rhs_source)

        # Interrogate shapes and units.
        for name, options in phase.control_options.items():

            shape, units = get_target_metadata(ode, name=name,
                                               user_targets=options['targets'],
                                               user_units=options['units'],
                                               user_shape=options['shape'],
                                               control_rate=True)

            options['units'] = units
            options['shape'] = shape

        if phase.control_options:
            phase.control_group.configure_io()
            phase.promotes('control_group',
                           any=['controls:*', 'control_values:*', 'control_rates:*'])

            phase.connect('dt_dstau', 'control_group.dt_dstau')

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
        ode = phase._get_subsystem(self._rhs_source)

        # Interrogate shapes and units.
        for name, options in phase.polynomial_control_options.items():

            shape, units = get_target_metadata(ode, name=name,
                                               user_targets=options['targets'],
                                               user_units=options['units'],
                                               user_shape=options['shape'],
                                               control_rate=True)

            options['units'] = units
            options['shape'] = shape

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

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        if phase.parameter_options:
            ode = phase._get_subsystem(self._rhs_source)

            for name, options in phase.parameter_options.items():
                src_name = 'parameters:{0}'.format(name)

                # Get units and shape from targets when needed.
                shape, units = get_target_metadata(ode, name=name,
                                                   user_targets=options['targets'],
                                                   user_shape=options['shape'],
                                                   user_units=options['units'],
                                                   dynamic=options['dynamic'])
                options['units'] = units
                options['shape'] = shape

                prom_name = 'parameters:{0}'.format(name)
                for tgts, src_idxs in self.get_parameter_connections(name, phase):
                    for pathname in tgts:
                        parts = pathname.split('.')
                        sub_sys = parts[0]
                        tgt_var = '.'.join(parts[1:])
                        if options['dynamic']:
                            phase.promotes(sub_sys, inputs=[(tgt_var, prom_name)],
                                           src_indices=src_idxs, flat_src_indices=True)
                        else:
                            phase.promotes(sub_sys, inputs=[(tgt_var, prom_name)])

                val = options['val']
                _shape = options['shape']
                shaped_val = np.broadcast_to(val, _shape)
                phase.set_input_defaults(name=src_name,
                                         val=shaped_val,
                                         units=options['units'])

    def configure_state_discovery(self, phase):
        """
        Searches phase output metadata for any declared states and adds them.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        state_options = phase.state_options
        ode = phase._get_subsystem(self._rhs_source)
        out_meta = ode.get_io_metadata(iotypes='output', metadata_keys=['tags'],
                                       get_remote=True)

        for name, meta in out_meta.items():
            tags = meta['tags']
            prom_name = meta['prom_name']
            state = None
            for tag in sorted(tags):

                # Declared as rate_source.
                if tag.startswith('state_rate_source:'):
                    state = tag[18:]
                    if state not in state_options:
                        state_options[state] = StateOptionsDictionary()
                        state_options[state]['name'] = state

                    if state_options[state]['rate_source'] is not None:
                        if state_options[state]['rate_source'] != prom_name:
                            raise ValueError(f"rate_source has been declared twice for state "
                                             f"'{state}' which is tagged on '{name}'.")

                    state_options[state]['rate_source'] = prom_name

                # Declares units for state.
                if tag.startswith('state_units:'):
                    if state is None:
                        raise ValueError(f"'state_units:' tag declared on '{name}' also requires "
                                         f"that the 'state_rate_source:' tag be declared.")
                    state_options[state]['units'] = tag[12:]

        # Check over all existing states and make sure we aren't missing any rate sources.
        for name, options in state_options.items():
            if options['rate_source'] is None:
                raise ValueError(f"State '{name}' is missing a rate_source.")

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
            bc_comp = phase.add_subsystem('{0}_boundary_constraints'.format(loc),
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

        for var, options in bc_dict.items():
            con_name = options['constraint_name']

            _, shape, units, linear = self._get_boundary_constraint_src(var, loc, phase)

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

            bc_comp._add_constraint(con_name, **con_options)

        if bc_comp:
            bc_comp.configure_io()

        for var, options in bc_dict.items():
            con_name = options['constraint_name']

            src, shape, units, linear = self._get_boundary_constraint_src(var, loc, phase)

            size = np.prod(shape)

            # Build the correct src_indices regardless of shape
            if loc == 'initial':
                src_idxs = np.arange(size, dtype=int).reshape(shape)
            else:
                src_idxs = np.arange(-size, 0, dtype=int).reshape(shape)

            if 'parameters:' in src:
                sys_name = '{0}_boundary_constraints'.format(loc)
                tgt_name = '{0}_value_in:{1}'.format(loc, con_name)
                phase.promotes(sys_name, inputs=[(tgt_name, src)],
                               src_indices=src_idxs, flat_src_indices=True)

            else:
                phase.connect(src,
                              f'{loc}_boundary_constraints.{loc}_value_in:{con_name}',
                              src_indices=src_idxs,
                              flat_src_indices=True)

    def setup_path_constraints(self, phase):
        """
        Add a path constraint component if necessary.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        gd = self.grid_data

        if phase._path_constraints:
            path_comp = PathConstraintComp(num_nodes=gd.num_nodes)
            phase.add_subsystem('path_constraints', subsys=path_comp)

    def configure_path_constraints(self, phase):
        """
        Handle the common operations for configuration of the path constraints.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        time_units = phase.time_options['units']

        for var, options in phase._path_constraints.items():
            constraint_kwargs = options.copy()
            con_units = constraint_kwargs['units'] = options.get('units', None)
            con_name = constraint_kwargs.pop('constraint_name')

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            if var_type == 'time':
                constraint_kwargs['shape'] = (1,)
                constraint_kwargs['units'] = time_units if con_units is None else con_units
                constraint_kwargs['linear'] = True

            elif var_type == 'time_phase':
                constraint_kwargs['shape'] = (1,)
                constraint_kwargs['units'] = time_units if con_units is None else con_units
                constraint_kwargs['linear'] = True

            elif var_type == 'state':
                state_shape = phase.state_options[var]['shape']
                state_units = phase.state_options[var]['units']
                constraint_kwargs['shape'] = state_shape
                constraint_kwargs['units'] = state_units if con_units is None else con_units
                constraint_kwargs['linear'] = False

            elif var_type == 'indep_control':
                control_shape = phase.control_options[var]['shape']
                control_units = phase.control_options[var]['units']

                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = True

            elif var_type == 'input_control':
                control_shape = phase.control_options[var]['shape']
                control_units = phase.control_options[var]['units']

                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = True

            elif var_type == 'indep_polynomial_control':
                control_shape = phase.polynomial_control_options[var]['shape']
                control_units = phase.polynomial_control_options[var]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = False

            elif var_type == 'input_polynomial_control':
                control_shape = phase.polynomial_control_options[var]['shape']
                control_units = phase.polynomial_control_options[var]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = False

            elif var_type == 'control_rate':
                control_name = var[:-5]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units

            elif var_type == 'control_rate2':
                control_name = var[:-6]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units

            elif var_type == 'polynomial_control_rate':
                control_name = var[:-5]
                control_shape = phase.polynomial_control_options[control_name]['shape']
                control_units = phase.polynomial_control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units

            elif var_type == 'polynomial_control_rate2':
                control_name = var[:-6]
                control_shape = phase.polynomial_control_options[control_name]['shape']
                control_units = phase.polynomial_control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units

            else:
                # Failed to find variable, assume it is in the ODE. This requires introspection.
                ode = phase._get_subsystem(self._rhs_source)

                shape, units = get_source_metadata(ode, src=var,
                                                   user_units=options['units'],
                                                   user_shape=options['shape'])

                constraint_kwargs['linear'] = False
                constraint_kwargs['shape'] = shape
                constraint_kwargs['units'] = units

            # Propagate the introspected shape back into the options dict.
            # Some transcriptions use this later.
            options['shape'] = constraint_kwargs['shape']

            constraint_kwargs.pop('constraint_name', None)
            phase._get_subsystem('path_constraints')._add_path_constraint_configure(con_name, **constraint_kwargs)

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
                                              index=obj_index, adder=options['adder'],
                                              scaler=options['scaler'],
                                              parallel_deriv_color=options['parallel_deriv_color'],
                                              vectorize_derivs=options['vectorize_derivs'])

    def _get_boundary_constraint_src(self, name, loc, phase):
        raise NotImplementedError('Transcription {0} does not implement method'
                                  '_get_boundary_constraint_source.'.format(self.__class__.__name__))

    def _get_rate_source_path(self, name, loc, phase):
        raise NotImplementedError('Transcription {0} does not implement method'
                                  '_get_rate_source_path.'.format(self.__class__.__name__))

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
        phase : dymos.Phase
            The phase to which this transcription applies.
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
        ode = phase._get_subsystem(self._rhs_source)
        ode_outputs = {opts['prom_name']: opts for (k, opts) in
                       ode.get_io_metadata(iotypes=('output',), get_remote=True).items()}
        ode_shape = ode_outputs[var]['shape']
        return ode_shape[0] != num_nodes
