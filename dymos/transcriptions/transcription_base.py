from __future__ import print_function, division, absolute_import

from collections import Sequence
import warnings

from six import iteritems

import numpy as np

import openmdao.api as om

from .common import BoundaryConstraintComp, InputParameterComp, ControlGroup, \
    PolynomialControlGroup
from ..utils.constants import INF_BOUND
from ..utils.indexing import get_src_indices_by_row
from ..utils.rk_methods import rk_methods


class TranscriptionBase(object):

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

    def _declare_options(self):
        pass

    def initialize(self):
        pass

    def init_grid(self):
        """
        Setup the GridData object for the Transcription
        """

        raise NotImplementedError('Transcription {0} does not implement method'
                                  'init_grid.'.format(self.__class__.__name__))

    def setup_time(self, phase):
        """
        Setup up the time component and time extents for the phase.

        Returns
        -------
        comps
            A list of the component names needed for time extents.
        """
        time_options = phase.time_options
        time_units = time_options['units']

        indeps = []
        default_vals = {'t_initial': phase.time_options['initial_val'],
                        't_duration': phase.time_options['duration_val']}
        externals = []
        comps = []

        # Warn about invalid options
        phase.check_time_options()

        if time_options['input_initial']:
            externals.append('t_initial')
        else:
            indeps.append('t_initial')
            # phase.connect('t_initial', 'time.t_initial')

        if time_options['input_duration']:
            externals.append('t_duration')
        else:
            indeps.append('t_duration')
            # phase.connect('t_duration', 'time.t_duration')

        if indeps:
            indep = om.IndepVarComp()

            for var in indeps:
                indep.add_output(var, val=default_vals[var], units=time_units)

            phase.add_subsystem('time_extents', indep, promotes_outputs=['*'])
            comps += ['time_extents']

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
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        phase._check_control_options()

        if phase.control_options:
            control_group = ControlGroup(control_options=phase.control_options,
                                         time_units=phase.time_options['units'],
                                         grid_data=self.grid_data)

            phase.add_subsystem('control_group',
                                subsys=control_group,
                                promotes=['controls:*', 'control_values:*', 'control_rates:*'])

            phase.connect('dt_dstau', 'control_group.dt_dstau')

    def setup_polynomial_controls(self, phase):
        """
        Adds the polynomial control group to the model if any polynomial controls are present.
        """
        if phase.polynomial_control_options:
            sys = PolynomialControlGroup(grid_data=self.grid_data,
                                         polynomial_control_options=phase.polynomial_control_options,
                                         time_units=phase.time_options['units'])
            phase.add_subsystem('polynomial_control_group', subsys=sys,
                                promotes_inputs=['*'], promotes_outputs=['*'])

    def setup_design_parameters(self, phase):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        phase._check_design_parameter_options()

        if phase.design_parameter_options:
            indep = phase.add_subsystem('design_params',
                                        subsys=om.IndepVarComp(),
                                        promotes_outputs=['*'])

            for name, options in iteritems(phase.design_parameter_options):
                src_name = 'design_parameters:{0}'.format(name)

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

                _shape = (1,) + options['shape']

                indep.add_output(name=src_name,
                                 val=options['val'],
                                 shape=_shape,
                                 units=options['units'])

                for tgts, src_idxs in self.get_parameter_connections(name, phase):
                    phase.connect(src_name, [t for t in tgts],
                                  src_indices=src_idxs, flat_src_indices=True)

    def setup_input_parameters(self, phase):
        """
        Adds a InputParameterComp to allow input parameters to be connected from sources
        external to the phase.
        """
        if phase.input_parameter_options:
            passthru = InputParameterComp(input_parameter_options=phase.input_parameter_options)

            phase.add_subsystem('input_params', subsys=passthru, promotes_inputs=['*'],
                                promotes_outputs=['*'])

        for name in phase.input_parameter_options:
            src_name = 'input_parameters:{0}_out'.format(name)

            for tgts, src_idxs in self.get_parameter_connections(name, phase):
                phase.connect(src_name, [t for t in tgts],
                              src_indices=src_idxs, flat_src_indices=True)

    def setup_traj_parameters(self, phase):
        """
        Adds a InputParameterComp to allow input parameters to be connected from sources
        external to the phase.
        """
        if phase.traj_parameter_options:
            passthru = \
                InputParameterComp(input_parameter_options=phase.traj_parameter_options,
                                   traj_params=True)

            phase.add_subsystem('traj_params', subsys=passthru, promotes_inputs=['*'],
                                promotes_outputs=['*'])

        for name, options in iteritems(phase.traj_parameter_options):
            src_name = 'traj_parameters:{0}_out'.format(name)
            for tgts, src_idxs in self.get_parameter_connections(name, phase):
                phase.connect(src_name, [t for t in tgts], src_indices=src_idxs)

    def setup_states(self, phase):
        raise NotImplementedError('Transcription {0} does not implement method '
                                  'setup_states.'.format(self.__class__.__name__))

    def setup_ode(self, phase):
        raise NotImplementedError('Transcription {0} does not implement method '
                                  'setup_ode.'.format(self.__class__.__name__))

    def setup_timeseries_outputs(self, phase):
        raise NotImplementedError('Transcription {0} does not implement method '
                                  'setup_timeseries_outputs.'.format(self.__class__.__name__))

    def setup_boundary_constraints(self, loc, phase):
        """
        Adds BoundaryConstraintComp for initial and/or final boundary constraints if necessary
        and issues appropriate connections.

        Parameters
        ----------
        loc : str
            The kind of boundary constraints being setup.  Must be one of 'initial' or 'final'.
        phase
            The phase object to which this transcription instance applies.

        """
        if loc not in ('initial', 'final'):
            raise ValueError('loc must be one of \'initial\' or \'final\'.')
        bc_comp = None

        bc_dict = phase._initial_boundary_constraints \
            if loc == 'initial' else phase._final_boundary_constraints

        if bc_dict:
            bc_comp = phase.add_subsystem('{0}_boundary_constraints'.format(loc),
                                          subsys=BoundaryConstraintComp(loc=loc))

        for var, options in iteritems(bc_dict):
            con_name = options['constraint_name']

            # Constraint options are a copy of options with constraint_name key removed.
            con_options = options.copy()
            con_options.pop('constraint_name')

            src, shape, units, linear = self._get_boundary_constraint_src(var, loc, phase)

            con_units = options.get('units', None)

            shape = options['shape'] if shape is None else shape
            if shape is None:
                shape = (1,)

            if options['indices'] is not None:
                # Indices are provided, make sure lower/upper/equals are compatible.
                con_shape = (len(options['indices']),)
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

            elif options['lower'] or options['upper'] or options['equals']:
                # Indices not provided, make sure lower/upper/equals have shape of source.
                if options['lower'] and not np.isscalar(options['lower']) and \
                        np.asarray(options['lower']).shape != shape:
                    raise ValueError('The lower bounds of boundary constraint on {0} are not '
                                     'compatible with its shape, and no indices were '
                                     'provided.'.format(var))

                if options['upper'] and not np.isscalar(options['upper']) and \
                        np.asarray(options['upper']).shape != shape:
                    raise ValueError('The upper bounds of boundary constraint on {0} are not '
                                     'compatible with its shape, and no indices were '
                                     'provided.'.format(var))

                if options['equals'] and not np.isscalar(options['equals']) \
                        and np.asarray(options['equals']).shape != shape:
                    raise ValueError('The equality boundary constraint value on {0} is not '
                                     'compatible with its shape, and no indices were '
                                     'provided.'.format(var))
                con_shape = (np.prod(shape),)

            size = np.prod(shape)
            con_options['shape'] = shape if shape is not None else con_shape
            con_options['units'] = units if con_units is None else con_units
            con_options['linear'] = linear

            # Build the correct src_indices regardless of shape
            if loc == 'initial':
                src_idxs = np.arange(size, dtype=int).reshape(shape)
            else:
                src_idxs = np.arange(-size, 0, dtype=int).reshape(shape)

            bc_comp._add_constraint(con_name, **con_options)

            phase.connect(src,
                          '{0}_boundary_constraints.{0}_value_in:{1}'.format(loc, con_name),
                          src_indices=src_idxs,
                          flat_src_indices=True)

    def setup_objective(self, phase):
        """
        Find the path of the objective(s) and add the objective using the standard OpenMDAO method.
        """
        for name, options in iteritems(phase._objectives):
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
        Returns a list containing tuples of each path and related indices to which the
        given parameter name is to be connected.

        Parameters
        ----------
        name : str
            The name of the parameter for which connection information is desired.
        phase
            The phase object to which this transcription applies.

        Returns
        -------
        connection_info : list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design variable is to be connected.
        """
        raise NotImplementedError('Transcription {0} does not implement method '
                                  'get_parameter_connections.'.format(self.__class__.__name__))

    def check_config(self, phase, logger):

        for var, options in iteritems(phase._path_constraints):
            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            if var_type == 'ode':
                # Failed to find variable, assume it is in the ODE
                if options['shape'] is None:
                    logger.warning('Unable to infer shape of path constraint \'{0}\' in '
                                   'phase \'{1}\'. Scalar assumed.  If this ODE output is '
                                   'is not scalar, connection errors will '
                                   'result.'.format(var, phase.name))

        for var, options in iteritems(phase._initial_boundary_constraints):
            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            if var_type == 'ode':
                # Failed to find variable, assume it is in the ODE
                if options['shape'] is None:
                    logger.warning('Unable to infer shape of boundary constraint \'{0}\' in '
                                   'phase \'{1}\'. Scalar assumed.  If this ODE output is '
                                   'is not scalar, connection errors will '
                                   'result.'.format(var, phase.name))

        for var, options in iteritems(phase._final_boundary_constraints):
            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            if var_type == 'ode':
                # Failed to find variable, assume it is in the ODE
                if options['shape'] is None:
                    logger.warning('Unable to infer shape of boundary constraint \'{0}\' in '
                                   'phase \'{1}\'. Scalar assumed.  If this ODE output is '
                                   'is not scalar, connection errors will '
                                   'result.'.format(var, phase.name))

        for name, timeseries_options in iteritems(phase._timeseries):
            for var, options in iteritems(phase._timeseries[name]['outputs']):

                # Determine the path to the variable which we will be constraining
                # This is more complicated for path constraints since, for instance,
                # a single state variable has two sources which must be connected to
                # the path component.
                var_type = phase.classify_var(var)

                # Ignore any variables that we've already added (states, times, controls, etc)
                if var_type != 'ode':
                    continue

                # Assume scalar shape here, but check config will warn that it's inferred.
                if options['shape'] is None:
                    logger.warning('Unable to infer shape of timeseries output \'{0}\' in '
                                   'phase \'{1}\'. Scalar assumed.  If this ODE output is '
                                   'is not scalar, connection errors will '
                                   'result.'.format(var, phase.name))
