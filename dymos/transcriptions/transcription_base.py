from __future__ import print_function, division, absolute_import

from collections import Sequence
import warnings

from six import iteritems

import numpy as np

from openmdao.api import IndepVarComp, OptionsDictionary

from ..utils.constants import INF_BOUND


class TranscriptionBase(object):

    def __init__(self, **kwargs):

        self.options = OptionsDictionary()

        self.options.declare('num_segments', types=int, desc='Number of segments')
        self.options.declare('segment_ends', default=None, types=Sequence, allow_none=True,
                             desc='Iterable of locations of segment ends or None for equally'
                                  'spaced segments')
        self.options.declare('order', default=3, types=(int, Sequence),
                             desc='Order of the state transcription')
        self.options.declare('compressed', default=True, types=bool,
                             desc='Use compressed transcription')

        self._declare_options()
        self.initialize()
        self.options.update(kwargs)

    def _declare_options(self):
        pass

    def initialize(self):
        pass

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
            indep = IndepVarComp()

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

    def setup_states(self, phase):
        pass

    def setup_ode(self, phase):
        pass

    def setup_timeseries_outputs(self, phase):
        pass

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

            src, shape, units, linear = self._get_boundary_constraint_src(var, loc)

            con_units = options.get('units', None)

            shape = options['shape'] if shape is None else shape
            if shape is None:
                warnings.warn('\nUnable to infer shape of boundary constraint {0}. Assuming scalar. '
                              '\nIf variable is not scalar, provide shape in '
                              'add_boundary_constraint. \nIn Dymos 1.0 an error will be raised if '
                              'a constrained ODE output shape is not specified in '
                              'add_boundary_constraint.'.format(var), DeprecationWarning)
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

            self.connect(src,
                         '{0}_boundary_constraints.{0}_value_in:{1}'.format(loc, con_name),
                         src_indices=src_idxs,
                         flat_src_indices=True)

    def _get_boundary_constraint_src(self, name, loc, phase):
        raise NotImplementedError('This transcription does not implement _get_boundary_constraint_src')

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

            phase.add_objective(obj_path, ref=options['ref'], ref0=options['ref0'],
                                index=obj_index, adder=options['adder'],
                                scaler=options['scaler'],
                                parallel_deriv_color=options['parallel_deriv_color'],
                                vectorize_derivs=options['vectorize_derivs'])