from __future__ import print_function, division, absolute_import

from collections import OrderedDict
from collections.abc import Sequence, Iterable
import itertools
import warnings
from six import iteritems, string_types

try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np

import openmdao.api as om

from ..utils.constants import INF_BOUND

from ..transcriptions.common import InputParameterComp, PhaseLinkageComp
from ..phase.options import TrajDesignParameterOptionsDictionary, \
    TrajInputParameterOptionsDictionary


_unspecified = object()


class Trajectory(om.Group):
    """
    A Trajectory object serves as a container for one or more Phases, as well as the linkage
    conditions between phases.
    """
    def __init__(self, **kwargs):
        super(Trajectory, self).__init__(**kwargs)

        self.input_parameter_options = {}
        self.design_parameter_options = {}
        self._linkages = OrderedDict()
        self._phases = OrderedDict()
        self._phase_add_kwargs = {}

    def initialize(self):
        """
        Declare any options for Trajectory.
        """
        self.options.declare('sim_mode', types=bool, default=False,
                             desc='Used internally by Dymos when invoking simulate on a trajectory')

    def add_phase(self, name, phase, **kwargs):
        """
        Add a phase to the trajectory.

        Phases will be added to the Trajectory's `phases` subgroup.

        Parameters
        ----------
        name : str
            The name of the phase being added.
        phase : dymos Phase object
            The Phase object to be added.

        Returns
        -------
        phase : PhaseBase
            The Phase object added to the trajectory.
        """
        self._phases[name] = phase
        self._phase_add_kwargs[name] = kwargs
        return phase

    def add_input_parameter(self, name, units, val=_unspecified, desc=_unspecified,
                            targets=_unspecified, custom_targets=_unspecified,
                            shape=_unspecified, dynamic=_unspecified):
        """
        Add an input parameter to the trajectory.

        Parameters
        ----------
        name : str
            Name of the input parameter.
        val : float or ndarray
            Default value of the input parameter at all nodes.
        desc : str
            A description of the input parameter.
        targets : dict or None
            A dictionary mapping the name of each phase in the trajectory to a sequence of ODE
            targets for this parameter in each phase.
        units : str or None or 0
            Units in which the input parameter is defined.  If 0, use the units declared
            for the parameter in the ODE.
        shape : Sequence of int
            The shape of the input parameter.
        dynamic : bool
            True if the targets in the ODE may be dynamic (if the inputs are sized to the number
            of nodes) else False.
        """
        if name not in self.input_parameter_options:
            self.input_parameter_options[name] = TrajInputParameterOptionsDictionary()

        if units is not _unspecified:
            self.input_parameter_options[name]['units'] = units

        if val is not _unspecified:
            self.input_parameter_options[name]['val'] = val

        if desc is not _unspecified:
            self.input_parameter_options[name]['desc'] = desc

        if targets is not _unspecified:
            if isinstance(targets, string_types):
                self.input_parameter_options[name]['targets'] = (targets,)
            else:
                self.input_parameter_options[name]['targets'] = targets

        if custom_targets is not _unspecified:
            warnings.warn('Option custom_targets is now targets, and should provide the ode '
                          'targets for the parameter in each phase', DeprecationWarning)

            if isinstance(custom_targets, string_types):
                self.input_parameter_options[name]['targets'] = (custom_targets,)
            else:
                self.input_parameter_options[name]['targets'] = custom_targets

        if shape is not _unspecified:
            self.input_parameter_options[name]['shape'] = shape

        if dynamic is not _unspecified:
            self.input_parameter_options[name]['dynamic'] = dynamic

    def add_design_parameter(self, name, units, val=_unspecified, desc=_unspecified, opt=_unspecified,
                             targets=_unspecified, custom_targets=_unspecified,
                             lower=_unspecified, upper=_unspecified, scaler=_unspecified,
                             adder=_unspecified, ref0=_unspecified, ref=_unspecified,
                             shape=_unspecified, dynamic=_unspecified):
        """
        Add a design parameter (static control) to the trajectory.

        Parameters
        ----------
        name : str
            Name of the design parameter.
        val : float or ndarray
            Default value of the design parameter at all nodes.
        desc : str
            A description of the design parameter.
        targets : dict or None
            If None, then the design parameter will be connected to the controllable parameter
            in the ODE of each phase.  For each phase where no such controllable parameter exists,
            a warning will be issued.  If targets is given as a dict, the dict should provide
            the relevant phase names as keys, each associated with the respective controllable
            parameter as a value.
        custom_targets : dict or None
            By default, the input parameter will be connect to the parameter/targets of the given
            name in each phase.  This argument can be used to override that behavior on a phase
            by phase basis.
        units : str or None or 0
            Units in which the design parameter is defined.  If 0, use the units declared
            for the parameter in the ODE.
        opt : bool
            If True (default) the value(s) of this design parameter will be design variables in
            the optimization problem, in the path
            'traj_name.design_params.design_parameters:name'.  If False, the this design
            parameter will still be owned by an IndepVarComp in the phase, but it will not be a
            design variable in the optimization.
        lower : float or ndarray
            The lower bound of the design parameter value.
        upper : float or ndarray
            The upper bound of the design parameter value.
        scaler : float or ndarray
            The scaler of the design parameter value for the optimizer.
        adder : float or ndarray
            The adder of the design parameter value for the optimizer.
        ref0 : float or ndarray
            The zero-reference value of the design parameter for the optimizer.
        ref : float or ndarray
            The unit-reference value of the design parameter for the optimizer.
        shape : Sequence of int
            The shape of the design parameter.
        dynamic : bool
            True if the targets in the ODE may be dynamic (if the inputs are sized to the number
            of nodes) else False.
        """
        if name not in self.design_parameter_options:
            self.design_parameter_options[name] = TrajDesignParameterOptionsDictionary()

        if units is not _unspecified:
            self.design_parameter_options[name]['units'] = units

        if opt is not _unspecified:
            self.design_parameter_options[name]['opt'] = opt

        if val is not _unspecified:
            self.design_parameter_options[name]['val'] = val

        if desc is not _unspecified:
            self.design_parameter_options[name]['desc'] = desc

        if lower is not _unspecified:
            self.design_parameter_options[name]['lower'] = lower

        if upper is not _unspecified:
            self.design_parameter_options[name]['upper'] = upper

        if scaler is not _unspecified:
            self.design_parameter_options[name]['scaler'] = scaler

        if adder is not _unspecified:
            self.design_parameter_options[name]['adder'] = adder

        if ref0 is not _unspecified:
            self.design_parameter_options[name]['ref0'] = ref0

        if ref is not _unspecified:
            self.design_parameter_options[name]['ref'] = ref

        if targets is not _unspecified:
            if isinstance(targets, string_types):
                self.design_parameter_options[name]['targets'] = (targets,)
            else:
                self.design_parameter_options[name]['targets'] = targets

        if custom_targets is not _unspecified:
            warnings.warn('Option custom_targets is now targets, and should provide the ode '
                          'targets for the parameter in each phase', DeprecationWarning)

            if isinstance(custom_targets, string_types):
                self.design_parameter_options[name]['targets'] = (custom_targets,)
            else:
                self.design_parameter_options[name]['targets'] = custom_targets

        if shape is not _unspecified:
            self.design_parameter_options[name]['shape'] = shape

        if dynamic is not _unspecified:
            self.design_parameter_options[name]['dynamic'] = dynamic

    def _setup_input_parameters(self):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        if self.input_parameter_options:
            passthru = InputParameterComp(input_parameter_options=self.input_parameter_options)

            self.add_subsystem('input_params', subsys=passthru, promotes_inputs=['*'],
                               promotes_outputs=['*'])

            for name, options in iteritems(self.input_parameter_options):

                tgts = options['targets']

                if tgts is None:
                    # The user is implicitly connecting to input parameters in all phases.
                    # No need to create input parameters in each phase.
                    continue

                for phase_name, phs in iteritems(self._phases):

                    if tgts[phase_name] is None:
                        # Targets for this phase are explicitly None.
                        # Skip addition of input parameter to this phase.
                        continue
                    elif isinstance(tgts[phase_name], string_types):
                        # User specified name of an input parameter in the phase for this parameter.
                        # Skip addition of input parameter to this phase.
                        continue
                    elif isinstance(tgts[phase_name], Sequence):
                        # User specified ODE targets for this parameter in this phase.
                        # We need to add an input parameter to this phase.

                        # The default target in the phase is name unless otherwise specified.
                        kwargs = {'dynamic': options['dynamic'],
                                  'units': options['units'],
                                  'val': options['val'],
                                  'targets': tgts[phase_name]}

                        if not self.options['sim_mode']:
                            phs.add_input_parameter(name, **kwargs)

    def _setup_design_parameters(self):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        if self.design_parameter_options:
            indep = self.add_subsystem('design_params', subsys=om.IndepVarComp(),
                                       promotes_outputs=['*'])

            for name, options in iteritems(self.design_parameter_options):
                if options['opt']:
                    lb = -INF_BOUND if options['lower'] is None else options['lower']
                    ub = INF_BOUND if options['upper'] is None else options['upper']

                    self.add_design_var(name='design_parameters:{0}'.format(name),
                                        lower=lb,
                                        upper=ub,
                                        scaler=options['scaler'],
                                        adder=options['adder'],
                                        ref0=options['ref0'],
                                        ref=options['ref'])

                indep.add_output(name='design_parameters:{0}'.format(name),
                                 val=options['val'],
                                 shape=(1, np.prod(options['shape'])),
                                 units=options['units'])

                tgts = options['targets']

                if tgts is None:
                    # The user is implicitly connecting to input parameters in all phases.
                    # No need to create input parameters in each phase.
                    continue

                for phase_name, phs in iteritems(self._phases):
                    if tgts[phase_name] is None:
                        # Targets for this phase are explicitly None.
                        # Skip addition of input parameter to this phase.
                        continue
                    elif isinstance(tgts[phase_name], string_types):
                        # User specified name of an input parameter in the phase for this parameter.
                        # Skip addition of input parameter to this phase.
                        continue
                    elif isinstance(tgts[phase_name], Sequence):
                        # User specified ODE targets for this parameter in this phase.
                        # We need to add an input parameter to this phase.

                        # The default target in the phase is name unless otherwise specified.
                        kwargs = {'dynamic': options['dynamic'],
                                  'units': options['units'],
                                  'val': options['val'],
                                  'targets': tgts[phase_name]}

                        if not self.options['sim_mode']:
                            phs.add_input_parameter(name, **kwargs)

    def _setup_linkages(self):
        link_comp = None

        for pair, vars in iteritems(self._linkages):
            phase_name1, phase_name2 = pair

            for name in pair:
                if name not in self._phases:
                    raise ValueError('Invalid linkage.  Phase \'{0}\' does not exist in '
                                     'trajectory \'{1}\'.'.format(name, self.pathname))

            p1 = self._phases[phase_name1]
            p2 = self._phases[phase_name2]

            p1_states = set([key for key in p1.state_options])
            p2_states = set([key for key in p2.state_options])

            p1_controls = set([key for key in p1.control_options])

            # Dict of vars that expands '*' to include time and states
            _vars = {}
            for var in sorted(vars.keys()):
                if var == '*':
                    _vars['time'] = vars[var].copy()
                    for state in p2_states:
                        _vars[state] = vars[var].copy()
                else:
                    _vars[var] = vars[var].copy()

            units_map = {}
            shape_map = {}
            vars_to_constrain = []

            for var, options in iteritems(_vars):
                if options['connected']:
                    # If this is a state, and we are linking it, we need to do some checks.
                    if var in p2_states:
                        # Trajectory linkage modifies these options in connected states.
                        p2.add_state(var, connected_initial=True)
                    elif var == 'time':
                        p2.set_time_options(input_initial=True)
                else:
                    vars_to_constrain.append(var)
                    if var in p1_states:
                        units_map[var] = p1.state_options[var]['units']
                        shape_map[var] = p1.state_options[var]['shape']
                    elif var in p1_controls:
                        units_map[var] = p1.control_options[var]['units']
                        shape_map[var] = p1.control_options[var]['shape']
                    elif var == 'time':
                        units_map[var] = p1.time_options['units']
                        shape_map[var] = (1,)
                    else:
                        units_map[var] = None
                        shape_map[var] = (1,)

            if vars_to_constrain:
                if not link_comp:
                    link_comp = self.add_subsystem('linkages', PhaseLinkageComp())

                linkage_name = '{0}|{1}'.format(phase_name1, phase_name2)
                link_comp.add_linkage(name=linkage_name,
                                      vars=vars_to_constrain,
                                      shape=shape_map,
                                      units=units_map)

    def setup(self):
        """
        Setup the Trajectory Group.
        """
        super(Trajectory, self).setup()

        if self.design_parameter_options:
            self._setup_design_parameters()

        if self.input_parameter_options:
            self._setup_input_parameters()

        phases_group = self.add_subsystem('phases', subsys=om.ParallelGroup(), promotes_inputs=['*'],
                                          promotes_outputs=['*'])

        for name, phs in iteritems(self._phases):
            g = phases_group.add_subsystem(name, phs, **self._phase_add_kwargs[name])
            # DirectSolvers were moved down into the phases for use with MPI
            g.linear_solver = om.DirectSolver()

            phs.finalize_variables()

        if self._linkages:
            self._setup_linkages()

    def _configure_parameters(self, mode):
        """
        Configure connections from input or design parameters to the appropriate targets
        in each phase.

        Parameters
        ----------
        mode : str
            One of 'input' or 'design' to specify if we are dealing with input parameters or
            design parameters.
        """
        parameter_options = self.input_parameter_options if mode == 'input' \
            else self.design_parameter_options

        for name, options in iteritems(parameter_options):

            # Connect the design parameter to its target in each phase
            suffix = '' if mode == 'design' else '_out'
            src_name = '{0}_parameters:{1}{2}'.format(mode, name, suffix)

            targets = options['targets']

            for phase_name, phs in iteritems(self._phases):

                if targets is None or phase_name not in targets:
                    # Attempt to connect to an input parameter of the same name in the phase, if
                    # it exists.
                    if name in phs.user_input_parameter_options:
                        tgt = '{0}.input_parameters:{1}'.format(phase_name, name)
                    else:
                        continue
                elif targets[phase_name] is None:
                    # Connections to this phase are explicitly omitted
                    continue
                elif isinstance(targets[phase_name], string_types) and \
                        targets[phase_name] in phs.user_input_parameter_options:
                    # Connect to an input parameter with a different name in this phase
                    tgt = '{0}.input_parameters:{1}'.format(phase_name, targets[phase_name])
                elif isinstance(targets[phase_name], Sequence) and \
                        name in phs.user_input_parameter_options:
                    # User gave a list of ODE targets which were passed to the creation of a
                    # new input parameter in setup, just connect to that new input parameter
                    tgt = '{0}.input_parameters:{1}'.format(phase_name, name)
                else:
                    raise ValueError('Unhandled {0} parameter target in '
                                     'phase {1}'.format(mode, phase_name))

                self.connect(src_name=src_name, tgt_name=tgt)

    def _configure_linkages(self):

        print('--- Linkage Report [{0}] ---'.format(self.pathname))

        indent = '    '

        for pair, vars in iteritems(self._linkages):
            phase_name1, phase_name2 = pair

            for name in pair:
                if name not in self._phases:
                    raise ValueError('Invalid linkage.  Phase \'{0}\' does not exist in '
                                     'trajectory \'{1}\'.'.format(name, self.pathname))

            p1 = self._phases[phase_name1]
            p2 = self._phases[phase_name2]

            print(indent * 1, phase_name1, '    ', phase_name2)

            p1_states = set([key for key in p1.state_options])
            p2_states = set([key for key in p2.state_options])

            p1_controls = set([key for key in p1.control_options])
            p2_controls = set([key for key in p2.control_options])

            p1_design_parameters = set([key for key in p1.design_parameter_options])
            p2_design_parameters = set([key for key in p2.design_parameter_options])

            p1_input_parameters = set([key for key in p1.input_parameter_options])
            p2_input_parameters = set([key for key in p2.input_parameter_options])

            # Dict of vars that expands '*' to include time and states
            _vars = {}
            for var in sorted(vars.keys()):
                if var == '*':
                    _vars['time'] = vars[var].copy()
                    for state in p2_states:
                        _vars[state] = vars[var].copy()
                else:
                    _vars[var] = vars[var].copy()

            max_varname_length = max(len(name) for name in _vars.keys())

            units_map = {}
            shape_map = {}
            vars_to_constrain = []

            for var, options in iteritems(_vars):
                if not options['connected']:
                    vars_to_constrain.append(var)
                    if var in p1_states:
                        units_map[var] = p1.state_options[var]['units']
                        shape_map[var] = p1.state_options[var]['shape']
                    elif var in p1_controls:
                        units_map[var] = p1.control_options[var]['units']
                        shape_map[var] = p1.control_options[var]['shape']
                    elif var == 'time':
                        units_map[var] = p1.time_options['units']
                        shape_map[var] = (1,)
                    else:
                        units_map[var] = None
                        shape_map[var] = (1,)

            if vars_to_constrain:

                linkage_name = '{0}|{1}'.format(phase_name1, phase_name2)

            for var, options in iteritems(_vars):
                loc1, loc2 = options['locs']

                if var in p1_states:
                    source1 = 'states:{0}{1}'.format(var, loc1)
                elif var in p1_controls:
                    source1 = 'controls:{0}{1}'.format(var, loc1)
                elif var == 'time':
                    source1 = '{0}{1}'.format(var, loc1)
                elif var in p1_design_parameters:
                    source1 = 'design_parameters:{0}'.format(var)
                elif var in p1_input_parameters:
                    source1 = 'input_parameters:{0}'.format(var)
                else:
                    raise ValueError('Cannot find linkage variable \'{0}\' in '
                                     'phase \'{1}\'.  Only states, time, controls, or parameters '
                                     'may be linked via link_phases.'.format(var, pair[0]))

                if var in p2_states:
                    source2 = 'states:{0}{1}'.format(var, loc2)
                elif var in p2_controls:
                    source2 = 'controls:{0}{1}'.format(var, loc2)
                elif var == 'time':
                    source2 = '{0}{1}'.format(var, loc2)
                elif var in p2_design_parameters:
                    source2 = 'design_parameters:{0}'.format(var)
                elif var in p2_input_parameters:
                    source2 = 'input_parameters:{0}'.format(var)
                else:
                    raise ValueError('Cannot find linkage variable \'{0}\' in '
                                     'phase \'{1}\'.  Only states, time, controls, or parameters '
                                     'may be linked via link_phases.'.format(var, pair[1]))

                if options['connected']:
                    if var == 'time':
                        src = '{0}.{1}'.format(phase_name1, source1)
                        path = 't_initial'
                        self.connect(src, '{0}.{1}'.format(phase_name2, path))
                    else:
                        path = 'initial_states:{0}'.format(var)
                        self.connect('{0}.{1}'.format(phase_name1, source1),
                                     '{0}.{1}'.format(phase_name2, path))
                    print('{3}{0:<{2}s} --> {1:<{2}s}'.format(source1, source2,
                                                              max_varname_length + 9,
                                                              indent*2))
                else:

                    self.connect('{0}.{1}'.format(phase_name1, source1),
                                 'linkages.{0}_{1}:lhs'.format(linkage_name, var))

                    self.connect('{0}.{1}'.format(phase_name2, source2),
                                 'linkages.{0}_{1}:rhs'.format(linkage_name, var))

                    print('{3}{0:<{2}s}  =  {1:<{2}s}'.format(source1, source2,
                                                              max_varname_length + 9,
                                                              indent*2))

        print('----------------------------')

    def configure(self):
        """
        Configure the Trajectory Group.

        This method is used to handle connections to the phases in the Trajectory, since
        setup has already been called on all children of the Trajectory, we can query them for
        variables at this point.
        """
        if self.design_parameter_options:
            self._configure_parameters(mode='design')
        if self.input_parameter_options:
            self._configure_parameters(mode='input')
        if self._linkages:
            self._configure_linkages()

    def link_phases(self, phases, vars=None, locs=('++', '--'), connected=False):
        """
        Specifies that phases in the given sequence are to be assume continuity of the given
        variables.

        This method caches the phase linkages, and may be called multiple times to express more
        complex behavior (branching phases, phases only continuous in some variables, etc).

        The location at which the variables should be coupled in the two phases are provided
        with a two character string:

        - '--' specifies the value at the start of the phase before an initial state or control jump
        - '-+' specifies the value at the start of the phase after an initial state or control jump
        - '+-' specifies the value at the end of the phase before a final state or control jump
        - '++' specifies the value at the end of the phase after a final state or control jump

        Parameters
        ----------
        phases : sequence of str
            The names of the phases in this trajectory to be sequentially linked.
        vars : sequence of str
            The variables in the phases to be linked, or '*'.  Providing '*' will time and all
            states.  Linking control values or rates requires them to be listed explicitly.
        locs : tuple of str.
            A two-element tuple of the two-character location specification.  For every pair in
            phases, the location specification refers to which location in the first phase is
            connected to which location in the second phase.  If the user wishes to specify
            different locations for different phase pairings, those phase pairings must be made
            in separate calls to link_phases.
        units : int, str, dict of {str: str or None}, or None
            The units of the linkage residual.  If an integer (default), then automatically
            determine the units of each variable in the linkage if possible.  Those that cannot
            be determined to be a time, state, control, design parameter, or control rate will
            be assumed to have units None.  If given as a dict, it should map the name of each
            variable in vars to the approprite units.
        connected : bool
            Set to True to directly connect the phases being linked. Otherwise, create constraints
            for the optimizer to solve.

        Examples
        --------
        **Typical Phase Linkage Sequence**

        A typical phase linkage sequence, where all phases use the same ODE (and therefore have
        the same states), and are simply linked sequentially in time.

        >>> t.link_phases(['phase1', 'phase2', 'phase3'])

        **Adding an Additional Linkage**

        If we want some control variable, u, to be continuous in value between phase2 and
        phase3 only, we could subsequently issue the following:

        >>> t.link_phases(['phase2', 'phase3'], vars=['u'])

        **Branching Trajectories**

        For a more complex example, consider the case where we have two phases which branch off
        from the same point, such as the case of a jettisoned stage.  The nominal trajectory
        consists of the phase sequence ['a', 'b', 'c'].  Let phase ['d'] be the phase that tracks
        the jettisoned component to its impact with the ground.  The linkages in this case
        would be defined as:

        >>> t.link_phases(['a', 'b', 'c'])
        >>> t.link_phases(['b', 'd'])

        **Specifying Linkage Locations**

        Phase linkages assume that, for each pair, the state/control values after any discontinuous
        jump in the first phase ('++') are linked to the state/control values before any
        discontinuous jump in the second phase ('--').  The user can override this behavior, but
        they must specify a pair of location strings for each pair given in `phases`.  For instance,
        in the following example phases 'a' and 'b' have the same initial time and state, but
        phase 'c' follows phase 'b'.  Note since there are three phases provided, there are two
        linkages and thus two pairs of location specifiers given.

        >>> t.link_phases(['a', 'b', 'c'], locs=[('--', '--'), ('++', '--')])

        """
        num_links = len(phases) - 1

        if num_links <= 0:
            raise ValueError('Phase sequence must consists of at least two phases.')
        if isinstance(locs, Sequence) and len(locs) != 2:
            raise ValueError('If given, locs must be a sequence of two-element tuples, one pair '
                             'for each phase pair in the phases sequence')

        _vars = ['*'] if vars is None else vars

        # Resolve linkage pairs from the phases sequence
        a, b = itertools.tee(phases)
        next(b, None)
        phase_pairs = zip(a, b)

        for phase1_name, phase2_name in phase_pairs:
            if (phase1_name, phase2_name) not in self._linkages:
                self._linkages[phase1_name, phase2_name] = OrderedDict()

            for var in _vars:
                self._linkages[phase1_name, phase2_name][var] = {'locs': locs, 'units': None,
                                                                 'connected': connected}

    def simulate(self, times_per_seg=10, method='RK45', atol=1.0E-9, rtol=1.0E-9, record_file=None):
        """
        Simulate the Trajectory using scipy.integrate.solve_ivp.

        Parameters
        ----------
        times_per_seg : int or None
            Number of equally spaced times per segment at which output is requested.  If None,
            output will be provided at all Nodes.
        method : str
            The scipy.integrate.solve_ivp integration method.
        atol : float
            Absolute convergence tolerance for scipy.integrate.solve_ivp.
        rtol : float
            Relative convergence tolerance for scipy.integrate.solve_ivp.
        record_file : str or None
            If a string, the file to which the result of the simulation will be saved.
            If None, no record of the simulation will be saved.

        Returns
        -------
        problem
            An OpenMDAO Problem in which the simulation is implemented.  This Problem interface
            can be interrogated to obtain timeseries outputs in the same manner as other Phases
            to obtain results at the requested times.
        """
        sim_traj = Trajectory(sim_mode=True)

        for name, phs in iteritems(self._phases):
            sim_phs = phs.get_simulation_phase(times_per_seg=times_per_seg, method=method,
                                               atol=atol, rtol=rtol)
            sim_traj.add_phase(name, sim_phs)

        sim_traj.design_parameter_options.update(self.design_parameter_options)
        sim_traj.input_parameter_options.update(self.input_parameter_options)

        sim_prob = om.Problem(model=om.Group())

        if self.name:
            sim_prob.model.add_subsystem(self.name, sim_traj)
        else:
            sim_prob.model.add_subsystem('sim_traj', sim_traj)

        if record_file is not None:
            rec = om.SqliteRecorder(record_file)
            sim_prob.model.recording_options['includes'] = ['*.timeseries.*']
            sim_prob.model.add_recorder(rec)

        sim_prob.setup(check=True)

        traj_op_dict = dict([(name, opts) for (name, opts) in self.list_outputs(units=True,
                                                                                out_stream=None)])

        # Assign trajectory design parameter values
        for name, options in iteritems(self.design_parameter_options):
            op = traj_op_dict['{0}.design_params.design_parameters:{1}'.format(self.pathname, name)]
            var_name = '{0}.design_parameters:{1}'.format(self.name, name)
            sim_prob[var_name] = op['value'][0, ...]

        # Assign trajectory input parameter values
        for name, options in iteritems(self.input_parameter_options):
                op = traj_op_dict['{0}.input_params.input_parameters:{1}_out'.format(self.pathname,
                                                                                     name)]
                var_name = '{0}.input_parameters:{1}'.format(self.name, name)
                sim_prob[var_name] = op['value'][0, ...]

        for phase_name, phs in iteritems(sim_traj._phases):
            phs.initialize_values_from_phase(sim_prob, self._phases[phase_name])

        print('\nSimulating trajectory {0}'.format(self.pathname))
        sim_prob.run_model()
        print('Done simulating trajectory {0}'.format(self.pathname))

        return sim_prob
