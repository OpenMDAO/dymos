from __future__ import print_function, division, absolute_import

from collections import Sequence, OrderedDict
import itertools
from six import iteritems, string_types
import warnings

try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np

from openmdao.api import Group, ParallelGroup, IndepVarComp, DirectSolver, Problem
from openmdao.api import SqliteRecorder, BalanceComp, NewtonSolver, BoundsEnforceLS

from ..utils.constants import INF_BOUND
from ..phases.components.phase_linkage_comp import PhaseLinkageComp
from ..phases.phase_base import PhaseBase
from ..phases.components.input_parameter_comp import InputParameterComp
from ..phases.options import DesignParameterOptionsDictionary, InputParameterOptionsDictionary
from ..phases.simulation.simulation_trajectory import SimulationTrajectory


class Trajectory(Group):
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
        self.options.declare('phase_linkages', default='constrained',
                             values=('solved', 'constrained'),
                             desc="Method to handle linkages. Set to 'constrained' (default) to "
                                  "let the optimizer handle them, or set to 'solved' to create a "
                                  "BalanceComponent for a solver to converge.")

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

    def add_input_parameter(self, name, target_params=None, val=0.0, units=0):
        """
        Add a design parameter (static control) to the trajectory.

        Parameters
        ----------
        name : str
            Name of the design parameter.
        val : float or ndarray
            Default value of the design parameter at all nodes.
        targets : dict or None
            If None, then the design parameter will be connected to the controllable parameter
            in the ODE of each phase.  For each phase where no such controllable parameter exists,
            a warning will be issued.  If targets is given as a dict, the dict should provide
            the relevant phase names as keys, each associated with the respective controllable
            parameteter as a value.
        units : str or None or 0
            Units in which the design parameter is defined.  If 0, use the units declared
            for the parameter in the ODE.
        """
        if name in self.input_parameter_options:
            raise ValueError('{0} has already been added as an input parameter.'.format(name))

        self.input_parameter_options[name] = InputParameterOptionsDictionary()

        self.input_parameter_options[name]['val'] = val
        self.input_parameter_options[name]['target_params'] = target_params

        if units != 0:
            self.input_parameter_options[name]['units'] = units

    def add_design_parameter(self, name, target_params=None, val=0.0, units=0, opt=True,
                             lower=None, upper=None, scaler=None, adder=None,
                             ref=None, ref0=None):
        """
        Add a design parameter (static control) to the trajectory.

        Parameters
        ----------
        name : str
            Name of the design parameter.
        val : float or ndarray
            Default value of the design parameter at all nodes.
        target_params : dict or None
            If None, then the design parameter will be connected to the controllable parameter
            in the ODE of each phase.  For each phase where no such controllable parameter exists,
            a warning will be issued.  If targets is given as a dict, the dict should provide
            the relevant phase names as keys, each associated with the respective controllable
            parameteter as a value.
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

        """
        if name in self.design_parameter_options:
            raise ValueError('{0} has already been added as a design parameter.'.format(name))

        self.design_parameter_options[name] = DesignParameterOptionsDictionary()

        # Don't allow the user to provide desvar options if the design parameter is not a desvar
        if not opt:
            illegal_options = []
            if lower is not None:
                illegal_options.append('lower')
            if upper is not None:
                illegal_options.append('upper')
            if scaler is not None:
                illegal_options.append('scaler')
            if adder is not None:
                illegal_options.append('adder')
            if ref is not None:
                illegal_options.append('ref')
            if ref0 is not None:
                illegal_options.append('ref0')
            if illegal_options:
                msg = 'Invalid options for non-optimal/input design parameter "{0}":'.format(name) \
                      + ', '.join(illegal_options)
                warnings.warn(msg, RuntimeWarning)

        self.design_parameter_options[name]['val'] = val
        self.design_parameter_options[name]['opt'] = opt
        self.design_parameter_options[name]['target_params'] = target_params
        self.design_parameter_options[name]['lower'] = lower
        self.design_parameter_options[name]['upper'] = upper
        self.design_parameter_options[name]['scaler'] = scaler
        self.design_parameter_options[name]['adder'] = adder
        self.design_parameter_options[name]['ref'] = ref
        self.design_parameter_options[name]['ref0'] = ref0

        if units != 0:
            self.design_parameter_options[name]['units'] = units

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

            # Connect the input parameter to its target in each phase
            src_name = 'input_parameters:{0}_out'.format(name)

            target_params = options['target_params']
            for phase_name, phs in iteritems(self._phases):
                tgt_param_name = target_params.get(phase_name, None) \
                    if isinstance(target_params, dict) else name
                if tgt_param_name:
                    if tgt_param_name not in phs.traj_parameter_options:
                        phs._add_traj_parameter(tgt_param_name, val=options['val'],
                                                units=options['units'])
                    tgt = '{0}.traj_parameters:{1}'.format(phase_name, tgt_param_name)
                    self.connect(src_name=src_name, tgt_name=tgt)

    def _setup_design_parameters(self):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        if self.design_parameter_options:
            indep = self.add_subsystem('design_params', subsys=IndepVarComp(),
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

            # Connect the design parameter to its target in each phase
            src_name = 'design_parameters:{0}'.format(name)

            target_params = options['target_params']
            for phase_name, phs in iteritems(self._phases):
                tgt_param_name = target_params.get(phase_name, None) \
                    if isinstance(target_params, dict) else name
                if tgt_param_name:
                    if tgt_param_name not in phs.traj_parameter_options:
                        phs._add_traj_parameter(tgt_param_name, val=options['val'],
                                                units=options['units'])
                    tgt = '{0}.traj_parameters:{1}'.format(phase_name, tgt_param_name)
                    self.connect(src_name=src_name, tgt_name=tgt)

    def _setup_linkages(self):
        if self.options['phase_linkages'] == 'constrained':
            link_comp = self.add_subsystem('linkages', PhaseLinkageComp())

        print('--- Linkage Report [{0}] ---'.format(self.pathname))

        for pair, vars in iteritems(self._linkages):
            phase_name1, phase_name2 = pair
            p1 = self._phases[phase_name1]
            p2 = self._phases[phase_name2]

            print('  ', phase_name1, '   ', phase_name2)

            p1_states = set([key for key in p1.state_options])
            p2_states = set([key for key in p2.state_options])

            p1_controls = set([key for key in p1.control_options])
            p2_controls = set([key for key in p2.control_options])

            p1_design_parameters = set([key for key in p1.design_parameter_options])
            p2_design_parameters = set([key for key in p2.design_parameter_options])

            varnames = vars.keys()
            linkage_name = '{0}|{1}'.format(phase_name1, phase_name2)
            max_varname_length = max(len(name) for name in varnames)

            units_map = {}
            for var, options in iteritems(vars):
                if var in p1_states:
                    units_map[var] = p1.state_options[var]['units']
                elif var in p1_controls:
                    units_map[var] = p1.control_options[var]['units']
                elif var == 'time':
                    units_map[var] = p1.time_options['units']
                else:
                    units_map[var] = None

            if self.options['phase_linkages'] == 'constrained':
                link_comp.add_linkage(name=linkage_name,
                                      vars=varnames,
                                      units=units_map)

            for var, options in iteritems(vars):
                loc1, loc2 = options['locs']

                if var in p1_states:
                    source1 = 'states:{0}{1}'.format(var, loc1)
                elif var in p1_controls:
                    source1 = 'controls:{0}{1}'.format(var, loc1)
                elif var == 'time':
                    source1 = '{0}{1}'.format(var, loc1)
                elif var in p1_design_parameters:
                    source1 = 'design_parameters:{0}'.format(var)

                if self.options['phase_linkages'] == 'constrained':
                    self.connect('{0}.{1}'.format(phase_name1, source1),
                                 'linkages.{0}_{1}:lhs'.format(linkage_name, var))

                if var in p2_states:
                    source2 = 'states:{0}{1}'.format(var, loc2)
                elif var in p2_controls:
                    source2 = 'controls:{0}{1}'.format(var, loc2)
                elif var == 'time':
                    source2 = '{0}{1}'.format(var, loc2)
                elif var in p2_design_parameters:
                    source2 = 'design_parameters:{0}'.format(var)

                if self.options['phase_linkages'] == 'constrained':
                    self.connect('{0}.{1}'.format(phase_name2, source2),
                                 'linkages.{0}_{1}:rhs'.format(linkage_name, var))

                if self.options['phase_linkages'] == 'solved':

                    if var == 'time':
                        path = 'time_extents.initial_state_continuity:t_initial'

                        self.connect('{0}.{1}'.format(phase_name1, source1),
                                     '{0}.{1}'.format(phase_name2, path))
                    else:
                        path1 = 'initial_conditions.initial_value:{0}'.format(var)
                        path2 = 'collocation_constraint.initial_state_continuity:{0}'.format(var)

                        self.connect('{0}.{1}'.format(phase_name1, source1),
                                     '{0}.{1}'.format(phase_name2, path1))

                        self.connect('{0}.{1}'.format(phase_name2, source2),
                                     '{0}.{1}'.format(phase_name2, path2))

                print('       {0:<{2}s} --> {1:<{2}s}'.format(source1, source2,
                                                              max_varname_length + 9))

        print('----------------------------')

    def setup(self):
        """
        Setup the Trajectory Group.
        """
        super(Trajectory, self).setup()

        if self.design_parameter_options:
            self._setup_design_parameters()

        if self.input_parameter_options:
            self._setup_input_parameters()

        phases_group = self.add_subsystem('phases', subsys=ParallelGroup(), promotes_inputs=['*'],
                                          promotes_outputs=['*'])

        for name, phs in iteritems(self._phases):
            g = phases_group.add_subsystem(name, phs, **self._phase_add_kwargs[name])
            # DirectSolvers were moved down into the phases for use with MPI
            g.linear_solver = DirectSolver()

        if self._linkages:
            self._setup_linkages()

        if self.options['phase_linkages'] == 'solved':
            newton = self.nonlinear_solver = NewtonSolver()
            newton.options['solve_subsystems'] = True
            newton.options['iprint'] = 0
            newton.linesearch = BoundsEnforceLS()

            self.linear_solver = DirectSolver()

    def link_phases(self, phases, vars=None, locs=('++', '--')):
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

            if '*' in _vars:
                p1_states = set([key for key in self._phases[phase1_name].state_options])
                p2_states = set([key for key in self._phases[phase2_name].state_options])
                implicitly_linked_vars = p1_states.intersection(p2_states)
                implicitly_linked_vars.add('time')
            else:
                implicitly_linked_vars = set()

            explicitly_linked_vars = [var for var in _vars if var != '*']

            for var in sorted(implicitly_linked_vars.union(explicitly_linked_vars)):
                self._linkages[phase1_name, phase2_name][var] = {'locs': locs, 'units': None}

    def simulate(self, times='all', record=True, record_file=None, time_units='s'):
        """
        Simulate the Trajectory using scipy.integrate.solve_ivp.

        Parameters
        ----------
        times : str or Sequence of float
            Times at which outputs of the simulation are requested.  If given as a str, it should
            be one of the node subsets (default is 'all').  If given as a sequence, output will
            be provided at those times *in addition to times at the boundary of each segment*.
        record_file : str or None
            If recording is enabled, the name of the file to which the results will be recorded.
            If None, use the default filename '<phase_name>_sim.db'.
        record : bool
            If True, recording the results of the simulation is enabled.
        time_units : str
            Units in which times are specified, if numeric.

        Returns
        -------
        problem
            An OpenMDAO Problem in which the simulation is implemented.  This Problem interface
            can be interrogated to obtain timeseries outputs in the same manner as other Phases
            to obtain results at the requested times.
        """
        sim_traj = SimulationTrajectory(phases=self._phases, times=times, time_units=time_units)

        sim_traj.design_parameter_options.update(self.design_parameter_options)
        sim_traj.input_parameter_options.update(self.input_parameter_options)

        sim_prob = Problem(model=Group())

        sim_prob.model.add_subsystem(self.name, sim_traj)

        if record:
            filename = '{0}_sim.sql'.format(self.name) if record_file is None else record_file
            rec = SqliteRecorder(filename)
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
                op = traj_op_dict['{0}.input_params.input_parameters:'
                                  '{1}_out'.format(self.pathname, name)]
                var_name = '{0}.input_parameters:{1}'.format(self.name, name)
                sim_prob[var_name] = op['value'][0, ...]

        for phase_name, phs in iteritems(self._phases):

            op_dict = dict([(name, opts) for (name, opts) in phs.list_outputs(units=True,
                                                                              out_stream=None)])

            # Assign initial state values
            for name, options in iteritems(phs.state_options):
                op = op_dict['{0}.timeseries.states:{1}'.format(phs.pathname, name)]
                tgt_var = '{0}.{1}.initial_states:{2}'.format(self.name, phase_name, name)
                sim_prob[tgt_var] = op['value'][0, ...]

            # Assign control values at all nodes
            for name, options in iteritems(phs.control_options):
                op = op_dict['{0}.control_interp_comp.control_values:'
                             '{1}'.format(phs.pathname, name)]
                var_name = '{0}.{1}.implicit_controls:{2}'.format(self.name, phase_name, name)
                sim_prob[var_name] = op['value']

            # Assign design parameter values
            for name, options in iteritems(phs.design_parameter_options):
                op = op_dict['{0}.design_params.design_parameters:{1}'.format(phs.pathname, name)]
                var_name = '{0}.{1}.design_parameters:{2}'.format(self.name, phase_name, name)
                sim_prob[var_name] = op['value'][0, ...]

            # Assign input parameter values
            for name, options in iteritems(phs.input_parameter_options):
                op = op_dict['{0}.input_params.input_parameters:{1}_out'.format(phs.pathname, name)]
                var_name = '{0}.{1}.input_parameters:{2}'.format(self.name, phase_name, name)
                sim_prob[var_name] = op['value'][0, ...]

        print('\nSimulating trajectory {0}'.format(self.pathname))
        sim_prob.run_model()
        print('Done simulating phase {0}'.format(self.pathname))

        return sim_prob
