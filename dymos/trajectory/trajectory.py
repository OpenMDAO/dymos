from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
import itertools
import warnings
try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np

import openmdao.api as om
from openmdao.utils.general_utils import warn_deprecation
from openmdao.utils.mpi import MPI

from ..utils.constants import INF_BOUND

from .options import LinkageOptionsDictionary
from .phase_linkage_comp import PhaseLinkageComp
from ..phase.options import TrajParameterOptionsDictionary
from ..utils.misc import get_rate_units, get_source_metadata, _unspecified


class Trajectory(om.Group):
    """
    A Trajectory object serves as a container for one or more Phases, as well as the linkage
    conditions between phases.
    """
    def __init__(self, **kwargs):
        super(Trajectory, self).__init__(**kwargs)

        self.parameter_options = {}
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

    def add_parameter(self, name, units, val=_unspecified, desc=_unspecified, opt=False,
                      targets=_unspecified, custom_targets=_unspecified,
                      lower=_unspecified, upper=_unspecified, scaler=_unspecified,
                      adder=_unspecified, ref0=_unspecified, ref=_unspecified,
                      shape=_unspecified, dynamic=_unspecified):
        """
        Add a parameter (static control) to the trajectory.

        Parameters
        ----------
        name : str
            Name of the parameter.
        val : float or ndarray
            Default value of the parameter at all nodes.
        desc : str
            A description of the parameter.
        targets : dict or None
            If None, then the parameter will be connected to the controllable parameter
            in the ODE of each phase.  For each phase where no such controllable parameter exists,
            a warning will be issued.  If targets is given as a dict, the dict should provide
            the relevant phase names as keys, each associated with the respective controllable
            parameter as a value.
        custom_targets : dict or None
            By default, the parameter will be connect to the parameter/targets of the given
            name in each phase.  This argument can be used to override that behavior on a phase
            by phase basis.
        units : str or None or 0
            Units in which the parameter is defined.  If 0, use the units declared
            for the parameter in the ODE.
        opt : bool
            If True the value(s) of this parameter will be design variables in
            the optimization problem. The default is False.
        lower : float or ndarray
            The lower bound of the parameter value.
        upper : float or ndarray
            The upper bound of the parameter value.
        scaler : float or ndarray
            The scaler of the parameter value for the optimizer.
        adder : float or ndarray
            The adder of the parameter value for the optimizer.
        ref0 : float or ndarray
            The zero-reference value of the parameter for the optimizer.
        ref : float or ndarray
            The unit-reference value of the parameter for the optimizer.
        shape : Sequence of int
            The shape of the parameter.
        dynamic : bool
            True if the targets in the ODE may be dynamic (if the inputs are sized to the number
            of nodes) else False.
        """
        if name not in self.parameter_options:
            self.parameter_options[name] = TrajParameterOptionsDictionary()

        if units is not _unspecified:
            self.parameter_options[name]['units'] = units

        if opt is not _unspecified:
            self.parameter_options[name]['opt'] = opt

        if val is not _unspecified:
            self.parameter_options[name]['val'] = val

        if desc is not _unspecified:
            self.parameter_options[name]['desc'] = desc

        if lower is not _unspecified:
            self.parameter_options[name]['lower'] = lower

        if upper is not _unspecified:
            self.parameter_options[name]['upper'] = upper

        if scaler is not _unspecified:
            self.parameter_options[name]['scaler'] = scaler

        if adder is not _unspecified:
            self.parameter_options[name]['adder'] = adder

        if ref0 is not _unspecified:
            self.parameter_options[name]['ref0'] = ref0

        if ref is not _unspecified:
            self.parameter_options[name]['ref'] = ref

        if targets is not _unspecified:
            if isinstance(targets, str):
                self.parameter_options[name]['targets'] = (targets,)
            else:
                self.parameter_options[name]['targets'] = targets

        if custom_targets is not _unspecified:
            warnings.warn('Option custom_targets is now targets, and should provide the ode '
                          'targets for the parameter in each phase', DeprecationWarning)

            if isinstance(custom_targets, str):
                self.parameter_options[name]['targets'] = (custom_targets,)
            else:
                self.parameter_options[name]['targets'] = custom_targets

        if shape is not _unspecified:
            self.parameter_options[name]['shape'] = shape

        if dynamic is not _unspecified:
            self.parameter_options[name]['dynamic'] = dynamic

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
        msg = "DesignParameters and InputParameters are being replaced by Parameters in  " + \
            "Dymos 1.0.0. Please use add_parameter or set_parameter_options to remove this " + \
            "deprecation warning."
        warn_deprecation(msg)
        self.add_parameter(name, units, val=val, desc=desc, targets=targets, shape=shape,
                           dynamic=dynamic)

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
            If True (default) the value(s) of this parameter will be design variables in
            the optimization problem.
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
        msg = "DesignParameters and InputParameters are being replaced by Parameters in  " + \
            "Dymos 1.0.0. Please use add_parameter or set_parameter_options to remove this " + \
            "deprecation warning."
        warn_deprecation(msg)
        self.add_parameter(name, units, val, desc, opt, targets, custom_targets, lower, upper,
                           scaler, adder, ref0, ref, shape, dynamic)

    def _setup_parameters(self):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        if self.parameter_options:
            for name, options in self.parameter_options.items():

                if options['opt']:
                    lb = -INF_BOUND if options['lower'] is None else options['lower']
                    ub = INF_BOUND if options['upper'] is None else options['upper']

                    self.add_design_var(name='parameters:{0}'.format(name),
                                        lower=lb,
                                        upper=ub,
                                        scaler=options['scaler'],
                                        adder=options['adder'],
                                        ref0=options['ref0'],
                                        ref=options['ref'])

                tgts = options['targets']

                if tgts is None:
                    # The user is implicitly connecting to input parameters in all phases.
                    # No need to create input parameters in each phase.
                    continue

                for phase_name, phs in self._phases.items():
                    if phase_name not in tgts or isinstance(tgts[phase_name], str):
                        # If user omitted this phase from targets, we will try to connect
                        # to an existing input parameter in the phase.
                        # If the target for this phase is a string, assume the user specified the
                        # name of an input parameter in the phase for this parameter.
                        # Skip addition of input parameter to this phase.
                        continue
                    elif tgts[phase_name] is None:
                        # Targets for this phase are explicitly None.
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
                            phs.add_parameter(name, **kwargs)

    def _setup_linkages(self):
        has_linkage_constraints = False
        for pair, var_dict in self._linkages.items():

            for name in pair:
                if name not in self._phases:
                    raise ValueError(f'Invalid linkage.  Phase \'{name}\' does not exist in '
                                     f'trajectory \'{self.pathname}\'.')

            phase2 = self._phases[pair[1]]

            for var_pair, options in var_dict.items():
                var1, var2 = var_pair

                if options['connected']:
                    if var2 == 'time':
                        phase2.set_time_options(input_initial=True)
                    elif var2 == '*':
                        phase2.set_time_options(input_initial=True)
                        for state_name in phase2.state_options:
                            phase2.set_state_options(state_name, conected_initial=True)
                    elif var2 in phase2.state_options:
                        phase2.set_state_options(var2, connected_initial=True)
                else:
                    has_linkage_constraints = True

        if has_linkage_constraints:
            self.add_subsystem('linkages', PhaseLinkageComp())

    def setup(self):
        """
        Setup the Trajectory Group.
        """
        super(Trajectory, self).setup()

        if self.parameter_options:
            self._setup_parameters()

        phases_group = self.add_subsystem('phases', subsys=om.ParallelGroup())

        for name, phs in self._phases.items():
            g = phases_group.add_subsystem(name, phs, **self._phase_add_kwargs[name])
            # DirectSolvers were moved down into the phases for use with MPI
            g.linear_solver = om.DirectSolver()

        if self._linkages:
            self._setup_linkages()

    def _configure_parameters(self):
        """
        Configure connections from input or design parameters to the appropriate targets
        in each phase.
        """
        parameter_options = self.parameter_options
        promoted_inputs = []

        for name, options in parameter_options.items():
            prom_name = f'parameters:{name}'
            targets = options['targets']

            # For each phase, use introspection to get the units and shape.
            # If units do not match across all phases, require user to set them.
            # If shapes do not match across all phases, this is an error.
            tgt_units = {}
            tgt_shapes = {}

            for phase_name, phs in self._phases.items():

                if targets is None or phase_name not in targets:
                    # Attempt to connect to an input parameter of the same name in the phase, if
                    # it exists.
                    if name in phs.parameter_options:
                        tgt = f'{phase_name}.parameters:{name}'
                        tgt_shapes[phs.name] = phs.parameter_options[name]['shape']
                        tgt_units[phs.name] = phs.parameter_options[name]['units']
                    else:
                        continue
                elif targets[phase_name] is None:
                    # Connections to this phase are explicitly omitted
                    continue
                elif isinstance(targets[phase_name], str) and \
                        targets[phase_name] in phs.parameter_options:
                    # Connect to an input parameter with a different name in this phase
                    tgt = f'{phase_name}.parameters:{targets[phase_name]}'
                    tgt_shapes[phs.name] = phs.parameter_options[targets[phase_name]]['shape']
                    tgt_units[phs.name] = phs.parameter_options[targets[phase_name]]['units']
                elif isinstance(targets[phase_name], Sequence) and \
                        name in phs.parameter_options:
                    # User gave a list of ODE targets which were passed to the creation of a
                    # new input parameter in setup, just connect to that new input parameter
                    tgt = f'{phase_name}.parameters:{name}'
                    tgt_shapes[phs.name] = phs.parameter_options[name]['shape']
                    tgt_units[phs.name] = phs.parameter_options[name]['units']
                else:
                    raise ValueError(f'Unhandled parameter target in '
                                     f'phase {phase_name}')

                promoted_inputs.append(tgt)
                self.promotes('phases', inputs=[(tgt, prom_name)])

            if len(set(tgt_shapes.values())) == 1:
                options['shape'] = next(iter(tgt_shapes.values()))
            else:
                raise ValueError(f'Parameter {name} in Trajectory {self.pathname} is connected to '
                                 f'targets in multiple phases that have different shapes.')

            if len(set(tgt_units.values())) != 1:
                options['units'] = next(iter(tgt_units))
            else:
                ValueError(f'Parameter {name} in Trajectory {self.pathname} is connected to '
                           f'targets in multiple phases that have different units. You must '
                           f'explicitly provide units for the parameter since they cannot be '
                           f'inferred.')

            val = options['val']
            _shape = options['shape']
            shaped_val = np.broadcast_to(val, _shape)

            self.set_input_defaults(name=prom_name,
                                    val=shaped_val,
                                    units=options['units'])

        return promoted_inputs

    def _configure_phase_options_dicts(self):
        """
        Called during configure if we are under MPI. Loops over all phases and broacasts the shape
        and units options to all procs for all dymos variables.
        """
        for name, phase in self._phases.items():
            all_dicts = [phase.state_options, phase.control_options, phase.parameter_options,
                         phase.polynomial_control_options]

            for opt_dict in all_dicts:
                for options in opt_dict.values():

                    all_ranks = self.comm.allgather(options['shape'])
                    for item in all_ranks:
                        if item not in [None, _unspecified]:
                            options['shape'] = item
                            break
                    else:
                        raise RuntimeError('Unexpectedly found no valid shape.')

                    all_ranks = self.comm.allgather(options['units'])
                    for item in all_ranks:
                        if item is not _unspecified:
                            options['units'] = item
                            break
                    else:
                        raise RuntimeError('Unexpectedly found no valid units.')

    def _update_linkage_options_configure(self, linkage_options):
        """
        Called during configure to return the source paths, units, and shapes of variables
        in linkages.

        Parameters
        ----------
        phases : Sequence of (str, str)
            The names of the phases involved in the linkage.
        vars : Sequence of (str, str)
            The paths of the variables involved in the linkage.
        options : dict
            The linkage options set during `add_linkage_constraint`.

        Returns
        -------

        """
        phase_name_a = linkage_options['phase_a']
        phase_name_b = linkage_options['phase_b']
        var_a = linkage_options['var_a']
        var_b = linkage_options['var_b']
        loc_a = linkage_options['loc_a']
        loc_b = linkage_options['loc_b']

        info_str = f'Error in linking {var_a} from {phase_name_a} to {var_b} in {phase_name_b}'

        phase_a = self._get_subsystem(f'phases.{phase_name_a}')
        phase_b = self._get_subsystem(f'phases.{phase_name_b}')

        phases = {'a': phase_a, 'b': phase_b}

        classes = {'a': phase_a.classify_var(var_a),
                   'b': phase_b.classify_var(var_b)}

        sources = {'a': None, 'b': None}
        vars = {'a': var_a, 'b': var_b}
        units = {'a': _unspecified, 'b': _unspecified}
        shapes = {'a': _unspecified, 'b': _unspecified}

        for i in ('a', 'b'):
            if classes[i] == 'time':
                sources[i] = 'timeseries.time'
                shapes[i] = (1,)
                units[i] = phases[i].time_options['units']
            elif classes[i] == 'time_phase':
                sources[i] = 'timeseries.time_phase'
                units[i] = phases[i].time_options['units']
                shapes[i] = (1,)
            elif classes[i] == 'state':
                sources[i] = f'timeseries.states:{vars[i]}'
                units[i] = phases[i].state_options[vars[i]]['units']
                shapes[i] = phases[i].state_options[vars[i]]['shape']
            elif classes[i] in {'indep_control', 'input_control'}:
                sources[i] = f'timeseries.controls:{vars[i]}'
                units[i] = phases[i].control_options[vars[i]]['units']
                shapes[i] = phases[i].control_options[vars[i]]['shape']
            elif classes[i] in {'control_rate', 'control_rate2'}:
                sources[i] = f'timeseries.control_rates:{vars[i]}'
                control_name = vars[i][:-5] if classes[i] == 'control_rate' else vars[i][:-6]
                units[i] = phases[i].control_options[control_name]['units']
                deriv = 1 if classes[i].endswith('rate') else 2
                units[i] = get_rate_units(units[i], phases[i].time_options['units'], deriv=deriv)
                shapes[i] = phases[i].control_options[control_name]['shape']
            elif classes[i] in {'indep_polynomial_control', 'input_polynomial_control'}:
                sources[i] = f'timeseries.polynomial_controls:{vars[i]}'
                units[i] = phases[i].polynomial_control_options[vars[i]]['units']
                shapes[i] = phases[i].polynomial_control_options[vars[i]]['shape']
            elif classes[i] in {'polynomial_control_rate', 'polynomial_control_rate2'}:
                sources[i] = f'timeseries.polynomial_control_rates:{vars[i]}'
                control_name = vars[i][:-5] if classes[i] == 'polynomial_control_rate' else vars[i][:-6]
                control_units = phases[i].polynomial_control_options[control_name]['units']
                time_units = phases[i].time_options['units']
                deriv = 1 if classes[i].endswith('rate') else 2
                units[i] = get_rate_units(control_units, time_units, deriv=deriv)
                shapes[i] = phases[i].polynomial_control_options[control_name]['shape']
            elif classes[i] == 'parameter':
                sources[i] = f'timeseries.parameters:{vars[i]}'
                units[i] = phases[i].parameter_options[vars[i]]['units']
                shapes[i] = phases[i].parameter_options[vars[i]]['shape']
            else:
                rhs_source = phases[i].options['transcription']._rhs_source
                sources[i] = f'{rhs_source}.{vars[i]}'
                try:
                    shapes[i], units[i] = get_source_metadata(phases[i]._get_subsystem(rhs_source),
                                                              vars[i], user_units=units[i],
                                                              user_shape=_unspecified)
                except ValueError:
                    raise ValueError(f'{info_str}: Unable to find variable \'{vars[i]}\' in '
                                     f'phase \'{phases[i].pathname}\' or its ODE.')

        linkage_options._src_a = sources['a']
        linkage_options._src_b = sources['b']
        linkage_options['shape'] = shapes['b']

        if linkage_options['units'] is _unspecified:
            if units['a'] != units['b']:
                raise ValueError(f'{info_str}: Linkage units were not specified but the units of '
                                 f'var_a ({units["a"]}) and var_b ({units["b"]}) are not the same. '
                                 f'Units for this linkage constraint must be specified explicitly.')
            else:
                linkage_options['units'] = units['b']

    def _expand_star_linkage_configure(self):
        """
        Finds the variable pair ('*', '*') and expands it out to time and all states if found.

        Returns
        -------
        dict
            The updated dictionary of linkages with '*' expanded to match time and all states at
            a phase boundary.

        """
        linkages_copy = deepcopy(self._linkages)
        for phase_pair, var_dict in linkages_copy.items():
            phase_name_a, phase_name_b = phase_pair

            phase_b = self._get_subsystem(f'phases.{phase_name_b}')

            for var_pair in var_dict.keys():
                if tuple(var_pair) == ('*', '*'):
                    options = var_dict[var_pair]
                    self.add_linkage_constraint(phase_name_a, phase_name_b, var_a='time',
                                                var_b='time', loc_a=options['loc_a'],
                                                loc_b=options['loc_b'], sign_a=options['sign_a'],
                                                sign_b=options['sign_b'])
                    for state_name, state_options in phase_b.state_options.items():
                        self.add_linkage_constraint(phase_name_a, phase_name_b, var_a=state_name,
                                                    var_b=state_name, loc_a=options['loc_a'],
                                                    loc_b=options['loc_b'],
                                                    sign_a=options['sign_a'],
                                                    sign_b=options['sign_b'])
                    self._linkages[phase_pair].pop(var_pair)

    def _is_valid_linkage(self, phase_name_a, phase_name_b, loc_a, loc_b, var_a, var_b):
        """
        Validates linkage constraints.

        Ensures that the posed linkage constraint can be satisfied by checking that the optimizer
        has the freedom to change the linked variable value on either side of the linkage.

        This check errs on the side of permitting linkages if their validity cannot be confirmed.

        Parameters
        ----------
        phase_name_a : str
            The phase name on the first side of the linkage.
        phase_name_b : str
            The phase name on the second side of the linkage.
        loc_a : str
            The "location" of the first side of the linkage, either "initial" or "final".
        loc_b : str
            The "location" of the second side of the linkage, either "initial" or "final".
        var_a : str
            The variable name of the first side of the linkage.
        var_b : str
            The variable name of the second side of the linkage.

        Returns
        -------
        bool
            True if the linkage constraint is valid.
        str
            A message explaining why the linkage is not valid.  Empty for valid linkages.
        """
        phase_a = self._get_subsystem(f'phases.{phase_name_a}')
        phase_b = self._get_subsystem(f'phases.{phase_name_b}')

        var_cls_a = phase_a.classify_var(var_a)
        var_cls_b = phase_b.classify_var(var_b)

        if var_cls_a == 'time':
            var_a_fixed = phase_a.is_time_fixed(loc_a)
        elif var_cls_a == 'state':
            var_a_fixed = phase_a.is_state_fixed(var_a, loc_a)
        else:
            var_a_fixed = False

        if var_cls_b == 'time':
            var_b_fixed = phase_b.is_time_fixed(loc_b)
        elif var_cls_b == 'state':
            var_b_fixed = phase_b.is_state_fixed(var_b, loc_b)
        else:
            var_b_fixed = False

        if var_a_fixed and var_b_fixed:
            return False, f'Cannot link {loc_a} value of "{var_a}" in {phase_name_a} to {loc_b} ' \
                          f'value of "{var_b}" in {phase_name_b}.  Values on both sides of the linkage ' \
                          'are fixed.'
        else:
            return True, ''

    def _configure_linkages(self):
        connected_linkage_inputs = []

        # First, if the user requested all states and time be continuous ('*', '*'), then
        # expand it out.
        self._expand_star_linkage_configure()

        print(f'--- Linkage Report [{self.pathname}] ---')

        indent = '    '

        linkage_comp = self._get_subsystem('linkages')

        for phase_pair, var_dict in self._linkages.items():
            phase_name_a, phase_name_b = phase_pair
            print(f'{indent}--- {phase_name_a} - {phase_name_b} ---')

            phase_b = self._get_subsystem(f'phases.{phase_name_b}')

            # Pull out the maximum variable name length of all variables to make the print nicer.
            var_len = [(len(var_pair[0]), len(var_pair[1])) for var_pair in var_dict]
            max_varname_length = max(itertools.chain(*var_len))
            padding = max_varname_length + 1

            for var_pair, options in var_dict.items():
                var_a, var_b = var_pair
                loc_a = 'initial' if options['loc_a'] in {'initial', '--', '-+'} else 'final'
                loc_b = 'initial' if options['loc_b'] in {'initial', '--', '-+'} else 'final'

                self._update_linkage_options_configure(options)

                src_a = options._src_a
                src_b = options._src_b

                if options['connected']:
                    if phase_b.classify_var(var_b) == 'time':
                        self.connect(f'{phase_name_a}.{src_a}',
                                     f'{phase_name_b}.t_initial',
                                     src_indices=[-1])
                    elif phase_b.classify_var(var_b) == 'state':
                        tgt_b = f'initial_states:{var_b}'
                        self.connect(f'{phase_name_a}.{src_a}',
                                     f'{phase_name_b}.{tgt_b}',
                                     src_indices=om.slicer[-1, ...])
                    print(f'{indent * 2}{var_a:<{padding}s}[{loc_a}]  ->  {var_b:<{padding}s}[{loc_b}]')
                else:
                    is_valid, msg = self._is_valid_linkage(phase_name_a, phase_name_b,
                                                           loc_a, loc_b, var_a, var_b)

                    if not is_valid:
                        raise ValueError(f'Invalid linkage in Trajectory {self.pathname}: {msg}')

                    linkage_comp.add_linkage_configure(options)

                    if options._input_a not in connected_linkage_inputs:
                        self.connect(f'{phase_name_a}.{src_a}',
                                     f'linkages.{options._input_a}',
                                     src_indices=om.slicer[[0, -1], ...])
                        connected_linkage_inputs.append(options._input_a)

                    if options._input_b not in connected_linkage_inputs:
                        self.connect(f'{phase_name_b}.{src_b}',
                                     f'linkages.{options._input_b}',
                                     src_indices=om.slicer[[0, -1], ...])
                        connected_linkage_inputs.append(options._input_b)

                    print(f'{indent * 2}{var_a:<{padding}s}[{loc_a}]  ==  {var_b:<{padding}s}[{loc_b}]')

            print('----------------------------')

    def configure(self):
        """
        Configure the Trajectory Group.

        This method is used to handle connections to the phases in the Trajectory, since
        setup has already been called on all children of the Trajectory, we can query them for
        variables at this point.
        """
        promoted_parameter_inputs = self._configure_parameters() if self.parameter_options else []

        if self._linkages:
            if MPI:
                self._configure_phase_options_dicts()
            self._configure_linkages()
        # promote everything else out of phases that wasn't promoted as a parameter
        phases_group = self._get_subsystem('phases')
        inputs_set = {opts['prom_name'] for (k, opts) in
                      phases_group.get_io_metadata(iotypes=('input',), get_remote=True).items()}
        inputs_to_promote = inputs_set - set(promoted_parameter_inputs)
        self.promotes('phases', inputs=list(inputs_to_promote), outputs=['*'])

    def add_linkage_constraint(self, phase_a, phase_b, var_a, var_b, loc_a='final', loc_b='initial',
                               sign_a=1.0, sign_b=-1.0, units=_unspecified, lower=None, upper=None,
                               equals=None, scaler=None, adder=None, ref0=None, ref=None,
                               linear=False, connected=False):
        """
        Explicitly add a single phase linkage constraint.

        Phase linkage constraints are enforced by constraining the following equation:

        sign_a * var_a + sign_b * var_b

        The resulting value of this equation is constrained.  This can satisfy 'coupling' or
        'linkage' conditions across phase boundaries:  enforcing continuity,
        common initial conditions, or common final conditions.

        With default values, this equation can be used to enforce variable continuity at phase
        boundaries.  For instance, constraining some variable `x` (either a state, control,
        parameter, or output of the ODE) to have the same value at the final point of phase 'foo'
        and the initial point of phase 'bar' is accomplished by:

        ```
        add_linkage_constraint('foo', 'bar', 'x', 'x')
        ```

        We may sometimes want two phases to have the same value of some variable at the start of
        each phase:

        ```
        add_linkage_constraint('foo', 'bar', 'x', 'x', loc_a='initial', loc_b='initial')
        ```

        (Here the specification of loc_b is unnecessary but helps in the clarity of whats going on.)

        Or perhaps a phase has cyclic behavior.  We may not know the exact value of some variable
        `x` at the start and end of the phase `foo`, but it must be the same value at each point.

        ```
        add_linkage_constraint('foo', 'foo', 'x', 'x')
        ```

        If `lower`, `upper`, and `equals` are all `None`, then dymos will use `equals=0` by default.
        If the continuity condition is limited by some bounds instead, lower and upper can be used.
        For instance, perhaps the velocity ('vel') is allowed to have an impulsive change within
        a certain magnitude between two phases:

        ```
        add_linkage_constraint('foo', 'bar', 'vel', 'vel', lower=-100, upper=100, units='m/s')
        ```

        Parameters
        ----------
        phase_a : str
            The first phase in the linkage constraint.
        phase_b : str
            The second phase in the linkage constraint.
        var_a : str
            The linked variable from the first phase in the linkage constraint.
        var_b : str
            The linked variable from the second phase in the linkage constraint.
        loc_a : str
            The location of the variable in the first phase of the linkage constraint (one of
            'initial' or 'final'.)
        loc_b : str
            The location of the variable in the second phase of the linkage constraint (one of
            'initial' or 'final'.)
        sign_a
            The sign applied to the variable from the first phase in the linkage constraint.
        sign_b
            The sign applied to the variable from the second phase in the linkage constraint.
        units : str or None or _unspecified
            Units of the linkage.  If _unspecified, dymos will use the units from the variable
            in the first phase of the linkage.  Units of the two specified variables must be
            compatible.
        lower : float or array or None
            The lower bound applied as a constraint on the linkage equation.
        upper : float or array or None
            The upper bound applied as a constraint on the linkage equation.
        equals : float or array or None
            Specifies a targeted value for an equality constraint on the linkage equation.
        scaler : float or array or None
            The scaler of the linkage constraint.
        adder : float or array or None
            The adder of the linkage constraint.
        ref0 : float or array or None
            The zero-reference value of the linkage constraint.
        ref : float or array or None
            The unit-reference value of the linkage constraint.
        linear : bool
            If True, treat this variable as a linear constraint, otherwise False.  Linear
            constraints should only be applied if the variable on each end of the linkage is a
            design variable or a linear function of one.
        connected : bool
            If True, this constraint is enforced by direct connection rather than a constraint
            for the optimizer.  This is only valid for states and time.
        """
        if connected:
            invalid_options = []
            for arg in ['lower', 'upper', 'equals', 'scaler', 'adder', 'ref0', 'ref']:
                if locals()[arg] is not None:
                    invalid_options.append(arg)
            if locals()['linear']:
                invalid_options.append('linear')
            if invalid_options:
                msg = f'Invalid option in linkage between {phase_a}:{var_a} and {phase_b}:{var_b} ' \
                      f'in trajectory {self.pathname}. The following options for ' \
                      f'add_linkage_constraint were specified but not valid when ' \
                      f'option \'connected\' is True: ' + ' '.join(invalid_options)
                warnings.warn(msg)

        if (phase_a, phase_b) not in self._linkages:
            self._linkages[phase_a, phase_b] = OrderedDict()

        self._linkages[phase_a, phase_b][var_a, var_b] = d = LinkageOptionsDictionary()
        d['phase_a'] = phase_a
        d['phase_b'] = phase_b
        d['var_a'] = var_a
        d['var_b'] = var_b
        d['loc_a'] = loc_a
        d['loc_b'] = loc_b
        d['sign_a'] = sign_a
        d['sign_b'] = sign_b
        d['units'] = units
        d['lower'] = lower
        d['upper'] = upper
        d['equals'] = equals
        d['scaler'] = scaler
        d['adder'] = adder
        d['ref0'] = ref0
        d['ref'] = ref
        d['linear'] = linear
        d['connected'] = connected

    def link_phases(self, phases, vars=None, locs=('final', 'initial'), connected=False):
        """
        Specify that phases in the given sequence are to be assume continuity of the given variables.

        This method caches the phase linkages, and may be called multiple times to express more
        complex behavior (branching phases, phases only continuous in some variables, etc).

        The location at which the variables should be coupled in the two phases are provided
        with a two character string:

        - 'final' specifies the value at the end of the phase (at time t_initial + t_duration)
        - 'initial' specifies the value at the start of the phase (at time t_initial)

        Parameters
        ----------
        phases : sequence of str
            The names of the phases in this trajectory to be sequentially linked.
        vars : sequence of str
            The variables in the phases to be linked, or '*'.  Providing '*' will time and all
            states.  Linking control values or rates requires them to be listed explicitly.
        locs : tuple of str
            A two-element tuple of the two-character location specification.  For every pair in
            phases, the location specification refers to which location in the first phase is
            connected to which location in the second phase.  If the user wishes to specify
            different locations for different phase pairings, those phase pairings must be made
            in separate calls to link_phases.
        connected : bool
            Set to True to directly connect the phases being linked. Otherwise, create constraints
            for the optimizer to solve.

        See Also
        --------
        add_linkage_constraint : Explicitly add a single phase linkage constraint.

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

        Phase linkages assume that, for each pair, the state/control values at the end ('final')
        of the first phase are linked to the state/control values at the start of the second phase
        ('initial').

        The user can override this behavior, but they must specify a pair of location strings for
        each pair given in `phases`.  For instance, in the following example phases 'a' and 'b'
        have the same initial time and state, but phase 'c' follows phase 'b'.  Note since there
        are three phases provided, there are two linkages and thus two pairs of location
        specifiers given.

        >>> t.link_phases(['a', 'b', 'c'], locs=[('initial', 'initial'), ('final', 'initial')])
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
        phase_pairs = list(zip(a, b))

        if len(locs) == 1:
            _locs = num_links * locs
        elif len(locs) == 2:
            _locs = num_links * [locs]
        elif len(locs) == num_links:
            _locs = locs
        else:
            raise ValueError('The number of location tuples, if provided, must be one less than '
                             f'the number of phases specified.  There are {num_links} phase pairs '
                             f'but {len(locs)} location tuples specified.')

        for i in range(len(phase_pairs)):
            phase_name_a, phase_name_b = phase_pairs[i]
            loc_a, loc_b = _locs[i]
            for var in _vars:
                self.add_linkage_constraint(phase_a=phase_name_a, phase_b=phase_name_b,
                                            var_a=var, var_b=var, loc_a=loc_a, loc_b=loc_b,
                                            connected=connected)

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

        for name, phs in self._phases.items():
            sim_phs = phs.get_simulation_phase(times_per_seg=times_per_seg, method=method,
                                               atol=atol, rtol=rtol)
            sim_traj.add_phase(name, sim_phs)

        sim_traj.parameter_options.update(self.parameter_options)

        sim_prob = om.Problem(model=om.Group())

        traj_name = self.name if self.name else 'sim_traj'
        sim_prob.model.add_subsystem(traj_name, sim_traj)

        if record_file is not None:
            rec = om.SqliteRecorder(record_file)
            sim_prob.add_recorder(rec)

        sim_prob.setup()

        # Assign trajectory parameter values
        param_names = [key for key in self.parameter_options.keys()]
        for name in param_names:
            prom_path = f'{self.name}.parameters:{name}'
            src = self.get_source(prom_path)

            # We use this private function to grab the correctly sized variable from the
            # auto_ivc source.
            val = self._abs_get_val(src, False, None, 'nonlinear', 'output', False, from_root=True)
            sim_prob_prom_path = f'{traj_name}.parameters:{name}'
            sim_prob[sim_prob_prom_path][...] = val

        for phase_name, phs in sim_traj._phases.items():
            skip_params = set(param_names)
            for name in param_names:
                targets = self.parameter_options[name]['targets']
                if targets and phase_name in targets:
                    targets_phase = targets[phase_name]
                    if targets_phase is not None:
                        if isinstance(targets_phase, str):
                            targets_phase = [targets_phase]
                        skip_params = skip_params.union(targets_phase)

            phs.initialize_values_from_phase(sim_prob, self._phases[phase_name],
                                             phase_path=traj_name,
                                             skip_params=skip_params)

        print('\nSimulating trajectory {0}'.format(self.pathname))
        sim_prob.run_model()
        print('Done simulating trajectory {0}'.format(self.pathname))
        if record_file:
            sim_prob.record('final')
        sim_prob.cleanup()

        return sim_prob
