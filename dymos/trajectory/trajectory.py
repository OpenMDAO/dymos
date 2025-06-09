import warnings
from collections import OrderedDict
from collections.abc import Sequence
import itertools

from openmdao.utils.om_warnings import warn_deprecation
from openmdao.utils.units import unit_conversion

import networkx as nx

import openmdao.api as om
from openmdao.utils.mpi import MPI

from ..utils.constants import INF_BOUND

from .options import LinkageOptionsDictionary
from .phase_linkage_comp import PhaseLinkageComp
from ..phase.analytic_phase import AnalyticPhase
from ..phase.options import TrajParameterOptionsDictionary
from ..transcriptions.common import ParameterComp
from ..utils.misc import create_subprob, get_rate_units, \
    _unspecified, is_unspecified, is_none_or_unspecified
from ..utils.introspection import get_source_metadata, _get_common_metadata


class Trajectory(om.Group):
    """
    Class definition for a Trajectory group.

    A Trajectory object serves as a container for one or more Phases, as well as the linkage
    conditions between phases.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.

    Attributes
    ----------
    parameter_options : dict
        A dictionary of parameter names and their associated TrajectoryParameterOptionsDictionary
    phases : om.Group or om.ParallelGroup
        The Group which contains phases for this Trajectory.
    sim_prob : Problem or None
        The OpenMDAO problem used for trajectory simulation.
        This is None unless the simulate method has been called.

    _linkages : OrderedDict
        A dictionary containing phase linkage information for the Trajectory.
    _phases : dict
        A dictionary of phase names as keys with the Phase objects being their associated values.
    _phase_graph : nx.DiGraph
        A graph of linked phases.
    """
    def __init__(self, **kwargs):
        super(Trajectory, self).__init__(**kwargs)

        self._linkages = {}
        self._phases = {}
        self._phase_graph = nx.DiGraph()
        self._has_connected_phases = False
        self.sim_prob = None

        self.phases = om.ParallelGroup() if self.options['parallel_phases'] else om.Group()

    def initialize(self):
        """
        Declare any options for Trajectory.
        """
        self.options.declare('sim_mode', types=bool, default=False,
                             desc='Used internally by Dymos when invoking simulate on a trajectory')
        self.options.declare('parallel_phases',
                             default=True, types=bool,
                             desc='If True, the top-level container of all phases will be a ParallelGroup,'
                                  'otherwise it will be a standard OpenMDAO Group.')
        self.options.declare('auto_solvers', types=bool, default=True,
                             desc='If True, attempt to automatically assign solvers if necessary.')
        self.options.declare('parameter_options', types=dict, default={},
                             desc='Options for each parameter in this Trajectory')

    @property
    def parameter_options(self):
        return self.options['parameter_options']

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
        **kwargs : dict
            Additional arguments when adding the phase to the trajectory.

        Returns
        -------
        PhaseBase
            The Phase object added to the trajectory.
        """
        self._phases[name] = self.phases.add_subsystem(name, phase, **kwargs)
        return phase

    def set_parameter_options(self, name, units=_unspecified, val=_unspecified, desc=_unspecified, opt=False,
                              targets=_unspecified, lower=_unspecified, upper=_unspecified,
                              scaler=_unspecified, adder=_unspecified, ref0=_unspecified, ref=_unspecified,
                              shape=_unspecified, dynamic=_unspecified, static_target=_unspecified,
                              static_targets=_unspecified):
        """
        Set the options of an existing a parameter in the trajectory.

        Parameters
        ----------
        name : str
            Name of the parameter.
        units : str or None or _unspecified
            Units in which the parameter is defined.  If _unspecified, use the units declared
            for the parameter in the ODE.
        val : float or ndarray
            Default value of the parameter at all nodes.
        desc : str
            A description of the parameter.
        opt : bool
            If True the value(s) of this parameter will be design variables in
            the optimization problem. The default is False.
        targets : dict or None
            If None, then the parameter will be connected to the controllable parameter
            in the ODE of each phase.  For each phase where no such controllable parameter exists,
            a warning will be issued.  If targets is given as a dict, the dict should provide
            the relevant phase names as keys, each associated with the respective controllable
            parameter as a value.
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
            of nodes) else False. This argument is deprecated in favor of static_target.
        static_target : bool or _unspecified
            True if the targets in the ODE are not shaped with num_nodes as the first dimension
            (meaning they cannot have a unique value at each node).  Otherwise False.
        static_targets : bool or Sequence or _unspecified
            True if ALL targets in the ODE are not shaped with num_nodes as the first dimension
            (meaning they cannot have a unique value at each node).  If False, ALL targets are
            expected to be shaped with the first dimension as the number of nodes. If given
            as a Sequence, it provides those targets not shaped with num_nodes. If left _unspecified,
            static targets will be determined automatically.
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

        if shape is not _unspecified:
            self.parameter_options[name]['shape'] = shape

        if dynamic is not _unspecified:  # Deprecated
            self.parameter_options[name]['static_targets'] = not dynamic

        if static_target is not _unspecified:  # Deprecated
            self.parameter_options[name]['static_targets'] = static_target

        if static_targets is not _unspecified:
            self.parameter_options[name]['static_targets'] = static_target

        if dynamic is not _unspecified and static_target is not _unspecified:
            raise ValueError("Both the deprecated 'dynamic' option and option 'static_target' were\n"
                             f"specified for parameter '{name}'. Going forward, please use only\n"
                             "option static_targets. Options 'dynamic' and 'static_target'\n"
                             "will be removed in Dymos 2.0.0.")

        if dynamic is not _unspecified and static_targets is not _unspecified:
            raise ValueError("Both the deprecated 'dynamic' option and option 'static_targets' were "
                             f"specified for parameter '{name}'. "
                             f"Going forward, please use only option static_targets.  Option "
                             f"'dynamic' will be removed in Dymos 2.0.0.")

    def add_parameter(self, name, units=_unspecified, val=_unspecified, desc=_unspecified, opt=False,
                      targets=_unspecified, lower=_unspecified, upper=_unspecified,
                      scaler=_unspecified, adder=_unspecified, ref0=_unspecified, ref=_unspecified,
                      shape=_unspecified, dynamic=_unspecified, static_target=_unspecified,
                      static_targets=_unspecified):
        """
        Add a parameter (static control) to the trajectory.

        Parameters
        ----------
        name : str
            Name of the parameter.
        units : str or None or _unspecified
            Units in which the parameter is defined.  If _unspecified, use the units declared
            for the parameter in the ODE.
        val : float or ndarray
            Default value of the parameter at all nodes.
        desc : str
            A description of the parameter.
        opt : bool
            If True the value(s) of this parameter will be design variables in
            the optimization problem. The default is False.
        targets : dict or None
            If None, then the parameter will be connected to the controllable parameter
            in the ODE of each phase.  For each phase where no such controllable parameter exists,
            a warning will be issued.  If targets is given as a dict, the dict should provide
            the relevant phase names as keys, each associated with the respective controllable
            parameter as a value.
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
            of nodes) else False. This argument is deprecated in favor of static_target.
        static_target : bool or _unspecified
            True if the targets in the ODE are not shaped with num_nodes as the first dimension
            (meaning they cannot have a unique value at each node).  Otherwise False.
        static_targets : bool or Sequence or _unspecified
            True if ALL targets in the ODE are not shaped with num_nodes as the first dimension
            (meaning they cannot have a unique value at each node).  If False, ALL targets are
            expected to be shaped with the first dimension as the number of nodes. If given
            as a Sequence, it provides those targets not shaped with num_nodes. If left _unspecified,
            static targets will be determined automatically.
        """
        if name not in self.parameter_options:
            self.parameter_options[name] = TrajParameterOptionsDictionary()
            self.parameter_options[name]['name'] = name
        else:
            raise ValueError(f'Attempted to add parameter "{name}" to trajectory but trajectory already has a parameter'
                             ' of that name.')

        self.set_parameter_options(name, units=units, val=val, desc=desc, opt=opt, targets=targets, lower=lower,
                                   upper=upper, scaler=scaler, adder=adder, ref0=ref0, ref=ref, shape=shape,
                                   dynamic=dynamic, static_target=static_target)

    def set_parameter_val(self, name, val=None, units=None):
        """
        Set the parameter values.

        Parameters
        ----------
        name : str
            Name of the variable. This should be a parameter variable.
        val : ndarray or Sequence or None
            Array of state values.
        units : str, optional
            The units of the state values specified.
            If None, use the units associated with the target.
            If provided, must be compatible with the target units.
        """
        if units is None:
            units = self.parameter_options[name]['units']
        self.set_val(f'parameters:{name}', val=val, units=units)

    def _setup_parameters(self):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        if self.parameter_options:
            param_comp = ParameterComp()
            self.add_subsystem('param_comp', subsys=param_comp, promotes_inputs=['*'], promotes_outputs=['*'])

            for name, options in self.parameter_options.items():
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
                        kwargs = {'static_targets': options['static_targets'],
                                  'units': options['units'],
                                  'val': options['val'],
                                  'shape': options['shape'],
                                  'ref0': options['ref0'],
                                  'ref': options['ref'],
                                  'adder': options['adder'],
                                  'scaler': options['scaler'],
                                  # The phase-level parameter is never optimized
                                  # if governed by the trajectory-level parameter.
                                  'opt': False,
                                  'lower': options['lower'],
                                  'upper': options['upper'],
                                  'targets': tgts[phase_name]}

                        if not self.options['sim_mode']:
                            if name not in phs.parameter_options:
                                phs.add_parameter(name, **kwargs)
                            else:
                                phs.set_parameter_options(name, **kwargs)

    def _setup_linkages(self):

        has_linkage_constraints = False

        err_template = '{traj}: Phase `{phase1}` links variable `{var1}` to phase ' \
                       '`{phase2}` state variable `{var2}` by connection, but phase `{phase2}` ' \
                       'is an AnalyticPhase and does not support linking initial state ' \
                       'values with option `connected=True`.'

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
                        phase2.set_time_options(input_initial=True, fix_initial=False)
                    elif var2 == '*':
                        phase2.set_time_options(input_initial=True, fix_initial=False)
                        for state_name in phase2.state_options:
                            if isinstance(phase2, AnalyticPhase):
                                raise RuntimeError(err_template.format(traj=self.pathname, phase1=pair[0],
                                                                       phase2=pair[1], var1=var1, var2=var2))
                            else:
                                phase2.set_state_options(state_name, input_initial=True, fix_initial=_unspecified)
                    elif var2 in phase2.state_options:
                        if isinstance(phase2, AnalyticPhase):
                            raise RuntimeError(err_template.format(traj=self.pathname, phase1=pair[0],
                                                                   phase2=pair[1], var1=var1, var2=var2))
                        else:
                            phase2.set_state_options(var2, input_initial=True, fix_initial=False)
                    elif var2 in phase2.parameter_options:
                        phase2.set_parameter_options(var2, opt=False)
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

        # This will override the existing phases attribute with the same thing.
        self.add_subsystem('phases', subsys=self.phases)

        if self._linkages and not self.options['sim_mode']:
            self._setup_linkages()

    def _configure_parameters(self):
        """
        Configure connections from input or design parameters to the appropriate targets
        in each phase.
        """
        parameter_options = self.parameter_options
        promoted_inputs = []
        params_by_phase = {phase_name: phs.parameter_options for phase_name, phs in self._phases.items()}

        for name, options in parameter_options.items():
            promoted_inputs.append(f'parameters:{name}')
            targets = options['targets']

            # First check that the user gave valid phase names, if providing
            # parameter targets as a dict.
            if isinstance(targets, dict):
                target_phases = set(targets.keys())
                avail_phases = set(self._phases.keys())
                non_existing_phases = target_phases - avail_phases
                if non_existing_phases:
                    raise ValueError(f'The following phases were specified as\n'
                                     f'targets for trajectory parameter {name}\n'
                                     f'but are not valid phase names: {non_existing_phases}')

            # For each phase, use introspection to get the units and shape.
            # If units do not match across all phases, require user to set them.
            # If shapes do not match across all phases, this is an error.
            targets_per_phase = {}

            for phase_name, phs in self._phases.items():
                target_param = None

                if targets is None or phase_name not in targets:
                    # Attempt to connect to an input parameter of the same name in the phase, if
                    # it exists.
                    if name in phs.parameter_options:
                        target_param = name
                    else:
                        continue
                elif targets[phase_name] is None:
                    # Connections to this phase are explicitly omitted
                    continue
                elif isinstance(targets[phase_name], str):
                    if targets[phase_name] in phs.parameter_options:
                        # Connect to an input parameter with a different name in this phase
                        target_param = targets[phase_name]
                    else:
                        msg = f'Invalid target for trajectory `{self.pathname}` parameter `{name}` in phase ' \
                              f"`{phase_name}`.\nTarget for phase `{phase_name}` is '{targets[phase_name]}' but " \
                              f"the phase has no such parameter."
                        raise ValueError(msg)
                elif isinstance(targets[phase_name], Sequence):
                    if name in phs.parameter_options:
                        # User gave a list of ODE targets which were passed to the creation of a
                        # new input parameter in setup, just connect to that new input parameter
                        target_param = name
                    else:
                        msg = f'Invalid target for trajectory `{self.pathname}` parameter `{name}` in phase ' \
                              f"`{phase_name}`.\nThe phase did not add the parameter as expected. Please file an " \
                              f"issue with the Dymos development team at https://github.com/OpenMDAO/dymos"
                        raise RuntimeError(msg)
                else:
                    raise ValueError(f'Unhandled target(s) ({targets[phase_name]}) for parameter {name} in '
                                     f'phase {phase_name}. If connecting to ODE inputs in the phase, '
                                     f'format the targets as a sequence of strings.')

                if target_param is not None:
                    targets_per_phase[phase_name] = target_param

            if not targets_per_phase:
                # Find the reason
                if targets is None:
                    reason = f'Option `targets=None` but no phase in the trajectory has a parameter named `{name}`.'
                elif all([t is None for t in targets.values()]) and targets.keys() == self._phases.keys():
                    reason = 'Option `targets` is a dictionary keyed by phase name but target for each phase is None.'
                else:
                    reason = ''
                raise ValueError(f'No target was found for trajectory parameter `{name}` in any phase.\n{reason}')

            # If metadata is unspecified, use introspection to find
            # it based on common values among the targets.
            targets = {phase_name: phs_params[targets_per_phase[phase_name]]
                       for phase_name, phs_params in params_by_phase.items()
                       if phase_name in targets_per_phase and targets_per_phase[phase_name] in phs_params}

            if is_unspecified(options['units']):
                options['units'] = _get_common_metadata(targets, metadata_key='units')

            if is_none_or_unspecified(options['shape']):
                options['shape'] = _get_common_metadata(targets, metadata_key='shape')

            param_comp = self._get_subsystem('param_comp')
            param_comp.add_parameter(name, val=options['val'], shape=options['shape'], units=options['units'])
            if options['opt']:
                lb = -INF_BOUND if options['lower'] is None else options['lower']
                ub = INF_BOUND if options['upper'] is None else options['upper']

                self.add_design_var(name=f'parameters:{name}',
                                    lower=lb,
                                    upper=ub,
                                    scaler=options['scaler'],
                                    adder=options['adder'],
                                    ref0=options['ref0'],
                                    ref=options['ref'])

            tgts = [f'{phase_name}.parameters:{param_name}' for phase_name, param_name in targets_per_phase.items()]
            self.connect(f'parameter_vals:{name}', tgts)

        return promoted_inputs

    def _configure_phase_options_dicts(self):
        """
        Called during configure if we are under MPI. Loops over all phases and populates the
        phase options dictionaries in self._phases.

        Because each phase performs introspection, on MPI the trajectory may not know certain
        metadata for phase variables that is necessary for things like linkage calculations.

        Note the phase objects exist in two places.  Traj._phases stores the phases for the purposes
        of setup and configure. These instances are the ones being updated by this method. The `phases`
        attribute is the actual OpenMDAO subsytem used in the trajectory model.

        Each phase populates phase_options_dicts, which contains options for each dymos variable type.
        Once populated, we gather this data from each proc and then populate self._phases variable
        options dictionaries with the updated information.
        """
        phase_options_dicts = {phase_name: {} for phase_name in self._phases.keys()}
        for phs in self.phases._subsystems_myproc:
            phase_options_dicts[phs.name]['time_options'] = phs.time_options
            phase_options_dicts[phs.name]['state_options'] = phs.state_options
            phase_options_dicts[phs.name]['control_options'] = phs.control_options
            phase_options_dicts[phs.name]['parameter_options'] = phs.parameter_options

        all_ranks = self.comm.allgather(phase_options_dicts)

        for phase_name, phs in self._phases.items():
            for rank_i, data in enumerate(all_ranks):
                if phase_name in data and len(data[phase_name]) > 0:
                    phs.time_options.update(data[phase_name]['time_options'])
                    phs.state_options.update(data[phase_name]['state_options'])
                    phs.control_options.update(data[phase_name]['control_options'])
                    phs.parameter_options.update(data[phase_name]['parameter_options'])

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

        info_str = f'{self.pathname}: ' if self.pathname else ''

        phase_a = self._get_subsystem(f'phases.{phase_name_a}')
        phase_b = self._get_subsystem(f'phases.{phase_name_b}')

        phases = {'a': phase_a, 'b': phase_b}

        classes = {'a': phase_a.classify_var(var_a),
                   'b': phase_b.classify_var(var_b)}

        sources = {'a': None, 'b': None}
        vars = {'a': var_a, 'b': var_b}
        units = {'a': _unspecified, 'b': _unspecified}
        shapes = {'a': _unspecified, 'b': _unspecified}

        use_prefix = {'a': phase_a.timeseries_options['use_prefix'],
                      'b': phase_b.timeseries_options['use_prefix']}

        for i in ('a', 'b'):
            if classes[i] == 't':
                time_name = phases[i].time_options['name']
                sources[i] = f'timeseries.{time_name}'
                shapes[i] = (1,)
                units[i] = phases[i].time_options['units']
            elif classes[i] == 't_phase':
                time_name = phases[i].time_options['name']
                sources[i] = f'timeseries.{time_name}_phase'
                units[i] = phases[i].time_options['units']
                shapes[i] = (1,)
            elif classes[i] == 'state':
                prefix = 'states:' if use_prefix[i] else ''
                sources[i] = f'timeseries.{prefix}{vars[i]}'
                units[i] = phases[i].state_options[vars[i]]['units']
                shapes[i] = phases[i].state_options[vars[i]]['shape']
            elif classes[i] in {'control'}:
                prefix = 'controls:' if use_prefix[i] else ''
                sources[i] = f'timeseries.{prefix}{vars[i]}'
                units[i] = phases[i].control_options[vars[i]]['units']
                shapes[i] = phases[i].control_options[vars[i]]['shape']
            elif classes[i] in {'control_rate', 'control_rate2'}:
                prefix = 'control_rates:' if use_prefix[i] else ''
                sources[i] = f'timeseries.{prefix}{vars[i]}'
                control_name = vars[i][:-5] if classes[i] == 'control_rate' else vars[i][:-6]
                units[i] = phases[i].control_options[control_name]['units']
                deriv = 1 if classes[i].endswith('rate') else 2
                units[i] = get_rate_units(units[i], phases[i].time_options['units'], deriv=deriv)
                shapes[i] = phases[i].control_options[control_name]['shape']
            elif classes[i] == 'parameter':
                sources[i] = f'parameter_vals:{vars[i]}'
                units[i] = phases[i].parameter_options[vars[i]]['units']
                shapes[i] = phases[i].parameter_options[vars[i]]['shape']
            else:
                rhs_source = phases[i].options['transcription']._rhs_source
                sources[i] = f'{rhs_source}.{vars[i]}'
                try:
                    meta = get_source_metadata(phases[i]._get_subsystem(rhs_source), vars[i], user_units=units[i],
                                               user_shape=_unspecified)
                    shapes[i] = meta['shape']
                    units[i] = meta['units']
                except ValueError:
                    raise RuntimeError(f'{info_str}Error in linking {var_a} from {phase_name_a} to {var_b} in '
                                       f'{phase_name_b}. Unable to find variable \'{vars[i]}\' in phase '
                                       f'\'{phases[i].pathname}\' or its ODE.')

        linkage_options._src_a = sources['a']
        linkage_options._src_b = sources['b']
        linkage_options['shape'] = shapes['b']

        if is_unspecified(linkage_options['units_a']):
            linkage_options['units_a'] = units['a']

        if is_unspecified(linkage_options['units_b']):
            linkage_options['units_b'] = units['b']

        if units['a'] is not None and units['b'] is not None:
            conversion_scaler, conversion_offset = unit_conversion(units['a'], units['b'])
        else:
            conversion_scaler, conversion_offset = (1.0, 0.0)

        if not linkage_options['connected'] \
                and is_unspecified(linkage_options['units']) \
                and (abs(conversion_scaler - 1.0) > 1.0E-15 or abs(conversion_offset) > 1.0E-15):
            raise ValueError(f'{info_str}Linkage units were not specified but the units of {phase_name_a}.{var_a} '
                             f'({units["a"]}) and {phase_name_b}.{var_b} ({units["b"]}) are not equivalent. '
                             f'Units for this linkage constraint must be specified explicitly.')
        else:
            linkage_options['units'] = units['a']

    def _expand_star_linkage_configure(self):
        """
        Finds the variable pair ('*', '*') and expands it out to time and all states if found.

        Returns
        -------
        dict
            The updated dictionary of linkages with '*' expanded to match time and all states at
            a phase boundary.

        """
        for phase_pair, var_dict in self._linkages.items():
            phase_name_a, phase_name_b = phase_pair

            phase_b = self._get_subsystem(f'phases.{phase_name_b}')

            var_pair = ('*', '*')
            if var_pair in var_dict:
                options = var_dict[var_pair]
                time_name = phase_b.time_options['name']
                self.add_linkage_constraint(phase_name_a, phase_name_b,
                                            var_a=time_name, var_b=time_name,
                                            loc_a=options['loc_a'], loc_b=options['loc_b'],
                                            mult_a=options['mult_a'], mult_b=options['mult_b'],
                                            connected=options['connected'])
                for state_name in phase_b.state_options:
                    self.add_linkage_constraint(phase_name_a, phase_name_b,
                                                var_a=state_name, var_b=state_name,
                                                loc_a=options['loc_a'], loc_b=options['loc_b'],
                                                mult_a=options['mult_a'], mult_b=options['mult_b'],
                                                connected=options['connected'])
                self._linkages[phase_pair].pop(var_pair)

    def _is_valid_linkage(self, phase_name_a, phase_name_b, loc_a, loc_b, var_a, var_b, fixed_a, fixed_b):
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
        fixed_a : bool
            True if variable a is fixed at the linkage location.
        fixed_b : bool
            True if variable b is fixed at the linkage location

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

        def _print_on_rank(rank=0, *args, **kwargs):
            if self.comm.rank == rank:
                print(*args, **kwargs)

        prefixes = {'t': '',
                    't_phase': '',
                    'state': 'states:',
                    'parameter': 'parameters:',
                    'control': 'controls:',
                    'control_rate': 'control_rates:',
                    'control_rate2': 'control_rates:',
                    'ode': ''
                    }

        def _get_prefixed_var(var, phase):
            if phase.timeseries_options['use_prefix']:
                return f'{prefixes[phase.classify_var(var)]}{var}'
            else:
                return var

        # First, if the user requested all states and time be continuous ('*', '*'), then
        # expand it out.
        self._expand_star_linkage_configure()

        _print_on_rank(f'--- Linkage Report [{self.pathname}] ---')

        indent = '    '

        linkage_comp = self._get_subsystem('linkages')

        self._phase_graph = phase_graph = nx.DiGraph()

        for phase_pair, var_dict in self._linkages.items():
            phase_name_a, phase_name_b = phase_pair
            _print_on_rank(f'{indent}--- {phase_name_a} - {phase_name_b} ---')

            phase_a = self._get_subsystem(f'phases.{phase_name_a}')
            phase_b = self._get_subsystem(f'phases.{phase_name_b}')

            # Pull out the maximum variable name length of all variables to make the print nicer.
            var_len_a = [len(_get_prefixed_var(var, phase_a)) for var, _ in var_dict]
            var_len_b = [len(_get_prefixed_var(var, phase_b)) for _, var in var_dict]
            padding_a = max(var_len_a) + 2
            padding_b = max(var_len_b) + 2

            for var_pair, options in var_dict.items():
                var_a, var_b = var_pair
                loc_a = options['loc_a']
                loc_b = options['loc_b']

                class_a = phase_a.classify_var(var_a)
                class_b = phase_b.classify_var(var_b)

                self._update_linkage_options_configure(options)

                src_a = options._src_a
                src_b = options._src_b

                fixed_a = phase_a._is_fixed(var_a, class_a, loc_a)
                fixed_b = phase_b._is_fixed(var_b, class_b, loc_b)

                if class_a == 't' and class_b == 't':
                    phase_graph.add_node(phase_a.name, fix_initial=None, t_initial=None,
                                         duration_bounds=None, initial_bounds=None)
                    phase_graph.add_node(phase_b.name, fix_initial=None, t_initial=None,
                                         duration_bounds=None, initial_bounds=None)
                phase_graph.add_edge(phase_a.name, phase_b.name)

                prefixed_a = _get_prefixed_var(var_a, phase_a)
                prefixed_b = _get_prefixed_var(var_b, phase_b)

                str_fixed_a = '*' if fixed_a else ''
                str_fixed_b = '*' if fixed_b else ''

                if options['connected']:
                    if class_b == 't':
                        self.connect(f'{phase_name_a}.{src_a}', f'{phase_name_b}.t_initial',
                                     src_indices=[-1], flat_src_indices=True)
                    elif class_b == 'state':
                        tgt_b = f'initial_states:{var_b}'
                        self.connect(f'{phase_name_a}.{src_a}', f'{phase_name_b}.{tgt_b}',
                                     src_indices=om.slicer[-1, ...])
                    elif class_b == 'parameter':
                        tgt_b = f'parameters:{var_b}'
                        self.connect(f'{phase_name_a}.{src_a}', f'{phase_name_b}.{tgt_b}',
                                     src_indices=om.slicer[-1, ...])
                    else:
                        msg = f'Could not create connection linkage from phase `{phase_name_a}` ' \
                              f'variable `{var_a}` to phase `{phase_name_b}` variable `{var_b}`. ' \
                              f'For direct connections, the target variable must be `time`, a ' \
                              f'state, or a parameter in the phase.\nEither remove the linkage or specify ' \
                              f'`connected=False` to enforce it via an optimization constraint.'
                        raise om.OpenMDAOWarning(msg)
                    _print_on_rank(f'{indent * 2}{prefixed_a:<{padding_a}s} [{loc_a}{str_fixed_a}] ->  '
                                   f'{prefixed_b:<{padding_b}s} [{loc_b}{str_fixed_b}]')
                    self._has_connected_phases = True
                else:

                    if fixed_a and fixed_b:
                        msg = f'Cannot link {loc_a} value of "{var_a}" in {phase_name_a} to {loc_b} ' \
                              f'value of "{var_b}" in {phase_name_b}. Values on both sides of the linkage are fixed ' \
                              'and the linkage is enforced via constraint. Either link the variables via connection ' \
                              'or make the variables design variables on at least one side of the connection.'

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

                    _print_on_rank(f'{indent * 2}{prefixed_a:<{padding_a}s} [{loc_a}{str_fixed_a}] ==  '
                                   f'{prefixed_b:<{padding_b}s} [{loc_b}{str_fixed_b}]')

        _print_on_rank('\n* : Value is fixed or is an input.\n')

    def _check_phase_graph(self):

        # determine if we have any fixed t_initial that are outside of allowed bounds
        def get_final_tbounds(phase_name, node_data, old_tmin=-INF_BOUND, old_tmax=INF_BOUND):
            t_initial = node_data['t_initial']
            fixed_t_initial = node_data['fix_initial']
            duration_min, duration_max = node_data['duration_bounds']
            initial_min, initial_max = node_data['initial_bounds']

            duration_min = -INF_BOUND if duration_min is None else duration_min
            duration_max = INF_BOUND if duration_max is None else duration_max
            old_tmin = -INF_BOUND if old_tmin is None else old_tmin
            old_tmax = INF_BOUND if old_tmax is None else old_tmax

            errs = []
            new_min_t = -INF_BOUND
            new_max_t = INF_BOUND

            if fixed_t_initial:
                # if t_initial is fixed, first check if it's within the incoming bounds
                if not old_tmin <= t_initial <= old_tmax:
                    errs.append(f"Fixed t_initial of {t_initial} is outside of allowed bounds "
                                f"{(old_tmin, old_tmax)} for phase '{phase_name}'.")

                if duration_min > -INF_BOUND:
                    new_min_t = t_initial + duration_min

                if duration_max < INF_BOUND:
                    new_max_t = t_initial + duration_max

            else:
                initial_min = -INF_BOUND if initial_min is None else initial_min
                initial_max = INF_BOUND if initial_max is None else initial_max

                # check for overlap of incoming bounds on t and initial_bounds
                if not (old_tmin <= initial_min <= old_tmax or
                        old_tmin <= initial_max <= old_tmax or
                        initial_min <= old_tmin <= initial_max or
                        initial_min <= old_tmax <= initial_max):
                    errs.append(f"t_initial bounds of ({initial_min}, {initial_max}) do not overlap"
                                f" with allowed bounds {(old_tmin, old_tmax)} for phase "
                                f"'{phase_name}'.")

                if duration_min > -INF_BOUND:
                    tmin = max(old_tmin, initial_min)
                    if tmin > -INF_BOUND:
                        new_min_t = tmin + duration_min
                    else:
                        new_min_t = tmin

                if duration_max < INF_BOUND:
                    tmax = min(old_tmax, initial_max)
                    if tmax < INF_BOUND:
                        new_max_t = tmax + duration_max
                    else:
                        new_max_t = tmax

            return new_min_t, new_max_t, errs

        phase_graph = self._phase_graph

        node_data = phase_graph.nodes(data=True)

        # only keep the part of the graph where 't' connects phases
        G = phase_graph.subgraph([n for n, meta in node_data if 't_initial' in meta])

        # Update G nodes with final values for time properties.
        # We wait until now because users can set some values after setup.
        for name, data in G.nodes(data=True):
            phase = self._get_subsystem(f'phases.{name}')
            data['fix_initial'] = phase.time_options['fix_initial']
            data['duration_bounds'] = phase.time_options['duration_bounds']
            data['initial_bounds'] = phase.time_options['initial_bounds']
            data['t_initial'] = phase.time_options['initial_val']

        # find starting node(s) in the graph
        start_nodes = [n for n, degree in G.in_degree() if degree == 0]
        all_errs = []
        skip = set()
        for start_name in start_nodes:
            stack = [(start_name, -INF_BOUND, INF_BOUND)]
            while stack:
                src, oldlb, oldub = stack.pop()
                lb, ub, errs = get_final_tbounds(src, node_data[src],
                                                 old_tmin=oldlb, old_tmax=oldub)
                if errs:
                    all_errs.append((src, errs))
                    # skip nodes downstream of the failing node.
                    skip.update(list(nx.dfs_preorder_nodes(G, source=src)))

                for tgt in G[src]:
                    if tgt not in skip:
                        stack.append((tgt, lb, ub))

        if all_errs:
            errs = []
            for _, s in sorted(all_errs, key=lambda x: x[0]):
                errs.extend(s)
            err_lines = '\n'.join(errs) if len(errs) > 1 else errs[0]
            raise RuntimeError(f"{self.msginfo}: {err_lines}")

    def _configure_solvers(self):
        """
        Automatically configure solvers for the Trajectory.

        If operating under MPI and phases are connected, then
        the default nonlinear solver will be a NonlinearBlockJac,
        while the default linear solver will be PETScKrylov.

        These solvers can be changed through the
        'default_nonlinear_solver' and 'default_linear_solver' options.
        """
        if not self.options['auto_solvers']:
            return

        warn = False
        if self._has_connected_phases and isinstance(self.phases, om.ParallelGroup) and self.comm.size > 1:
            msg = (f'{self.msginfo}\n'
                   f'  Non-default solvers are required\n'
                   f'    connected phases in parallel: True\n')
            if isinstance(self.phases.nonlinear_solver, om.NonlinearRunOnce):
                warn = True
                msg += (f'  Setting \n'
                        f'    {self.pathname}.phases.nonlinear_solver = om.NonlinearBlockJac(iprint=0)\n'
                        f'  Explicitly set {self.pathname}.phases.nonlinear_solver to override.\n')
                self.phases.nonlinear_solver = om.NonlinearBlockJac(iprint=0)

            if isinstance(self.phases.linear_solver, om.LinearRunOnce):
                warn = True
                msg += (f'  Setting\n'
                        f'    {self.pathname}.phases.linear_solver = om.PETScKrylov(iprint=0)\n'
                        f'  Explicitly set {self.pathname}.phases.linear_solver to override.\n')
                self.phases.linear_solver = om.PETScKrylov(iprint=0)

            if warn:
                om.issue_warning(msg)

    def configure(self):
        """
        Configure the Trajectory Group.

        This method is used to handle connections to the phases in the Trajectory, since
        setup has already been called on all children of the Trajectory, we can query them for
        variables at this point.
        """
        if MPI:
            self._configure_phase_options_dicts()

        if self.parameter_options:
            self._configure_parameters()

        if self._linkages and not self.options['sim_mode']:
            self._configure_linkages()

        self._configure_solvers()

        # promote everything else out of phases
        self.promotes('phases', inputs=['*'], outputs=['*'])

    def add_linkage_constraint(self, phase_a, phase_b, var_a, var_b, loc_a='final', loc_b='initial',
                               sign_a=_unspecified, sign_b=_unspecified,
                               mult_a=_unspecified, mult_b=_unspecified,
                               units=_unspecified, lower=None, upper=None,
                               equals=None, scaler=None, adder=None, ref0=None, ref=None,
                               linear=False, connected=False):
        """
        Explicitly add a single phase linkage constraint.

        Phase linkage constraints are enforced by constraining the following equation:

        mult_a * var_a + mult_b * var_b

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
            'initial' or 'final').
        loc_b : str
            The location of the variable in the second phase of the linkage constraint (one of
            'initial' or 'final').
        sign_a : float
            The multiplier applied to the variable from the first phase in the linkage constraint.
            This argument is deprecated in favor of mult_a.
        sign_b : float
            The multiplier applied to the variable from the second phase in the linkage constraint.
            This argument is deprecated in favor of mult_b.
        mult_a : float
            The multiplier applied to the variable from the first phase in the linkage constraint.
        mult_b : float
            The multiplier applied to the variable from the second phase in the linkage constraint.
        units : str or None or _unspecified
            Units of the linkage. If _unspecified, dymos will use the units from the variable
            in the first phase of the linkage. Units of the two specified variables must be
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
            If True, treat this variable as a linear constraint, otherwise False. Linear
            constraints should only be applied if the variable on each end of the linkage is a
            design variable or a linear function of one.
        connected : bool
            If True, this constraint is enforced by direct connection rather than a constraint
            for the optimizer. This is only valid for states and time.
        """
        if not is_unspecified(sign_a):
            if not is_unspecified(mult_a):
                raise ValueError(
                    "Both the deprecated 'sign_a' option and option 'mult_a' were specified."
                    "Going forward, please use only option mult_a.")
            warn_deprecation("'sign_a' has been deprecated. Use 'mult_a' instead.")
            mult_a = sign_a
        else:
            if is_unspecified(mult_a):
                mult_a = 1.0

        if not is_unspecified(sign_b):
            if not is_unspecified(mult_b):
                raise ValueError(
                    "Both the deprecated 'sign_b' option and option 'mult_b' were specified."
                    "Going forward, please use only option mult_b.")
            warn_deprecation("'sign_b' has been deprecated. Use 'mult_b' instead.")
            mult_b = sign_b
        else:
            if is_unspecified(mult_b):
                mult_b = -1.0

        if connected:
            invalid_options = []
            for arg in ['lower', 'upper', 'equals', 'scaler', 'adder', 'ref0', 'ref', 'units']:
                if locals()[arg] is not None and not is_unspecified(locals()[arg]):
                    invalid_options.append(arg)
            if locals()['linear']:
                invalid_options.append('linear')
            if invalid_options:
                msg = f'Invalid option in linkage between {phase_a}:{var_a} and {phase_b}:{var_b} ' \
                      f'in trajectory {self.pathname}. The following options for ' \
                      f'add_linkage_constraint were specified but not valid when ' \
                      f'option \'connected\' is True: ' + ' '.join(invalid_options)
                om.issue_warning(msg, category=om.UnusedOptionWarning)

        if (phase_a, phase_b) not in self._linkages:
            self._linkages[phase_a, phase_b] = OrderedDict()

        self._linkages[phase_a, phase_b][var_a, var_b] = d = LinkageOptionsDictionary()
        d['phase_a'] = phase_a
        d['phase_b'] = phase_b
        d['var_a'] = var_a
        d['var_b'] = var_b
        d['loc_a'] = loc_a
        d['loc_b'] = loc_b
        d['mult_a'] = mult_a
        d['mult_b'] = mult_b
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

    def link_phases(self, phases, vars=None, locs=('final', 'initial'), connected=False,
                    units=_unspecified, scaler=None, adder=None, ref0=None, ref=None, linear=False):
        """
        Specify that phases in the given sequence are to assume continuity of the given variables.

        This method caches the phase linkages, and may be called multiple times to express more
        complex behavior (branching phases, phases only continuous in some variables, etc).

        The location at which the variables should be coupled in the two phases are provided
        with one of two strings:

        - 'final' specifies the value at the end of the phase (at time t_initial + t_duration)
        - 'initial' specifies the value at the start of the phase (at time t_initial)

        Parameters
        ----------
        phases : sequence of str
            The names of the phases in this trajectory to be sequentially linked.
        vars : sequence of str
            The variables in the phases to be linked, or '*'.  Providing '*' will link time and all
            states.  Linking control values or rates requires them to be listed explicitly.
        locs : tuple of str
            A two-element tuple of a location specification.  For every pair in
            phases, the location specification refers to which location in the first phase is
            connected to which location in the second phase.  If the user wishes to specify
            different locations for different phase pairings, those phase pairings must be made
            in separate calls to link_phases.
        connected : bool
            Set to True to directly connect the phases being linked. Otherwise, create constraints
            for the optimizer to solve.
        units : str or None or _unspecified
            Units of the linkage.  If _unspecified, dymos will use the units from the variable
            in the first phase of the linkage.  Units of the two specified variables must be
            compatible.
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

        See Also
        --------
        add_linkage_constraint : Explicitly add a single phase linkage constraint.
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

        for phase_name_a, phase_name_b, loctup in zip(a, b, _locs):
            loc_a, loc_b = loctup
            for var in _vars:
                self.add_linkage_constraint(phase_a=phase_name_a, phase_b=phase_name_b,
                                            var_a=var, var_b=var, loc_a=loc_a, loc_b=loc_b,
                                            connected=connected, units=units,
                                            scaler=scaler, adder=adder, ref0=ref0, ref=ref,
                                            linear=linear)

    def simulate(self, times_per_seg=_unspecified, method=_unspecified, atol=_unspecified, rtol=_unspecified,
                 first_step=_unspecified, max_step=_unspecified, record_file=None, case_prefix=None,
                 reset_iter_counts=True, reports=False, interpolant='cubic'):
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
        first_step : float
            Initial step size for the integration.
        max_step : float
            Maximum step size for the integration.
        record_file : str or None
            If a string, the file to which the result of the simulation will be saved.
            If None, no record of the simulation will be saved.
        case_prefix : str or None
            Prefix to prepend to coordinates when recording.
        reset_iter_counts : bool
            If True and model has been run previously, reset all iteration counters.
        reports : bool or None or str or Sequence
            Reports setting for the subproblems run under simualate.
        interpolant : str
            The interpolation method to be used for the controls in the simulation phase.

        Returns
        -------
        problem
            An OpenMDAO Problem in which the simulation is implemented.  This Problem interface
            can be interrogated to obtain timeseries outputs in the same manner as other Phases
            to obtain results at the requested times.
        """
        sim_traj = Trajectory(sim_mode=True)

        for name, phs in self._phases.items():
            if phs.simulate_options is None:
                continue
            sim_phs = phs.get_simulation_phase(times_per_seg=times_per_seg, method=method,
                                               atol=atol, rtol=rtol, first_step=first_step,
                                               max_step=max_step, reports=reports,
                                               interpolant=interpolant)
            sim_traj.add_phase(name, sim_phs)

        if not sim_traj._phases:
            raise RuntimeError(f'Trajectory `{self.pathname}` has no phases that support simulation.')

        sim_traj.parameter_options.update(self.parameter_options)

        self.sim_prob = sim_prob = create_subprob(base_name=f'{self.name}_simulation',
                                                  comm=self.comm,
                                                  reports=reports)

        traj_name = self.name if self.name else 'sim_traj'
        sim_prob.model.add_subsystem(traj_name, sim_traj)

        if record_file is not None:
            rec = om.SqliteRecorder(record_file)
            sim_prob.add_recorder(rec)
            # record_outputs is needed to capture the timeseries outputs
            sim_prob.recording_options['record_outputs'] = True

        # Support model options
        sim_prob.model_options = self._problem_meta['model_options']

        with warnings.catch_warnings():
            # Some timeseries options are duplicated (expression options may be provide duplicate shape)
            # These filters suppress these warnings during simulation when they are not the
            # fault of the user.
            warnings.filterwarnings(action='ignore', category=om.UnusedOptionWarning)
            warnings.filterwarnings(action='ignore', category=om.SetupWarning)
            sim_prob.setup(check=False, parent=self)
            sim_prob.final_setup()

        # Assign trajectory parameter values
        for name in self.parameter_options:
            sim_traj.set_val(f'parameters:{name}', self.get_val(f'parameters:{name}'))

        for sim_phase_name, sim_phase in sim_traj._phases.items():
            if sim_phase._is_local:
                sim_phase.set_vals_from_phase(from_phase=self._phases[sim_phase_name])

        if sim_traj.comm.rank == 0:
            print(f'\nSimulating trajectory {self.pathname}')
        sim_prob.run_model(case_prefix=case_prefix, reset_iter_counts=reset_iter_counts)
        if sim_traj.comm.rank == 0:
            print(f'Done simulating trajectory {self.pathname}')
        if record_file:
            _case_prefix = '' if case_prefix is None else f'{case_prefix}_'
            sim_prob.record(f'{_case_prefix}final')
        sim_prob.cleanup()

        return sim_prob
