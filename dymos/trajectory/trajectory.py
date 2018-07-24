from __future__ import print_function, division, absolute_import

from collections import Sequence
import itertools
import multiprocessing as mp
from six import iteritems, string_types

try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np

from openmdao.api import Group, ParallelGroup

from ..phases.components.phase_linkage_comp import PhaseLinkageComp
from ..phases.phase_base import PhaseBase
from ..utils.simulation import simulate_phase_map_unpack
from ..utils.simulation.trajectory_simulation_results import TrajectorySimulationResults


class Trajectory(Group):
    """
    A Trajectory object serves as a container for one or more Phases, as well as the linkage
    conditions between phases.
    """
    def __init__(self, **kwargs):
        super(Trajectory, self).__init__(**kwargs)

        self._linkages = {}
        self._phases = {}
        self._phase_add_kwargs = {}

    def initialize(self):
        """
        Declare any options for Trajectory.
        """
        pass

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

    def setup(self):
        """
        Setup the Trajectory Group.
        """

        phases_group = self.add_subsystem('phases', subsys=ParallelGroup(), promotes_inputs=['*'],
                                          promotes_outputs=['*'])

        for name, phs in iteritems(self._phases):
            phases_group.add_subsystem(name, phs, **self._phase_add_kwargs[name])

        print('--- Linkage Report [{0}] ---'.format(self.pathname))

        if self._linkages:
            link_comp = self.add_subsystem('linkages', PhaseLinkageComp())

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

                    self.connect('{0}.{1}'.format(phase_name2, source2),
                                 'linkages.{0}_{1}:rhs'.format(linkage_name, var))

                    print('       {0:<{2}s} --> {1:<{2}s}'.format(source1, source2,
                                                                  max_varname_length+9))

            print('----------------------------')

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
        - '+-' specifies the value at the end of the phase before an initial state or control jump
        - '++' specifies the value at the end of the phase after an initial state or control jump

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
            self._linkages[phase1_name, phase2_name] = {}

            explicitly_linked_vars = [var for var in _vars if var != '*']

            if '*' in _vars:
                p1_states = set([key for key in self._phases[phase1_name].state_options])
                p2_states = set([key for key in self._phases[phase2_name].state_options])
                common_states = p1_states.intersection(p2_states)

                implicitly_linked_vars = ['time'] + list(common_states)
            else:
                implicitly_linked_vars = []

            linked_vars = list(set(implicitly_linked_vars + explicitly_linked_vars))

            for var in linked_vars:
                self._linkages[phase1_name, phase2_name][var] = {'locs': locs, 'units': None}

    def simulate(self, times='all', num_procs=None, record=True, record_file=None):
        """
        Simulate all phases in the trajectory using the scipy odeint interface.  If the trajectory
        has multiple phases, they can be simulated in parallel.

        Parameters
        ----------
        times : str, int, ndarray, dict of {str: str}, dict of {str: int}, dict of {str: ndarray}.
            The times at which outputs are desired, given as a node subset, an integer number of
            times evenly distributed throughout the phase, or an ndarray of time values.  If more
            control is desired, times can be passed as a dict keyed by phase name and the associated
            value being one of the three options above.
        num_procs : int, None
            The number of processors that should be used to simulate the phases in the trajectory.
            If None, then the number is obtained from the multiprocessing.cpu_count function.
        record : bool
            If True, record the resulting phase simulations to file.

        Returns
        -------
        dict of {str: PhaseSimulationResults}
            A dictionary, keyed by phase name, containing the PhaseSimulationResults object
            from each phase simulation.

        """

        if isinstance(times, dict):
            times_dict = times
        else:
            times_dict = {}
            for phase_name in self._phases:
                times_dict[phase_name] = times

        data = []

        for phase_name, phase in iteritems(self._phases):
            ode_class = phase.options['ode_class']
            ode_init_kwargs = phase.options['ode_init_kwargs']
            time_values = phase.get_values('time')
            state_values = {}
            control_values = {}
            design_parameter_values = {}
            for state_name, options in iteritems(phase.state_options):
                state_values[state_name] = phase.get_values(state_name, nodes='all')
            for control_name, options in iteritems(phase.control_options):
                control_values[control_name] = phase.get_values(control_name, nodes='all')
            for dp_name, options in iteritems(phase.design_parameter_options):
                design_parameter_values[dp_name] = phase.get_values(dp_name, nodes='all')

            data.append((phase_name, ode_class, phase.time_options, phase.state_options,
                         phase.control_options, phase.design_parameter_options, time_values,
                         state_values, control_values, design_parameter_values, ode_init_kwargs,
                         phase.grid_data, times_dict[phase_name], False, None))

        num_procs = mp.cpu_count() if num_procs is None else num_procs

        pool = mp.Pool(processes=num_procs)

        exp_outs = pool.map(simulate_phase_map_unpack, data)

        pool.close()
        pool.join()

        exp_outs_map = {}
        for i, phase_name in enumerate(self._phases.keys()):
            exp_outs_map[phase_name] = exp_outs[i]

        results = TrajectorySimulationResults(exp_outs=exp_outs_map)

        if record:
            if record_file is None:
                traj_name = self.pathname.split('.')[-1]
                if not traj_name:
                    traj_name = 'traj'
                record_file = '{0}_sim.db'.format(traj_name)
            print('Recording Results to {0}...'.format(record_file), end='')
            results.record_results(self, exp_outs_map, filename=record_file)
            print('Done')

        return results

    def get_values(self, var, phases=None, nodes='all', units=None, flat=False):
        """
        Returns the values of the given variable from the given phases, if provided.
        If the variable is not present in one ore more phases, it will be returned as
        numpy.nan at each time step.  If the variable does not exist in any phase within the
        trajectory, KeyError is raised.

        Parameters
        ----------
        var : str
            The variable whose values are to be returned.
        phases : Sequence, None
            The phases from which the values are desired.  If None, included all Phases.
        units : str, None
            The units in which the values are desired.
        nodes : str or dict of {str: str}
            The node subset for which the values should be provided.  If given as a string,
            provide the same subset for all phases.  If given as a dictionary, provide a
            mapping of phase names to node subsets.  If an invalid node subset is requested
            in one or more phases, a ValueError is raised.
        flat : bool
            If False return the values in a dictionary keyed by phase name.  If True,
            return a single array incorporating values from all phases.

        Returns
        -------
        dict or np.array
            If flat=False, a dictionary of the values of the variable in each phase will be
            returned, keyed by Phase name.  If the values are not present in a subset of the phases,
            return numpy.nan at each time point in those phases.

        Raises
        ------
        KeyError
            If the given variable is not found in any phase, a KeyError is raised.

        """
        phase_names = phases if phases is not None else list(self._phases.keys())

        results = {}
        time_units = None
        times = {}

        if isinstance(nodes, string_types):
            node_map = dict([(phase_name, nodes) for phase_name in phase_names])
        else:
            node_map = nodes

        var_in_traj = False
        for phase_name in phase_names:
            p = self._phases[phase_name]

            if time_units is None:
                time_units = p.time_options['units']
            times[phase_name] = p.get_values('time', nodes=node_map[phase_name], units=time_units)

            try:
                results[phase_name] = p.get_values(var, nodes=node_map[phase_name], units=units)
                var_in_traj = True
            except KeyError:
                num_nodes = p.grid_data.subset_num_nodes[node_map[phase_name]]
                results[phase_name] = np.empty((num_nodes, 1))
                results[phase_name][:, :] = np.nan

        if not var_in_traj:
            raise KeyError('Variable "{0}" not found in trajectory.'.format(var))

        if flat:
            time_array = np.concatenate([times[pname] for pname in phase_names])
            sort_idxs = np.argsort(time_array, axis=0).ravel()
            results = np.concatenate([results[pname] for pname in phase_names])[sort_idxs, ...]

        return results
