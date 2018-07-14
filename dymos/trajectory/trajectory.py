from __future__ import print_function, division, absolute_import

import collections
import itertools
from six import iteritems

try:
    from itertools import izip
except ImportError:
    izip = zip

from openmdao.api import Group, ParallelGroup

from ..phases.components.phase_linkage_comp import PhaseLinkageComp


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
        # Required metadata
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
        min_procs
        max_procs
        proc_weight

        Returns
        -------

        """
        self._phases[name] = phase
        self._phase_add_kwargs[name] = kwargs

    def setup(self):

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

                varnames = vars.keys()
                linkage_name = '{0}|{1}'.format(phase_name1, phase_name2)
                max_varname_length = max(len(name) for name in varnames)

                link_comp.add_linkage(name=linkage_name,
                                      vars=varnames)

                for var, options in iteritems(vars):
                    loc1, loc2 = options['locs']

                    if var in p1_states:
                        source1 = 'states:{0}{1}'.format(var, loc1)
                    elif var in p1_controls:
                        source1 = 'controls:{0}{1}'.format(var, loc1)
                    elif var == 'time':
                        source1 = '{0}{1}'.format(var, loc1)

                    self.connect('{0}.{1}'.format(phase_name1, source1),
                                 'linkages.{0}_{1}:lhs'.format(linkage_name, var))

                    if var in p2_states:
                        source2 = 'states:{0}{1}'.format(var, loc2)
                    elif var in p2_controls:
                        source2 = 'controls:{0}{1}'.format(var, loc2)
                    elif var == 'time':
                        source2 = '{0}{1}'.format(var, loc2)

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

        Examples
        --------
        **Typical Phase Linkage Sequence**

        A typical phase linkage sequence, where all phases use the same ODE (and therefore have
        the same states), and are simply linked sequentially in time.

        >>> t.link_phases(['phase1', 'phase2', 'phase3'])

        **Adding an Additional Linkage**

        If we want some control variable, u, to be continuous in value and rate between phase2 and
        phase3 only, we could subsequently issue the following:

        >>> t.link_phases(['phase2', 'phase3'], vars=['u'])

        **Branching Trajectories**

        For a more complex example, consider the case where we have two phases which branch off
        from the same point, such as the case of a jettisonned stage.  The nominal trajectory
        consists of the phase sequence ['a', 'b', 'c'].  Let phase ['d'] be the phase that tracks
        the jettisonned component to its impact with the ground.  The linkages in this case
        would be defined as:

        >>> t.link_phases(['a', 'b', 'c'])
        >>> t.link_phases(['b', 'd'])

        **Specifying Linkage Locations**

        Phase linkages assume that, for each pair, the state/control values after any discontinuous
        jump in the first phase ('++') are linked to the state/control values before any
        discontinous jump in the second phase ('--').  The user can override this behavior, but
        they must specify a pair of location strings for each pair given in `phases`.  For instance,
        in the following example phases 'a' and 'b' have the same initial time and state, but
        phase 'c' follows phase 'b'.  Note since there are three phases provided, there are two
        linkages and thus two pairs of location specifiers given.

        >>> t.link_phases(['a', 'b', 'c'], locs=[('--', '--'), ('++', '--')])

        """
        num_links = len(phases) - 1

        if num_links <= 0:
            raise ValueError('Phase sequence must consists of at least two phases.')
        if isinstance(locs, collections.Sequence) and len(locs) != 2:
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
                self._linkages[phase1_name, phase2_name][var] = {'locs': locs}

    def simulate(self, parallel=True):
        raise NotImplementedError('Trajectory.simulate has not yet been implemented.')

    def get_values(self):
        raise NotImplementedError('Trajectory.get_values has not yet been implemented.')


