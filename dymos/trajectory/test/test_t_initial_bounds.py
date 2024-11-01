import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_warning


def make_phases(traj, time_option_kwargs):
    for pname, kwargs in time_option_kwargs.items():
        phase = dm.Phase(ode_class=OscillatorODE, transcription=dm.Radau(num_segments=10))
        traj.add_phase(pname, phase)

        phase.set_time_options(**kwargs)

        # Tell Dymos the states to be propagated using the given ODE.
        phase.add_state('v', rate_source='v_dot', targets=['v'], units='m/s')
        phase.add_state('x', rate_source='v', targets=['x'], units='m')

        # The spring constant, damping coefficient, and mass are inputs to the system
        # that are constant throughout the phase.
        phase.add_parameter('k', units='N/m', targets=['k'])
        phase.add_parameter('c', units='N*s/m', targets=['c'])
        phase.add_parameter('m', units='kg', targets=['m'])

    # Since we're using an optimization driver, an objective is required.  We'll minimize
    # the final time in this case.
    phase.add_objective('time', loc='final')


def connect_phases(traj, conns):
    for src, tgts in conns:
        for tgt in tgts:
            traj.link_phases((src, tgt), vars=['time'])


class OscillatorODE(om.ExplicitComponent):
    """
    A Dymos ODE for a damped harmonic oscillator.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('x', shape=(nn,), desc='displacement', units='m')
        self.add_input('v', shape=(nn,), desc='velocity', units='m/s')
        self.add_input('k', shape=(nn,), desc='spring constant', units='N/m')
        self.add_input('c', shape=(nn,), desc='damping coefficient', units='N*s/m')
        self.add_input('m', shape=(nn,), desc='mass', units='kg')

        self.add_output('v_dot', val=np.ones(nn), desc='rate of change of velocity', units='m/s**2')

        arange = np.arange(self.options['num_nodes'], dtype=int)
        self.declare_partials(of='*', wrt='*', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        x = inputs['x']
        v = inputs['v']
        k = inputs['k']
        c = inputs['c']
        m = inputs['m']

        f_spring = -k * x
        f_damper = -c * v

        outputs['v_dot'] = (f_spring + f_damper) / m

    def compute_partials(self, inputs, jacobian):
        x = inputs['x']
        v = inputs['v']
        k = inputs['k']
        c = inputs['c']
        m = inputs['m']

        jacobian['v_dot', 'x'] = -k / m
        jacobian['v_dot', 'v'] = -c / m
        jacobian['v_dot', 'k'] = -x / m
        jacobian['v_dot', 'c'] = -v / m
        jacobian['v_dot', 'm'] = -1. / (m * m) * (-k * x - c * v)


@use_tempdirs
class Test_t_initialBounds(unittest.TestCase):
    def try_model(self, probname, kwargs, conns):
        prob = om.Problem(name=probname)
        prob.driver = om.ScipyOptimizeDriver()
        traj = prob.model.add_subsystem('traj', dm.Trajectory())

        make_phases(traj, kwargs)
        connect_phases(traj, conns)

        prob.setup(force_alloc_complex=True)

        dm.run_problem(prob, run_driver=False)

    def test_pair_fixed_t_initial_below(self):
        kwargs = {
            'phase0': {
                'fix_initial': True,
                'initial_val': 5.,
                'initial_bounds': (None, None),
                'fix_duration': False,
                'duration_bounds': (5., 10.),
            },
            'phase1': {
                'fix_initial': True,
                'initial_val': 5.,
                'initial_bounds': (None, None),
                'fix_duration': False,
                'duration_bounds': (5., 10.),
            }
        }
        conns = [('phase0', ['phase1'])]

        with self.assertRaises(Exception) as cm:
            self.try_model('pair_fixed_t_initial_below', kwargs, conns)

        msg = ("'traj' <class Trajectory>: Fixed t_initial of 5.0 is outside of allowed bounds "
               "(10.0, 15.0) for phase 'phase1'.")
        self.assertEqual(cm.exception.args[0], msg)

    def test_pair_fixed_t_initial_above(self):
        kwargs = {
            'phase0': {
                'fix_initial': True,
                'initial_val': 5.,
                'initial_bounds': (None, None),
                'fix_duration': False,
                'duration_bounds': (5., 10.),
            },
            'phase1': {
                'fix_initial': True,
                'initial_val': 99.,
                'initial_bounds': (None, None),
                'fix_duration': False,
                'duration_bounds': (5., 10.),
            }
        }
        conns = [('phase0', ['phase1'])]

        with self.assertRaises(Exception) as cm:
            self.try_model('pair_fixed_t_initial_above', kwargs, conns)

        msg = ("'traj' <class Trajectory>: Fixed t_initial of 99.0 is outside of allowed bounds "
               "(10.0, 15.0) for phase 'phase1'.")
        self.assertEqual(cm.exception.args[0], msg)

    def test_pair_t_initial_bounds_below(self):
        kwargs = {
            'phase0': {
                'initial_bounds': (3., 7.),
                'fix_duration': False,
                'duration_bounds': (5., 10.),
            },
            'phase1': {
                'initial_bounds': (5., 7.),
                'fix_duration': False,
                'duration_bounds': (5., 10.),
            }
        }
        conns = [('phase0', ['phase1'])]

        with self.assertRaises(Exception) as cm:
            self.try_model('pair_t_initial_bounds_below', kwargs, conns)

        msg = ("'traj' <class Trajectory>: t_initial bounds of (5.0, 7.0) do not overlap with "
               "allowed bounds (8.0, 17.0) for phase 'phase1'.")
        self.assertEqual(cm.exception.args[0], msg)

    def test_pair_t_initial_bounds_above(self):
        kwargs = {
            'phase0': {
                'initial_bounds': (3., 7.),
                'fix_duration': False,
                'duration_bounds': (5., 10.),
            },
            'phase1': {
                'initial_bounds': (20., 22.),
                'fix_duration': False,
                'duration_bounds': (5., 10.),
            }
        }
        conns = [('phase0', ['phase1'])]

        with self.assertRaises(Exception) as cm:
            self.try_model('pair_t_initial_bounds_above', kwargs, conns)

        msg = ("'traj' <class Trajectory>: t_initial bounds of (20.0, 22.0) do not overlap with "
               "allowed bounds (8.0, 17.0) for phase 'phase1'.")
        self.assertEqual(cm.exception.args[0], msg)

    def test_pair_no_duration_bounds(self):
        kwargs = {
            'phase0': {
                'initial_bounds': (10., 15.),
                'fix_duration': False,
            },
            'phase1': {
                'initial_bounds': (5., 7.),
                'fix_duration': False,
            }
        }
        conns = [('phase0', ['phase1'])]

        # no exception should be raised
        self.try_model('pair_t_initial_bounds', kwargs, conns)

    def test_all_fixed_t_initial(self):
        nphases = 3

        kwargs = {}
        conns = []
        for i in range(nphases):
            pname = f'phase{i}'
            kwargs[pname] = dct = {}
            dct['fix_initial'] = True
            dct['fix_duration'] = False
            dct['initial_val'] = 5. * i if i < nphases - 1 else 5. * (i - 1)
            dct['duration_bounds'] = (5. * i, 5 * (i + 1))

            if i > 0:
                conns.append((f'phase{i-1}', [pname]))

        with self.assertRaises(Exception) as cm:
            self.try_model('all_fixed', kwargs, conns)

        msg = ("'traj' <class Trajectory>: Fixed t_initial of 5.0 is outside of allowed "
               "bounds (10.0, 15.0) for phase 'phase2'.")
        self.assertEqual(cm.exception.args[0], msg)

    def test_all_t_initial_bounds(self):
        nphases = 3

        kwargs = {}
        conns = []
        for i in range(nphases):
            pname = f'phase{i}'
            kwargs[pname] = dct = {}
            dct['fix_duration'] = False
            t = 5. * i if i < nphases - 1 else 99.
            dct['initial_bounds'] = (t, t + 5.)
            dct['duration_bounds'] = (5. * i, 5 * (i + 1))

            if i > 0:
                conns.append((f'phase{i-1}', [pname]))

        with self.assertRaises(Exception) as cm:
            self.try_model('all_t_initial_bounds', kwargs, conns)

        msg = ("'traj' <class Trajectory>: t_initial bounds of (99.0, 104.0) do not overlap with "
               "allowed bounds (10.0, 20.0) for phase 'phase2'.")
        self.assertEqual(cm.exception.args[0], msg)

    def test_odd_fixed_t_initial(self):
        nphases = 4

        kwargs = {}
        conns = []
        for i in range(nphases):
            pname = f'phase{i}'
            kwargs[pname] = dct = {}
            dct['fix_initial'] = i % 2 != 0
            dct['fix_duration'] = False
            dct['initial_val'] = 5. * i if i < nphases - 1 else 5. * (i-1)
            dct['duration_bounds'] = (5. * i, 5 * (i + 1))

            if i > 0:
                conns.append((f'phase{i - 1}', [pname]))

        msg = ("'traj' <class Trajectory>: Fixed t_initial of 10.0 is outside of allowed "
               "bounds (20.0, 30.0) for phase 'phase3'.")

        with self.assertRaises(Exception) as cm:
            self.try_model('odd_fixed', kwargs, conns)

        self.assertEqual(cm.exception.args[0], msg)

    def test_odd_t_initial_bounds(self):
        nphases = 4

        kwargs = {}
        conns = []
        for i in range(nphases):
            pname = f'phase{i}'
            kwargs[pname] = dct = {}
            if i % 2 != 0:
                t = 5. * i if i < nphases - 1 else 99.
                dct['initial_bounds'] = (t, t + 5.)
            dct['fix_duration'] = False
            dct['duration_bounds'] = (5. * i, 5 * (i + 1))

            if i > 0:
                conns.append((f'phase{i - 1}', [pname]))

        msg = ("'traj' <class Trajectory>: t_initial bounds of (99.0, 104.0) do not overlap with "
               "allowed bounds (20.0, 35.0) for phase 'phase3'.")

        with self.assertRaises(Exception) as cm:
            self.try_model('odd_t_initial_bounds', kwargs, conns)

        self.assertEqual(cm.exception.args[0], msg)

    def test_even_fixed_t_initial(self):
        nphases = 5

        kwargs = {}
        conns = []
        for i in range(nphases):
            pname = f'phase{i}'
            kwargs[pname] = dct = {}
            dct['fix_initial'] = i % 2 == 0
            dct['fix_duration'] = False
            dct['initial_val'] = 5. * i
            dct['duration_bounds'] = (5. * i, 5 * (i + 1))

            if i > 0:
                conns.append((f'phase{i - 1}', [pname]))

        with self.assertRaises(Exception) as cm:
            self.try_model('even_fixed', kwargs, conns)

        msg = ("'traj' <class Trajectory>: "
               "Fixed t_initial of 20.0 is outside of allowed bounds (35.0, 45.0) for phase "
               "'phase4'.")

        self.assertEqual(cm.exception.args[0], msg)

    def test_even_t_initial_bounds(self):
        nphases = 3

        kwargs = {}
        conns = []
        for i in range(nphases):
            pname = f'phase{i}'
            kwargs[pname] = dct = {}
            if i % 2 == 0:
                t = 5. * i if i < nphases - 1 else 99.
                dct['initial_bounds'] = (t, t + 5.)
            dct['fix_duration'] = False
            dct['duration_bounds'] = (5. * i, 5 * (i + 1))

            if i > 0:
                conns.append((f'phase{i - 1}', [pname]))

        msg = ("'traj' <class Trajectory>: t_initial bounds of (99.0, 104.0) do not overlap with "
               "allowed bounds (5.0, 20.0) for phase 'phase2'.")

        with self.assertRaises(Exception) as cm:
            self.try_model('even_t_initial_bounds', kwargs, conns)

        self.assertEqual(cm.exception.args[0], msg)

    def test_branching_all_fixed_t_initial(self):
        nphases = 3  # number of phases in trunk and each branch
        nbranches = 2

        kwargs = {}
        conns = []
        branches = [f'br{i}' for i in range(nbranches)]

        # trunk phases
        for i in range(nphases):
            pname = f'phase{i}'
            kwargs[pname] = dct = {}
            dct['fix_initial'] = True
            dct['fix_duration'] = False
            dct['initial_val'] = 5.*i
            dct['duration_bounds'] = (5, 10)

            if i > 0:
                conns.append((f'phase{i-1}', [pname]))

        last_trunk_phase = pname
        last_t_min = dct['initial_val'] + dct['duration_bounds'][0]

        for br in branches:
            for i in range(nphases):
                pname = f'{br}_phase{i}'
                kwargs[pname] = dct = {}
                dct['fix_initial'] = True
                dct['fix_duration'] = False
                dct['initial_val'] = last_t_min + 5.*(i-1)
                dct['duration_bounds'] = (5, 10)

                if i > 0:
                    conns.append((f'{br}_phase{i-1}', [pname]))

            # connect branch to trunk
            conns.append((last_trunk_phase, [f'{br}_phase0']))

        with self.assertRaises(Exception) as cm:
            self.try_model('branching_all_fixed', kwargs, conns)

        msg = ("'traj' <class Trajectory>: Fixed t_initial of 10.0 is outside of allowed bounds "
               "(15.0, 20.0) for phase 'br0_phase0'.\n"
               "Fixed t_initial of 10.0 is outside of allowed bounds (15.0, 20.0) for phase "
               "'br1_phase0'.")
        self.assertEqual(cm.exception.args[0], msg)

    def test_branching_all_t_initial_bounds(self):
        nphases = 3  # number of phases in trunk and each branch
        nbranches = 2

        kwargs = {}
        conns = []
        branches = [f'br{i}' for i in range(nbranches)]

        # trunk phases
        for i in range(nphases):
            pname = f'phase{i}'
            kwargs[pname] = dct = {}
            dct['fix_duration'] = False
            dct['initial_bounds'] = (5. * i, 5. * (i + 1))
            dct['duration_bounds'] = (5, 10)

            if i > 0:
                conns.append((f'phase{i-1}', [pname]))

        last_trunk_phase = pname
        last_t_min = dct['initial_bounds'][0] + dct['duration_bounds'][0]

        for br in branches:
            for i in range(nphases):
                pname = f'{br}_phase{i}'
                kwargs[pname] = dct = {}
                dct['fix_duration'] = False
                if i == 1:
                    dct['initial_bounds'] = (0., 5)
                else:
                    t = last_t_min + 5.*(i-1)
                    dct['initial_bounds'] = (t, t + 10.)
                dct['duration_bounds'] = (5, 10)

                if i > 0:
                    conns.append((f'{br}_phase{i-1}', [pname]))

            # connect branch to trunk
            conns.append((last_trunk_phase, [f'{br}_phase0']))

        with self.assertRaises(Exception) as cm:
            self.try_model('branching_all_t_initial_bounds', kwargs, conns)

        msg = ("'traj' <class Trajectory>: t_initial bounds of (0.0, 5) do not overlap with "
               "allowed bounds (20.0, 30.0) for phase 'br0_phase1'.\n"
               "t_initial bounds of (0.0, 5) do not overlap with allowed bounds (20.0, 30.0) "
               "for phase 'br1_phase1'.")
        self.assertEqual(cm.exception.args[0], msg)

    def test_branching_odd_fixed_t_initial(self):
        nphases = 3  # number of phases in trunk and each branch
        nbranches = 2
        nbrphases = 4

        kwargs = {}
        conns = []
        branches = [f'br{i}' for i in range(nbranches)]

        # trunk phases
        for i in range(nphases):
            pname = f'phase{i}'
            kwargs[pname] = dct = {}
            dct['fix_initial'] = i % 2 != 0
            dct['fix_duration'] = False
            dct['initial_val'] = 5.*i
            dct['duration_bounds'] = (5, 10)

            if i > 0:
                conns.append((f'phase{i-1}', [pname]))

        last_trunk_phase = pname
        last_t_min = dct['initial_val'] + dct['duration_bounds'][0]

        for br in branches:
            for i in range(nbrphases):
                pname = f'{br}_phase{i}'
                kwargs[pname] = dct = {}
                dct['fix_initial'] = i % 2 != 0
                dct['fix_duration'] = False
                dct['initial_val'] = last_t_min + 15.*i
                dct['duration_bounds'] = (5, 10)

                if i > 0:
                    conns.append((f'{br}_phase{i-1}', [pname]))

            # connect branch to trunk
            conns.append((last_trunk_phase, [f'{br}_phase0']))

        with self.assertRaises(Exception) as cm:
            self.try_model('branching_odd_fixed', kwargs, conns)

        msg = ("'traj' <class Trajectory>: Fixed t_initial of 60.0 is outside of allowed bounds "
               "(40.0, 50.0) for phase 'br0_phase3'.\n"
               "Fixed t_initial of 60.0 is outside of allowed bounds (40.0, 50.0) for phase "
               "'br1_phase3'.")
        self.assertEqual(cm.exception.args[0], msg)
