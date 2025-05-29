import unittest


import numpy as np
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
import dymos as dm


class _BrysonDenhamODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('x', shape=(nn,), units='m')
        self.add_input('v', shape=(nn,), units='m/s')
        self.add_input('u', shape=(nn,), units='m/s**2')

        self.add_output('J_dot', shape=(nn,), units='m**2/s**4',
                        tags=['dymos.state_rate_source:J',
                              'dymos.state_units:m**2/s**3'])

        ar = np.arange(nn, dtype=int)

        self.declare_partials(of='J_dot', wrt='u', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        u = inputs['u']
        outputs['J_dot'] = 0.5 * u ** 2

    def compute_partials(self, inputs, partials):
        partials['J_dot', 'u'] = inputs['u']


@use_tempdirs
class TestGridRefinement(unittest.TestCase):

    def test_refine_hp_non_ode_rate_sources(self):
        p = om.Problem()

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        tx = dm.Radau(num_segments=5)
        phase = traj.add_phase('phase0', dm.Phase(ode_class=_BrysonDenhamODE, transcription=tx))

        #
        # Set the variables
        #
        phase.set_time_options(fix_initial=True, fix_duration=True)

        phase.add_state('x', fix_initial=True, fix_final=True, rate_source='v')
        phase.add_state('v', fix_initial=True, fix_final=True, rate_source='u')
        phase.add_state('J', fix_initial=True, fix_final=False)
        phase.add_control('u', continuity=True, rate_continuity=False)

        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('J', loc='final', ref=1)
        phase.add_path_constraint('x', upper=1/9)

        #
        # Setup the Problem
        #
        p.setup()

        #
        # Set the initial values
        #
        phase.set_time_val(initial=0, duration=1)

        phase.set_state_val('x', [0, 0])
        phase.set_state_val('v', [1, -1])
        phase.set_state_val('J', [0, 1])
        phase.set_control_val('u', [0, 0])

        #
        # Solve for the optimal trajectory
        #
        dm.run_problem(p, run_driver=True, simulate=True, refine_iteration_limit=5)

        num_seg = p.model.traj.phases.phase0.options['transcription'].grid_data.num_segments
        seg_orders = p.model.traj.phases.phase0.options['transcription'].grid_data.transcription_order

        self.assertGreaterEqual(num_seg, 5)
        self.assertGreater(sum(seg_orders), 5 * 3)

    def test_refine_ph_non_ode_rate_sources(self):
        p = om.Problem()

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        tx = dm.Radau(num_segments=5, order=3)
        phase = traj.add_phase('phase0', dm.Phase(ode_class=_BrysonDenhamODE, transcription=tx))

        #
        # Set the variables
        #
        phase.set_time_options(fix_initial=True, fix_duration=True)

        phase.add_state('x', fix_initial=True, fix_final=True, rate_source='v')
        phase.add_state('v', fix_initial=True, fix_final=True, rate_source='u')
        phase.add_state('J', fix_initial=True, fix_final=False)
        phase.add_control('u', continuity=True, rate_continuity=False)

        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('J', loc='final', ref=1)
        phase.add_path_constraint('x', upper=1/9)

        #
        # Setup the Problem
        #
        p.setup()

        #
        # Set the initial values
        #
        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 1.0

        p.set_val('traj.phase0.states:x', phase.interp('x', ys=[0, 0]))
        p.set_val('traj.phase0.states:v', phase.interp('v', ys=[1, -1]))
        p.set_val('traj.phase0.states:J', phase.interp('J', ys=[0, 1]))
        p.set_val('traj.phase0.controls:u', np.sin(phase.interp('u', ys=[0, 0])))

        #
        # Solve for the optimal trajectory
        #
        dm.run_problem(p, run_driver=True, simulate=True,
                       refine_method='ph', refine_iteration_limit=5)

        num_seg = p.model.traj.phases.phase0.options['transcription'].grid_data.num_segments
        seg_orders = p.model.traj.phases.phase0.options['transcription'].grid_data.transcription_order

        self.assertGreaterEqual(num_seg, 5)
        self.assertGreater(sum(seg_orders), 5 * 3)
