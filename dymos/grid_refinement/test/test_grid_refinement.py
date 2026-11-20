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
        phase.set_time_val(initial=0.0, duration=1.0)
        phase.set_state_val('x', [0, 0])
        phase.set_state_val('v', [-1, 1])
        phase.set_state_val('J', [0, 1])
        phase.set_control_val('u', [0, 0])

        #
        # Solve for the optimal trajectory
        #
        dm.run_problem(p, run_driver=True, simulate=True,
                       refine_method='ph', refine_iteration_limit=5)

        num_seg = p.model.traj.phases.phase0.options['transcription'].grid_data.num_segments
        seg_orders = p.model.traj.phases.phase0.options['transcription'].grid_data.transcription_order

        self.assertGreaterEqual(num_seg, 5)
        self.assertGreaterEqual(sum(seg_orders), 5 * 3)

    def test_refine_keeps_values_not_on_refined_grid(self):
        from openmdao.utils.assert_utils import assert_near_equal
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        # The default driver runs the model once, so the design variables keep their values.
        p = om.Problem()

        # A variable outside of the phases, set after setup.
        p.model.add_subsystem('ivc', om.IndepVarComp('h', val=1.0, units='m'))
        p.model.add_subsystem('comp', om.ExecComp('z = 2 * h', h={'units': 'm'}, z={'units': 'm'}))
        p.model.connect('ivc.h', 'comp.h')

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phases = {}
        for name, num_segments in (('coarse', 2), ('fine', 8)):
            phase = dm.Phase(ode_class=BrachistochroneODE,
                             transcription=dm.Radau(num_segments=num_segments, order=3))
            traj.add_phase(name, phase)
            phase.set_time_options(fix_initial=(name == 'coarse'), duration_bounds=(0.2, 10))
            phase.add_state('x', fix_initial=(name == 'coarse'))
            phase.add_state('y', fix_initial=(name == 'coarse'))
            phase.add_state('v', fix_initial=(name == 'coarse'))
            phase.add_control('theta', units='deg', lower=0.01, upper=179.9)
            phase.add_parameter('g', units='m/s**2', val=1.0, opt=False)
            phases[name] = phase

        phases['fine'].add_objective('time', loc='final')
        traj.link_phases(['coarse', 'fine'], vars=['time', 'x', 'y', 'v', 'theta'])

        # Only the first phase is refined.
        phases['coarse'].set_refine_options(refine=True, tol=1.0E-8)
        phases['fine'].set_refine_options(refine=False)

        p.setup()

        p.set_val('ivc.h', 5.0)

        phases['coarse'].set_time_val(initial=0.0, duration=0.9)
        phases['coarse'].set_state_val('x', [0, 3])
        phases['coarse'].set_state_val('y', [10, 7])
        phases['coarse'].set_state_val('v', [0, 7])
        phases['coarse'].set_control_val('theta', [5, 50])
        phases['coarse'].set_parameter_val('g', 9.80665)

        phases['fine'].set_time_val(initial=0.9, duration=0.9)
        phases['fine'].set_state_val('x', [3, 10])
        phases['fine'].set_state_val('y', [7, 5])
        phases['fine'].set_state_val('v', [7, 9.9])
        phases['fine'].set_control_val('theta', [50, 100])
        phases['fine'].set_parameter_val('g', 9.80665)

        p.final_setup()

        names = ['ivc.h', 'traj.fine.t_initial', 'traj.fine.t_duration', 'traj.fine.parameters:g',
                 'traj.fine.states:x', 'traj.fine.states:y', 'traj.fine.states:v',
                 'traj.fine.controls:theta']
        expected = {name: np.copy(p.get_val(name)) for name in names}
        coarse_nodes = p.get_val('traj.coarse.states:x').shape[0]

        dm.run_problem(p, refine_iteration_limit=1, simulate=False, make_plots=False)

        # The first phase is on a new grid.
        self.assertGreater(p.get_val('traj.coarse.states:x').shape[0], coarse_nodes)
        assert_near_equal(p.get_val('traj.coarse.parameters:g'), 9.80665)
        assert_near_equal(p.get_val('traj.coarse.timeseries.x')[-1, 0], 3.0, tolerance=1.0E-12)

        # The second phase and the variables outside of the phases keep their values.
        for name in names:
            with self.subTest(name=name):
                assert_near_equal(p.get_val(name), expected[name], tolerance=1.0E-12)


if __name__ == '__main__':
    unittest.main()
