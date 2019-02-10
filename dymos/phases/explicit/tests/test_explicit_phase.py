from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent, NonlinearBlockGS
from openmdao.utils.assert_utils import assert_rel_error

from dymos import ExplicitPhase, declare_time, declare_state
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


@declare_time(targets=['t'], units='s')
@declare_state('y', targets=['y'], rate_source='ydot', units='m')
class TestODE(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        self.add_input('t', val=np.ones(self.options['num_nodes']), units='s')
        self.add_input('y', val=np.ones(self.options['num_nodes']), units='m')
        self.add_output('ydot', val=np.ones(self.options['num_nodes']), units='m/s')

        ar = np.arange(self.options['num_nodes'])
        self.declare_partials(of='ydot', wrt='t', rows=ar, cols=ar)
        self.declare_partials(of='ydot', wrt='y', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        t = inputs['t']
        y = inputs['y']
        outputs['ydot'] = y - t ** 2 + 1

    def compute_partials(self, inputs, partials):
        partials['ydot', 't'] = -2 * inputs['t']


class TestExplicitPhaseNLRK(unittest.TestCase):

    def test_single_segment_simple_integration(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      ExplicitPhase(num_segments=1, transcription_order=3,
                                                    num_steps=4, ode_class=TestODE))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:y'] = 0.5

        p.run_model()

        np.set_printoptions(linewidth=1024)
        p.check_partials(compact_print=True, method='cs', out_stream=None)

        t = p['phase0.seg_0.t_step']
        y = p['phase0.seg_0.step_states:y']

        assert_rel_error(self,
                         t,
                         [0.0, 0.5, 1.0, 1.5, 2.0],
                         tolerance=1.0E-11)

        assert_rel_error(self,
                         y[:, 0],
                         [0.5, 1.425130208333333, 2.639602661132812, 4.006818970044454,
                          5.301605229265987],
                         tolerance=1.0E-11)

    def test_single_segment_with_controls(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      ExplicitPhase(num_segments=1, transcription_order=5,
                                                    num_steps=40, ode_class=BrachistochroneODE))

        phase.set_time_options(fix_initial=True, fix_duration=False)
        phase.set_state_options('x', fix_initial=True, fix_final=False)
        phase.set_state_options('y', fix_initial=True, fix_final=False)
        phase.set_state_options('v', fix_initial=True, fix_final=False)
        phase.add_control('theta', lower=0.0, upper=180.0, units='deg')
        phase.add_design_parameter('g', opt=False, val=9.80665)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_model()

        x = p['phase0.seg_0.step_states:x']
        y = p['phase0.seg_0.step_states:y']

        assert_rel_error(self, y[-1], 4.2513636, tolerance=1.0E-4)
        assert_rel_error(self, x[-1], 12.2137034, tolerance=1.0E-4)

    def test_multiple_segment_single_shooting_simple_integration(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      ExplicitPhase(num_segments=2, transcription_order=3,
                                                    num_steps=2, ode_class=TestODE))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:y'] = 0.5

        p.run_model()

        p.check_partials(compact_print=True, method='cs', out_stream=None)

        t_0 = p['phase0.seg_0.t_step']
        y_0 = p['phase0.seg_0.step_states:y']

        assert_rel_error(self,
                         t_0,
                         [0.0, 0.5, 1.0],
                         tolerance=1.0E-12)

        assert_rel_error(self,
                         y_0[:, 0],
                         [0.5, 1.425130208333333, 2.639602661132812],
                         tolerance=1.0E-12)

        t_1 = p['phase0.seg_1.t_step']
        y_1 = p['phase0.seg_1.step_states:y']

        assert_rel_error(self,
                         t_1,
                         [1.0, 1.5, 2.0],
                         tolerance=1.0E-12)

        assert_rel_error(self,
                         y_1[:, 0],
                         [2.639602661132812, 4.006818970044454, 5.301605229265987],
                         tolerance=1.0E-12)

    def test_multiple_segment_single_shooting_with_controls(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      ExplicitPhase(num_segments=2, transcription_order=5,
                                                    num_steps=40, ode_class=BrachistochroneODE))

        phase.set_time_options(fix_initial=True, fix_duration=False)
        phase.set_state_options('x', fix_initial=True, fix_final=False)
        phase.set_state_options('y', fix_initial=True, fix_final=False)
        phase.set_state_options('v', fix_initial=True, fix_final=False)
        phase.add_control('theta', lower=0.0, upper=180.0, units='deg')
        phase.add_design_parameter('g', opt=False, val=9.80665)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_model()

        x = p['phase0.seg_1.step_states:x']
        y = p['phase0.seg_1.step_states:y']

        assert_rel_error(self, y[-1], 4.2513636, tolerance=1.0E-4)
        assert_rel_error(self, x[-1], 12.2137034, tolerance=1.0E-4)

    def test_multiple_segment_hybrid_shooting_simple_integration(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      ExplicitPhase(num_segments=2, transcription_order=3,
                                                    num_steps=2, ode_class=TestODE,
                                                    shooting='hybrid'))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:y'] = 0.5

        p.run_model()

        np.set_printoptions(linewidth=1024)
        p.check_partials(compact_print=True, method='cs', out_stream=None)

        t_0 = p['phase0.seg_0.t_step']
        y_0 = p['phase0.seg_0.step_states:y']

        assert_rel_error(self,
                         t_0,
                         [0.0, 0.5, 1.0],
                         tolerance=1.0E-12)

        assert_rel_error(self,
                         y_0[:, 0],
                         [0.5, 1.425130208333333, 2.639602661132812],
                         tolerance=1.0E-12)

        t_1 = p['phase0.seg_1.t_step']
        y_1 = p['phase0.seg_1.step_states:y']

        assert_rel_error(self,
                         t_1,
                         [1.0, 1.5, 2.0],
                         tolerance=1.0E-12)

        assert_rel_error(self,
                         y_1[:, 0],
                         [2.639602661132812, 4.006818970044454, 5.301605229265987],
                         tolerance=1.0E-12)


class TestExplicitPhaseNLBGS(unittest.TestCase):

    def test_single_segment_simple_integration(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      ExplicitPhase(num_segments=1, transcription_order=3,
                                                    num_steps=4, ode_class=TestODE,
                                                    seg_solver_class=NonlinearBlockGS))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:y'] = 0.5

        p.run_model()

        np.set_printoptions(linewidth=1024)
        p.check_partials(compact_print=True, method='cs', out_stream=None)

        t = p['phase0.seg_0.t_step']
        y = p['phase0.seg_0.step_states:y']

        assert_rel_error(self,
                         t,
                         [0.0, 0.5, 1.0, 1.5, 2.0],
                         tolerance=1.0E-11)

        assert_rel_error(self,
                         y[:, 0],
                         [0.5, 1.425130208333333, 2.639602661132812, 4.006818970044454,
                          5.301605229265987],
                         tolerance=1.0E-11)

    def test_single_segment_with_controls(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      ExplicitPhase(num_segments=1, transcription_order=5,
                                                    num_steps=40, ode_class=BrachistochroneODE,
                                                    seg_solver_class=NonlinearBlockGS))

        phase.set_time_options(fix_initial=True, fix_duration=False)
        phase.set_state_options('x', fix_initial=True, fix_final=False)
        phase.set_state_options('y', fix_initial=True, fix_final=False)
        phase.set_state_options('v', fix_initial=True, fix_final=False)
        phase.add_control('theta', lower=0.0, upper=180.0, units='deg')
        phase.add_design_parameter('g', opt=False, val=9.80665)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_model()

        x = p['phase0.seg_0.step_states:x']
        y = p['phase0.seg_0.step_states:y']

        assert_rel_error(self, y[-1], 4.2513636, tolerance=1.0E-4)
        assert_rel_error(self, x[-1], 12.2137034, tolerance=1.0E-4)

    def test_multiple_segment_single_shooting_simple_integration(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      ExplicitPhase(num_segments=2, transcription_order=3,
                                                    num_steps=2, ode_class=TestODE,
                                                    seg_solver_class=NonlinearBlockGS))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:y'] = 0.5

        p.run_model()

        p.check_partials(compact_print=True, method='cs', out_stream=None)

        t_0 = p['phase0.seg_0.t_step']
        y_0 = p['phase0.seg_0.step_states:y']

        assert_rel_error(self,
                         t_0,
                         [0.0, 0.5, 1.0],
                         tolerance=1.0E-12)

        assert_rel_error(self,
                         y_0[:, 0],
                         [0.5, 1.425130208333333, 2.639602661132812],
                         tolerance=1.0E-12)

        t_1 = p['phase0.seg_1.t_step']
        y_1 = p['phase0.seg_1.step_states:y']

        assert_rel_error(self,
                         t_1,
                         [1.0, 1.5, 2.0],
                         tolerance=1.0E-12)

        assert_rel_error(self,
                         y_1[:, 0],
                         [2.639602661132812, 4.006818970044454, 5.301605229265987],
                         tolerance=1.0E-12)

    def test_multiple_segment_single_shooting_with_controls(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      ExplicitPhase(num_segments=2, transcription_order=5,
                                                    num_steps=40, ode_class=BrachistochroneODE,
                                                    seg_solver_class=NonlinearBlockGS))

        phase.set_time_options(fix_initial=True, fix_duration=False)
        phase.set_state_options('x', fix_initial=True, fix_final=False)
        phase.set_state_options('y', fix_initial=True, fix_final=False)
        phase.set_state_options('v', fix_initial=True, fix_final=False)
        phase.add_control('theta', lower=0.0, upper=180.0, units='deg')
        phase.add_design_parameter('g', opt=False, val=9.80665)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_model()

        x = p['phase0.seg_1.step_states:x']
        y = p['phase0.seg_1.step_states:y']

        print(p['phase0.time'])

        assert_rel_error(self, y[-1], 4.2513636, tolerance=1.0E-4)
        assert_rel_error(self, x[-1], 12.2137034, tolerance=1.0E-4)

    def test_multiple_segment_hybrid_shooting_simple_integration(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      ExplicitPhase(num_segments=2, transcription_order=3,
                                                    num_steps=2, ode_class=TestODE,
                                                    shooting='hybrid',
                                                    seg_solver_class=NonlinearBlockGS))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:y'] = 0.5

        p.run_model()

        np.set_printoptions(linewidth=1024)
        p.check_partials(compact_print=True, method='cs', out_stream=None)

        t_0 = p['phase0.seg_0.t_step']
        y_0 = p['phase0.seg_0.step_states:y']

        assert_rel_error(self,
                         t_0,
                         [0.0, 0.5, 1.0],
                         tolerance=1.0E-12)

        assert_rel_error(self,
                         y_0[:, 0],
                         [0.5, 1.425130208333333, 2.639602661132812],
                         tolerance=1.0E-12)

        t_1 = p['phase0.seg_1.t_step']
        y_1 = p['phase0.seg_1.step_states:y']

        assert_rel_error(self,
                         t_1,
                         [1.0, 1.5, 2.0],
                         tolerance=1.0E-12)

        assert_rel_error(self,
                         y_1[:, 0],
                         [2.639602661132812, 4.006818970044454, 5.301605229265987],
                         tolerance=1.0E-12)


class TestExplicitPhasePathConstraints(unittest.TestCase):

    def test_brachistochrone_explicit_path_constrained(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('explicit',
                      ode_class=BrachistochroneODE,
                      num_segments=4,
                      transcription_order=3,
                      num_steps=10)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9, ref0=0, ref=180.0)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        phase.add_path_constraint('y', lower=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=1)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p['phase0.time'][-1], 1.80297, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']

        exp_out = phase.simulate(times=np.linspace(t0, tf, 50), record=False)

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_explicit_path_constrained_nonlinear_rk(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase
        from dymos.phases.explicit.solvers.nl_rk_solver import NonlinearRK
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('explicit',
                      ode_class=BrachistochroneODE,
                      num_segments=4,
                      transcription_order=3,
                      num_steps=10,
                      seg_solver_class=NonlinearRK)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9, ref0=0, ref=180.0)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        phase.add_path_constraint('y', lower=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=1)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p['phase0.time'][-1], 1.80297, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']
        exp_out = phase.simulate(times=np.linspace(t0, tf, 50), record=False)

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()
