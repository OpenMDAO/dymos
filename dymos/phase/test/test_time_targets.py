import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm


class _BrachistochroneTestODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

        self.add_input('g', val=9.80665 * np.ones(nn), desc='grav. acceleration', units='m/s/s')

        self.add_input('theta', val=np.zeros(nn), desc='angle of wire', units='rad')

        self.add_input('t_initial', val=-88.0, desc='start time of phase', units='s')

        self.add_input('t_duration', val=-89.0, desc='total duration of phase', units='s')

        self.add_input('time_phase', val=np.zeros(nn), desc='elapsed time of phase', units='s')

        self.add_input('dt_dstau', val=np.zeros(nn), desc='segment time ratio', units='s')

        self.add_input('time', val=np.zeros(nn), desc='time of phase', units='s')

        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s')

        self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s')

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2')

        self.add_output('check', val=np.zeros(nn), desc='check solution: v/sin(theta) = constant',
                        units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange)
        self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='check', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='check', wrt='theta', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g * cos_theta
        outputs['xdot'] = v * sin_theta
        outputs['ydot'] = -v * cos_theta
        outputs['check'] = v / sin_theta

    def compute_partials(self, inputs, jacobian):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        jacobian['vdot', 'g'] = cos_theta
        jacobian['vdot', 'theta'] = -g * sin_theta

        jacobian['xdot', 'v'] = sin_theta
        jacobian['xdot', 'theta'] = v * cos_theta

        jacobian['ydot', 'v'] = -cos_theta
        jacobian['ydot', 'theta'] = v * sin_theta

        jacobian['check', 'v'] = 1 / sin_theta
        jacobian['check', 'theta'] = -v * cos_theta / sin_theta**2


@use_tempdirs
class TestPhaseTimeTargets(unittest.TestCase):

    def _make_problem(self, transcription, num_seg, transcription_order=3, input_initial=False,
                      input_duration=False, time_name='time', dt_dstau_targets=None):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        # Compute sparsity/coloring when run_driver is called
        p.driver.declare_coloring()

        t = {'gauss-lobatto': dm.GaussLobatto(num_segments=num_seg, order=transcription_order),
             'radau-ps': dm.Radau(num_segments=num_seg, order=transcription_order),
             'explicit-shooting': dm.ExplicitShooting(num_segments=num_seg, grid='radau-ps'),
             'birkhoff': dm.Birkhoff(num_nodes=20),
             'picard-shooting': dm.PicardShooting(num_segments=1, nodes_per_seg=20)}

        phase = dm.Phase(ode_class=_BrachistochroneTestODE, transcription=t[transcription])

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(1, 1), duration_bounds=(.5, 10), units='s',
                               time_phase_targets=['time_phase'], t_duration_targets=['t_duration'],
                               t_initial_targets=['t_initial'], targets=['time'], dt_dstau_targets=dt_dstau_targets,
                               input_initial=input_initial, input_duration=input_duration, name=time_name)

        phase.add_state('x', fix_initial=True, rate_source='xdot', units='m')
        phase.add_state('y', fix_initial=True, rate_source='ydot', units='m')
        phase.add_state('v', fix_initial=True, rate_source='vdot', targets=['v'], units='m/s')

        phase.add_control('theta', units='deg', rate_continuity=True, lower=0.01, upper=179.9, targets=['theta'])

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665, targets=['g'])

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        phase.timeseries_options['include_t_phase'] = True

        # Minimize time at the end of the phase
        phase.add_objective(time_name, loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        if input_duration:
            p.model.add_design_var('phase0.t_duration', lower=0, upper=3, scaler=1.0)

        p.setup(force_alloc_complex=True)

        phase.set_time_val(initial=0, duration=2.0)
        phase.set_state_val('x', (0, 10))
        phase.set_state_val('y', (10, 5))
        phase.set_state_val('v', (0, 9.9))
        phase.set_control_val('theta', (5, 100))
        phase.set_parameter_val('g', 9.80665)

        return p

    def test_gauss_lobatto(self):
        num_seg = 20

        for time_name in ('time', 'elapsed_time'):
            with self.subTest():

                p = self._make_problem('gauss-lobatto', num_seg, time_name=time_name)

                # Solve for the optimal trajectory
                p.run_driver()

                gd = p.model.phase0.options['transcription'].grid_data

                time_all = p[f'phase0.timeseries.{time_name}'].ravel()
                time_col = time_all[gd.subset_node_indices['col']]
                time_disc = time_all[gd.subset_node_indices['state_disc']]

                time_phase_all = p[f'phase0.timeseries.{time_name}_phase'].ravel()
                time_phase_col = time_phase_all[gd.subset_node_indices['col']]
                time_phase_disc = time_phase_all[gd.subset_node_indices['state_disc']]

                assert_near_equal(p['phase0.rhs_disc.time_phase'][-1], 1.8016, tolerance=1.0E-3)

                assert_near_equal(p['phase0.rhs_disc.t_initial'], p['phase0.t_initial'])
                assert_near_equal(p['phase0.rhs_col.t_initial'], p['phase0.t_initial'])

                assert_near_equal(p['phase0.rhs_disc.t_duration'], p['phase0.t_duration'])
                assert_near_equal(p['phase0.rhs_col.t_duration'], p['phase0.t_duration'])

                assert_near_equal(p['phase0.rhs_disc.time_phase'], time_phase_disc)
                assert_near_equal(p['phase0.rhs_col.time_phase'], time_phase_col)

                assert_near_equal(p['phase0.rhs_disc.time'], time_disc)
                assert_near_equal(p['phase0.rhs_col.time'], time_col)

                exp_out = p.model._get_subsystem('phase0').simulate()

                time_comp = exp_out.model.phase0._get_subsystem('time')
                integrator_comp = exp_out.model.phase0._get_subsystem('integrator')
                ode = exp_out.model.phase0._get_subsystem('ode')
                timeseries_comp = exp_out.model.phase0._get_subsystem('timeseries')

                time_comp_t_initial = time_comp.get_val('t_initial')
                integrator_comp_t_initial = integrator_comp.get_val('t_initial')
                ode_t_initial = ode.get_val('t_initial')

                time_comp_t_duration = time_comp.get_val('t_duration')
                integrator_comp_t_duration = integrator_comp.get_val('t_duration')
                ode_t_duration = ode.get_val('t_duration')
                ode_time_phase = ode.get_val('time_phase')
                timeseries_time_phase = timeseries_comp.get_val(f'{time_name}_phase')

                assert_near_equal(time_comp_t_initial, p['phase0.t_initial'])
                assert_near_equal(integrator_comp_t_initial, p['phase0.t_initial'])
                assert_near_equal(ode_t_initial, p['phase0.t_initial'])

                assert_near_equal(time_comp_t_duration, p['phase0.t_duration'])
                assert_near_equal(integrator_comp_t_duration, p['phase0.t_duration'])
                assert_near_equal(ode_t_duration, p['phase0.t_duration'])

                assert_near_equal(ode_time_phase.ravel(), timeseries_time_phase.ravel())

    def test_radau(self):
        for time_name in ('time', 'elapsed_time'):
            for dt_dstau_targets in ('dt_dstau', []):
                with self.subTest():
                    num_seg = 20
                    p = self._make_problem('radau-ps', num_seg, time_name=time_name,
                                           dt_dstau_targets=dt_dstau_targets)

                    # Solve for the optimal trajectory
                    p.run_driver()

                    time_all = p[f'phase0.timeseries.{time_name}'].ravel()

                    time_phase_all = p[f'phase0.timeseries.{time_name}_phase'].ravel()

                    assert_near_equal(p['phase0.rhs_all.time_phase'][-1], 1.8016, tolerance=1.0E-3)

                    assert_near_equal(p['phase0.rhs_all.t_initial'], p['phase0.t_initial'])

                    assert_near_equal(p['phase0.rhs_all.t_duration'], p['phase0.t_duration'])

                    assert_near_equal(p['phase0.rhs_all.time_phase'], time_phase_all)

                    assert_near_equal(p['phase0.rhs_all.time'], time_all)

                    if dt_dstau_targets:
                        assert_near_equal(p['phase0.rhs_all.dt_dstau'], p['phase0.dt_dstau'])
                        with self.assertRaises(ValueError) as e:
                            exp_out = p.model.phase0.simulate()

                        expected = 'dt_dstau_targets in ExplicitShooting are not supported at this time.'
                        self.assertEqual(expected, str(e.exception))
                        continue
                    else:
                        exp_out = p.model.phase0.simulate()

                    time_comp = exp_out.model.phase0._get_subsystem('time')
                    integrator_comp = exp_out.model.phase0._get_subsystem('integrator')
                    ode = exp_out.model.phase0._get_subsystem('ode')
                    timeseries_comp = exp_out.model.phase0._get_subsystem('timeseries')

                    time_comp_t_initial = time_comp.get_val('t_initial')
                    integrator_comp_t_initial = integrator_comp.get_val('t_initial')
                    ode_t_initial = ode.get_val('t_initial')

                    time_comp_t_duration = time_comp.get_val('t_duration')
                    integrator_comp_t_duration = integrator_comp.get_val('t_duration')
                    ode_t_duration = ode.get_val('t_duration')
                    ode_time_phase = ode.get_val('time_phase')
                    timeseries_time_phase = timeseries_comp.get_val(f'{time_name}_phase')

                    assert_near_equal(time_comp_t_initial, p['phase0.t_initial'])
                    assert_near_equal(integrator_comp_t_initial, p['phase0.t_initial'])
                    assert_near_equal(ode_t_initial, p['phase0.t_initial'])

                    assert_near_equal(time_comp_t_duration, p['phase0.t_duration'])
                    assert_near_equal(integrator_comp_t_duration, p['phase0.t_duration'])
                    assert_near_equal(ode_t_duration, p['phase0.t_duration'])

                    assert_near_equal(ode_time_phase.ravel(), timeseries_time_phase.ravel())

    def test_explicit_shooting(self):
        num_seg = 5

        for time_name in ('time', 'elapsed_time'):
            with self.subTest():
                p = self._make_problem('explicit-shooting', num_seg, time_name=time_name)

                # Solve for the optimal trajectory
                p.run_driver()

                time_all = p[f'phase0.timeseries.{time_name}']

                time_phase_all = p[f'phase0.timeseries.{time_name}_phase']

                assert_near_equal(p['phase0.t_phase'][-1], 1.8016, tolerance=1.0E-3)

                assert_near_equal(p['phase0.t_initial'], p['phase0.integrator.t_initial'])

                assert_near_equal(p['phase0.t_duration'], p['phase0.integrator.t_duration'])

                assert_near_equal(np.atleast_2d(p['phase0.t_phase']).T, time_phase_all)

                assert_near_equal(np.atleast_2d(p['phase0.t']).T, time_all)

    def test_gauss_lobatto_targets_are_inputs(self):
        num_seg = 20
        p = self._make_problem('gauss-lobatto', num_seg, input_initial=True, input_duration=True)

        # Solve for the optimal trajectory
        p.run_driver()

        gd = p.model.phase0.options['transcription'].grid_data

        time_all = p['phase0.t']
        time_col = time_all[gd.subset_node_indices['col']]
        time_disc = time_all[gd.subset_node_indices['state_disc']]
        time_segends = np.reshape(time_all[gd.subset_node_indices['segment_ends']],  # noqa: F841
                                  (gd.num_segments, 2))

        time_phase_all = p['phase0.t_phase']
        time_phase_col = time_phase_all[gd.subset_node_indices['col']]
        time_phase_disc = time_phase_all[gd.subset_node_indices['state_disc']]
        time_phase_segends = np.reshape(time_phase_all[gd.subset_node_indices['segment_ends']],  # noqa: F841
                                        (gd.num_segments, 2))

        assert_near_equal(p['phase0.rhs_disc.time_phase'][-1], 1.8016, tolerance=1.0E-3)

        assert_near_equal(p['phase0.rhs_disc.t_initial'], p['phase0.t_initial'])
        assert_near_equal(p['phase0.rhs_col.t_initial'], p['phase0.t_initial'])

        assert_near_equal(p['phase0.rhs_disc.t_duration'], p['phase0.t_duration'])
        assert_near_equal(p['phase0.rhs_col.t_duration'], p['phase0.t_duration'])

        assert_near_equal(p['phase0.rhs_disc.time_phase'], time_phase_disc)
        assert_near_equal(p['phase0.rhs_col.time_phase'], time_phase_col)

        assert_near_equal(p['phase0.rhs_disc.time'], time_disc)
        assert_near_equal(p['phase0.rhs_col.time'], time_col)

        exp_out = p.model._get_subsystem('phase0').simulate()

        time_comp = exp_out.model.phase0._get_subsystem('time')
        integrator_comp = exp_out.model.phase0._get_subsystem('integrator')
        ode = exp_out.model.phase0._get_subsystem('ode')
        timeseries_comp = exp_out.model.phase0._get_subsystem('timeseries')

        time_comp_t_initial = time_comp.get_val('t_initial')
        integrator_comp_t_initial = integrator_comp.get_val('t_initial')
        ode_t_initial = ode.get_val('t_initial')

        time_comp_t_duration = time_comp.get_val('t_duration')
        integrator_comp_t_duration = integrator_comp.get_val('t_duration')
        ode_t_duration = ode.get_val('t_duration')
        ode_time_phase = ode.get_val('time_phase')
        timeseries_time_phase = timeseries_comp.get_val('time_phase')

        assert_near_equal(time_comp_t_initial, p['phase0.t_initial'])
        assert_near_equal(integrator_comp_t_initial, p['phase0.t_initial'])
        assert_near_equal(ode_t_initial, p['phase0.t_initial'])

        assert_near_equal(time_comp_t_duration, p['phase0.t_duration'])
        assert_near_equal(integrator_comp_t_duration, p['phase0.t_duration'])
        assert_near_equal(ode_t_duration, p['phase0.t_duration'])

        assert_near_equal(ode_time_phase.ravel(), timeseries_time_phase.ravel())

    def test_radau_targets_are_inputs(self):
        num_seg = 20
        p = self._make_problem('radau-ps', num_seg, input_initial=True, input_duration=True)

        # Solve for the optimal trajectory
        p.run_driver()

        time_all = p['phase0.t']

        time_phase_all = p['phase0.t_phase']

        assert_near_equal(p['phase0.rhs_all.time_phase'][-1], 1.8016, tolerance=1.0E-3)

        assert_near_equal(p['phase0.rhs_all.t_initial'], p['phase0.t_initial'])

        assert_near_equal(p['phase0.rhs_all.t_duration'], p['phase0.t_duration'])

        assert_near_equal(p['phase0.rhs_all.time_phase'], time_phase_all)

        assert_near_equal(p['phase0.rhs_all.time'], time_all)

        exp_out = p.model._get_subsystem('phase0').simulate()

        time_comp = exp_out.model.phase0._get_subsystem('time')
        integrator_comp = exp_out.model.phase0._get_subsystem('integrator')
        ode = exp_out.model.phase0._get_subsystem('ode')
        timeseries_comp = exp_out.model.phase0._get_subsystem('timeseries')

        time_comp_t_initial = time_comp.get_val('t_initial')
        integrator_comp_t_initial = integrator_comp.get_val('t_initial')
        ode_t_initial = ode.get_val('t_initial')

        time_comp_t_duration = time_comp.get_val('t_duration')
        integrator_comp_t_duration = integrator_comp.get_val('t_duration')
        ode_t_duration = ode.get_val('t_duration')
        ode_time_phase = ode.get_val('time_phase')
        timeseries_time_phase = timeseries_comp.get_val('time_phase')

        assert_near_equal(time_comp_t_initial, p['phase0.t_initial'])
        assert_near_equal(integrator_comp_t_initial, p['phase0.t_initial'])
        assert_near_equal(ode_t_initial, p['phase0.t_initial'])

        assert_near_equal(time_comp_t_duration, p['phase0.t_duration'])
        assert_near_equal(integrator_comp_t_duration, p['phase0.t_duration'])
        assert_near_equal(ode_t_duration, p['phase0.t_duration'])

        assert_near_equal(ode_time_phase.ravel(), timeseries_time_phase.ravel())


if __name__ == "__main__":
    unittest.main()
