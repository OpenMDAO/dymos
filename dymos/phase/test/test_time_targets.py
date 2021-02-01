import unittest

import numpy as np
import matplotlib
matplotlib.use('Agg')

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

    def _make_problem(self, transcription, num_seg, transcription_order=3):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        # Compute sparsity/coloring when run_driver is called
        p.driver.declare_coloring()

        t = {'gauss-lobatto': dm.GaussLobatto(num_segments=num_seg, order=transcription_order),
             'radau-ps': dm.Radau(num_segments=num_seg, order=transcription_order),
             'runge-kutta': dm.RungeKutta(num_segments=num_seg)}

        phase = dm.Phase(ode_class=_BrachistochroneTestODE, transcription=t[transcription])

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(1, 1), duration_bounds=(.5, 10), units='s',
                               time_phase_targets=['time_phase'], t_duration_targets=['t_duration'],
                               t_initial_targets=['t_initial'], targets=['time'])

        phase.add_state('x', fix_initial=True, rate_source='xdot', units='m')
        phase.add_state('y', fix_initial=True, rate_source='ydot', units='m')
        phase.add_state('v', fix_initial=True, rate_source='vdot', targets=['v'], units='m/s')

        phase.add_control('theta', units='deg', rate_continuity=True, lower=0.01, upper=179.9, targets=['theta'])

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665, targets=['g'])

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 1.0
        p['phase0.t_duration'] = 3.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        return p

    def test_gauss_lobatto(self):
        num_seg = 20
        p = self._make_problem('gauss-lobatto', num_seg)

        # Solve for the optimal trajectory
        p.run_driver()

        gd = p.model.phase0.options['transcription'].grid_data

        time_all = p['phase0.time']
        time_col = time_all[gd.subset_node_indices['col']]
        time_disc = time_all[gd.subset_node_indices['disc']]
        time_segends = np.reshape(time_all[gd.subset_node_indices['segment_ends']],
                                  newshape=(gd.num_segments, 2))

        time_phase_all = p['phase0.time_phase']
        time_phase_col = time_phase_all[gd.subset_node_indices['col']]
        time_phase_disc = time_phase_all[gd.subset_node_indices['disc']]
        time_phase_segends = np.reshape(time_phase_all[gd.subset_node_indices['segment_ends']],
                                        newshape=(gd.num_segments, 2))

        assert_near_equal(p['phase0.rhs_disc.time_phase'][-1], 1.8016, tolerance=1.0E-3)

        assert_near_equal(p['phase0.rhs_disc.t_initial'], p['phase0.t_initial'])
        assert_near_equal(p['phase0.rhs_col.t_initial'], p['phase0.t_initial'])

        assert_near_equal(p['phase0.rhs_disc.t_duration'], p['phase0.t_duration'])
        assert_near_equal(p['phase0.rhs_col.t_duration'], p['phase0.t_duration'])

        assert_near_equal(p['phase0.rhs_disc.time_phase'], time_phase_disc)
        assert_near_equal(p['phase0.rhs_col.time_phase'], time_phase_col)

        assert_near_equal(p['phase0.rhs_disc.time'], time_disc)
        assert_near_equal(p['phase0.rhs_col.time'], time_col)

        exp_out = p.model.phase0.simulate()

        for iseg in range(num_seg):
            seg_comp_i = exp_out.model.phase0._get_subsystem('segments.segment_{0}'.format(iseg))
            iface = seg_comp_i.options['ode_integration_interface']
            t_initial_i = iface.prob.get_val('ode.t_initial')
            t_duration_i = iface.prob.get_val('ode.t_duration')
            time_phase_i = iface.prob.get_val('ode.time_phase')
            time_i = iface.prob.get_val('ode.time')

            # Since the phase has simulated, all times should be equal to their respective value
            # at the end of each segment.
            assert_near_equal(t_initial_i, p['phase0.t_initial'])
            assert_near_equal(t_duration_i, p['phase0.t_duration'])
            assert_near_equal(time_phase_i, time_phase_segends[iseg, 1], tolerance=1.0E-12)
            assert_near_equal(time_i, time_segends[iseg, 1], tolerance=1.0E-12)

    def test_radau(self):
        num_seg = 20
        p = self._make_problem('radau-ps', num_seg)

        # Solve for the optimal trajectory
        p.run_driver()

        gd = p.model.phase0.options['transcription'].grid_data

        time_all = p['phase0.time']
        time_segends = np.reshape(time_all[gd.subset_node_indices['segment_ends']],
                                  newshape=(gd.num_segments, 2))

        time_phase_all = p['phase0.time_phase']
        time_phase_segends = np.reshape(time_phase_all[gd.subset_node_indices['segment_ends']],
                                        newshape=(gd.num_segments, 2))

        assert_near_equal(p['phase0.rhs_all.time_phase'][-1], 1.8016, tolerance=1.0E-3)

        assert_near_equal(p['phase0.rhs_all.t_initial'], p['phase0.t_initial'])

        assert_near_equal(p['phase0.rhs_all.t_duration'], p['phase0.t_duration'])

        assert_near_equal(p['phase0.rhs_all.time_phase'], time_phase_all)

        assert_near_equal(p['phase0.rhs_all.time'], time_all)

        exp_out = p.model.phase0.simulate()

        for iseg in range(num_seg):
            seg_comp_i = exp_out.model.phase0._get_subsystem('segments.segment_{0}'.format(iseg))
            iface = seg_comp_i.options['ode_integration_interface']
            t_initial_i = iface.prob.get_val('ode.t_initial')
            t_duration_i = iface.prob.get_val('ode.t_duration')
            time_phase_i = iface.prob.get_val('ode.time_phase')
            time_i = iface.prob.get_val('ode.time')

            # Since the phase has simulated, all times should be equal to their respective value
            # at the end of each segment.
            assert_near_equal(t_initial_i, p['phase0.t_initial'])
            assert_near_equal(t_duration_i, p['phase0.t_duration'])
            assert_near_equal(time_phase_i, time_phase_segends[iseg, 1], tolerance=1.0E-12)
            assert_near_equal(time_i, time_segends[iseg, 1], tolerance=1.0E-12)

    def test_runge_kutta(self):
        num_seg = 20
        p = self._make_problem('runge-kutta', num_seg)

        # Solve for the optimal trajectory
        p.run_driver()

        gd = p.model.phase0.options['transcription'].grid_data

        time_all = p['phase0.time']
        time_segends = np.reshape(time_all[gd.subset_node_indices['segment_ends']],
                                  newshape=(gd.num_segments, 2))

        time_phase_all = p['phase0.time_phase']
        time_phase_segends = np.reshape(time_phase_all[gd.subset_node_indices['segment_ends']],
                                        newshape=(gd.num_segments, 2))

        # Test the iteration ODE

        assert_near_equal(p['phase0.rk_solve_group.ode.time_phase'][-1], 1.8016,
                          tolerance=1.0E-3)

        assert_near_equal(p['phase0.rk_solve_group.ode.t_initial'], p['phase0.t_initial'])

        assert_near_equal(p['phase0.rk_solve_group.ode.t_duration'], p['phase0.t_duration'])

        assert_near_equal(p['phase0.rk_solve_group.ode.time_phase'], time_phase_all)

        assert_near_equal(p['phase0.rk_solve_group.ode.time'], time_all)

        # Now test the final ODE

        assert_near_equal(p['phase0.ode.time_phase'][-1], 1.8016,
                          tolerance=1.0E-3)

        assert_near_equal(p['phase0.ode.t_initial'], p['phase0.t_initial'])

        assert_near_equal(p['phase0.ode.t_duration'], p['phase0.t_duration'])

        assert_near_equal(p['phase0.ode.time_phase'], time_phase_segends.ravel())

        assert_near_equal(p['phase0.ode.time'], time_segends.ravel())

        exp_out = p.model.phase0.simulate()

        for iseg in range(num_seg):
            seg_comp_i = exp_out.model.phase0._get_subsystem('segments.segment_{0}'.format(iseg))
            iface = seg_comp_i.options['ode_integration_interface']
            t_initial_i = iface.prob.get_val('ode.t_initial')
            t_duration_i = iface.prob.get_val('ode.t_duration')
            time_phase_i = iface.prob.get_val('ode.time_phase')
            time_i = iface.prob.get_val('ode.time')

            # Since the phase has simulated, all times should be equal to their respective value
            # at the end of each segment.
            assert_near_equal(t_initial_i, p['phase0.t_initial'])
            assert_near_equal(t_duration_i, p['phase0.t_duration'])
            assert_near_equal(time_phase_i, time_phase_segends[iseg, 1], tolerance=1.0E-12)
            assert_near_equal(time_i, time_segends[iseg, 1], tolerance=1.0E-12)


if __name__ == "__main__":
    unittest.main()
