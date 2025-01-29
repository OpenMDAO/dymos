import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs


# Modify class so we can run it standalone.
import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.examples.brachistochrone.test.ex_brachistochrone import brachistochrone_min_time as brach


@use_tempdirs
class TestCollocationBalanceApplyNL(unittest.TestCase):

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def make_prob(self, transcription, num_segments, transcription_order, compressed):

        p = om.Problem(model=om.Group())

        if transcription == 'gauss-lobatto':
            t = dm.GaussLobatto(num_segments=num_segments,
                                order=transcription_order,
                                compressed=compressed,)
        elif transcription == 'radau-ps':
            t = dm.RadauDeprecated(num_segments=num_segments,
                                   order=transcription_order,
                                   compressed=compressed)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False, solve_segments='forward')
        phase.add_state('y', fix_initial=True, fix_final=False, solve_segments='forward')

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=True, fix_final=False, solve_segments='forward')

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], units='m/s**2')

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.setup(force_alloc_complex=True)

        p['traj0.phase0.t_initial'] = 0.0
        p['traj0.phase0.t_duration'] = 2.0

        p['traj0.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj0.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj0.phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['traj0.phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['traj0.phase0.parameters:g'] = 9.80665

        return p

    def test_apply_nonlinear_gl(self):
        dm.options['include_check_partials'] = True
        p = self.make_prob(transcription='gauss-lobatto', num_segments=3, transcription_order=3,
                           compressed=True)

        p.final_setup()
        p.model.run_apply_nonlinear()  # need to make sure residuals are computed

        expected = np.array([[0., 1., 1., 1.]]).T

        outputs = p.model.traj0.phases.phase0.indep_states.list_outputs(residuals=True, out_stream=None)
        resids = {k: v['resids'] for k, v in outputs}

        assert_almost_equal(resids['states:x'], expected)
        assert_almost_equal(resids['states:v'], expected)

    def test_apply_nonlinear_radau(self):
        dm.options['include_check_partials'] = True
        p = self.make_prob(transcription='radau-ps', num_segments=3, transcription_order=3,
                           compressed=True)

        p.final_setup()
        p.model.run_apply_nonlinear()  # need to make sure residuals are computed

        expected = np.array([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]).T

        outputs = p.model.traj0.phases.phase0.indep_states.list_outputs(residuals=True, out_stream=None)
        resids = {k: v['resids'] for k, v in outputs}

        assert_almost_equal(resids['states:x'], expected)
        assert_almost_equal(resids['states:v'], expected)

    def test_partials_gl(self):
        dm.options['include_check_partials'] = True
        p = self.make_prob(transcription='gauss-lobatto', num_segments=3, transcription_order=3,
                           compressed=True)

        def assert_partials(data):
            for of, wrt in data:
                if of == wrt:
                    # IndepVarComp like outputs have correct derivs, but FD is wrong so we skip
                    # them (should be some form of -I)
                    continue
                check_data = data[(of, wrt)]
                self.assertLess(check_data['abs error'].forward, 1e-8)

        cpd = p.check_partials(compact_print=True, method='fd', out_stream=None)
        data = cpd['traj0.phases.phase0.indep_states']
        assert_partials(data)

    def test_partials_radau(self):
        dm.options['include_check_partials'] = True
        p = self.make_prob(transcription='radau-ps', num_segments=3, transcription_order=3,
                           compressed=True)

        def assert_partials(data):
            for of, wrt in data:
                if of == wrt:
                    # IndepVarComp like outputs have correct derivs, but FD is wrong so we skip
                    # them (should be some form of -I)
                    continue
                check_data = data[(of, wrt)]
                self.assertLess(check_data['abs error'].forward, 1e-8)

        cpd = p.check_partials(compact_print=True, method='fd', out_stream=None)
        data = cpd['traj0.phases.phase0.indep_states']
        assert_partials(data)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
