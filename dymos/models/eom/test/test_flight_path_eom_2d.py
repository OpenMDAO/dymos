import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from dymos.utils.testing_utils import assert_check_partials

import dymos as dm
from dymos.models.eom import FlightPathEOM2D

OPTIMIZER = 'SLSQP'
SHOW_PLOTS = False


class _CannonballODE(FlightPathEOM2D):
    pass


@use_tempdirs
@require_pyoptsparse(optimizer='SLSQP')
class TestFlightPathEOM2D(unittest.TestCase):

    def setUp(self):
        self.p = om.Problem(model=om.Group())

        self.p.driver = om.pyOptSparseDriver()
        self.p.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            self.p.driver.opt_settings['Major iterations limit'] = 50
            self.p.driver.opt_settings['iSumm'] = 6
            self.p.driver.opt_settings['Verify level'] = 3

        phase = dm.Phase(ode_class=_CannonballODE,
                         transcription=dm.GaussLobatto(num_segments=5, order=3, compressed=False))

        self.p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(10, 20), units='s')

        phase.add_state('r', rate_source='r_dot', units='m',
                        fix_initial=True, fix_final=False,
                        scaler=0.001, defect_scaler=0.001)

        phase.add_state('h', rate_source='h_dot', units='m',
                        fix_initial=True, fix_final=True,
                        scaler=0.001, defect_scaler=0.001)

        phase.add_state('v', rate_source='v_dot', targets='v', units='m/s',
                        fix_initial=True, fix_final=False,
                        scaler=0.01, defect_scaler=0.01)

        phase.add_state('gam', rate_source='gam_dot', targets='gam', units='rad',
                        fix_final=False, scaler=1.0, defect_scaler=1.0)

        # Maximize final range by varying initial flight path angle
        phase.add_objective('r', loc='final', scaler=-0.01)

    def test_cannonball_simulate(self):
        self.p.setup()

        v0 = 100.0
        gam0 = np.radians(45)
        g = 9.80665
        t_duration = 2 * v0 * np.sin(gam0) / g

        phase = self.p.model.phase0

        phase.set_time_val(0, t_duration)
        phase.set_state_val('r', [0, v0 * np.cos(gam0) * t_duration])
        phase.set_state_val('h', [0, 0])
        phase.set_state_val('v', [v0, v0])
        phase.set_state_val('gam', [gam0, -gam0])

        self.p.run_model()

        exp_out = phase.simulate()

        assert_near_equal(exp_out.get_val('phase0.timeseries.h', units='km')[-1], 0.0,
                          tolerance=0.001)
        assert_near_equal(exp_out.get_val('phase0.timeseries.r')[-1], v0**2 / g,
                          tolerance=0.001)
        assert_near_equal(exp_out.get_val('phase0.timeseries.gam')[-1], -gam0,
                          tolerance=0.001)
        assert_near_equal(exp_out.get_val('phase0.timeseries.v')[-1], v0,
                          tolerance=0.001)

    def test_cannonball_max_range(self):
        self.p.setup(force_alloc_complex=True)

        v0 = 100.0
        gam0 = np.radians(45.0)
        g = 9.80665
        t_duration = 10.0

        phase = self.p.model.phase0

        phase.set_time_val(0, t_duration)
        phase.set_state_val('r', [0, v0 * np.cos(gam0) * t_duration])
        phase.set_state_val('h', [0, 0])
        phase.set_state_val('v', [v0, v0])
        phase.set_state_val('gam', [gam0, -gam0])

        self.p.run_driver()

        with np.printoptions(linewidth=1024):
            dm.options['include_check_partials'] = True
            self.p.check_partials(method='cs')

        exp_out = phase.simulate(times_per_seg=None)

        assert_near_equal(exp_out.get_val('phase0.timeseries.r')[-1], v0**2 / g,
                          tolerance=0.001)
        assert_near_equal(exp_out.get_val('phase0.timeseries.gam')[-1], -np.radians(45),
                          tolerance=0.001)
        assert_near_equal(exp_out.get_val('phase0.timeseries.v')[-1], v0,
                          tolerance=0.001)

    def test_partials(self):
        self.p.setup(force_alloc_complex=True)

        v0 = 100.0
        gam0 = np.radians(30.0)
        t_duration = 10.0

        phase = self.p.model.phase0

        self.p['phase0.t_initial'] = 0.0
        self.p['phase0.t_duration'] = t_duration

        self.p['phase0.states:r'] = phase.interp('r', [0, v0 * np.cos(gam0) * t_duration])
        self.p['phase0.states:h'] = phase.interp('h', [0, 0])
        self.p['phase0.states:v'] = phase.interp('v', [v0, v0])
        self.p['phase0.states:gam'] = phase.interp('gam', [gam0, -gam0])

        self.p.run_model()

        cpd = self.p.check_partials(compact_print=True, method='cs')

        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
