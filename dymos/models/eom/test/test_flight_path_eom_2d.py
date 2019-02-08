from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, pyOptSparseDriver
from openmdao.utils.assert_utils import assert_rel_error

from dymos import Phase, ODEOptions
from dymos.models.eom import FlightPathEOM2D

OPTIMIZER = 'SLSQP'
SHOW_PLOTS = False


class _CannonballODE(FlightPathEOM2D):

    ode_options = ODEOptions()

    ode_options.declare_time(units='s')

    ode_options.declare_state(name='r', rate_source='r_dot', units='m')
    ode_options.declare_state(name='h', rate_source='h_dot', units='m')
    ode_options.declare_state(name='gam', rate_source='gam_dot', targets='gam', units='rad')
    ode_options.declare_state(name='v', rate_source='v_dot', targets='v', units='m/s')

    def __init__(self, **kwargs):
        super(_CannonballODE, self).__init__(**kwargs)


class TestFlightPathEOM2D(unittest.TestCase):

    def setUp(self):
        self.p = Problem(model=Group())

        self.p.driver = pyOptSparseDriver()
        self.p.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            self.p.driver.opt_settings['Major iterations limit'] = 50
            self.p.driver.opt_settings['iSumm'] = 6
            self.p.driver.opt_settings['Verify level'] = 3

        phase = Phase(transcription='gauss-lobatto',
                      ode_class=_CannonballODE,
                      num_segments=15,
                      transcription_order=3,
                      compressed=False)

        self.p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, 20))

        phase.set_state_options('r', fix_initial=True, fix_final=False,
                                scaler=0.001, defect_scaler=0.001)

        phase.set_state_options('h', fix_initial=True, fix_final=True,  # Require final altitude
                                scaler=0.001, defect_scaler=0.001)

        phase.set_state_options('v', fix_initial=True, fix_final=False,
                                scaler=0.01, defect_scaler=0.01)

        phase.set_state_options('gam', fix_final=False,
                                scaler=1.0, defect_scaler=1.0)

        # Maximize final range by varying initial flight path angle
        phase.add_objective('r', loc='final', scaler=-0.01)

    def test_cannonball_simulate(self):
        self.p.setup()

        v0 = 100.0
        gam0 = np.radians(45)
        g = 9.80665
        t_duration = 2 * v0 * np.sin(gam0) / g

        phase = self.p.model.phase0

        self.p['phase0.t_initial'] = 0.0
        self.p['phase0.t_duration'] = t_duration

        self.p['phase0.states:r'] = phase.interpolate(ys=[0, 700.0], nodes='state_input')
        self.p['phase0.states:h'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        self.p['phase0.states:v'] = phase.interpolate(ys=[v0, v0], nodes='state_input')
        self.p['phase0.states:gam'] = phase.interpolate(ys=[gam0, -gam0], nodes='state_input')

        self.p.run_model()

        exp_out = phase.simulate(times='all')

        assert_rel_error(self, exp_out.get_val('phase0.timeseries.states:h', units='km')[-1], 0.0,
                         tolerance=0.001)
        assert_rel_error(self, exp_out.get_val('phase0.timeseries.states:r')[-1], v0**2 / g,
                         tolerance=0.001)
        assert_rel_error(self, exp_out.get_val('phase0.timeseries.states:gam')[-1], -gam0,
                         tolerance=0.001)
        assert_rel_error(self, exp_out.get_val('phase0.timeseries.states:v')[-1], v0,
                         tolerance=0.001)

    def test_cannonball_max_range(self):
        self.p.setup()

        v0 = 100.0
        gam0 = np.radians(45.0)
        g = 9.80665
        t_duration = 10.0

        phase = self.p.model.phase0

        self.p['phase0.t_initial'] = 0.0
        self.p['phase0.t_duration'] = t_duration

        self.p['phase0.states:r'] = phase.interpolate(ys=[0, v0 * np.cos(gam0) * t_duration],
                                                      nodes='state_disc')
        self.p['phase0.states:h'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        self.p['phase0.states:v'] = phase.interpolate(ys=[v0, v0], nodes='state_input')
        self.p['phase0.states:gam'] = phase.interpolate(ys=[gam0, -gam0], nodes='state_input')

        self.p.run_driver()

        exp_out = phase.simulate(times='all')

        assert_rel_error(self, exp_out.get_val('phase0.timeseries.states:r')[-1], v0**2 / g,
                         tolerance=0.001)
        assert_rel_error(self, exp_out.get_val('phase0.timeseries.states:gam')[-1], -np.radians(45),
                         tolerance=0.001)
        assert_rel_error(self, exp_out.get_val('phase0.timeseries.states:v')[-1], v0,
                         tolerance=0.001)

    def test_partials(self):
        self.p.setup(force_alloc_complex=True)

        v0 = 100.0
        gam0 = np.radians(30.0)
        t_duration = 10.0

        phase = self.p.model.phase0

        self.p['phase0.t_initial'] = 0.0
        self.p['phase0.t_duration'] = t_duration

        self.p['phase0.states:r'] = phase.interpolate(ys=[0, v0 * np.cos(gam0) * t_duration],
                                                      nodes='state_disc')
        self.p['phase0.states:h'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        self.p['phase0.states:v'] = phase.interpolate(ys=[v0, v0], nodes='state_input')
        self.p['phase0.states:gam'] = phase.interpolate(ys=[gam0, -gam0], nodes='state_input')

        self.p.run_model()

        cpd = self.p.check_partials(compact_print=True, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                assert_almost_equal(cpd[comp][var, wrt]['abs error'], 0.0, decimal=2,
                                    err_msg='error in partial of'
                                            ' {0} wrt {1} in {2}'.format(var, wrt, comp))


if __name__ == '__main__':
    unittest.main()
