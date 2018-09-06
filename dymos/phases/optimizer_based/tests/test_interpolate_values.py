from __future__ import print_function, division, absolute_import

import os
import os.path
import unittest

import numpy as np
import scipy.interpolate as interpolate

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

OPTIMIZER = 'SLSQP'


class TestInterpolateValuesAndRates(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db', 'brachistochrone_sim.db', 'coloring.json']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_interpolate_values_and_rates(self, transcription='gauss-lobatto',
                                          top_level_jacobian='csc', optimizer='slsqp'):
        p = Problem(model=Group())

        if optimizer == 'SNOPT':
            p.driver = pyOptSparseDriver()
            p.driver.options['optimizer'] = OPTIMIZER
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['iSumm'] = 6
            p.driver.opt_settings['Verify level'] = 3
        else:
            p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase(transcription,
                      ode_class=BrachistochroneODE,
                      num_segments=15,
                      transcription_order=3)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', rate_continuity=True, lower=0.01, upper=179.9)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = top_level_jacobian.lower()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        p.run_driver()

        for var in ['theta', 'x', 'y', 'time']:
            var_1 = phase.get_values(var)

            var_2 = phase.interpolate_values(var,
                                             times=np.linspace(0, p['phase0.t_duration'], 20))

            f1 = interpolate.interp1d(phase.get_values('time').ravel(),
                                      var_1.ravel(),
                                      fill_value='extrapolate')
            f2 = interpolate.interp1d(np.linspace(0, p['phase0.t_duration'], 20),
                                      var_2.ravel(),
                                      fill_value='extrapolate')

            t1 = f1(np.linspace(0, p['phase0.t_duration'], 20))
            t2 = f2(np.linspace(0, p['phase0.t_duration'], 20))

            assert_rel_error(self, t1, t2, tolerance=5.0E-3)

        # Test rate interpolation
        var_1 = phase.get_values('theta_rate')

        var_2 = phase.interpolate_rates('theta',
                                        times=np.linspace(0, p['phase0.t_duration'], 20))

        f1 = interpolate.interp1d(phase.get_values('time').ravel(),
                                  var_1.ravel(),
                                  fill_value='extrapolate')
        f2 = interpolate.interp1d(np.linspace(0, p['phase0.t_duration'], 20),
                                  var_2.ravel(),
                                  fill_value='extrapolate')

        t1 = f1(np.linspace(0, p['phase0.t_duration'], 20))
        t2 = f2(np.linspace(0, p['phase0.t_duration'], 20))

        assert_rel_error(self, t1, t2, tolerance=5.0E-3)


if __name__ == '__main__':
    unittest.main()
