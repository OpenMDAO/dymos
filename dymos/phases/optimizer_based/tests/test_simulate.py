from __future__ import print_function, division, absolute_import

import os
import os.path
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver, \
    CaseReader

from dymos import Phase
from dymos.utils.simulation import SimulationResults
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

OPTIMIZER = 'SLSQP'
SHOW_PLOTS = False


class TestSimulateRecording(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db', 'brachistochrone_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_record_default_file(self, transcription='gauss-lobatto', top_level_jacobian='csc',
                                 optimizer='slsqp'):
        p = Problem(model=Group())

        if optimizer == 'SNOPT':
            p.driver = pyOptSparseDriver()
            p.driver.options['optimizer'] = OPTIMIZER
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['iSumm'] = 6
            p.driver.opt_settings['Verify level'] = 3
        else:
            p.driver = ScipyOptimizeDriver()

        phase = Phase(transcription,
                      ode_class=BrachistochroneODE,
                      num_segments=8,
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

        p.model.linear_solver = DirectSolver()

        p.setup(mode='fwd')

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_disc')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_disc')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_disc')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_disc')

        p.run_driver()

        exp_out = phase.simulate(times=np.linspace(p['phase0.t_initial'],
                                                   p['phase0.t_initial'] + p['phase0.t_duration'],
                                                   50))

        cr = CaseReader('phase0_sim.db')
        last_case = cr.system_cases.get_case(-1)

        for var in ['time', 'states:x', 'states:y', 'states:v', 'controls:theta']:
            if ':' in var:
                _var = var.split(':')[-1]
            else:
                _var = var
            assert_almost_equal(last_case.outputs[var].ravel(), exp_out.get_values(_var).ravel())

        loaded_exp_out = SimulationResults('phase0_sim.db')

        for var in ['time', 'x', 'y', 'v', 'theta']:
            assert_almost_equal(exp_out.get_values(var).ravel(),
                                loaded_exp_out.get_values(var).ravel())

    def test_record_specified_file(self, transcription='gauss-lobatto',
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

        phase = Phase(transcription,
                      ode_class=BrachistochroneODE,
                      num_segments=8,
                      transcription_order=3)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', rate_continuity=True, lower=0.01, upper=179.9)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.options['assembled_jac_type'] = top_level_jacobian.lower()
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.setup(mode='fwd')

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_disc')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_disc')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_disc')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_disc')

        p.run_driver()

        exp_out = phase.simulate(times=np.linspace(p['phase0.t_initial'],
                                                   p['phase0.t_initial'] + p['phase0.t_duration'],
                                                   50),
                                 record_file='brachistochrone_sim.db')

        loaded_exp_out = SimulationResults('brachistochrone_sim.db')

        for var in ['time', 'x', 'y', 'v', 'theta']:
            assert_almost_equal(exp_out.get_values(var).ravel(),
                                loaded_exp_out.get_values(var).ravel())


if __name__ == '__main__':
    unittest.main()
