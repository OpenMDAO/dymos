from __future__ import print_function, absolute_import, division

import os
import unittest


class TestBrachistochroneSimulDerivsSetupExample(unittest.TestCase):

    @unittest.skip('skipped until SNOPT is available on CI')
    def test_brachistochrone_for_docs_gauss_lobatto_simul_derivs(self):
        from openmdao.api import Problem, Group, pyOptSparseDriver, CSCJacobian, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())

        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.opt_settings['Major iterations limit'] = 100

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=100,
                      transcription_order=3)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', dynamic=True,
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_control('g', units='m/s**2', dynamic=False, opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.jacobian = CSCJacobian()
        p.model.linear_solver = DirectSolver()

        p.setup(mode='fwd')


if __name__ == '__main__':
    unittest.main()
