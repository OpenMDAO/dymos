import unittest

import numpy as np

import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestExplicitShooting(unittest.TestCase):

    def test_brachistochrone_explicit_shooting(self):
        prob = om.Problem()

        prob.driver = om.pyOptSparseDriver(optimizer='SNOPT')
        prob.driver.opt_settings['Verify level'] = 3
        prob.driver.opt_settings['iSumm'] = 6

        tx = dm.transcriptions.ExplicitShooting(num_segments=10, grid='gauss-lobatto',
                                                order=3, num_steps_per_segment=20, compressed=True)

        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

        phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 2.0))

        # automatically discover states
        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_parameter('g', val=9.80665, units='m/s**2', opt=False)
        phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9)

        phase.add_boundary_constraint('x', loc='final', equals=10.0)
        phase.add_boundary_constraint('y', loc='final', equals=5.0)

        prob.model.add_subsystem('phase0', phase)

        phase.add_objective('time', loc='final')

        prob.setup(force_alloc_complex=True)

        prob.set_val('phase0.t_initial', 0.0)
        prob.set_val('phase0.t_duration', 2)
        prob.set_val('phase0.states:x', 0.0)
        prob.set_val('phase0.states:y', 10.0)
        prob.set_val('phase0.states:v', 1.0E-6)
        prob.set_val('phase0.parameters:g', 9.80665, units='m/s**2')
        prob.set_val('phase0.controls:theta', phase.interp('theta', ys=[0.01, 90]), units='deg')

        prob.run_driver()

        with np.printoptions(linewidth=1024):
            cpd = prob.check_partials(compact_print=True, method='cs')
            assert_check_partials(cpd)

        prob.model.list_outputs(print_arrays=True)

        prob.list_problem_vars()
