from __future__ import print_function, absolute_import, division

import itertools
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import matplotlib
matplotlib.use('Agg')

from parameterized import parameterized

from openmdao.utils.assert_utils import assert_rel_error

import dymos.examples.ssto.ex_ssto_moon_linear_tangent as ex_ssto_moon_lintan


class TestDocSSTOPolynomialControl(unittest.TestCase):

    def test_doc_ssto_polynomial_control(self):
        from openmdao.api import Problem, Group, DirectSolver, pyOptSparseDriver
        import dymos as dm
        from dymos.examples.ssto.launch_vehicle_linear_tangent_ode import LaunchVehicleLinearTangentODE2

        p = Problem(model=Group())

        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        # p.driver.opt_settings['iSumm'] = 6
        p.driver.options['dynamic_simul_derivs'] = True

        phase = dm.Phase(ode_class=LaunchVehicleLinearTangentODE2,
                         ode_init_kwargs={'central_body': 'moon'},
                         transcription=dm.Radau(num_segments=20, order=3, compressed=False))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, 1000))

        phase.set_state_options('x', fix_initial=True, lower=0)
        phase.set_state_options('y', fix_initial=True, lower=0)
        phase.set_state_options('vx', fix_initial=True, lower=0)
        phase.set_state_options('vy', fix_initial=True)
        phase.set_state_options('m', fix_initial=True)

        phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
        phase.add_boundary_constraint('vx', loc='final', equals=1627.0)
        phase.add_boundary_constraint('vy', loc='final', equals=0)

        phase.add_polynomial_control('tan_theta', order=1, units=None, opt=True)
        phase.add_design_parameter('thrust', units='N', opt=False, val=3.0 * 50000.0 * 1.61544)
        phase.add_design_parameter('Isp', units='s', opt=False, val=1.0E6)

        phase.add_objective('time', index=-1, scaler=0.01)

        p.model.linear_solver = DirectSolver()

        phase.add_timeseries_output('guidance.theta', units='deg')

        p.setup(force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 500.0
        p['phase0.states:x'] = phase.interpolate(ys=[0, 350000.0], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[0, 185000.0], nodes='state_input')
        p['phase0.states:vx'] = phase.interpolate(ys=[0, 1627.0], nodes='state_input')
        p['phase0.states:vy'] = phase.interpolate(ys=[1.0E-6, 0], nodes='state_input')
        p['phase0.states:m'] = phase.interpolate(ys=[50000, 50000], nodes='state_input')
        p['phase0.polynomial_controls:tan_theta'] = [[0.5 * np.pi], [0.0]]

        p.run_driver()

        return p


if __name__ == "__main__":
    unittest.main()
