from __future__ import print_function, absolute_import, division

import itertools
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import matplotlib
matplotlib.use('Agg')

from parameterized import parameterized

from openmdao.utils.assert_utils import assert_rel_error

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver

from dymos import Phase, declare_time, declare_state, declare_parameter

from dymos.examples.ssto.log_atmosphere_comp import LogAtmosphereComp
from dymos.examples.ssto.launch_vehicle_2d_eom_comp import LaunchVehicle2DEOM
from dymos.examples.ssto.linear_tangent_guidance_comp import LinearTangentGuidanceComp


@declare_time(units='s', time_phase_targets=['guidance.time'])
@declare_state('x', rate_source='eom.xdot', units='m')
@declare_state('y', rate_source='eom.ydot', targets=['atmos.y'], units='m')
@declare_state('vx', rate_source='eom.vxdot', targets=['eom.vx'], units='m/s')
@declare_state('vy', rate_source='eom.vydot', targets=['eom.vy'], units='m/s')
@declare_state('m', rate_source='eom.mdot', targets=['eom.m'], units='kg')
@declare_parameter('thrust', targets=['eom.thrust'], units='N')
@declare_parameter('a_ctrl', targets=['guidance.a_ctrl'], units='1/s')
@declare_parameter('b_ctrl', targets=['guidance.b_ctrl'], units=None)
@declare_parameter('Isp', targets=['eom.Isp'], units='s')
class _LaunchVehicleLinearTangentODE2(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

        self.options.declare('central_body', values=['earth', 'moon'], default='earth',
                             desc='The central graviational body for the launch vehicle.')

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']

        if cb == 'earth':
            rho_ref = 1.225
            h_scale = 8.44E3
        elif cb == 'moon':
            rho_ref = 0.0
            h_scale = 1.0
        else:
            raise RuntimeError('Unrecognized value for central_body: {0}'.format(cb))

        self.add_subsystem('atmos',
                           LogAtmosphereComp(num_nodes=nn, rho_ref=rho_ref, h_scale=h_scale))

        self.add_subsystem('guidance', LinearTangentGuidanceComp(num_nodes=nn))

        self.add_subsystem('eom', LaunchVehicle2DEOM(num_nodes=nn, central_body=cb))

        self.connect('atmos.rho', 'eom.rho')
        self.connect('guidance.theta', 'eom.theta')


class TestTimePhaseTargets(unittest.TestCase):

    @parameterized.expand(
        itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                          ['fwd'],  # derivative_mode
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
                                                                          p.args[0],
                                                                          p.args[1]])
    )
    def test_results(self, transcription='gauss-lobatto', derivative_mode='rev', compressed=True):
        p = Problem(model=Group())

        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase(transcription,
                      ode_class=_LaunchVehicleLinearTangentODE2,
                      ode_init_kwargs={'central_body': 'moon'},
                      num_segments=10,
                      transcription_order=5,
                      compressed=compressed)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, 1000))

        phase.set_state_options('x', fix_initial=True, scaler=1.0E-5, lower=0)
        phase.set_state_options('y', fix_initial=True, scaler=1.0E-5, lower=0)
        phase.set_state_options('vx', fix_initial=True, scaler=1.0E-3, lower=0)
        phase.set_state_options('vy', fix_initial=True, scaler=1.0E-3)
        phase.set_state_options('m', fix_initial=True, scaler=1.0E-3)

        phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
        phase.add_boundary_constraint('vx', loc='final', equals=1627.0)
        phase.add_boundary_constraint('vy', loc='final', equals=0)

        phase.add_design_parameter('a_ctrl', units='1/s', opt=True)
        phase.add_design_parameter('b_ctrl', units=None, opt=True)
        phase.add_design_parameter('thrust', units='N', opt=False, val=3.0 * 50000.0 * 1.61544)
        phase.add_design_parameter('Isp', units='s', opt=False, val=1.0E6)

        phase.add_objective('time', index=-1, scaler=0.01)

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.setup(force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 500.0
        p['phase0.states:x'] = phase.interpolate(ys=[0, 350000.0], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[0, 185000.0], nodes='state_input')
        p['phase0.states:vx'] = phase.interpolate(ys=[0, 1627.0], nodes='state_input')
        p['phase0.states:vy'] = phase.interpolate(ys=[1.0E-6, 0], nodes='state_input')
        p['phase0.states:m'] = phase.interpolate(ys=[50000, 50000], nodes='state_input')
        p['phase0.design_parameters:a_ctrl'] = -0.01
        p['phase0.design_parameters:b_ctrl'] = 3.0

        p.run_driver()

        # Ensure defects are zero
        for state in ['x', 'y', 'vx', 'vy', 'm']:
            assert_rel_error(self, p['phase0.collocation_constraint.defects:{0}'.format(state)],
                             np.zeros_like(p['phase0.collocation_constraint.'
                                             'defects:{0}'.format(state)]),
                             tolerance=1.0E-3)

            if not compressed:
                assert_rel_error(self, p['phase0.continuity_comp.defect_states:{0}'.format(state)],
                                 0.0, tolerance=1.0E-3,
                                 err_msg='error in state continuity for state {0}'.format(state))

        # Ensure time found is the known solution
        assert_rel_error(self, p['phase0.t_duration'], 481.8, tolerance=1.0E-3)

        # Does this case find the same answer as using theta as a dynamic control?
        assert_rel_error(self, p['phase0.design_parameters:a_ctrl'], [[-0.0082805]],
                         tolerance=1.0E-3)
        assert_rel_error(self, p['phase0.design_parameters:b_ctrl'], [[2.74740137]],
                         tolerance=1.0E-3)

if __name__ == "__main__":
    unittest.main()
