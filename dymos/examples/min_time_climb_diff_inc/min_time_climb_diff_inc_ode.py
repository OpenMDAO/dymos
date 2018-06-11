from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, BalanceComp, DirectSolver, NewtonSolver, BoundsEnforceLS, \
    ArmijoGoldsteinLS

from dymos import declare_time, declare_state, declare_parameter

from ...models.atmosphere import StandardAtmosphereGroup
from .aero import AeroGroup
from .prop import PropGroup
from ...models.eom import FlightPathEOM2D


@declare_time(units='s')
@declare_state('r', units='m', rate_source='flight_dynamics.r_dot')
@declare_state('h', units='m', rate_source='flight_dynamics.h_dot', targets=['atmos.h', 'prop.h'])
@declare_state('m', units='kg', rate_source='prop.m_dot', targets=['flight_dynamics.m'])
@declare_parameter('gam', units='rad', targets=['flight_dynamics.gam'])
@declare_parameter('v', units='m/s', targets=['aero.v', 'flight_dynamics.v'])
@declare_parameter('Isp', targets=['prop.Isp'], units='s')
@declare_parameter('S', targets=['aero.S'], units='m**2')
@declare_parameter('v_rate', targets=['balance_comp.v_dot_approx'], units='m/s**2')
@declare_parameter('gam_rate', targets=['balance_comp.gam_dot_approx'], units='rad/s')
class MinTimeClimbDiffIncODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='atmos',
                           subsys=StandardAtmosphereGroup(num_nodes=nn))

        self.connect('atmos.sos', 'aero.sos')
        self.connect('atmos.rho', 'aero.rho')

        feg = self.add_subsystem('flight_equlibrium_group',
                                 subsys=Group(),
                                 promotes_inputs=['*'],
                                 promotes_outputs=['*'])

        feg.add_subsystem(name='aero',
                          subsys=AeroGroup(num_nodes=nn))

        feg.add_subsystem(name='prop',
                          subsys=PropGroup(num_nodes=nn))

        feg.connect('aero.mach', 'prop.mach')

        feg.add_subsystem(name='flight_dynamics',
                          subsys=FlightPathEOM2D(num_nodes=nn))

        feg.connect('aero.f_drag', 'flight_dynamics.D')
        feg.connect('aero.f_lift', 'flight_dynamics.L')
        feg.connect('prop.thrust', 'flight_dynamics.T')

        bal = feg.add_subsystem(name='balance_comp',
                                subsys=BalanceComp(),
                                promotes_outputs=['alpha', 'throttle'])

        bal.add_balance('throttle', units=None, eq_units='km/s**2', lhs_name='v_dot_approx',
                        rhs_name='v_dot_computed', val=np.ones(nn), res_ref=0.01, ref=1.0,
                        lower=-100, upper=100)

        bal.add_balance('alpha', units='deg', val=np.ones(nn), eq_units='rad/s',
                        lhs_name='gam_dot_approx', rhs_name='gam_dot_computed', res_ref=1.0,
                        lower=-50, upper=50)

        feg.connect('flight_dynamics.v_dot', 'balance_comp.v_dot_computed')
        feg.connect('flight_dynamics.gam_dot', 'balance_comp.gam_dot_computed')

        feg.connect('alpha', ('aero.alpha', 'flight_dynamics.alpha'))
        feg.connect('throttle', 'prop.throttle')

        feg.linear_solver = DirectSolver()
        feg.nonlinear_solver = NewtonSolver()
        feg.nonlinear_solver.options['debug_print'] = True
        feg.nonlinear_solver.options['atol'] = 1e-8
        feg.nonlinear_solver.options['rtol'] = 1e-8
        feg.nonlinear_solver.options['solve_subsystems'] = True
        feg.nonlinear_solver.options['err_on_maxiter'] = True
        feg.nonlinear_solver.options['max_sub_solves'] = 10
        feg.nonlinear_solver.options['maxiter'] = 20
        feg.nonlinear_solver.options['iprint'] = -1
        # feg.nonlinear_solver.linesearch = ArmijoGoldsteinLS()
        feg.nonlinear_solver.linesearch = BoundsEnforceLS()
        feg.nonlinear_solver.linesearch.options['print_bound_enforce'] = True
