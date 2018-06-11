from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, BalanceComp, NewtonSolver, DirectSolver, ArmijoGoldsteinLS, \
    BoundsEnforceLS

from dymos.models.eom.flight_path_eom_2d import FlightPathEOM2D

from ..aero.aero import AeroGroup
from ..prop.prop import PropGroup


class UnsteadyFlightEquilibriumGroup(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='aero', subsys=AeroGroup(num_nodes=nn))

        self.add_subsystem(name='prop', subsys=PropGroup(num_nodes=nn))

        self.connect('aero.mach', 'prop.mach')

        self.add_subsystem(name='flight_dynamics', subsys=FlightPathEOM2D(num_nodes=nn))

        self.connect('aero.f_drag', 'flight_dynamics.D')
        self.connect('aero.f_lift', 'flight_dynamics.L')
        self.connect('prop.thrust', 'flight_dynamics.T')

        bal = self.add_subsystem(name='balance_comp', subsys=BalanceComp(),
                                 promotes_outputs=['alpha', 'throttle'])

        # bal.add_balance('alpha', units='deg', eq_units='m/s**2', lhs_name='v_dot_approx',
        #                 rhs_name='v_dot_computed', val=np.ones(nn), res_ref=1.0,
        #                 lower=-10, upper=10)
        #
        # bal.add_balance('throttle', units=None, val=np.ones(nn), eq_units='rad/s',
        #                 lhs_name='gam_dot_approx', rhs_name='gam_dot_computed', res_ref=1.0,
        #                 lower=-1, upper=1)

        bal.add_balance('throttle', units=None, eq_units='km/s**2', lhs_name='v_dot_approx',
                        rhs_name='v_dot_computed', val=np.ones(nn), res_ref=1.0, ref=1.0,
                        lower=-5, upper=5)

        bal.add_balance('alpha', units='deg', val=np.ones(nn), eq_units='rad/s',
                        lhs_name='gam_dot_approx', rhs_name='gam_dot_computed', res_ref=1.0,
                        lower=-14, upper=14)

        self.connect('flight_dynamics.v_dot', 'balance_comp.v_dot_computed')
        self.connect('flight_dynamics.gam_dot', 'balance_comp.gam_dot_computed')

        self.connect('alpha', ('aero.alpha', 'flight_dynamics.alpha'))
        self.connect('throttle', 'prop.throttle')

        self.linear_solver = DirectSolver()
        self.nonlinear_solver = NewtonSolver()
        self.nonlinear_solver.options['debug_print'] = True
        self.nonlinear_solver.options['atol'] = 1e-8
        self.nonlinear_solver.options['rtol'] = 1e-8
        self.nonlinear_solver.options['solve_subsystems'] = True
        # self.nonlinear_solver.options['err_on_maxiter'] = True
        self.nonlinear_solver.options['max_sub_solves'] = 10
        self.nonlinear_solver.options['maxiter'] = 100
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.linesearch = ArmijoGoldsteinLS()
        self.nonlinear_solver.linesearch.options['iprint'] = 2
        # self.nonlinear_solver.linesearch = BoundsEnforceLS()
        self.nonlinear_solver.linesearch.options['print_bound_enforce'] = True
