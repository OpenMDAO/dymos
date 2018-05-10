from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, BalanceComp, NewtonSolver, DirectSolver, ArmijoGoldsteinLS

from ..aero.aerodynamics_group import AerodynamicsGroup
from .unsteady_flight_dynamics_comp import UnsteadyFlightDynamicsComp


class FlightEquilibriumGroup(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('aero',
                           subsys=AerodynamicsGroup(num_nodes=nn))

        self.add_subsystem('flight_dynamics',
                           subsys=UnsteadyFlightDynamicsComp(num_nodes=nn))

        self.connect('aero.L', 'flight_dynamics.L')
        self.connect('aero.D', 'flight_dynamics.D')
        self.connect('aero.CM', 'TAS_gam_balance.CM')

        bal = self.add_subsystem(name='TAS_gam_balance',
                                 subsys=BalanceComp(),
                                 promotes_inputs=['TAS_rate', 'gam_rate'],
                                 promotes_outputs=['alpha', 'thrust', 'eta'])

        self.connect('alpha', ('aero.alpha', 'flight_dynamics.alpha'))
        self.connect('thrust', ('flight_dynamics.thrust'))
        self.connect('eta', ('aero.eta'))

        bal.add_balance('alpha', units='rad', eq_units='m/s**2', lhs_name='TAS_rate_computed',
                        rhs_name='TAS_rate', val=0.01*np.ones(nn), lower=-20, upper=30)

        bal.add_balance('thrust', units='N', eq_units='rad/s', lhs_name='gam_rate_computed',
                        rhs_name='gam_rate', val=1.0E5*np.ones(nn), lower=0.0)

        bal.add_balance('eta', units='rad', val=0.01*np.ones(nn), eq_units=None, lhs_name='CM',
                        lower=-30, upper=30, rhs_val=0.0)

        self.connect('flight_dynamics.TAS_rate_computed', 'TAS_gam_balance.TAS_rate_computed')
        self.connect('flight_dynamics.gam_rate_computed', 'TAS_gam_balance.gam_rate_computed')

        self.linear_solver = DirectSolver()
        self.nonlinear_solver = NewtonSolver()
        self.nonlinear_solver.linesearch = ArmijoGoldsteinLS()
        self.nonlinear_solver.options['solve_subsystems'] = True
        self.nonlinear_solver.linesearch.options['print_bound_enforce'] = True
        self.nonlinear_solver.options['err_on_maxiter'] = True
        self.nonlinear_solver.options['max_sub_solves'] = 10
        self.nonlinear_solver.options['iprint'] = -1
