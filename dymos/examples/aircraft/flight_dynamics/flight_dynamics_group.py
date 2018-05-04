from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, BalanceComp, NewtonSolver, DirectSolver, ArmijoGoldsteinLS

from ..aero.aerodynamics_group import AerodynamicsGroup
from .unsteady_flight_dynamics_comp import UnsteadyFlightDynamicsComp

class FlightDynamicsGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int,
                              desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_subsystem('aero',
                           subsys=AerodynamicsGroup(num_nodes=nn),
                           promotes_inputs=['mach', 'alpha', 'alt', 'eta', 'q'])

        self.add_subsystem('dynamics',
                           subsys=UnsteadyFlightDynamicsComp(num_nodes=nn),
                           promotes_inputs=['thrust', 'gam', 'alpha', 'm', 'TAS'])

        bal = self.add_subsystem(name='balance',
                                 subsys=BalanceComp(),
                                 promotes_inputs=['TAS_rate', 'gam_rate'],
                                 promotes_outputs=['alpha', 'thrust', 'eta'])

        bal.add_balance('alpha', units='rad', eq_units='m/s**2', lhs_name='TAS_rate_computed',
                        rhs_name='TAS_rate', val=0.01*np.ones(nn), lower=-20, upper=30)

        bal.add_balance('thrust', units='N', eq_units='rad/s', lhs_name='gam_rate_computed',
                        rhs_name='gam_rate', val=1.0E5*np.ones(nn), lower=0.0)

        bal.add_balance('eta', units='rad', val=0.01*np.ones(nn), eq_units=None, lhs_name='CM',
                        lower=-30, upper=30, rhs_val=0.0)

        self.connect('aero.CM', 'balance.CM')
        self.connect('aero.D', 'dynamics.D')
        self.connect('aero.L', 'dynamics.L')

        self.connect('dynamics.TAS_rate_computed', 'balance.TAS_rate_computed')
        self.connect('dynamics.gam_rate_computed', 'balance.gam_rate_computed')


        self.linear_solver = DirectSolver()
        self.nonlinear_solver = NewtonSolver()
        # self.jacobian = DenseJacobian()
        # flight_equilibrium_group.nonlinear_solver.linesearch = BoundsEnforceLS()
        # flight_equilibrium_group.nonlinear_solver.linesearch.options['bound_enforcement'] = 'vector'
        self.nonlinear_solver.linesearch = ArmijoGoldsteinLS()
        self.nonlinear_solver.options['solve_subsystems'] = True
        self.nonlinear_solver.linesearch.options['print_bound_enforce'] = True
        self.nonlinear_solver.options['err_on_maxiter'] = True
        self.nonlinear_solver.options['max_sub_solves'] = 10
