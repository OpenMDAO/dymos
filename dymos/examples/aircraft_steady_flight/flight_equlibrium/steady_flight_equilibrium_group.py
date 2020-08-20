import numpy as np

import openmdao.api as om

from ..aero.aerodynamics_group import AerodynamicsGroup
from .lift_equilibrium_comp import LiftEquilibriumComp
from .thrust_equilibrium_comp import ThrustEquilibriumComp


class SteadyFlightEquilibriumGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('aero',
                           subsys=AerodynamicsGroup(num_nodes=nn),
                           promotes_inputs=['alt'])

        self.add_subsystem('thrust_eq_comp',
                           subsys=ThrustEquilibriumComp(num_nodes=nn),
                           promotes_inputs=['q', 'S', 'gam', 'alpha', 'W_total'],
                           promotes_outputs=['CT'])

        self.add_subsystem('lift_eq_comp',
                           subsys=LiftEquilibriumComp(num_nodes=nn),
                           promotes_inputs=['q', 'S', 'gam', 'alpha', 'W_total', 'CT'],
                           promotes_outputs=['CL_eq'])

        bal = self.add_subsystem(name='alpha_eta_balance',
                                 subsys=om.BalanceComp(),
                                 promotes_outputs=['alpha', 'eta'])

        self.connect('alpha', ('aero.alpha'))
        self.connect('eta', ('aero.eta'))

        bal.add_balance('alpha', units='rad', eq_units=None, lhs_name='CL_eq',
                        rhs_name='CL', val=0.01*np.ones(nn), lower=-20, upper=30, res_ref=1.0)

        bal.add_balance('eta', units='rad', val=0.01*np.ones(nn), eq_units=None, lhs_name='CM',
                        lower=-30, upper=30, res_ref=1.0)

        self.connect('aero.CL', 'alpha_eta_balance.CL')
        self.connect('aero.CD', 'thrust_eq_comp.CD')
        self.connect('aero.CM', 'alpha_eta_balance.CM')
        self.connect('CL_eq', ('alpha_eta_balance.CL_eq'))

        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver()
        self.nonlinear_solver.options['atol'] = 1e-14
        self.nonlinear_solver.options['rtol'] = 1e-14
        self.nonlinear_solver.options['solve_subsystems'] = True
        self.nonlinear_solver.options['err_on_non_converge'] = True
        self.nonlinear_solver.options['max_sub_solves'] = 10
        self.nonlinear_solver.options['maxiter'] = 150
        self.nonlinear_solver.options['iprint'] = -1
        self.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        self.nonlinear_solver.linesearch.options['print_bound_enforce'] = True
