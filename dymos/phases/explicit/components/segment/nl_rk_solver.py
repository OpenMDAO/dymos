"""
Define the NonlinearRK class.

Custom nonlinear solver that time-steps through the integration of an ODE using rk4
"""

import numpy as np

from openmdao.core.system import System
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.solver import NonlinearSolver
from openmdao.utils.mpi import multi_proc_fail_check

from dymos.utils.simulation import ScipyODEIntegrator


def _single_rk4_step(f, h, t0, y0):
    """takes a single RK4 step in time"""

    K_ = np.zeros(4)


    K0 = K_[0] = h*f([t0],[y0])
    K1 = K_[1] = h*f([t0+h/2.],[y0+K0/2.])
    K2 = K_[2] = h*f([t0+h/2.],[y0+K1/2.])
    K_[3] = h*f([t0+h],[y0+K2])

    next_y = y0 + np.sum(K_*np.array([1, 2, 2, 1]))/6.
    return next_y, K_

class NonlinearRK(NonlinearSolver):

    SOLVER = 'NL: RK'

    def solve(self):
        """
        Run the solver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        system = self._system

        f = self.options['ODE_integrator']._f_ode

        N_STEPS = system.options['num_steps']
        seg_t0, seg_tf = system._inputs['seg_t0_tf']
        h = ((seg_tf - seg_t0) / N_STEPS) * np.ones(N_STEPS)
        t = np.linspace(seg_t0, seg_tf, N_STEPS + 1)
        y0 = system._inputs['initial_states:y']


        y = system._outputs['step_states:y']
        y[0]= y0

        K =system._outputs['k:y']

        for i in range(N_STEPS):
            # print(i, h[i], t[i], y[i])
            yn, Kn = _single_rk4_step(f, h[i], t[i], y[i])
            y[i+1] = yn
            system._outputs['k:y'][i] = Kn.reshape((4,1))

        # TODO: optionally check the residual values to ensure the RK was stable
        #       only do this optionally, because it will require an additional
        #       call to ODE which might be expensive

        #TODO: optionally have one more _solve_nonlinear to make sure the whole
        #      group has the correct values



    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """

        # NOTE: if possible we probably don't want to leave this integrator allocated when solve is not being done.
        #       it might make sense to have the solver create this itself, using information from the parent class.
        #       For now this works fine though.
        self.options.declare('ODE_integrator', types=ScipyODEIntegrator, desc='callable function to compute y_dot = f(u,y,args)')

