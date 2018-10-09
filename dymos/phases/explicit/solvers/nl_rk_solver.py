"""
Define the NonlinearRK class.

Custom nonlinear solver that time-steps through the integration of an ODE using rk4
"""
from six import iteritems

import numpy as np

from openmdao.core.system import System
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.solver import NonlinearSolver
from openmdao.utils.mpi import multi_proc_fail_check

from dymos.utils.simulation import ScipyODEIntegrator
from dymos.utils.rk_methods import rk_methods


def _single_rk4_step(f, h, t0, y0):
    """takes a single RK4 step in time"""
    size_y = len(y0)
    K_ = np.zeros((size_y, 4))

    K0 = K_[:, 0] = h*f([t0], [y0])
    K1 = K_[:, 1] = h*f([t0+h/2.], [y0+K0/2.])
    K2 = K_[:, 2] = h*f([t0+h/2.], [y0+K1/2.])
    K_[:, 3] = h*f([t0+h], [y0+K2])

    next_y = y0 + np.sum(K_*np.array([1, 2, 2, 1]))/6.
    return next_y, K_


class NonlinearRK(NonlinearSolver):

    SOLVER = 'NL: RK'

    def _setup_solvers(self, system, depth):
        super(NonlinearRK, self)._setup_solvers(system, depth)

        ops = self._system.options

        self.ode_wrap = ScipyODEIntegrator('seg', ode_class=ops['ode_class'],
                                           time_options=ops['time_options'],
                                           state_options=ops['state_options'],
                                           control_options=ops['control_options'],
                                           design_parameter_options=ops['design_parameter_options'],
                                           input_parameter_options=ops['input_parameter_options'],
                                           ode_init_kwargs=ops['ode_init_kwargs'])

        self.ode_wrap.setup(check=False)

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
        method = system.options['method']
        num_stages = rk_methods[method]['num_stages']

        f = self.ode_wrap._f_ode

        num_steps = system.options['num_steps']
        seg_t0, seg_tf = system._inputs['seg_t0_tf']
        h = ((seg_tf - seg_t0) / num_steps) * np.ones(num_steps)
        t = np.linspace(seg_t0, seg_tf, num_steps + 1)

        y0 = {}
        y = {}
        for state_name, options in iteritems(self.ode_wrap.state_options):
            y0[state_name] = system._inputs['initial_states:{0}'.format(state_name)]
            y[state_name] = system._outputs['step_states:{0}'.format(state_name)]
            y[state_name][0, ...] = y0[state_name]

        for i in range(num_steps):

            # Pack the state vector
            y_i = self.ode_wrap._pack_state_vec(y, index=i)

            yn, Kn = _single_rk4_step(f, h[i], t[i], y_i)

            # Unpack the output state vector and k vector
            for state_name, options in iteritems(self.ode_wrap.state_options):
                pos = options['pos']
                size = options['size']
                y[state_name][i + 1, ...] = yn[pos:pos + size]
                system._outputs['k:{0}'.format(state_name)][i] = \
                    Kn[pos:pos + size].reshape((num_stages, size))

        # TODO: optionally check the residual values to ensure the RK was stable
        #       only do this optionally, because it will require an additional
        #       call to ODE which might be expensive

        #TODO: optionally have one more _solve_nonlinear to make sure the whole
        with Recording('NLRunOnce', 0, self) as rec:
            # If this is a parallel group, transfer all at once then run each subsystem.
            if len(system._subsystems_myproc) != len(system._subsystems_allprocs):
                system._transfer('nonlinear', 'fwd')

                with multi_proc_fail_check(system.comm):
                    for subsys in system._subsystems_myproc:
                        subsys._solve_nonlinear()

                system._check_reconf_update()

            # If this is not a parallel group, transfer for each subsystem just prior to running it.
            else:
                for isub, subsys in enumerate(system._subsystems_myproc):
                    system._transfer('nonlinear', 'fwd', isub)
                    subsys._solve_nonlinear()
                    system._check_reconf_update()
            rec.abs = 0.0
            rec.rel = 0.0

        return False, 0.0, 0.0
