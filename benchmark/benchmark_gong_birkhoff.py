import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse


# First define a system which computes the equations of motion
class GongODE(om.ExplicitComponent):
    """
    Define the ODE for the "Gong Challenge Problem"

    Computational Optimal Control - Theory & Tools Beyond Nonlinear Programming
    Dr. Michael Ross
    Naval Postgraduate School
    https://nps.edu/documents/103424443/116151573/Ross.pdf/2c85d1a1-ff5b-4f60-9700-2ee5e1f3f65f?t=1580766209000
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', shape=(nn,), units='m/s', desc='velocity')
        self.add_input('x', shape=(nn,), units='m', desc='position')
        self.add_input('u', shape=(nn,), units='m/s', desc='control')

        self.add_output('xdot', shape=(nn,), units='m/s', desc='x rate of change', tags=['dymos.state_rate_source:x'])
        self.add_output('vdot', shape=(nn,), units='m/s**2', desc='y rate of change', tags=['dymos.state_rate_source:v'])
        self.add_output('Jdot', shape=(nn,), units='m**2/s', desc='J rate of change', tags=['dymos.state_rate_source:J'])

        # Ask OpenMDAO to compute the partial derivatives using complex-step
        # with a partial coloring algorithm for improved performance
        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='xdot', wrt='v', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='vdot', wrt='v', rows=ar, cols=ar, val=-1.0)
        self.declare_partials(of='vdot', wrt='u', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='Jdot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='Jdot', wrt='u', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        v, x, u = inputs.values()
        outputs['vdot'] = -v + u
        outputs['xdot'] = v
        outputs['Jdot'] = v * u

    def compute_partials(self, inputs, partials):
        v, x, u = inputs.values()
        partials['Jdot', 'v'] = u
        partials['Jdot', 'u'] = v


def _run_gong_problem(tx, simulate=True):

    p = om.Problem()

    # Define a Trajectory object
    traj = p.model.add_subsystem('traj', dm.Trajectory())

    # Define a Dymos Phase object with GaussLobatto Transcription
    phase = dm.Phase(ode_class=GongODE, transcription=tx)

    traj.add_phase(name='phase0', phase=phase)

    # Set the time options
    phase.set_time_options(fix_initial=True, fix_duration=True)

    # Set the state options
    phase.set_state_options('x', units='m', fix_initial=True, fix_final=True)
    phase.set_state_options('v', units='m/s', fix_initial=True, fix_final=True, lower=0)
    phase.set_state_options('J', units='m**2', fix_initial=True, fix_final=False)

    # Define theta as a control.
    phase.add_control(name='u', units='m/s', lower=0, upper=2, opt=True)

    # Minimize final time.
    phase.add_objective('J', loc='final', scaler=1)

    # Set the driver.

    p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
    p.driver.opt_settings['print_level'] = 0
    p.driver.opt_settings['mu_init'] = 1e-3
    # p.driver.opt_settings['max_iter'] = 500
    # p.driver.opt_settings['acceptable_tol'] = 1e-5
    # p.driver.opt_settings['constr_viol_tol'] = 1e-6
    # p.driver.opt_settings['compl_inf_tol'] = 1e-6
    p.driver.opt_settings['tol'] = 1e-5
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence
    # p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
    p.driver.opt_settings['mu_strategy'] = 'monotone'
    p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'

    # p.driver = om.pyOptSparseDriver(optimizer='SNOPT')
    # p.driver.opt_settings['iSumm'] = 6
    # p.driver.opt_settings['Proximal point method'] = 1

    p.driver.declare_coloring()
    phase.simulate_options['times_per_seg'] = 200

    # Setup the problem
    p.setup()

    # Now that the OpenMDAO problem is setup, we can guess the
    # values of time, states, and controls.
    phase.set_time_val(initial=0.0, duration=1.0)

    # States and controls here use a linearly interpolated
    # initial guess along the trajectory.
    phase.set_state_val('x', vals=(0, 1))
    phase.set_state_val('v', vals=(1, 1))
    phase.set_state_val('J', vals=(0, 1))
    phase.set_control_val('u', vals=(0.0, 1.0))

    # Run the driver to solve the problem and generate default plots of
    # state and control values vs time
    dm.run_problem(p, run_driver=True, simulate=simulate, make_plots=False)


@use_tempdirs
@require_pyoptsparse()
class BenchmarkGongBirkhoff(unittest.TestCase):

    def benchmark_gong_birkhoff_100(self):
        _run_gong_problem(tx=dm.Birkhoff(num_nodes=100))

    def benchmark_gong_birkhoff_100_nosim(self):
        _run_gong_problem(tx=dm.Birkhoff(num_nodes=100), simulate=False)

    def benchmark_gong_birkhoff_200(self):
        _run_gong_problem(tx=dm.Birkhoff(num_nodes=200))

    def benchmark_gong_birkhoff_200_nosim(self):
        _run_gong_problem(tx=dm.Birkhoff(num_nodes=200), simulate=False)

    def benchmark_gong_birkhoff_300_nosim(self):
        _run_gong_problem(tx=dm.Birkhoff(num_nodes=300), simulate=False)
