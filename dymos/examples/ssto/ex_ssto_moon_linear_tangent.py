from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver

from dymos import Phase
from dymos.examples.ssto.launch_vehicle_linear_tangent_ode import LaunchVehicleLinearTangentODE


def ssto_moon_linear_tangent(transcription='gauss-lobatto', num_seg=10, transcription_order=5,
                             optimizer='SLSQP', compressed=True):
    """
    Returns an instance of the SSTO problem for ascent from the lunar surface using linear
    tangent guidance.

    Parameters
    ----------
    transcription : str ('gauss-lobatto')
        The transcription method for optimal control:  'gauss-lobatto', 'radau-ps'.
    num_seg : int (10)
        The number of segments in the phase.
    transcription_order : int or sequence (5)
        The transcription order for the states in each segment.
    optimizer : str ('SLSQP')
        The optimization driver to use for the problem:  'SLSQP' or 'SNOPT'.
    compressed : bool (True)
        If True, run with compressed transcription.

    Returns
    -------
    prob : openmdao.Problem
        The OpenMDAO problem instance for the optimal control problem.

    """

    p = Problem(model=Group())

    if optimizer == 'SNOPT':
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.options['dynamic_simul_derivs'] = True
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Verify level'] = 3
    else:
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

    phase = Phase(transcription,
                  ode_class=LaunchVehicleLinearTangentODE,
                  ode_init_kwargs={'central_body': 'moon'},
                  num_segments=num_seg,
                  transcription_order=transcription_order,
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

    phase.add_timeseries_output('guidance.theta', units='deg')

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

    return p


if __name__ == '__main__':
    ssto_moon_linear_tangent(optimizer='SLSQP')
