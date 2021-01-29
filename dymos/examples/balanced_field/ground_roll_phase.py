
__all__ = ['ground_roll_phase']

import openmdao.api as om
import dymos as dm

from dymos.examples.balanced_field.ground_roll_ode_comp import GroundRollODEComp


def ground_roll_phase(transcription=dm.Radau(num_segments=3), h=(0., 'ft'), T=(27000. * 2, 'lbf'),
                      m=(174200., 'lbm'), alpha=(0.0, 'deg'), mu_r=(0.03, None),
                      initial_bounds=(None, None), duration_bounds=(None, None)):
    """
    Instantiate a Phase to model ground roll for the balanced field length problem.

    Parameters
    ----------
    transcription : dymos Transcription
        The transcription to be used with this Phase.
    h : (float, str)
        The altitude (value, units) to be used for this Phase.
    T : (float, str)
        The thrust (value, units) to be used for this Phase.
    m : (float, str)
        The mass (value, units) to be used for this Phase.
    alpha : (float, str)
        The angle of attack (value, units) to be used for this Phase.
    mu_r : (float, str or None)
        The runway friction coefficient (value, units) to be used for this Phase.
    initial_bounds : (float, float)
        Bounds (lower, upper) on initial time in seconds
    duration_bounds : (float, float)
        Bounds (lower, upper) on duration in seconds

    Returns
    -------
    phs
        An instantiated Phase for modeling Groundroll.  The phase instantiation includes the addition
        of the necessary states (r, v) and the parameters from the argument list.  No boundary
        constraints, path constraints, or objectives are included on the returned Phase.

    """
    phs = dm.Phase(ode_class=GroundRollODEComp, transcription=transcription)

    #
    # Set the options on the optimization variables
    #
    # phs.set_time_options(initial_bounds=initial_bounds, duration_bounds=duration_bounds, duration_ref=10.0)
    #
    # # Add states to be integrated
    # phs.add_state('r', lower=0, ref=1000.0, defect_ref=1000.0)
    # phs.add_state('v', lower=0.001, ref=100.0, defect_ref=100.0)

    # Setup the constant inputs for this phase and their default values
    # phs.add_parameter('h', val=h[0], opt=False, units=h[1], dynamic=False)
    # phs.add_parameter('T', val=T[0], opt=False, units=T[1], dynamic=False)
    # phs.add_parameter('m', val=m[0], opt=False, units=m[1], dynamic=True)
    # phs.add_parameter('alpha', val=alpha[0], opt=False, units=alpha[1], dynamic=True)
    # phs.add_parameter('mu_r', val=mu_r[0], opt=False, units=mu_r[1], dynamic=False)

    # Send all ODE outputs to the timeseries
    phs.add_timeseries_output('*')

    return phs

