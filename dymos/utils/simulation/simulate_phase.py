from __future__ import print_function, division, absolute_import

from collections import Iterable
from six import iteritems, string_types

try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np
from numpy import ndarray

from . import ScipyODEIntegrator, SimulationResults
from ..interpolate import LagrangeBarycentricInterpolant


def simulate_phase(phase_name, ode_class, time_options, state_options, control_options,
                   design_parameter_options, time_values, state_values, control_values,
                   design_parameter_values, ode_init_kwargs, grid_data, times, record=True,
                   record_file=None):
    """
    Provides a way of simulating a phase that can be called in a multiprocessing pool.

    Parameters
    ----------
    phase_name : str
        The name of the phase being simulated.
    ode_class : object
        The OpenMDAO system class providing the ODE to be simulated.
    time_options : OptionsDictionary
        The time options for the phase being simulated.
    state_options : OptionsDictionary
        The state options for the phase being simulated.
    control_options : OptionsDictionary
        The control options for the phase being simulated.
    design_parameter_options : OptionsDictionary
        The design parameter options for the phase being simulated.
    time_values : ndarray
        The time values from the phase being simulated.
    state_values : dict of {str : ndarray}
        A dictionary keyed by state name containing the values of the state at the state input
        nodes of the given phase.
    control_values : dict of {str : ndarray}
        A dictionary keyed by control name containing the values of the state at all
        nodes of the given phase.
    design_parameter_values : dict of {str : ndarray}
        A dictionary keyed by design parameter name containing the values of all design parameters.
    ode_init_kwargs : dict
        Keyword initialization arguments for the ODE class.
    grid_data : GridData
        GridData object associated with the phase to be simulated.
    times : str, int, Sequence
        The times at which outputs of the ODE are desired.  If given as an int n, return values
        at n equally distributed points in time throughout the phase.  If str, must be a valid
        node subset for the phase.  If sequence, provides the times at which ODE outputs are
        desired.
    record : bool, optional
        If True, record the results of the simulation.
    record_file : str, optional
        If given, provides the file path for the recorded simulation.  Defaults to
        '<phase_name>_sim.db'.

    Returns
    -------
    dict of {str : SimulationResults}
        The SimulationResults object resulting from each Phase simulation, keyed by phase name.

    """
    print('simulating ', phase_name)

    if not state_options:
        msg = 'Phase has no states, nothing to simulate. \n' \
              'Call run_model() on the containing problem instead.'
        raise RuntimeError(msg)

    if isinstance(times, int):
        times = np.linspace(time_values[0], time_values[-1], times)

    rhs_integrator = ScipyODEIntegrator(ode_class=ode_class,
                                        time_options=time_options,
                                        state_options=state_options,
                                        control_options=control_options,
                                        design_parameter_options=design_parameter_options,
                                        ode_init_kwargs=ode_init_kwargs)

    x0 = {}

    for state_name, options in iteritems(state_options):
        x0[state_name] = state_values[state_name][0, ...]

    rhs_integrator.setup(check=False)

    exp_out = SimulationResults(time_options=time_options,
                                state_options=state_options,
                                control_options=control_options,
                                design_parameter_options=design_parameter_options)

    seg_sequence = range(grid_data.num_segments)

    for param_name, options in iteritems(design_parameter_options):
        val = design_parameter_values[param_name]
        rhs_integrator.set_design_param_value(param_name, val[0, ...], options['units'])

    first_seg = True
    for seg_i in seg_sequence:
        seg_idxs = grid_data.segment_indices[seg_i, :]

        seg_times = time_values[seg_idxs[0]:seg_idxs[1]]

        for control_name, options in iteritems(control_options):
            interp = LagrangeBarycentricInterpolant(grid_data.node_stau[seg_idxs[0]:seg_idxs[1]])
            ctrl_vals = control_values[control_name][seg_idxs[0]:seg_idxs[1]].ravel()
            interp.setup(x0=seg_times[0], xf=seg_times[-1], f_j=ctrl_vals)
            rhs_integrator.set_interpolant(control_name, interp)

        if not first_seg:
            for state_name, options in iteritems(state_options):
                x0[state_name] = seg_out.outputs['states:{0}'.format(state_name)]['value'][-1, ...]

        if not isinstance(times, string_types) and isinstance(times, Iterable):
            idxs_times_in_seg = np.where(np.logical_and(times > seg_times[0],
                                                        times < seg_times[-1]))[0]
            t_out = np.zeros(len(idxs_times_in_seg) + 2, dtype=float)
            t_out[1:-1] = times[idxs_times_in_seg]
            t_out[0] = seg_times[0]
            t_out[-1] = seg_times[-1]
        elif times in ('disc', 'state_disc'):
            t_out = seg_times[::2]
        elif times == 'all':
            t_out = seg_times
        elif times == 'col':
            t_out = seg_times[1::2]
        else:
            raise ValueError('Invalid value for option times. '
                             'Must be \'disc\', \'all\', \'col\', or Iterable')

        seg_out = rhs_integrator.integrate_times(x0, t_out,
                                                 integrator='vode',
                                                 integrator_params=None,
                                                 observer=None)

        if first_seg:
            exp_out.outputs.update(seg_out.outputs)
        else:
            for var in seg_out.outputs:
                exp_out.outputs[var]['value'] = np.concatenate((exp_out.outputs[var]['value'],
                                                                seg_out.outputs[var]['value']),
                                                               axis=0)
        first_seg = False
    # Save
    if record:
        filepath = record_file if record_file else '{0}_sim.db'.format(phase_name)
        exp_out.record_results(filepath, ode_class, ode_init_kwargs)
    print(phase_name, 'simulation complete')

    return exp_out


def simulate_phase_map_unpack(args):
    """
    Provides argument unpacking for versions of Python without multiprocessing.Pool.starmap.

    Parameters
    ----------
    args : tuple
        A tuple of the positional arguments to be sent to simulate_phase.

    Returns
    -------
    dict of {str : SimulationResults}
        The SimulationResults object resulting from each Phase simulation, keyed by phase name.

    """
    return simulate_phase(*args)
