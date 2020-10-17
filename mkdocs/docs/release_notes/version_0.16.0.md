# dymos 0.16.0

## Backwards incompatible changes

### Design parameters and input parameters are now just parameters

There is no longer a distinction between input parameters and design parameters.
While the add_input_parameter and add_design_paramaters will still function until dymos 1.0.0, the path to the parameters themselves is changed in a backwards incompatible way.

{{ upgrade_doc('dymos.test.test_upgrade_guide.TestUpgrade_0_16_0.test_parameters',
               feature='set_val',
               old_label='dymos < 0.16.0',
               new_label='dymos >= 0.16.0') }}

Likewise, the path to the parameter values in the timeseries outputs is changed:

{{ upgrade_doc('dymos.test.test_upgrade_guide.TestUpgrade_0_16_0.test_parameters',
               feature='parameter_timeseries',
               old_label='dymos < 0.16.0',
               new_label='dymos >= 0.16.0') }}

### The location specification for link phases is simplified

When using the `link_phases` method, previously a "location" of `'--'` was used to specify the initial time in a phase and `'++'` was used to specify the final time.
These have been changed to `initial` and `final` for more consistency across the codebase.
The previous behavior is deprecated and will be removed in v1.0.0.

### The endpoint conditions are no longer separately computed.

We had toyed with they notion of allowing discontinuous jumps in states and controls at the beginning or end of a phase.
Going forward, we'll handle this through special phase linkage conditions.
In the mean time, the components used to compute the values of states and controls before/after these jumps have been removed.

The initial and final value of a time, state, or control variable in a phase should now be retrieved from the timeseries.

{{ upgrade_doc('dymos.test.test_upgrade_guide.TestUpgrade_0_16_0.test_sequence_timeseries_outputs',
               feature='state_endpoint_values',
               old_label='dymos < 0.16.0',
               new_label='dymos >= 0.16.0') }}

## Enhancements

### Omitting parameter values from a timeseries

Since parameters are static, including their values in the timeseries can lead to more data being stored than necessary.
If desired, one can avoid storing parameters in timeseries using the `include_timeseries=False` directive.

{{ upgrade_doc('dymos.test.test_upgrade_guide.TestUpgrade_0_16_0.test_parameter_no_timeseries',
               feature='parameter_no_timeseries',
               old_label='dymos < 0.16.0',
               new_label='dymos >= 0.16.0') }}

!!! note "Phase linkages rely on timeseries values"
    Currently in Dymos, when linking phases the values of the linked parameters are required to be present in the timeseries.  
    Omitting a parameter from the timeseries will cause errors if it's used in a phase linkage constraint.

### Timeseries improvements

In previous dymos versions, when adding ODE outputs to a timeseries, the user was required to specify the units and shape of the output variable.
This is redundant since that information was defined in the timeseries itself.
However, dymos had no way to determine that variable metadata.
With recent changes to OpenMDAO, this is now possible and the addition of ODE variables is simplified by no longer having to specify units or shape of the timeseries outputs.

{{ upgrade_doc('dymos.test.test_upgrade_guide.TestUpgrade_0_16_0.test_simplified_ode_timeseries_output',
               feature='simplified_ode_output_timeseries',
               old_label='dymos < 0.16.0',
               new_label='dymos >= 0.16.0') }}

Adding timeseries outputs for a large ODE in previous versions of Dymos was a very verbose process, with a call to `add_timeseries_output` for each ODE-output of interest.
The `add_timeseries_output` method now supports the addition of multiple outputs with a single call, through providing the timeseries outputs as a sequence of strings, or via a glob pattern.
Timeseries outputs and glob patterns are matched based on their promoted path _relative to the top of the ODE_.

The block below shows the code required to create timeseries for all outputs of the `aero` subsystem in the minimum-time-climb ODE in version 0.15.0 compared to 0.16.0.

{{ upgrade_doc('dymos.test.test_upgrade_guide.TestUpgrade_0_16_0.test_glob_timeseries_outputs',
               feature='glob_timeseries_outputs',
               old_label='dymos < 0.16.0',
               new_label='dymos >= 0.16.0') }}

Adding a handful of outputs can be done by providing a sequence of output paths in the ODE.
Units can also be provided as a dictionary in which the keys are one or more of the output names provided.

{{ upgrade_doc('dymos.test.test_upgrade_guide.TestUpgrade_0_16_0.test_sequence_timeseries_outputs',
               feature='sequence_timeseries_outputs',
               old_label='dymos < 0.16.0',
               new_label='dymos >= 0.16.0') }}

Finally, the rates of a state variables as computed by the ODE (or provided by another variable) are now avilable in a timeseries as `timeseries.state_rates:{state_name}`.

### Phase Linkage improvements

Phases may now be linked by any variable: times, states, controls, control rates, polynomial controls, polynomial control rates, parameters, or ODE outputs).
The `link_phases` method continues to provide a convenient way of specifying continuity across multiple phases.
A new Trajectory method, `add_linkage_constraint`, has been added to provide more general phase linkage conditions.
`add_phase_linkage` allows inequality constraints to govern the way a variable's value changes between phases.

### dymos.run_problem

Dymos now features a top-level function `run_problem`.
While the standard OpenMDAO `run_driver` and `run_model` methods still work, `run_problem` automates some aspects of the workflow typically found in Dymos problems.

1. Automatically load a case from a previously recorded file
2. Iteratively call run_driver or run_model through a grid refinement algorithm.
3. Simulate a trajectory with-or-without having previously run it.

   This gives an explicit simulation output file that can be used to seed a reasonable initial guess.

### Grid refinement updates

Dymos now implements two grid refinement algorithm, dubbed 'hp' and 'ph'.
The 'hp' method was developed by Liu, Hager, and Rao[@liu2015adaptive].
It utilizes curvature information in the collocating polynomials to determine whether segments should be added, removed, or have their order changed.
The 'ph' method was developed by Patterson, Hager and Rao[@patterson2015ph].
It tends to prefer first changing the order of segments before increasing the number of segments in a phase.
The ph method does not reduce the number of segments in a phase.

While the same refinement method is used problem-wide, refinement options may be set per-phase using the Phase method `set_refine_options`.

Invoking grid refinement is done by using the `dymos.run_problem` method with `refine_iterations_limit > 1`.

## References

\bibliography