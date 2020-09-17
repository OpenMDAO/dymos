# dymos 0.16.0

## Backwards incompatible changes

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

## Enhancements

### Omitting parameter values from a timeseries

Since parameters are static, including their values in the timeseries can lead to more data being stored than necessary.
If desired, one can avoid storing parameters in timeseries using the `include_timeseries=False` directive.

{{ upgrade_doc('dymos.test.test_upgrade_guide.TestUpgrade_0_16_0.test_parameter_no_include_timeseries',
               feature='parameter_no_timeseries',
               old_label='dymos < 0.16.0',
               new_label='dymos >= 0.16.0') }}

### Timeseries improvements

In previous dymos versions, when adding ODE outputs to a timeseries, the user was required to specify the units and shape of the output variable.
This is redundant since that information was defined in the timeseries itself.
However, dymos had no way to determine that variable metadata.
With recent changes to OpenMDAO, this is now possible and the addition of ODE variables is simplified by no longer having to specify units or shape of the timeseries outputs.

=== "Adding timseries otuputs in dymos < 0.16.0"
    ```
    phase.add_timeseries_output('aero.lift_force', units='N', shape=(1,))
    ```

=== "Adding timeseries outputs in dymos >= 0.16.0"
    ```
    phase.add_timeseries_output('aero.lift_force')
    ```

Adding timeseries outputs for a large ODE in previous versions of Dymos was a very verbose process, with a call to `add_timeseries_output` for each ODE-output of interest.
The `add_timeseries_output` method now supports the addition of multiple outputs with a single call, through providing the timeseries outputs as a sequence of strings, or via a glob pattern.
Timeseries outputs and glob patterns are matched based on their promoted path _relative to the top of the ODE_.

For instance, adding all ODE outputs to a timeseries is as simple as:

=== "Adding timeseries outputs with glob pattern"
    ```
    phase.add_timeseries_output('*')
    ```

Adding all outputs of a particular subsystem in the ODE (in this case named `aero`) is done as follows:

=== "Adding all ODE outputs from a particular subsystem"
    ```
    phase.add_timeseries_output('aero.*')
    ```

Added a handful of outputs can be done by providing a sequence of output paths in the ODE.
Units can also be provided as a dictionary in which the keys are one or more of the output names provided.

=== "Adding multiple variables with a sequence"
    ```
    phase.add_timeseries_output(['aero.*', 'prop.thrust', 'Q_body_inertial'], units={'prop.thrust': 'lbf', 'aero.lift': 'lbf'})
    ```

The rate of a state variable as computed by the ODE (or provided by another variable) is now avilable in a timeseries as `timeseries.state_rates:{state_name}`.

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