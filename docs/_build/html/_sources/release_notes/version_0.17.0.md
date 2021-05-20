# dymos 0.17.0

## Enhancements

### OpenMDAO variable tags may be used to specify rate sources in ODE systems.

OpenMDAO supports tagging of inputs and outputs.
Dymos now can use this capability to specify the rate source and default units of a state from within the ODE.
The following code demonstrates how to tag state rate sources and provide their default units.

{{ upgrade_doc('dymos.test.test_upgrade_guide.TestUpgrade_0_17_0.test_tags',
               feature='tag_rate_source',
               old_label='dymos < 0.17.0',
               new_label='dymos >= 0.17.0') }}

This allows the user to avoid repetitive specification of the rate sources every time a specific ODE is used.

{{ upgrade_doc('dymos.test.test_upgrade_guide.TestUpgrade_0_17_0.test_tags',
               feature='declare_rate_source',
               old_label='dymos < 0.17.0',
               new_label='dymos >= 0.17.0') }}

### Dymos can now automatically generate plots of timeseries outputs.

Typically Dymos usage featured a lot of boilerplate code for plotting the results after a run and, optionally, a simulation of the resulting control profile.
As of 0.17.0, Dymos can now generate plots of all timeseries outputs by giving the `make_plots=True` option to `dymos.run_problem`.

### Dymos now uses introspection to automatically determine shapes of states, controls, and parameters if they are left unspecified.

In a continuing effort to minimize the amount of boilerplate code required, Dymos can now automatically determine the default units and shapes of states, controls, polynomial controls, parameters, and time.

