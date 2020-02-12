=============================================
Subclassing phases to reduce code duplication
=============================================

In many of the Dymos examples, you might notice that in each phase we metadata for time, states,
controls, and parameters which determines targets for various variables in the ODE, the variable
path which serves as the rate for states being integrated, and default units to be used for these
variables.  In a trajectory with many phases, this can lead to a lot of unnecessary code duplication.
Afterall, an ODE system will generally be associated with a given set of state variables, and the
rate sources, targets, and default units of those state variables *shouldn't* need to be specified
each time you decide to use it.  This is a pretty significant violation of the
<DRY https://en.wikipedia.org/wiki/Don%27t_repeat_yourself>_ principle.


The developers didn't want to require that non-standard OpenMDAO systems be used in ODEs.
Thanks to recent updates to the OpenMDAO setup stack, you can now subclass
Phase to associate that subclass with a particular ODE class.  The `initialize`` method for that Phase-derived
class can include declarations for `add_state`, `set_time_options`, `add_control`, etc., that set
default behavior for states, times, controls, and parameters when that phase is used.  Just remember
to invoke `super(DerivedPhase, self).initialize()` at the beginning of your subclass' `initialize` method.


If you want to override any options for time, states, controls, or parameters you can use the
`set_time_options`, `set_state_options`, `set_control_options`, `set_polynomial_control_options`,
`set_input_parameter_options`, or `set_design_parameter_options` to change any of the settings
whose defaults were set in the phase definition.


.. warning::
   Dymos Trajectory objects need to know about the optimal control variables in Phases
   (time, states, controls, parameters).  Therefore all of these variables must be added
   to a phase **prior** to setup!


Consider the two-phase cannonball example.  Here we use the same ODE in two phases.  To reduce the
amount of code needed when setting up the problem, we can create `Cannonball` phase, which always
uses the CannonballODE.  Since the default units, targets, and rate_source of the state variables
are unchanged in each phase in which we use it, we can define CannonballPhase as follows:


.. embed-code::
    dymos.examples.cannonball.cannonball_phase.CannonballPhase
    :layout: code


While the setup method could also have been used here, there will be some unit issues in the phase
linkages.  Definining this state metadata in `initialize`, before setup is called, ensures that Dymos
has all of the necessary unit information when setting up the trajectory.
