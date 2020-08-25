# Variables

An Optimal Control problem in Dymos consists of the following variable types.

## Time

Optimal control problems in Dymos assume a system that is evolving in time.  State variables
typically obey some ordinary differential equation that provides the derivative of the states
w.r.t. time.

Users can specify various options relating to time with the `set_time_options` method of Phase.

In Dymos, the phase time is controlled by two inputs:

- `t_initial` - The initial time of the phase
- `t_duration` - The duration of the phase

The bounds, scaling, and units of these variables may be set using `set_time_options`.  In addition,
the user can specify that the initial time or duration of a phase is to be connected to some
external output by specifying `input_initial = True` or `input_duration = True`.  In the case of
fixed initial time or duration, or input initial time or duration, the optimization-related options
have no effect and a warning will be raised if they are used.

The variables `t_initial` and `t_duration` are converted to time values at the nodes within the phase.
Dymos computes the following time values, which can be used inside the ODE:

- `time` - The canonical time. At the start of the phase `time = t_initial`, and `time = t_initial + t_duration` at the end of the phase.
- `time_phase` - The elasped time since the beginning of the phase.  `time_phase = time - t_initial`
- `t_initial` - The initial time of the current phase (this value is the same at all nodes within the phase).
- `t_duration` - The phase time duration of the current phase (this value is the same at all nodes within the phase).

{{ embed_options('dymos.phase.options.TimeOptionsDictionary', '###Options for Time Variables') }}

## States

States are variables that define the current condition of the system.  For instance, in trajectory
optimization they are typically coordinates that define the position and velocity of the vehicle.
They can also be things like component bulk temperatures or battery state-of-charge.  In most
dynamic systems, states obey some ordinary differential equation.  As such, these are defined
in an `ODE` object.

At the phase level, we assume that states evolve continuously such that they can be modeled as a
series of one or more polynomials.  The phase duration is broken into one or more *segments* on
which each state (and each dynamic control) is modeled as a polynomial.  The order of the
polynomial is specified using the *transcription_order* method.  **In Dymos, the minimum state
transcription order is 3.**

Users can specify bounds, scaling, and units of the state variables with the phase method `add_state`.
Settings on a previously-added state variable may be changed using the `set_state_options` method.
The following options are valid:

{{ embed_options('dymos.phase.options.StateOptionsDictionary', '###Options for State Variables') }}

The Radau Pseudospectral and Gauss Lobatto phases types in Dymos use differential defects to
approximate the evolution of the state variables with respect to time.  In addition to scaling
the state values, scaling the defect constraints correctly is important to good performance of
the collocation algorithms.  This is accomplished with the `defect_scaler` or `defect_ref` options.
As the name implies, `defect_scaler` is multiplied by the defect value to provide the defect
constraint value to the optimizer.  Alternatively, the user can specify `defect_ref`.  If provided,
`defect_ref` overrides `defect_scaler` and is the value of the defect seen as `1` by the optimizer.

If the ODE is explicitly depending on a state's value (for example, the brachistochrone ODE is a function of the bead's speed), then the user specifies those inputs in the ODE to which the state is to be connected using the `targets` option.
It can take the following values:

- (default)
    If left unspecified, targets assumes a special `dymos.utils.misc._unspecified` value.
    In this case, dymos will attempt to connect to an input of the same name at the top of the ODE (either promoted there, or there because the ODE is a single component).

- None
    The state is explicitly not connected to any inputs in the ODE.
- str or sequence of str
    The state values are connected to inputs of the given name or names in the ODE.
    These targets are specified by their path relative to the top level of the ODE.

To simplify state specifications, using the first option (not specifying targets) and promoting targets of the state to inputs of the same name at the top-level of the ODE.

## Controls

Typically, an ODE will have inputs that impact its values but, unlike states, don't define the
system itself.  Such inputs include things like throttle level, elevator deflection angle,
or spring constants.  In Dymos, dynamic inputs are referred to as controls, while
static inputs are called parameters.

Dynamic controls are values which we might expect to vary continuously throughout a trajectory, like an elevator deflection angle for instance.
The value of these controls are often determined by an optimizer.

!!! note
    The order of a dynamic control polynomial in a segment is one less than the state
    transcription order (i.e. a dynamic control in a phase with `transcription_order=3` will
    be represented by a second-order polynomial.

{{ embed_options('dymos.phase.options.ControlOptionsDictionary', '###Options for Control Variables') }}

## Polynomial Controls

Sometimes it can be easier to optimize a problem by reducing the freedom in the controls.
For instance, one might want the control to be linearly or quadratically varying throughout a phase, rather than having a different value specified at each node.
In Dymos, this role is filled by the PolynomialControl.
Polynomial controls are specified at some limited number of points throughout a _phase_, and then have their values interpolated to each node in each segment.

{{ embed_options('dymos.phase.options.PolynomialControlOptionsDictionary', '###Options for Polynomial Control Variables') }}

## Parameters

Some inputs impact the system but have one set value throughout the trajectory.
We refer to these non-time-varying inputs as *design parameters*, since they typically involve parameters which define a system.
Parameters could include things like the wingspan of a vehicle or the mass of a heatsink.

{{ embed_options('dymos.phase.options.DesignParameterOptionsDictionary', '###Options for Parameters') }}

Parameters and controls can have their values determined by the optimizer, or they can be passed in from an external source.

