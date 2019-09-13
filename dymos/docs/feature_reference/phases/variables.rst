Variables
---------

An Optimal Control problem in Dymos consists of the following variable types.

Time
~~~~

Optimal control problems in Dymos assume a system that is evolving in time.  State variables
typically obey some ordinary differential equation that provides the derivative of the states
w.r.t. time.

Users can specify various options relating to time with the `set_time_options` method of Phase.

In Dymos, time is characterized by two variables:

* `t_initial` - The initial time of the phase
* `t_duration` - The duration of the phase

The bounds, scaling, and units of these variables may be set using `set_time_options`.  In addition,
the user can specify that the initial time or duration of a phase is to be connected to some
external output by specifying `input_initial = True` or `input_duration = True`.  In the case of
fixed initial time or duration, or input initial time or duration, the optimization-related options
have no effect and a warning will be raised if they are used.

.. embed-options::
    dymos.phase.options
    _ForDocs
    time_options


States
~~~~~~

States are variables that define the current condition of the system.  For instance, in trajectory
optimization they are typically coordinates that define the position and velocity of the vehicle.
They can also be things like component bulk temperatures or battery state-of-charge.  In most
dynamic systems, states obey some ordinary differential equation.  As such, these are defined
in an `ODE` object.

At the phase level, we assume that states evolve continuously such that they can be modeled as a
series of one or more polynomials.  The phase duration is broken into one or more *segments* on
which each state (and each dynamic control) is modeled as a polynomial.  The order of the
polynomial is specified using the *transcription_order* method.  In Dymos, the minimum state
transcription order is 3.

Users can specify bounds, scaling, and units of the state variables with the
phase method `add_state`.  The following options are valid:

.. embed-options::
    dymos.phase.options
    _ForDocs
    state_options

The Radau Pseudospectral and Gauss Lobatto phases types in Dymos use differential defects to
approximate the evolution of the state variables with respect to time.  In addition to scaling
the state values, scaling the defect constraints correctly is important to good performance of
the collocation algorithms.  This is accomplished with the `defect_scaler` or `defect_ref` options.
As the name implies, `defect_scaler` is multiplied by the defect value to provide the defect
constraint value to the optimizer.  Alternatively, the user can specify `defect_ref`.  If provided,
`defect_ref` overrides `defect_scaler` and is the value of the defect seen as `1` by the optimizer.


Controls and Design Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typically, an ODE will have inputs that impact its values but, unlike states, don't define the
system itself.  Such inputs include things like throttle level, elevator deflection angle,
or spring constants.

Some of these inputs are values which we might expect to vary continuously throughout a trajectory,
like an elevator deflection angle for instance.  When we allow the optimizer to determine the dynamic
value of these parameters, we call them *controls*.

Other inputs are values which impact the system but have one set value throughout the trajectory.
We refer to these non-time-varying inputs as *design parameters*, since they typically involve
parameters which define a system. Design parameters include things like the wingspan of a vehicle
or the mass of a heatsink.  A design parameter can be declared using the phase method
`add_design_parameter`.

States are defined in the ODE.  Controls and design parameters, on the other hand, are optionally
assigned to controllable parameters in the ODE.  Therefore, the method name to add a control to
a phase is `add_control`. Valid options for controls and design parameters are as follows:

.. embed-options::
    dymos.phase.options
    _ForDocs
    control_options

.. embed-options::
    dymos.phase.options
    _ForDocs
    design_parameter_options

Like states, *dynamic* controls are modeled as polynomials within each segment.  When
transcribed to a nonlinear programming problem, a dynamic control is given a unique value at each
node within the phase.  Design parameters are modeled as a singular value that is broadcast to all
nodes in the phase before the ODE function is evaluated.  If you can parameterize your problem in
such a way that static controls can be used, performance may be significantly better due to the
size of the NLP problem being much smaller, despite the fact that they reduce the *sparsity* of
the jacobian matrix of the resulting optimal control problem.

.. note::
    The order of a dynamic control polynomial in a segment is one less than the state
    transcription order (i.e. a dynamic control in a phase with `transcription_order=3` will
    be represented by a second-order polynomial.

