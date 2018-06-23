Variables
---------

An Optimal Control problem in |project| consists of the following variable types.

Time
~~~~

Optimal control problems in |project| assume a system that is evolving in time.  State variables
typically obey some ordinary differential equation that provides the derivative of the states
w.r.t. time.

Users can specify various options relating to time with the `set_time_options` method of Phase.

In |project|, time is characterized by two variables:

* `t_initial` - The initial time of the phase
* `t_duration` - The duration of the phase

The bounds, scaling, and units of these variables may be set using `set_time_options`.

.. embed-options::
    dymos.phases.options
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
polynomial is specified using the *transcription_order* method.  In |project|, the minimum state
transcription order is 3.

Users can specify bounds, scaling, and units of the state variables with the
phase method `set_state_options`.  The following options are valid:

.. embed-options::
    dymos.phases.options
    _ForDocs
    state_options


Controls and Design Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typically, and ODE will have inputs that impact its values but, unlike states, don't define the
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
assigned to controllable parameters to the ODE.  Therefore, the method name to add a control to
a phase is `add_control`. Valid options for controls and design parameters are as follows:

.. embed-options::
    dymos.phases.options
    _ForDocs
    control_options

.. embed-options::
    dymos.phases.options
    _ForDocs
    design_parameter_options

Like states, *dynamic* controls are modeled as polynomials.  When
transcribed to a nonlinear programming problem, a dynamic control is given a unique value at each
node within the phase.  Design parameters are modeled as a singular value that is broadcast to all
nodes in the phase before the ODE function is evaluated.  If you can parameterize your problem in
such a way that static controls can be used, performance may be significantly better due to the
size of the NLP problem being much smaller.

.. note::
    The order of a dynamic control polynomial in a segment is one less than the state
    transcription order (i.e. a dynamic control in a phase with `transcription_order=3` will
    be represented by a second-order polynomial.

Example: Solving the Linear Tangent Launch Vehicle Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following problem demonstrates the use of |project| to solve for the minimum time for a single
stage launch vehicle to reach lunar orbit from the surface of the moon.  Optimal control theory
dictates that the tangent of the pitch angle of the thrust vector varies linearly with time.
Therefore, rather than using a dynamic control to specify the thrust angle at each instance in
time, we can instead specify two paramters (`a` and `b`) as design parameters.  These parameters
dictate the slope and intercept of the tangent of the thrust angle w.r.t. time.


