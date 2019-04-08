================================================================
Commercial Aircraft Range Maximization by Differential Inclusion
================================================================

In this example we seek to maximize the range of a commercial aircraft
flying at a constant Mach number with a fixed initial fuel load.
The initial and final altitudes are constrained to 10000 ft
(that is, in this simplified example we're ignoring
the aircraft behavior near the ground).

This example is simplified in several ways.  The Mach number is fixed
at 0.8.  When only considering fuel burn, the aircraft will tend to fly
slower to conserve fuel, ignoring the running costs of being airborne.

For a more detailed optimization of commercial aircraft performance, see [BettsAircraft1995]_.

Differential Inclusion
----------------------

In this case we also use the steady flight assumption to allow the problem
to be parameterized with a different control set.  Rather than providing
throttle/thrust and angle of attack as control variables, we use a *differential
inclusion* approach.  Whereas altitude and velocity are often treated as
state variables to be integrated, here we take a slightly different approach.

Rather than using the differential equations to compute rates for the
flight path angle and true airspeed, we use a nonlinear solver to determine
the alpha and thrust time-history which is needed to make the given
altitude and airspeed time-history possible.  This technique is known
as differential inclusion.  It was demonstrated by Seywald [Seywald1994]_.  Since
the collocation techniques in Dymos are based on polynomial approximations,
rates of the control variables can be easily calculated, which can then be used
to the reordered dynamics equations to solve for angle of attack and thrust.

The use of collocation/psuedospectral techniques in solving problems via
differential inclusion has been examined by Fahroo and Ross [FahrooRoss2001]_ and
Kumar and Seywald [Kumar1996]_. Since the differential inclusion approach demonstrated
here relies on nonlinear solvers running within the ODE model, obtaining accurate
derivatives is paramount.  Computing derivatives with finite differences in the
presence of nonlinear solvers can lead to inaccurate derivatives based on the properties
of the solver.  With OpenMDAO's unified derivatives framework, the sensitivy to
solver tolerances is greatly reduced.


Pitfalls of Differential Inclusion
==================================

Differential inclusion works well in this case but it is highly dependent
on the parameterization of the problem.  For instance, it is possible for the
optimizer to suggest a flight profile that requires thrust well beyond the capabilities
of the vehicle.  In our case, the throttle parameter :math:`$\tau$` is path constrained
within the optimizer, so that the optimizer can make the solution physically attainable
by our model.

We have two options for how to tackle this problem.  We could specify altitude as a control
variable, and use it's approximate derivative (`alt_rate`) to provide the altitude rate for
the flight equilibrium conditions.  In practice, however, this noisy derivative of altitude
can cause numerical difficulties.  Other times, the optimizer will shape the control to satisfy
our equilibrium conditions at the collocation nodes but be wildly wrong in between these nodes.
One simple way to get around this is to still integrate altitude, but to provide `climb_rate`
as the control variable.  This makes it easy to constraint `climb_rate` with bounds constraints,
and the time-history of altitude will be smoother since it is the time integral of the rate of
climb.


Solver Options
##############

In general, `openmdao.api.AnalysisError` should be raised when the design variables
provided by the optimizer result in nonconvergence of the solvers. This informs pyoptsparse
to backtrack along its line search for a solution.  This error can be raised by the `compute`
method of a component.  The nonlinear solver can also raise this error if it fails to converge
by setting the following options:

.. code-block:: python

        self.nonlinear_solver.options['maxiter'] = 150
        self.nonlinear_solver.options['err_on_maxiter'] = True

In the course of solving, an outer NewtonSolver won't necessarily force subsystems to be solved,
instead relying on the derivative at the at the previous solution.  In the context of OpenMDAO,
even explicit components constitute subsystems. This can lead to poorly converged solutions.

Setting the `solve_subystems` option to True will force subsystems to be updated, making sure that
all derivative information is up-to-date.

.. code-block:: python

        self.nonlinear_solver.options['solve_subsystems'] = True
        self.nonlinear_solver.options['max_sub_solves'] = 10

The Impact of Choice of Transcription
#####################################

The high-order Gauss-Lobatto method used by Dymos evaluates the ordinary differential equations
in a two-step process.  First, the ODEs are evaluated at the *state discretization* nodes.  Using
the state values and derivatives, values and rates at the *collocation* nodes are then interpolated.
If a large rate is found at one of the state discretization nodes, the interpolation at the
collocation node can be outside of the expected range, leading to nonconvergence of solvers within
the ODEs.

There is also an overhead cost to invoking solvers.  While differential inclusion reduces the
number of design variables for the optimizer, it also slows down each evaluation of the ODE
somewhat.  Since evaluating the collocation constraints in the Gauss-Lobatto method requires two
evaluations of the ODE, it typically suffers greater slowdown that the Radau Pseudospectral method,
which evaluates the defects of the ODE with a single evaluation that encompasses all the nodes.

For these reasons the Radau Pseudospectral transcription is typically preferable when using
a differential inclusion approach to problems.

Problem Formulation
-------------------

State Variables
===============

=========  ==============================  =================== ===================
Name       Description                     Fixed Initial Value Fixed Final Value
=========  ==============================  =================== ===================
range      distance flown along ground     True (0 NM)         False
mass_fuel  mass of fuel aboard aircraft    True (30000 lbm)    True (0 lbm)
alt        aircraft altitude               True (10000 ft)     True (10000 ft)
=========  ==============================  =================== ===================

Dynamic Controls
================

==========       ==============================  =========  =================== ===================
Name             Description                     Optimized  Fixed Initial Value Fixed Final Value
==========       ==============================  =========  =================== ===================
climb_rate       aircraft rate of climb          True       False               False
==========       ==============================  =========  =================== ===================

Design Parameters
=================

============  ==============================  ======================
Name          Description                     Optimized
============  ==============================  ======================
mach          mach number                     False (0.8)
S             aerodynamic reference area      False (427.8 m**2)
mass_empty    aircraft empty mass             False (330693.393 lbm)
mass_payload  aircraft payload mass           False (74100 lbm)
============  ==============================  ======================

Objective
=========

============  ==============================  ======================
Name          Location (initial or final)     Minimized or Maximized
============  ==============================  ======================
range         final                           Maximized
============  ==============================  ======================

Nonlinear Path Constraints
==========================

============  ==============================  ============  ==============
Name          Location (initial or final)     Lower         Upper
============  ==============================  ============  ==============
tau           engine throttle parameter       0.01          1.0
============  ==============================  ============  ==============

Nonlinear Boundary Constraints
==============================

None

Models
------

Atmosphere
==========

This problem uses an analytic fit to the 1976 standard atmosphere.

============  ==============================  ======================
Name          Description                     Input or Output
============  ==============================  ======================
alt           altitude (m)                    input
pres          static pressure (Pa)            output
temp          temperature (K)                 output
sos           speed of sound (m/s)            output
rho           density (kg/m**3)               output
============  ==============================  ======================

True Airspeed
=============
`TrueAirspeedComp` uses the Mach number, provided as a control, and the speed of
sound from the atmosphere model to compute the true airspeed of the aircraft.

.. math ::
    TAS = mach \cdot sos

============  ==============================  ======================
Name          Description                     Input or Output
============  ==============================  ======================
mach          Mach number                     input
sos           speed of sound (m/s)            input
TAS           true airspeed (m/s)             output
============  ==============================  ======================

Flight Path Angle
=================
`SteadyFlightPathAngleComp` uses the true airspeed and the climb rate, obtained
by differentiating the altitude time history at the nodes, to compute the
flight path angle.

.. math ::
    \gamma = \arctan{\frac{TAS}{\dot{h}}}

============  ==============================  ======================
Name          Description                     Input or Output
============  ==============================  ======================
TAS           true airspeed (m/s)             input
alt_rate      climb rate (m/s)                input
gam           flight path angle (rad)         output
============  ==============================  ======================

Range Rate
==========

`RangeRateComp` uses the true airspeed and the flight path angle to
determine the velocity projected along the ground.  This is the
derivative of the state variable `range`.

.. math ::
    \dot{range} = TAS \cdot \cos{\gamma}

============  ==============================  ======================
Name          Description                     Input or Output
============  ==============================  ======================
TAS           true airspeed (m/s)             input
gam           flight path angle (rad)         input
dXdt:range    range rate (m/s)                output
============  ==============================  ======================

Mass
====

The component `MassComp` defined in `mass_comp.py` computes the aircraft
total mass based on its empty mass, payload mass, and current fuel mass.
It also computes total weight which simplifies some equations later on.

.. math ::
    mass_{total} = mass_{empty} + mass_{payload} + mass_{fuel}
    W_{total} = 9.80665 mass_{total}

============  ==============================  ======================
Name          Description                     Input or Output
============  ==============================  ======================
mass_empty    aircraft empty mass (kg)        input
mass_payload  payload mass (kg)               input
mass_fuel     fuel mass (kg)                  input
mass_total    total aircraft mass (kg)        output
W_total       total aircraft weight (N)       output
============  ==============================  ======================

Dynamic Pressure
================

The `DynamicPressureComp` computes the dynamic pressure from true airspeed
and atmospheric density.

.. math ::
    q = \frac{1}{2}\rho TAS^2

============  ==============================  ======================
Name          Description                     Input or Output
============  ==============================  ======================
TAS           true airspeed (m/s)             input
rho           atmospheric density (kg/m**3)   input
q             dynamic pressure (Pa)           output
============  ==============================  ======================

Aerodynamics
============

The aerodynamics group computes the aerodynamic coefficients and forces
on the vehicle.  It consists of an interpolation component which outputs
lift, drag, and moment coefficients as a function of Mach number, angle of attack,
altitude, and tail rotation angle.  A second component then uses these
coefficients, along with dynamic pressure and aerodynamic reference area,
to compute the lift and drag forces on the vehicle.

The aerodynamics group resides within the flight equilibrium group.  As that
group iterates to find the combination of thrust coefficient, angle of attack,
and tail rotation angle, aerodynamics needs to update the values of the
interpolated coefficients and resulting forces.

Organizationally speaking, we logically could have put the dynamic pressure component
within the aerodynamics group.  However, since that group doesn't need to be
updated with changes in alpha and tail angle, it's more efficient to leave it
outside of flight equilibrium group.

============  ==============================  ======================
Name          Description                     Input or Output
============  ==============================  ======================
mach          mach number                     input
alt           altitude (m)                    input
alpha         angle of attack (deg)           input
eta           tail rotation angle (deg)       input
CL            lift coefficient                output
CD            drag coefficient                output
CM            moment coefficient              output
L             lift force (N)                  output
D             drag force (N)                  output
============  ==============================  ======================

.. note::
    This example uses `openmdao.api.MetaModelStructuredComp` to interpolate aerodynamic properties
    of the vehicle.  This component is somewhat easier to use since it is distributed as part of
    OpenMDAO, but it can be significantly slower than alternatives such as
    `MBI <https://github.com/OpenMDAO/MBI>`_ and `SMT <https://github.com/SMTorg/smt>`_.  The
    Aerodynamics group includes an implementation of the aerodynamics coefficients interpolant
    which uses MBI that is commented out.  By switching to MBI for this component, this problem
    will solve roughly 20 times faster.

Flight Equilibrium
==================

The steady flight equilibrium group uses balances to solve for the angle of attack and
tail plane rotation angle such that the aircraft is in steady
flight (the rates of change in flight path angle and true airspeed are zero) and the
aerodynamic moment in the pitch axis is zero.

Of course, in reality the vehicle will accelerate, but the flight profile being modeled
is so benign that assuming steady flight at discrete points (nodes) in the trajectory is
not terribly inaccurate.

The thrust coefficient necessary for steady flight is computed by balancing the drag equation

.. math ::
    C_T = W_{total} * \frac{\sin{\gamma}}{\cos{\alpha} * q \cdot S} + \frac{C_D}{\cos{\alpha}}

The lift coefficient required for steady flight is found by balancing lift and weight:

.. math ::
    \tilde{C_L} = W_{total} * \frac{\cos{\gamma}}{q \cdot S} - C_T * \sin{\alpha}

Using coefficients in the balance equations is better scaled from a numerical standpoint.

Propulsion
==========

Having determined the coefficient of thrust, the propulsion group converts that to an
actual thrust value. This is, in turn, used to compute the rate of fuel burn.  In addition,
by normalizing thrust at any point by the maximum possible thrust, we obtain the throttle
parameter :math:`\tau`.  The propulsion group uses a number of components to perform these
calculations.  First, thrust is computed from the thrust coefficient, dynamic pressure,
and reference area:

.. math ::
    T &= C_T \cdot q \cdot S

Maximum thrust is computed by multiplying sea-level thrust by the ratio of pressure to
sea-level atmospheric pressure.

.. math ::
    T_{max} = T_{max,sl} \frac{P}{P_{sl}}

The throttle parameter is then the ratio current thrust to maximum possible thrust.

.. math ::
    \tau = \frac{T}{T_{max}}

The thrust specific fuel consumption is computed as follows:

.. math ::
    TSFC = TSFC_{sl} - 1.5E-10 \cdot 9.80665 \cdot alt

Finally, fuel burn rate is:

.. math ::
    \dot{mass_{fuel}} = -TSFC \frac{T}{9.80665}

1. The ODE System: aircraft_ode.py
----------------------------------

.. embed-code::
    dymos.examples.aircraft_steady_flight.aircraft_ode.AircraftODE
    :layout: code

In this case the system has only two integrated states: `range` and `mass_fuel`.  There are six parameters.
Two of them, `alt` and `climb_rate`, will be varied dynamically in the phase, and the other four,
`mach`, `S`, `mass_empty`, and `mass_payload`, will be set to fixed values as non-optimized design
parameters.  More details on the various models involved can be found in the examples code.

2. Building and running the problem
-----------------------------------

In the following code we define and solve the optimal control problem.  Note that we demonstrate
the use of externally-connected design parameters in this case.  The four design parameters have
`input_value = True`, and are connected to a source provided by the `assumptions` IndepVarComp.

.. embed-code::
    dymos.examples.aircraft_steady_flight.doc.test_doc_aircraft_steady_flight.TestSteadyAircraftFlightForDocs.test_steady_aircraft_for_docs
    :layout: code, output, plot

References
----------
.. [BettsAircraft1995] Betts, John T., and Evin J. Cramer. “Application of Direct Transcription to Commercial Aircraft Trajectory Optimization.” Journal of Guidance, Control, and Dynamics 18.1 (1995): 151–159.
.. [Seywald1994] Seywald, Hans. “Trajectory Optimization Based on Differential Inclusion (Revised).” Journal of Guidance, Control, and Dynamics 17.3 (1994): 480–487.
.. [Kumar1996] Kumar, Renjith R, and Hans Seywald. “Should Controls Be Eliminated While Solving Optimal Control Problems via Direct Methods?” Journal of Guidance, Control, and Dynamics 19.2 (1996): 418–423.
.. [FahrooRoss2001] Fahroo, F., and Ross, I. M., “A Second Look at Approximating Differential Inclusions,” Journal of Guidance, Control, and Dynamics, Vol. 24, No. 1, 2001, pp. 131–133.