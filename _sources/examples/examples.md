# Dymos by Example

The goal of these examples is to walk users through the process of formulating an optimal control problem and solving it using Dymos.

In working through these examples, we'll try to emphasize the following process:

1.  Formulate the optimal control problem
2.  Identify state and control variables, and the ordinary differential equations (ODE) which govern the dynamics.
3.  Build the ODE as an OpenMDAO system.
4.  Test the evaluation of the ODE.
5.  Define the partial derivatives for the ODE.
6.  Test the partial derivatives of the ODE against finite-difference or (preferably) complex-step approximations.


## Prerequisites

These examples assume that the user has a working knowledge of the following:

-   Python
-   The numpy package for numerical computing with Python
-   The matplotlib package for plotting results

Some examples may require additional optional packages that are noted in those notebooks.

```{toctree}
:hidden:

brachistochrone/brachistochrone
brachistochrone/brachistochrone_upstream_init_duration_states
vanderpol/vanderpol
balanced_field/balanced_field
balanced_field/balanced_field_funccomp
bryson_denham/bryson_denham
commercial_aircraft/commercial_aircraft
double_integrator/double_integrator
hull/hull_problem
hypersensitive/hypersensitive
finite_burn_orbit_raise/finite_burn_orbit_raise
length_constrained_brachistochrone/length_constrained_brachistochrone
min_time_climb/min_time_climb
mountain_car/mountain_car
moon_landing/moon_landing
multi_phase_cannonball/multi_phase_cannonball
multibranch_trajectory/multibranch_trajectory
racecar/racecar
reentry/reentry
ssto_earth/ssto_earth
ssto_moon_linear_tangent/ssto_moon_linear_tangent
ssto_moon_polynomial_controls/ssto_moon_polynomial_controls
robertson_problem/robertson_problem
water_rocket/water_rocket
cart_pole/cart_pole
brachistochrone/brachistochrone_tandem_phases
implicit_duration/brachistochrone_implicit_duration
```
