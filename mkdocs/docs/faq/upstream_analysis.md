# How do connect the outputs of an upstream analysis as inputs to Dymos?

One of the design goals of Dymos is to allow the trajectory to be a part of a larger multidisciplinary optimization problem.
Sometimes, you may want to take the results of some upstream design and use them as an input to a Dymos trajectory.
There are different ways of doing this depending on what is being connected from the upstream source.

## Upstream outputs as parameters

If an upstream component performs some calculations to compute an output that is static through the trajectory, this is a use-case for parameters.
For instance, we might use an upstream analysis to compute a wing area, spacecraft dry mass, or rotor radius.


Example:  [Multi-phase cannonball](../examples/multi_phase_cannonball/multi_phase_cannonball.md)


The [multi-phase cannonball example](../examples/multi_phase_cannonball/multi_phase_cannonball.md) uses an upstream analysis to compute the aerodynamic reference area and mass of a cannonball based on its radius.
The problem then seeks to find the radius which provides the maximum range for a given initial muzzle energy.
If the cannonball is too large, it will be heavy and have lower initial velocity given the fixed initial energy.
If the cannonball is too small, its ballistic coefficient will be low and it will not maintain velocity in the presence of drag.

## Upstream outputs as controls

It is possible to connect a control time history as inputs to a phase.
This is somewhat more complex compared to static parameters because we now need to know the points in time at which the control input values are needed.
When the controls in a phase are not optimized (`opt=False`) in the control specification, the controls exist as an input that can accept a connection.

Example: [Brachistochrone with upstream controls](../examples/brachistochrone/brachistochrone_upstream_controls.md)

In the [Brachistochrone with upstream controls example](../examples/brachistochrone/brachistochrone_upstream_controls.md), an upstream IndepVarComp provides the control input values, rather than the phase itself.
Optimization settings have to be applied using the OpenMDAO `add_design_var` method, since these controls are not "owned" by the Dymos phase.

## Upstream outputs as initial state values

States behave a bit differently than controls.
Rather than connecting the entire time history at the state input nodes, those are still managed by Dymos.
If using `solve_segments=True`, one can connect the initial value of a state from an external source.
When using optimizer-driven collocation (`solve_segments=False`), states should be "connected" via constraint.
In fact, this is how phases are connected in trajectories...new constraints are added that constrain the value of states at phase junctions.

Example: [Brachistochrone with upstream state](../examples/brachistochrone/brachistochrone_upstream_states.md)

## Upstream outputs as phase initial time or duration
