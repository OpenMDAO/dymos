# Transcriptions

Dymos supports multiple ways of solving optimal control problems.
While the goal of optimal control is typically to find a continuous control that minimizes the objective, Dymos utilizes some well-known _transcriptions_ which convert the problem of finding a continuous control function into a discrete nonlinear programming (NLP) problem.
The transcriptions used by Dymos transcribe the problem into a discrete problem using OpenMDAO, which in turn uses a third-party optimizer to solve the NLP.

In this section, four common transcriptions are used to solve the brachistochrone optimal control problem.

### Single Shooting

Perhaps the easiest way to conceptualize solving the problem is single shooting.

First, the optimizer proposes a time duration for the trajectory.

Next, the optimizer proposes a time history for the control $\theta$.

In Dymos, controls are generally modeled as a series of one or more polynomials defined at discrete _nodes_ across the trajectory.

The design variables controlled by the optimizer in this case are therefore the time duration $t_{duration}$ and a sequence of control values at some number of points throughout the trajectory $\vec{\theta}$.

The states are then propagated from their initial conditions for the proposed time duration.

- The solution is said to be _feasible_ when the constraints (given by the final conditions above) are satisfied.
- The solution is said to be _optimal_ when it is feasible and no control profile can provide a better objective value (a shorter time duration in this case).

The cannonball trajectory had no control variables.
Given the initial conditions, at a duration of flight, there is a single possible solution for its final states.
In this case, the presence of the control variables mean that there are an infinite number of possible control profiles that can bring the point to the final desired state.
For a well-posed optimization problem, one combination of feasible control values will provide a better objective value than any other.

### Multiple Shooting

Single shooting tends to be easily grasped, but it can encounter problems in practical application.
It can be difficult to guess a control profile that results in a feasible solution.
Sometimes infeasible solutions don't fail gracefully.
For instance, when optimizing the trajectory or an aircraft, the assumed control profile may result in the aircraft altitude becoming negative.
But computing the aerodynamic forces on the vehicle require us to know the atmospheric properties at the given state.
If the atmospheric model doesn't provide results at this negative altitude, an error will be raised and the optimization terminated.

One way to resolve this problem is to employ _multiple shooting_.
Rather than propagating the entire trajectory from start to finish in one continuous propagation, the trajectory is broken up into a number of segments.
The optimizer controls the state values at the beginning of each segment (except the first, since that state is fixed per the initial conditions in this case).
Now the trajectory is propagated across each segment.
The state values at the end of one segment, and the beginning of the next, will need to be added as a constraint to the optimizer.
Unless these states are continuous across the segment bounds, the trajectory are not physically realizable.
Unlike single-shooting, multiple shooting relies on a feasible solution to guarantee the trajectory is physically realizable.

For multiple shooting, the design variables include not only the time duration and control values, but the states at the start of all but the first segment.
In addition to the terminal constraints, the constraints now include the _continuity defects_ for each state at each segment boundary.
The resulting NLP is significantly larger than that of single shooting, but it is more numerically stable.

### Collocation/Pseudospectral Techniques

Multiple shooting alleviates some of the issues associated with single shooting, but it does not completely eliminate them.
Another family of transcriptions, collocation methods, impose constraints that enforce the physical realization of the trajectory.
When these constraints are to be satisfied by the optimizer, the trajectory can be more easily constrained to avoid infeasible regions of the search space where the models are invalid.
With collocation methods, the state values at discrete points in each segment throughout the trajectory are included in the design variables.
The _defect constraints_ which enforce physical accuracy, are added to the constraints.
The result is an even larger problem, but a few properties of this approach allow for good performance.
First, theres no propagation performed during each iteration, ODE simply needs to be evaluated.
Second, by breaking the trajectory into segments, the _defect constraints_ in any given segment are only impacted by the state and control values within that segment.
This makes the _jacobian_ matrix, the derivatives of the objective and constraints with respect to the design variables, very _sparse_.
OpenMDAO and some optimizers can capitalize on this sparsity to evaluate the derivatives for the optimizer much more quickly.
