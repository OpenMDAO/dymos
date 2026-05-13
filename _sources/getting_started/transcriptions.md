# Optimal Control Transcriptions

Solving optimal control problems using classical techniques is typically a matter of finding a continuous control function that moves a system to the desired state while extremizing some objective (minimizing time, fuel, cost, etc.).

To solve these problems directly using nonlinear optimizers, we have to find a way to discretize this continuous problem into a form that can be solved by a nonlinear optimizer. This process is called _transcription_.

Dymos supports two forms of direct transcription: [collocation](getting_started:collocation:what_is_collocation) and explicit shooting.

Collocation-based optimal control methods, also known as pseudospectral methods, are implicit techniques.
The entire state and control history of the system is proposed at some discrete set of points in time.
We can then calculate how accurately the proposed state and control history obeyed the ODE governing the system dynamics - this is a residual we call the _defects_.
By iteratively varying the states, controls, and elapsed time of the trajectory we can reduce the defects to zero, meaning that the proposed state history is a solution to the ODE of the system, with the given control history and time of propagation.

In explicit shooting, an explicit numerical integration technique is used to propagate the initial state over the given time duration subject to the given controls. The process of integrating the trajectory itself eliminates the defects.

## Differences between collocation and explicit shooting

### Design Variables

For implicit collocation techniques, the design variables include:
- the states at some set of points along the trajectory (the initial and final ones may not be included if they're fixed)
- the controls at discrete points in time
- the elapsed time of the trajectory.

For explicit shooting techniques, the design variables include:
- the initial state (assuming it's not fixed, and assuming a single integration interval (_single shooting_))
- the initial state in each integration interval (for _multiple shooting_)
- the control values at some set of discrete points in time
- the elapsed time of the trajectory.

The design variable vector can be considerably larger for the implicit collocation techniques, depending on the size of the state vector and the number of discrete points in time.

### Constraints

For implicit collocation techniques, the constraints include:
- the defect constraints for each state at various discrete points in time throughout the trajectory
- continuity constraints that ensure the state values remain continuous (for `compressed=False`)
- continuity constraints on the control values (for `compressed=False`) and (optionally) their rates
- any other path or boundary constraints imposed

For explicit shooting techniques, the constraints include:
- continuity constraints that ensure the state values remain continuous (assuming multiple integration intervals (_multiple shooting_))
- continuity constraints on the control values (for `compressed=False`) and (optionally) their rates
- any other path or boundary constraints imposed

The constraint vector can be considerably larger for implicit collocation techniques.
Again this depends on the size of the state vector and the number of points into which the trajectory has been discretized.

We should also note that for the implicit collocation techniques, the state values at the start and the end of the trajectory are design variables.
This means that these values can be trivially _fixed_ (removed from the design variable vector) or bound using simple bounds on the corresponding design variable.

**The only way to impose a constraint on the path or the final value of a state in a shooting method is with a nonlinear path or boundary constraint.**

### Performance

Due to the fact that the design variable and constraint vectors are considerably larger for implicit collocation problems, it might be logical to conclude they they're slower than explicit shooting for solving a corresponding problem, but this is not the case.

In a single shooting method, the value of some state later in time is (at least potentially) a function of everything that happened before it.
The corresponding _jacobian matrix_ of derivatives for these state values along a trajectory is lower triangular.

For the implicit collocation techniques, the state in one integration _segment_ is not a function of anything that happens outside of that integration segment.
The jacobian matrix is much more sparse.
This increased sparsity helps in two ways.

First, we can determine the derivatives much more efficiently.
Consider a finite differencing technique.
If we know that a state value in the first integration segment of a trajectory only impacts that segment, and there are 100 segments, then we can simultaneously perturb a state value in each of the 100 segments and compare the resulting defect vector to the nominal defect value to compute the sensitivity via finite difference.
This is a massive performance gain that's not possible with a single shooting technique, though multiple shooting can help this.

Second, some optimizers can capitalize on the sparsity of jacobian matrices to provide significantly improved performance vs those which operate only on dense matrices.

Because of these factors, **collocation techniques can be orders of magnitude faster than their explicit shooting counterparts**.

### Robustness

While faster, implicit techniques impose far more defect constraints on the problem.
In some cases, scaling the optimization problem can be a challenge and convergence is poor.
For instance, if the thrust of a rocket engine is taken from experimental data, it can be extremely noisy.
Without first smoothing the data, an implicit simulation of the resulting trajectory may be difficult.
In this case, explicit integration methods can power through and provide _an_ answer, even if it is subject to some amount of error.

**Explicit integration, at least using a fixed-step form, doesn't fail to provide an answer - but it's important that the user verify its accuracy.**

On the other hand, explicit integration is subservient to the ODE, and sometimes this feature is problematic.
When following the prescribed control profile can take the system into a region where the dynamics become singular, the resulting errors will cause the optimization to fail.
The [minimum time-to-climb](examples:minimum_time_climb) problem is a classic example of this.
The equations of motion used in this problem are singular in vertical flight - there is division by zero in the derivatives of the equations of motion in this case.
It is relatively easy to prescribe a profile of the angle-of-attack history (the control) that sends the aircraft into vertical flight when the integration is required to follow it.

Conversely, the collocation techniques decouple the proposed control history and the trajectory.
Rather than being governed by the control, the flight path angle at various times throughout the trajectory is itself a design variable, and can be bound to values such that the singularities are avoided.
The optimizer will then work to make the control history compatible with the corresponding state trajectory, but the two are only compatible when the defect constraints are satisfied.
