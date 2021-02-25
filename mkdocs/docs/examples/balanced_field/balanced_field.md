# Aircraft Balanced Field Length Calculation

!!! info "Things you'll learn through this example"
    - How to perform branching trajectories
    - How to constraint the difference between values at the end of two different phases
    - Using complex-step differentiation on a monolithic ODE component

The United States Federal Aviation Regulations Part 25 defines a
balanced field length for the aircraft as the shortest field which can
accommodate a "balanced takeoff".  In a balanced takeoff the aircraft
accelerates down the runway to some critical speed "V1".

Before achieving V1, the aircraft must be capable of rejecting the takeoff
and coming to a stop before the end of the runway.

After V1, the aircraft must be capable of achieving an altitude of 35 ft
above the end of the runway with a speed of V2 (the minimum safe takeoff
speed or 1.2 x the stall speed) while on a single engine (for two engine aircraft).

At V1, both options must be available. The nominal phase sequence for this trajectory is:

1. Break Release to V1 (br_to_v1)

    Accelerate down the runway under the power of two engines.
    Where V1 is some as-yet-undetermined speed.

2. V1 to Vr (v1_to_vr)

    Accelerate down the runway under the power of a single engine.
    End at "Vr" or the rotation speed.
    The rotation speed here is defined as 1.2 times the stall speed.

3. Rotate (rotate)

    Operating under the power of a single engine, begin pitching the nose
    up (increasing alpha) while rolling down the runway.
    In this example, the rate of change of alpha is linear over some time.

4. Climb (climb)

    Still operating with one engine, begin climbing to a flight path angle
    of no more than 5 degrees.
    This phase ends when the aircraft is 35 feet above the ground with an airspeed
    of 1.25 x the stall speed.

5. Rejected Takeoff (rto)

    Shut down all engines (zero thrust) and apply brakes (increase of
    runway $\mu$ coefficient to 0.3) until the aircraft airspeed is zero.

    This phase is continuous in time and state with the first phase, and so
    forms a branch off of the nominal trajectory.

Since the RTO phase and the climb phase both must end such that they
do not exceed the end of the runway, the final value of range in each
of these two phases must be the same.  We don't know what this value is
until we've solved the problem, so we cannot simply constrain both to
the same fixed value.

Instead, we'll use a trajectory linkage constraint to ensure that `range`
at the end of the RTO phase is equal to `range` and the end of the climb phase.

More information on calculating the balanced field length is available
in section 17.8 of _Aircraft Design: A Conceptual Approach_ by
Daniel Raymer[@raymer2012aircraft].

## The ODE System

In this problem two sets of ordinary differential equations are used:
one for the aircraft motion on the runway, and one for the aircraft motion once airborne.

For simplification, we're going to assume a constant atmospheric model since the aircraft will never exceed 35 feet of altitude.
Also, since the duration of the trajectory is small, we're going to assume that the vehicle fuel burn is negligible

### The Aerodynamic Model

Both sets of equations of motion have common aerodynamic models.

First, the lift coefficient is computed using a model which assumes linearity in lift wrt the angle of attack.

\begin{align}
    C_L &= C_{L0} + \frac{\alpha}{\alpha_{max}} \left(C_{L-max} - C_{L0}\right)
\end{align}

Next, the drag-due-to-lift factor is computed (Equations 12.48 and 12.61 in Raymer[@raymer2012aircraft]).

\begin{align}
    K_{nom} &= \frac{1}{ \pi  AR  e} \\
    b &= \frac{span}{2} \\
    K &= 33 K_{nom} \frac{ \left(\frac{h + h_w}{b} \right) ^{\frac{3}{2}}}{1.0 + 33 \left( \frac{h + h_w}{b}\right) ^{\frac{3}{2}}}
\end{align}

Note the singularity in the equation for $K$ when $h + h_w$ is negative.
This causes this problem to be difficult to solve using a shooting method.
If the optimizer proposes a combination of initial states and a control history that results in altitude dropping significantly below zero, the propagation will fail.

Finally, the lift and drag are computed after computing the dynamic pressure.

\begin{align}
    q &= 0.5 \rho v^2 \\
    L &= q  S  C_L \\
    D &= q S \left( C_{D0} + K C_{L}^2 \right)
\end{align}

### Stall Speed

This model relies on the ratio of the current true airspeed to stall speed ($\frac{v}{v_{stall}}$).
This constraint is used to trigger the beginning of rotation and is used as a boundary constraint at the end of the initial climb.
Stall speed is given by Equation 5.6 in Raymer[@raymer2012aircraft].

\begin{align}
    W &= m g \\
    v_{stall} &= \sqrt{\frac{2 * W}{\rho S C_{L-max}}}
\end{align}

### Runway Equations of Motion

The runway equations of motion are used to integrate range and speed as the vehicle rolls along the runway.

\begin{align}
  F_r &= mg - L \cos \alpha - T \sin \alpha \\
  \dot{v} &= \frac{T \cos \alpha - D - F_r \mu_r}{m} \\
  \dot{r} &= v
\end{align}

|State  | Description           |Units        |
|-------|-----------------------|-------------|
|r      |range                  |$m$          |
|v      |true airspeed          |$m s^{-1}$   |

### Flight Equations of Motion

The flight equations of motion include two additional state variables: the flight-path angle ($\gamma$) and altitude ($h$).

\begin{align}
  \dot{v} &= \frac{T}{m} \cos \alpha - \frac{D}{m} - g \sin \gamma \\
  \dot{\gamma} &= \frac{T}{m v} \sin \alpha + \frac{L}{m v} - \frac{g \cos \gamma}{v} \\
  \dot{h} &= v \sin \gamma \\
  \dot{r} &= v \cos \gamma
\end{align}

|State    | Description           |Units       |
|---------|-----------------------|------------|
|v        |true airspeed          |$m s^{-1}$  |
|$\gamma$ |flight path angle      |$rad$       |
|r        |range                  |$m$         |
|h        |altitude               |$m$         |

### Treatment of the angle-of-attack ($\alpha$)

In three of the runway phases (break release to $V1$, $V1$ to $V_r$, and rejected takeoff) $\alpha$ is treated as a fixed static parameter.

In the rotation phase, $\alpha$ is treated as a polynomial control of order 1.
$\alpha$ starts at an initial value of zero and increases at a linear rate until the upward forces on the aircraft due to lift and thrust cancel the downward force due to gravity.

In the climb phase, $\alpha$ is treated as a dynamic control to be chosen by the optimizer.

Phase linkage constraints are used to maintain continuity in $\alpha$ throughout the trajectory.

### Parameters

The following parameters define properties of the aircraft and environment for the problem.

|Parameters  | Description               |Units        |Value                    |
|------------|---------------------------|-------------|-------------------------|
|m           |mass                       |$kg$         | 79015.7909              |
|g           |gravitational acceleration |$m s^{-2}$   | 9.80665                 |
|T           |thrust                     |$N$          | 2 x 120101.98 (nominal) |
|$\mu$       |runway friction coefficient|-            | 0.03 (nominal)          |
|h           |altitude                   |$m$          | 0                       |
|$\rho$      |atmospheric density        |$kg\,m^{3}$  | 1.225                   |
|S           |aerodynamic reference area |$m^2$        | 124.7                   |
|CD0         |zero-lift drag coefficient |-            | 0.03                    |
|AR          |wing aspect ratio          |-            | 9.45                    |
|e           |Oswald's wing efficiency   |-            | 801                     |
|span        |wingspan                   |$m$          | 35.7                    |
|h_w         |height of wing above CoG   |$m$          | 1.0                     |
|CL0         |aerodynamic reference area |-            | 0.5                     |
|CL_max      |aerodynamic reference area |-            | 2.0                     |

## The Optimal Control Problem

The following constraints and objective complete the definition of this optimal control problem.

### Objective

|Name   | Phase     | Location | Description | Minimized or Maximized | Ref  |
|-------|-----------|----------|-------------|------------------------|------|
| r     | rto       | final    | range       | Minimized              | 1000 |

### Nonlinear Boundary Constraints

|Name              | Phase    | Description           | Loc   | Units     | Lower  | Upper | Equals | Ref    |
|------------------|----------|-----------------------|-------|-----------|--------|-------|--------|--------|
| v_over_v_stall   | v1_to_vr | $\frac{v}{v_{stall}}$ | final | -         | 1.2    |       |        | 1.2    |
| v                | rto      | airspeed              | final | $ms^{-1}$ |        |       | 0      | 100    |
| F_r              | rotate   | downforce on gear     | final | $N$       |        |       | 0      | 100000 |
| h                | climb    | altitude              | final | $ft$      |        |       | 35     | 35     |
| gam              | climb    | flight path angle     | final | $rad$     |        |       | 5      | 5      |
| v_over_v_stall   | climb    | $\frac{v}{v_{stall}}$ | final | -         | 1.25   |       |        | 1.25   |

### Nonlinear Path Constraints

|Name              | Phase    | Description         | Units     | Lower  | Upper | Equals | Ref    |
|------------------|----------|---------------------|-----------|--------|-------|--------|--------|
| gam              | climb    | flight path angle   | $rad$     | 0      | 5     |        | 5      |

### Phase Continuity Constraints

| First Phase      | Second Phase      | Variables                   |
|------------------|-------------------|-----------------------------|
| br_to_v1[final]  | v1_to_vr[initial] | $time$, $r$, $v$            |
| vr_to_v1[final]  | rotate[initial]   | $time$, $r$, $v$, $\alpha$  |
| rotate[final]    | climb[initial]    | $time$, $r$, $v$, $\alpha$  |
| br_to_v1[final]  | rto[initial]      | $time$, $r$, $v$            |
| climb[final]     | rto[final]        | $r$                         |

## Source Code

Unlike most other Dymos examples, which use analytic derivatives, the ODE in this case is a single component.
All calculations within the ODE are complex-safe and thus we can use complex-step, in conjunction with
[partial derivative coloring](http://openmdao.org/twodocs/versions/latest/features/experimental/simul_coloring_fd_cs.html),
to automatically compute the derivatives using complex-step with reasonable speed.

Since there is significant commonality between the ODEs for the runway roll and the climb, this implementation
uses a single ODE class with an option `mode` that can be set to either `'runway'` or `'climb'`.
Based on the value of `mode`, the component conditionally changes its inputs and outputs.

### BalancedFieldODEComp

=== "balanced_field_ode.py : BalancedFieldODEComp"
{{ inline_source('dymos.examples.balanced_field.balanced_field_ode.BalancedFieldODEComp',
include_def=True,
include_docstring=True,
indent_level=0)
}}

## Building and running the problem

In the following code we define and solve the optimal control problem.
Note the use of `add_linkage_constraint` to handle the less common phase
linkage condition, where the range must be equal at the end of the `rto` and `climb` phases.

{{ embed_test('dymos.examples.balanced_field.doc.test_doc_balanced_field_length.TestBalancedFieldLengthForDocs.test_balanced_field_length_for_docs') }}

## References

\bibliography
