# Two-Burn Orbit Raise

This example demonstrates the use of a Trajectory to encapsulate a
three-phase orbit raising maneuver with a burn-coast-burn phase
sequence. This example is based on the problem provided in
Enright[@enright1991optimal].

The dynamics are given by

\begin{align}
  \frac{dr}{dt} &= v_r \\
  \frac{d\theta}{dt} &= \frac{v_\theta}{r} \\
  \frac{dv_r}{dt} &= \frac{v^2_\theta}{r} - \frac{1}{r^2} + a_{thrust} \sin u_1 \\
  \frac{dv_\theta}{dt} &= - \frac{v_r v_\theta}{r} + a_{thrust} \cos u_1 \\
  \frac{da_{thrust}}{dt} &= \frac{a^2_{thrust}}{c} \\
  \frac{d \Delta v}{dt} &= a_{thrust}
\end{align}

The initial conditions are

\begin{align}
  r &= 1 \rm{\,DU} \\
  \theta &= 0 \rm{\,rad} \\
  v_r &= 0 \rm{\,DU/TU}\\
  v_\theta &= 1 \rm{\,DU/TU}\\
  a_{thrust} &= 0.1 \rm{\,DU/TU^2}\\
  \Delta v &= 0 \rm{\,DU/TU}
\end{align}

and the final conditions are

\begin{align}
  r &= 3 \rm{\,DU} \\
  \theta &= \rm{free} \\
  v_r &= 0 \rm{\,DU/TU}\\
  v_\theta &= \sqrt{\frac{1}{3}} \rm{\,DU/TU}\\
  a_{thrust} &= \rm{free}\\
  \Delta v &= \rm{free}
\end{align}

## Building and running the problem

The following code instantiates our problem, our trajectory, three
phases, and links them accordingly. The spacecraft initial position,
velocity, and acceleration magnitude are fixed. The objective is to
minimize the delta-V needed to raise the spacecraft into a circular
orbit at 3 Earth radii.

Note the call to _link\_phases_ which provides time,
position, velocity, and delta-V continuity across all phases, and
acceleration continuity between the first and second burn phases.
Acceleration is 0 during the coast phase. Alternatively, we could have
specified a different ODE for the coast phase, as in the example.

This example runs inconsistently with SLSQP but is solved handily by
SNOPT.

{{ embed_test('dymos.examples.finite_burn_orbit_raise.doc.test_doc_finite_burn_orbit_raise.TestDocFiniteBurnOrbitRaise.test_doc_finite_burn_orbit_raise') }}

## References

\bibliography
