Runge-Kutta Shooting Method
---------------------------

Dymos implements a Runge-Kutta method that allows optimal control problems to be solved
via a shooting method.  Perhaps the key feature of the Runge-Kutta method in Dymos is that, unlike
the pseudospectral methods which require the optimizer to satisfy constraints in order for the
trajectory to be physically realizable, the Runge-Kutta method provides a physically realizable
trajectory without the involvement of the optimizer.

Advantages of the Runge-Kutta Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Runge-Kutta transcription is useful for pure analysis of a dynamical system where
no control is involved.  If the user simply wants to track some integrated state over some
given amount of time, the Runge-Kutta transcription (or alternatively a pseudospecral transcription
with solved segments) would be a good choice.


Disdvantages of the Runge-Kutta Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

when solving an optimal control problem with Runge-Kutta transcription, each iteration
of the trajectory is physically realizable.  While this is useful when trying to parse what is
happening from a physical perspective in each iteration, this can be problematic.  The Runge-Kutta
method is effectively a "single shooting" method in which the states are propagated across the entire
duration of the phase.  This can lead the optimizer to get stuck in local optima more easily since
the optimizer may be unable to traverse an infeasible region on the way to a better solution.

Furthermore, demanding physicality at each iteration can lead to numerical issues.  Consider the
single-stage-to-orbit example.  In this problem the vehicle reaches its terminal condition with
just a small fraction of its initial mass remaining.  If the optimizer happens to attempt a time
duration that causes the entirety of the initial mass to be expelled as propellant, the equations
of motion will become *singular* and in all likelihood the optimization will fail.
