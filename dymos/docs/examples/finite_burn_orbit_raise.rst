====================
Two-Burn Orbit Raise
====================

This example demonstrates the use of a Trajectory to encapsulate a three-phase orbit raising
maneuver with a burn-coast-burn phase sequence.  This example is based on the problem provided
in [EnrightConway91]_.

The dynamics are given by

.. math ::
    \frac{d r}{d t} &= v_r \\
    \frac{d \theta}{d t} &= \frac{v_{\theta}}{r} \\
    \frac{d v_r}{d t} &= \frac{v_{\theta}^2}{r} - \frac{1}{r^2} + a_{thrust} \sin{u_1} \\
    \frac{d v_{\theta}}{d t} &= -\frac{v_r v_{\theta}}{r} + a_{thrust} \cos{u_1} \\
    \frac{d a_{thrust}}{d t} &= \frac{a_{thrust}^2}{c} \\
    \frac{d \Delta v}{d t} &= a_{thrust}

The initial conditions are

.. math ::
    r &= 1 \, \mathrm{DU} \\
    \theta &= 0 \, \mathrm{rad} \\
    v_{r} &= 0 \, \mathrm{DU/TU} \\
    v_{\theta} &= 1 \, \mathrm{DU/TU} \\
    a_{thrust} &= 0.1 \, \mathrm{DU/TU^2} \\
    \Delta v &= 0 \, \mathrm{DU/TU}

and the final conditions are

.. math ::
    r &= 3 \, \mathrm{DU} \\
    \theta &= \, \mathrm{free} \\
    v_r &= 0 \, \mathrm{DU/TU} \\
    v_{\theta} &= \sqrt{\frac{1}{3}} \, \mathrm{DU/TU} \\
    a_{thrust} &= \mathrm{free} \\
    \Delta v &= \mathrm{free}

Building and running the problem
--------------------------------

The following code instantiates our problem, our trajectory, three phases, and links them
accordingly.  The spacecraft initial position, velocity, and acceleration magnitude are fixed.
The objective is to minimize the delta-V needed to raise the spacecraft into a circular orbit
at 3 Earth radii.

Note the call to `link_phases` which provides time, position, velocity, and delta-V continuity
across all phases, but acceleration continuity between the first and second burn phases.
Acceleration is 0 during the coast phase.  Alternatively, we could have specified a different
ODE for the coast phase, as in the example.

This example runs inconsistently with SLSQP but is solved handily by SNOPT.

.. embed-code::
    dymos.examples.finite_burn_orbit_raise.doc.test_doc_finite_burn_orbit_raise.TestFiniteBurnOrbitRaise.test_finite_burn_orbit_raise
    :layout: code, output, plot

References
----------
.. [EnrightConway91] Enright, Paul J, and Bruce A Conway. “Optimal Finite-Thrust Spacecraft Trajectories Using Collocation and Nonlinear Programming.” Journal of Guidance, Control, and Dynamics 14.5 (1991): 981–985.
