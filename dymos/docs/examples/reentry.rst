==================================
Single-Phase Space Shuttle Reentry
==================================

The equations of motion for the reentry of the space shuttle into Earth's atmosphere are 
limited by the heating rate on the leading edges of the shuttle's wings. There are six 
governing equations of motion:

.. math ::
    \frac{dh}{dt} = v * \sin{\gamma}
    \frac{d{\phi}}{dt} = \frac{v}{r} * \cos{\gamma} * \frac{sin{\psi}}{cos{\theta}}
    \frac{d{\theta}}{dt} = \frac{v}{r} * \cos{\gamma} * \cos{\psi}
    \frac{dv}{dt} = -\frac{D}{m} - g * \sin{\gamma}
    \frac{d{\gamma}}{dt} = \frac{L}{m*v} * \cos{\beta} + \cos{\gamma} * (\frac{v}{r} - \frac{g}{v})
    \frac{d{\psi}}{dt} = \frac{L * \sin{\beta}}{m*v*\cos{\gamma}} + \frac{v}{r*\cos{\theta}} * \cos{\gamma} * \sin{\psi} * \sin{\theta}

where :math:`v` is airspeed in ft/s,  :math:`\gamma` is flight path angle in radians, :math:`r` 
is distance from the center of the Earth, :math:`\psi` is azimuthal angle in radians, 
:math:`theta` is latitude in radians, :math:`D` is drag in lb, :math:`m` is mass in sl, :math:`g`
is the local gravitational acceleration in ft/s/s, :math:`L` is lift in lb, :math:`\beta` 
is bank angle in radians, :math:`h` is altitude in ft, and :math:`\phi` is longitude in 
radians. Mass is considered to be a constant because the model encompasses the time from
when the space shuttle begins reentry to the time right before the space shuttle starts its 
engines, so no mass is consumed during the model's timespan. The goal is to maximize the 
crossrange (latitude) while constraining the heat at the leading edges, subject to the 
heating constraint

.. math ::
    q <= 70

where q is the heating rate in Btu/ft/ft/s. The initial conditions are

.. math ::
    h_0 = 260000
    v_0 = 25600
    {\phi}_0 = 0
    {\gamma}_0 = -.01745
    {\theta}_0 = 0
    {\psi}_0 = \frac{\pi}{2}

and the final conditions are

.. math ::
    h_0 = 80000
    v_0 = 2500
    {\gamma}_0 = -.08727
    {\theta} = free
    {\psi} = free
more info here
There are additional atmospheric, aerocynamic, and heating components that calculate local
density, lift, drag, and heating rate. The following component computes the flight dynamics
whose equations are listed above.

.. embed-code::
    examples.shuttle_reentry.flight_dynamics_comp
    :layout: code

more info here
The following code generates the ode required by dymos in order to solve the optimization
problem.

.. embed-code::
    examples.shuttle_reentry.shuttle_ode
    :layout: code

more info here
The following code implements the dymos problem to solve the optimization problem.

.. embed-code:
    examples.shuttle_reentry.doc.test_doc_reentry.TestReentryForDocs.test_doc_reentry
    :layout: code, output, plot

more info here

