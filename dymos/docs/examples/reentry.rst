==================================
Single-Phase Space Shuttle Reentry
==================================

The problem of the space shuttle reentering Earth's atmosphere is an optimal control problem governed by six equations of 
motion and limited by the aerodynamic heating rate. For a detailed layout of this problem and other optimal control problems see
[BettsNLP2010]_. The governing equations of motion for this problem are:

.. math ::
    \frac{dh}{dt} &= v\sin{\gamma} \\
    \frac{d{\phi}}{dt} &= \frac{v}{r}\cos{\gamma}\frac{sin{\psi}}{cos{\theta}} \\
    \frac{d{\theta}}{dt} &= \frac{v}{r}\cos{\gamma}\cos{\psi} \\
    \frac{dv}{dt} &= -\frac{D}{m} - g\sin{\gamma} \\
    \frac{d{\gamma}}{dt} &= \frac{L}{mv}\cos{\beta} + \cos{\gamma}(\frac{v}{r} - \frac{g}{v}) \\
    \frac{d{\psi}}{dt} &= \frac{L\sin{\beta}}{mv\cos{\gamma}} + \frac{v}{r\cos{\theta}}\cos{\gamma}\sin{\psi}\sin{\theta} \\

where :math:`v` :math:`[ft/s]` is airspeed,  :math:`\gamma` :math:`[rad]` is flight path angle, :math:`r` :math:`[ft]`
is distance from the center of the Earth, :math:`\psi` :math:`[rad]` is azimuth, 
:math:`\theta` :math:`[rad]` is latitude, :math:`D` :math:`[lb]` is drag, :math:`m` :math:`[sl]` is mass, :math:`g` :math:`[{ft}/{{s}^2}]`
is the local gravitational acceleration, :math:`L` :math:`[lb]` is lift, :math:`\beta` :math:`[rad]` 
is bank angle, :math:`h` :math:`[ft]` is altitude, and :math:`\phi` :math:`[rad]` is longitude. Mass is considered to be a constant for this 
case, because the model spans the time from when the shuttle begins reentry to the time right before the shuttle starts its 
engines. The engines are not actually running at any time during the model, so there is no thrust and thus no mass lost.
The goal is to maximize the crossrange (latitude) that the shuttle can cover before reaching the final altitude, without exceding a maximum heat rate at the leading edges. 
This heat rate is constrained by :math:`q \leq 70` where q :math:`[{btu}/{{ft}^2s}]` is the heating rate. The initial conditions are

.. math ::
    h_0 &= 260000 \\
    v_0 &= 25600 \\
    {\phi}_0 &= 0 \\
    {\gamma}_0 &= -.01745 \\
    {\theta}_0 &= 0 \\
    {\psi}_0 &= \frac{\pi}{2} \\

and the final conditions are

.. math ::
    h_0 &= 80000 \\
    v_0 &= 2500 \\
    {\gamma}_0 &= -.08727 \\
    {\theta} &= free \\
    {\psi} &= free \\

Notice that no final condition appears for :math:`\phi`. This is because none of the equations of motion actually depend on :math:`\phi`, and as a result, while
:math:`\phi` exists in the dymos model (last code block below) as a state variable, it does not exist as either an input or output in the ode 
(ShuttleODE group, second to last code block below).

This model uses four explicit OpenMDAO components. The first component computes the local atmospheric condition at the shuttle's altitude. 
The second component computes the aerodynamic forces of lift and drag on the shuttle. The third component is where the heating rate on the leading edge of
the shuttle's wings is computed. The heating rate is given by :math:`q = q_aq_r` where 

.. math::
    q_a &= c_0 + c_1\alpha + c_2{\alpha}^2 + c_3{\alpha}^3 \\

and

.. math::
    q_r &= 17700{\rho}^.5{(.0001v)}^{3.07} \\

where :math:`c_0, c_1, c_2,` and :math:`c_3` are constants, :math:`\alpha` :math:`[deg]` is the angle of attack,
:math:`\rho` :math:`[{sl}/{{ft}^3}]` is local atmospheric density, and :math:`v` :math:`[{ft}/{s}]` is velocity. The final component is where the equations of 
motion are implemented. These four components are put together in the ShuttleODE group, which is the top level ode that the dymos model sees.

Below is the code for the atmospheric component:

.. embed-code::
    examples.shuttle_reentry.atmosphere_comp
    :layout: code

Below is the code for the aerodynamic component:

.. embed-code::
    examples.shuttle_reentry.aerodynamics_comp
    :layout: code

Below is the code for the heating component:

.. embed-code::
    examples.shuttle_reentry.heating_comp
    :layout: code

Below is the code for the component containing the equations of motion:

.. embed-code::
    examples.shuttle_reentry.flight_dynamics_comp
    :layout: code

Below is the code for the top level ode group that will be fed to dymos:

.. embed-code::
    examples.shuttle_reentry.shuttle_ode
    :layout: code

The following code is the dymos implementation of the model. As the code shows, there are six states, two controls, and one constraint in the model. The states are :math:`h, v, 
\phi, \gamma, \theta,` and :math:`\psi`. The two controls are :math:`\alpha` and :math:`\beta`, and the constraint is :math:`q`.

.. embed-code::
    dymos.examples.shuttle_reentry.doc.test_doc_reentry.TestReentryForDocs.test_reentry
    :layout: code, output, plot

References
----------
.. [BettsNLP2010] Betts, John. *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming.* 2nd ed., Society for Industrial and Applied Mathematics, 2010.
