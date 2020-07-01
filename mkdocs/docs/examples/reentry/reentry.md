# Single-Phase Space Shuttle Reentry

The problem of the space shuttle reentering Earth's atmosphere is an
optimal control problem governed by six equations of motion and limited
by the aerodynamic heating rate. For a detailed layout of this problem
and other optimal control problems see Betts[@betts2010practical].
The governing equations of motion for this problem are:

\begin{align}
  \frac{dh}{dt} &= v \sin \gamma \\
  \frac{d\phi}{dt} &= \frac{v}{r} \cos \gamma \frac{\sin \psi}{\cos \theta} \\
  \frac{d\theta}{dt} &= \frac{v}{r} \cos \gamma \cos \psi  \\
  \frac{dv}{dt} &= - \frac{D}{m} - g \sin \gamma \\
  \frac{d\gamma}{dt} &= \frac{L}{mv} \cos \beta + \cos \gamma (\frac{v}{r} - \frac{g}{v}) \\
  \frac{d\psi}{dt} &= \frac{L \sin \beta}{mv \cos \gamma} + \frac{v}{r \cos \theta} \cos \gamma \sin \psi \sin \theta
\end{align}

where $v$ $[ft/s]$ is airspeed, $\gamma$ $[rad]$ is flight path angle,
$r$ $[ft]$ is distance from the center of the Earth, $\psi$ $[rad]$ is
azimuth, $\theta$ $[rad]$ is latitude, $D$ $[lb]$ is drag, $m$ $[sl]$ is
mass, $g$ $[\frac{ft}{s^2}]$ is the local gravitational acceleration, $L$
$[lb]$ is lift, $\beta$ $[rad]$ is bank angle, $h$ $[ft]$ is altitude,
and $\phi$ $[rad]$ is longitude. Mass is considered to be a constant for
this case, because the model spans the time from when the shuttle begins
reentry to the time right before the shuttle starts its engines. The
engines are not actually running at any time during the model, so there
is no thrust and thus no mass lost. The goal is to maximize the
crossrange (latitude) that the shuttle can cover before reaching the
final altitude, without exceding a maximum heat rate at the leading
edges. This heat rate is constrained by $q \leq 70$ where q
$[\frac{btu}{ft^2s}]$ is the heating rate.

The initial conditions are

\begin{align}
  h_0 &= 26000 \\
  v_0 &= 25600 \\
  \phi_0 &= 0 \\
  \gamma_0 &= -0.01745 \\
  \theta_0 &= 0 \\
  \psi_0 &= \frac{\pi}{2}
\end{align}

and the final conditions are

\begin{align}
  h_0 &= 80000 \\
  v_0 &= 2500 \\
  \gamma_0 &= -0.08727 \\
  \theta &= \rm{free} \\
  \psi &= \rm{free}
\end{align}

Notice that no final condition appears for $\phi$. This is because none
of the equations of motion actually depend on $\phi$, and as a result,
while $\phi$ exists in the dymos model (last code block below) as a
state variable, it does not exist as either an input or output in the
ode (ShuttleODE group, second to last code block below).

This model uses four explicit OpenMDAO components. The first component
computes the local atmospheric condition at the shuttle's altitude. The
second component computes the aerodynamic forces of lift and drag on the
shuttle. The third component is where the heating rate on the leading
edge of the shuttles wings is computed. The heating rate is given by
$q = q_a q_r$ where

\begin{align}
  q_a &= c_0 + c_1\alpha + c_2 \alpha^2 + c_3 \alpha^3
\end{align}

and

\begin{align}
  q_r &= 17700 \rho^.5 (.0001v)^{3.07}
\end{align}

where $c_0, c_1, c_2,$ and $c_3$ are constants, $\alpha$ $[deg]$ is the
angle of attack, $\rho$ $[\frac{sl}{ft^3}]$ is local atmospheric density,
and $v$ $[\frac{ft}{s}]$ is velocity. The final component is where the
equations of motion are implemented. These four components are put
together in the ShuttleODE group, which is the top level ode that the
dymos model sees.

## Component Models

Below is the code for the atmospheric component:

=== "atmosphere_comp.py"
{{ inline_source('dymos.examples.shuttle_reentry.atmosphere_comp',
include_def=True,
include_docstring=True,
indent_level=0)
}}

Below is the code for the aerodynamic component:

=== "aerodynamics_comp.py"
{{ inline_source('dymos.examples.shuttle_reentry.aerodynamics_comp',
include_def=True,
include_docstring=True,
indent_level=0)
}}

Below is the code for the heating component:

=== "heating_comp.py"
{{ inline_source('dymos.examples.shuttle_reentry.heating_comp',
include_def=True,
include_docstring=True,
indent_level=0)
}}

Below is the code for the component containing the equations of motion:

=== "flight_dynamics_comp.py"
{{ inline_source('dymos.examples.shuttle_reentry.flight_dynamics_comp',
include_def=True,
include_docstring=True,
indent_level=0)
}}

## Defining the ODE

Below is the code for the top level ode group that will be fed to dymos:

=== "shuttle_ode.py"
{{ inline_source('dymos.examples.shuttle_reentry.shuttle_ode',
include_def=True,
include_docstring=True,
indent_level=0)
}}

## Building and running the problem

The following code is the dymos implementation of the model. As the code
shows, there are six states, two controls, and one constraint in the
model. The states are $h, v, \phi, \gamma, \theta,$ and $\psi$.
The two controls are $\alpha$ and $\beta$, and the constraint is $q$.

{{ embed_test('dymos.examples.shuttle_reentry.doc.test_doc_reentry.TestReentryForDocs.test_reentry') }}

## References

\bibliography
