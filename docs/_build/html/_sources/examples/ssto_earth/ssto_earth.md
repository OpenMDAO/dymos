# SSTO Earth Launch

This example is based on the _Time-Optimal Launch of a Titan II_
example given in Appendix B of Longuski[@longuski2014optimal].
It finds the pitch profile for a single-stage-to-orbit launch vehicle that minimizes the time
required to reach orbit insertion under constant thrust.

![The free-body-diagram of the single-stage-to-orbit problem.](ssto_fbd.png)

The vehicle dynamics are given by

\begin{align}
  \frac{dx}{dt} &= v_x \\
  \frac{dy}{dt} &= v_y \\
  \frac{dv_x}{dt} &= \frac{1}{m} (T \cos \theta - D \cos \gamma) \\
  \frac{dv_y}{dt} &= \frac{1}{m} (T \sin \theta - D \sin \gamma) - g \\
  \frac{dm}{dt} &= \frac{T}{g I_{sp}}
\end{align}

The initial conditions are

\begin{align}
  x_0 &= 0 \\
  y_0 &= 0 \\
  v_{x0} &= 0 \\
  v_{y0} &= 0 \\
  m_0 &= 117000 \rm{\,kg}
\end{align}

and the final conditions are

\begin{align}
  x_f &= \rm{free} \\
  y_f &= 185 \rm{\,km} \\
  v_{xf} &= V_{circ} \\
  v_{yf} &= 0 \\
  m_f &= \rm{free}
\end{align}

## Defining the ODE

Generally, one could define the ODE system as a composite group of multile components.
The atmosphere component computes density ($\rho$).
The eom component computes the state rates.
Decomposing the ODE into smaller calculations makes it easier to derive the analytic derivatives.

![The notional XDSM diagram for the ODE system in the SSTO problem.](ssto_xdsm.png)

However, for this example we will demonstrate the use of complex-step differentiation and define the ODE as a single component.
This saves time up front in the deevlopment of the ODE at a minor cost in execution time.

The unconnected inputs to the EOM at the top of the diagram are provided by the Dymos phase as states, controls, or time values.
The outputs, including the state rates, are shown on the right side of the diagram.
The Dymos phases use state rate values to ensure that the integration technique satisfies the dynamics of the system.

=== "launch_vehicle_ode.py"
{{ inline_source('dymos.examples.ssto.launch_vehicle_ode',
include_def=True,
include_docstring=True,
indent_level=0)
}}

## Solving the problem

{{ embed_test('dymos.examples.ssto.doc.test_doc_ssto_earth.TestDocSSTOEarth.test_doc_ssto_earth') }}

## References

\bibliography
