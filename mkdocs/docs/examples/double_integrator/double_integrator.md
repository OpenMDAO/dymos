# Double Integrator

In the double integrator problem, we seek to maximize the distance
traveled by a block (that starts and ends at rest) sliding without
friction along a horizontal surface, with acceleration as the control.

We minimize the final time, $t_f$, by varying the dynamic control,
$u$, subject to the dynamics:

\begin{align}
  \frac{dx}{dt} &= v \\
  \frac{dv}{dt} &= u
\end{align}

The initial conditions are

\begin{align}
  x_0 &= 0 \\
  v_0 &= 0
\end{align}

and the final conditions are

\begin{align}
  x_f &= \rm{free} \\
  v_f &= 0
\end{align}

The control $u$ is constrained to fall between -1 and 1. Due to the fact
that the control appears linearly in the equations of motion, we should
expect _bang-bang_ behavior in the control (alternation between its extreme values).

## The ODE System: double\_integrator\_ode.py

This problem is unique in that we do not actually have to calculate
anything in the Dymos formulation of the ODE. We create an
_ExplicitComponent_ and provide it with the _num\_nodes_
option, but it has no inputs and no outputs. The rates for the states
are entirely provided by the other states and controls.

=== "double_integrator_ode.py"
{{ inline_source('dymos.examples.double_integrator.double_integrator_ode',
include_def=True,
include_docstring=True,
indent_level=0)
}}

## Building and running the problem

In order to facilitate the bang-bang behavior in the control, we disable
continuity and rate continuity in the control value.

{{ embed_test('dymos.examples.double_integrator.doc.test_doc_double_integrator.TestDoubleIntegratorForDocs.test_double_integrator_for_docs') }}
