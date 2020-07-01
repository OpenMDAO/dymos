# Hyper-Sensitive Problem

This example is based on the Hyper-Sensitive problem given in
Patterson[@patterson2015ph]. In this problem we seek to minimize both
the distance traveled when moving between fixed boundary conditions and
also to minimize the control $u$ used. The cost function to be minimized is:

\begin{align}
    J &= \frac{1}{2} \int_{0}^{t_f} (x^2 + u^2) dt
\end{align}

The system is subject to the dynamic constraints:

\begin{align}
    \frac{dx}{dt} &= -x + u
\end{align}

The boundary conditions are:

\begin{align}
    x(t_0) &= 1.5 \\
    x(t_f) &= 1
\end{align}

The control $u$ is unconstrained while the final time $t_f$ is fixed.

Due to the nature of dynamics, for sufficiently large values of $t_f$,
the problem exhibits a _dive_, _cruise_, and _resurface_ type
structure, where the all interesting behavior occurs at the beginning and
end while remaining relatively constant in the middle.

This problem has a known analytic optimal solution:

\begin{align}
    x^*(t) &= c_1 e^{\sqrt{2} t} + c_2 e^{-\sqrt{2} t} \\
      u^*(t) &= \dot{x}^*(t) + x^*(t)
\end{align}

where:

\begin{align}
    c_1 &= \frac{1.5 e^{-\sqrt{2} t_f} - 1}{e^{-\sqrt{2} t_f} - e^{\sqrt{2} t_f}} \\
    c_2 &= \frac{1 - 1.5 e^{\sqrt{2} t_f}}{e^{-\sqrt{2} t_f} - e^{\sqrt{2} t_f}}
\end{align}

## The ODE System: hyper\_sensitive\_ode.py

=== "hyper_sensitive_ode.py"
{{ inline_source('dymos.examples.hyper_sensitive.hyper_sensitive_ode',
include_def=True,
include_docstring=True,
indent_level=0)
}}

## Building and running the problem

The following code shows the procedure for solving the problem

{{ embed_test('dymos.examples.hyper_sensitive.doc.test_doc_hyper_sensitive.TestHyperSensitive.test_hyper_sensitive_for_docs') }}

## References

\bibliography
