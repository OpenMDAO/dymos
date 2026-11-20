# Optimal Control


Optimal control implies the optimization of a dynamical system.  Typically this takes the form
of a trajectory in which the *states* of the system evolve with time.  The evolution of the states
$\left(\bar{x}\right)$ are typically governed by an ordinary differential equation (ODE) or
a differential algebraic equation (DAE).  In Dymos we characterize all dynamics as an (ODE),
although solving systems of DAEs is also possible.

\begin{align}
  \dot{\bar{x}} = f_{ode}(\bar{x},t,\bar{u},\bar{d})
\end{align}

# Controls and Parameters

In order to impact the behavior of such systems we need *controls*.
Controls may be allowed to vary with time, such as the angle-of-attack of an aircraft during flight.
We refer to these as _dynamic_ controls $\left(\bar{u}\right)$.
Other controllable parameters might be fixed with time, such as the wingspan of an aircraft.
We refer to these as _parameters_ $\left(\bar{d}\right)$, although in the literature they may also be referred to as static controls.
The endpoints in time, state values, control values, and design parameter values define the independent variables for our optimization problem.
In Dymos, we discretize these variables in time to convert a continuous-time optimal control problem into a nonlinear programming (NLP) problem.

# Constraints

When optimizing problems there will always be constraints on the system.
Some of these constraints can be characterized as bounds on our independent variables.
For instance, fixing the initial conditions of a trajectory can be accomplished by bounding the initial states and time to the desired value.

\begin{align}
  \mathrm{Time:}& \qquad {t}_{lb} \leq t \leq {t}_{ub} \\
  \mathrm{State \, Variables:}& \qquad \bar{x}_{lb} \leq \bar{x} \leq \bar{x}_{ub} \\
  \mathrm{Dynamic \, Controls:}& \qquad \bar{u}_{lb} \leq \bar{u} \leq \bar{u}_{ub} \\
  \mathrm{Design \, Parameters:}& \qquad \bar{d}_{lb} \leq \bar{d} \leq \bar{d}_{ub} \\
\end{align}

Other times we may want to constrain an output of our system at a specific point along the trajectory (boundary constraints) or along the entire trajectory (path constraints).
These are so-called nonlinear constraints because they *may* be nonlinear functions of our independent variables.

\begin{align}
\mathrm{Initial \, Boundary \, Constraints:}& \qquad \bar{g}_{0,lb} \leq g_{0}(\bar{x}_0,t_0,\bar{u}_0, \bar{d}) \leq \bar{g}_{0,ub} \\
\mathrm{Final \, Boundary \, Constraints:}& \qquad \bar{g}_{f,lb} \leq g_{f}(\bar{x}_f,t_f,\bar{u}_f, \bar{d}) \leq \bar{g}_{f,ub} \\
\mathrm{Path \, Constraints:}& \qquad \bar{p}_{f,lb} \leq p_{f}(\bar{x},t,\bar{u},\bar{d}) \leq \bar{p}_{f,ub} \\
\end{align}

In practice, bound constraints typically govern the limits which the optimizer must observe when providing values of our independent variables.  
Nonlinear constraints, on the other hand, are constraints applied to the outputs of a system.  
To put another way, bound constraints are always observed, while nonlinear constraints are not necessarily observed until the optimizer has finished solving the optimization problem.

# The Objective

Dymos may be used to both simulate and optimize dynamical systems.
The phase construct is generally used in optimization contexts.
Within each phase, the user can set the objective:

\begin{align*}
  \mathrm{J} = f_{obj}(\bar{x},t,\bar{u},\bar{d})
\end{align*}

As with constraints, the objective may be any output within the Phase.  Phases can also be
incorporated into larger models wherein the objective is defined in some subsystem outside of the
phase.  In this case, the standard OpenMDAO method `add_objective` can be used.

# The Overall Optimization Problem

The optimization problem as defined by Dymos can thus be stated as:

\begin{align*}
    \mathrm{Minimize}& \qquad \mathrm{J} = f_{obj}(\bar{x},t,\bar{u},\bar{d}) \\
    \mathrm{Subject \, to:}& \\
    \mathrm{Dynamic \, Constraints:}& \qquad \dot{\bar{x}} = f_{ode}(\bar{x},t,\bar{u},\bar{d}) \\
    \mathrm{Time:}& \qquad {t}_{lb} \leq t \leq {t}_{ub} \\
    \mathrm{State \, Variables:}& \qquad \bar{x}_{lb} \leq \bar{x} \leq \bar{x}_{ub} \\
    \mathrm{Dynamic \, Controls:}& \qquad \bar{u}_{lb} \leq \bar{u} \leq \bar{u}_{ub} \\
    \mathrm{Design \, Parameters:}& \qquad \bar{d}_{lb} \leq \bar{d} \leq \bar{d}_{ub} \\
    \mathrm{Initial \, Boundary \, Constraints:}& \qquad \bar{g}_{0,lb} \leq g_{0}(\bar{x}_0,t_0,\bar{u}_0, \bar{d}) \leq \bar{g}_{0,ub} \\
    \mathrm{Final \, Boundary \, Constraints:}& \qquad \bar{g}_{f,lb} \leq g_{f}(\bar{x}_f,t_f,\bar{u}_f, \bar{d}) \leq \bar{g}_{f,ub} \\
    \mathrm{Path \, Constraints:}& \qquad \bar{p}_{f,lb} \leq p_{f}(\bar{x},t,\bar{u},\bar{d}) \leq \bar{p}_{f,ub}
\end{align*}
