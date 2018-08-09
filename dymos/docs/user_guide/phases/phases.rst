======================
Phases of a Trajectory
======================

|project| uses the concept of *phases* to support intermediate boundary constraints and path constraints on variables in the system.
Each phase represents the trajectory of a dynamical system, and may be subject to different equations of motion, force models, and constraints.
Multiple phases may be assembled to form one or more trajectories by enforcing compatibility constraints between them.

For implicit and explicit phases, the equations-of-motion or process equations are defined via an ordinary differential equation.

An ODE is of the form

.. math ::
  \frac{\partial \textbf x}{\partial t} = \textbf f(t, \textbf x, \textbf u)

where
:math:`\textbf x` is the vector of *state variables* (the variable being integrated),
:math:`t` is *time* (or *time-like*),
:math:`\textbf u` is the vector of *parameters* (an input to the ODE),
and
:math:`\textbf f` is the *ODE function*.

|project| can treat the parameters :math:`\textbf u` as either **design parameters** or dynamic **controls**.
In addition, |project| automatically calculates the first and second time-derivatives of the controls.
These derivatives can then be utilized as via constraints or as additional parameters to the ODE.
Subsequently, the optimal control problem as solved by |project| can be expressed as:

.. math::

  \textrm{Minimize}:& \quad J = \textbf f_{obj}(t, \textbf x, \textbf u, \dot{\textbf u}, \ddot{\textbf u}) \\
  \textrm{subject to:}& \\
  &\textrm{system dynamics} \quad &\frac{\partial \textbf x}{\partial t} &= \textbf f_{ode}(t, \textbf x, \textbf u, \dot{\textbf u}, \ddot{\textbf u}) \\
  &\textrm{initial time bounds} \quad &t_{0,lb} &\,\le\, t_0 \,\le\, t_{0,ub} \\
  &\textrm{elapsed time bounds} \quad &t_{p,lb} &\,\le\, t_p \,\le\, t_{p,ub} \\
  &\textrm{state bounds} \quad &\textbf x_{lb} &\,\le\, \textbf x \,\le\, \textbf x_{ub} \\
  &\textrm{control bounds} \quad &\textbf u_{lb} &\,\le\, \textbf u \,\le\, \textbf u_{ub} \\
  &\textrm{nonlinear boundary constraints} \quad &\textbf g_{b,lb} &\,\le\, \textbf g_{b}(t, \textbf x, \textbf u, \dot{\textbf u}, \ddot{\textbf u}) \,\le\, \textbf g_{b,ub} \\
  &\textrm{nonlinear path constraints} \quad &\textbf g_{p,lb} &\,\le\, \textbf g_{p}(t, \textbf x, \textbf u, \dot{\textbf u}, \ddot{\textbf u}) \,\le\, \textbf g_{p,ub} \\

The ability to utilize control derivatives in the equations of motion provides some unique capabilities, namely the ability to
easily solve problems using *differential inclusion*, which will be demonstrated in the examples.

The solution techniques used by the Phase classes in |project| generally fall into two categories:
implicit and explicit phases.  They differ in underlying details but both allow for the same
general form of the optimal control problem.

Implicit Phases
---------------
Implicit phases in |project| use collocation techniques to find a state-time history that satisfies
the ordinary differential equation for each state.  If the given problem is an initial value or
well-posed boundary value problem, then there is a single unique solution.

Implicit phases are so-called because the state-time history of the trajectory is provided a priori
at each iteration by the optimizer.  In |project|, implicit phases are subdivided into a series
of one or more *segments*.  On each segment, the time-history of each state variable is represented
as a polynomial.  The number of segments and the order of the state polynomial on each segment
define a series of *nodes* in dimensionless time at which the state and control values are
specified.  This distribution of segments/nodes is often referred to as the *Grid* (or in other
places the *mesh*) of the phase.

Since the state-time histories are assumed to be continuous the slopes of those polynomials
in time are compared with the outputs of the *ODE* at a subset of the nodes called the
collocation nodes. We call the difference between these two quantities the differential *defects*,
or more commonly just *defects* (:math:`\Delta`).

.. math::

  \Delta = \frac{dx}{dt} - \textbf f_{ode}(t, \textbf x, \textbf u, \dot{\textbf u}, \ddot{\textbf u}) \\


.. toctree::
    :maxdepth: 2
    :titlesonly:

    segments
    variables
    constraints
    objective
    transcriptions/transcriptions
    simulate/simulate
