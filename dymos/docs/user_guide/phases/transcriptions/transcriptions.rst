==============
Transcriptions
==============

|project| implements several different *transcriptions* which are used to convert a continuous
optimal control problem into an analogous nonlinear programming (NLP) problem.

|project| implements multi-phase trajectory optimization.  That is to say, in each phase
the state and time histories are continuous.  At the beginning or end of each phase,
a discrete jump in the value of a state variable is permitted.  This comes in useful when
modeling things like stage or store (mass) jettisons or impulsive changes in velocity.
By linking phases together with continuity constraints, a trajectory or series of trajectories
can be assembled.

Different transcription methods are supported via different Phase types in |project|.  Currently,
the software supports the following techniques:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    gauss-lobatto
    radau-pseudospectral
