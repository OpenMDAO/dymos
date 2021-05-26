# Multibranch Trajectory

This example demonstrates the use of a Trajectory to encapsulate a
series of branching phases.

## Overview

For this example, we build a system that contains two components: the
first component represents a battery pack that contains multiple cells
in parallel, and the second component represents a bank of DC electric
motors (also in parallel) driving a gearbox to achieve a desired power
output. The battery cells have a state of charge that decays as current
is drawn from the battery. The open circuit voltage of the battery is a
function of the state of charge. At any point in time, the coupling
between the battery and the motor component is solved with a Newton
solver in the containing group for a line current that satisfies the
equations.

Both the battery and the motor models allow the number of cells and the
number of motors to be modified by setting the _n\_parallel_ option in
their respective options dictionaries. For this model, we start with 3
cells and 3 motors. We will simulate failure of a cell or battery by
setting _n\_parallel_ to 2.

Branching phases are a set of linked phases in a trajectory where the
input ends of multiple phases are connected to the output of a single
phase. This way you can simulate alternative trajectory paths in the
same model. For this example, we will start with a single phase
(_phase0_) that simulates the model for one hour. Three follow-on
phases will be linked to the output of the first phase: _phase1_ will
run as normal, _phase1\_bfail_ will fail one of the battery cells, and
_phase1\_mfail_ will fail a motor. All three of these phases start
where _phase0_ leaves off, so they share the same initial time and
state of charge.

## Battery and Motor models

The models are loosely based on the work done in Chin[@chin2019battery].

=== "batteries.py"
{{ inline_source('dymos.examples.battery_multibranch.batteries',
include_def=True,
include_docstring=True,
indent_level=0)
}}

=== "motors.py"
{{ inline_source('dymos.examples.battery_multibranch.motors',
include_def=True,
include_docstring=True,
indent_level=0)
}}

=== "battery_multibranch_ode.py"
{{ inline_source('dymos.examples.battery_multibranch.battery_multibranch_ode',
include_def=True,
include_docstring=True,
indent_level=0)
}}

## Building and running the problem

{{ embed_test('dymos.examples.battery_multibranch.doc.test_multibranch_trajectory_for_docs.TestBatteryBranchingPhasesForDocs.test_basic') }}

## References

\bibliography
