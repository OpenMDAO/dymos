# The Brachistochrone with Externally-Sourced Controls

!!! info "Things you'll learn through this example"
    - How to provide trajectory control values from an external source.

This example is the same as the other brachistochrone example with one exception:  the control values come from an external source upstream of the trajectory.

The following script fully defines the brachistochrone problem with Dymos and solves it.
A new `IndepVarComp` is added before the trajectory.
The transcription used in the relevant phase is defined first so that we can obtain the number of control input nodes.
The IndepVarComp then provides the control $\theta$ at the correct number of nodes, and sends them to the trajectory.
Since the control values are no longer managed by Dymos, they are added as design variables using the OpenMDAO `add_design_var` method.

{{ embed_test('dymos.examples.brachistochrone.doc.test_doc_brachistochrone_upstream_control.TestBrachistochroneUpstreamControl.test_brachistochrone_upstream_control') }}
