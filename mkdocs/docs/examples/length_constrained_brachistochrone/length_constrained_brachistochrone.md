# The Length-Constrained Brachistochrone

!!! info "Things you'll learn through this example"
    - How to connect the outputs from a trajectory to a downstream system.

This is a modified take on the brachistochrone problem.
In this instance, we assume that the quantity of wire available is limited.
Now, we seek to find the minimum time brachistochrone trajectory subject to a upper-limit on the arclength of the wire.

The most efficient way to approach this problem would be to treat the arc-length $S$ as an integrated state variable.
In this case, as is often the case in real-world MDO analyses, the implementation of our arc-length function is not integrated into our pseudospectral approach.
Rather than rewrite an analysis tool to accommodate the pseudospectral approach, the arc-length analysis simply takes the result of the trajectory in its entirety and computes the arc-length constraint via the trapezoidal rule:\

\begin{align}
    S &= \frac{1}{2} \left( \sum_{i=1}^{N-1} \sqrt{1 + \frac{1}{\tan{\theta_{i-1}}}} + \sqrt{1 + \frac{1}{\tan{\theta_{i}}}} \right) \left(x_{i-1} - x_i \right)
\end{align}

The OpenMDAO component used to compute the arclength is defined as follows:

=== "arc_length_comp.py"
{{ inline_source('dymos.examples.length_constrained_brachistochrone.arc_length_comp',
include_def=True,  
include_docstring=True,
indent_level=0)
}}

!!! note
  In this example, the number of nodes used to compute the arclength is needed when building the problem.
  The transcription object is initialized and its attribute `grid_data.num_nodes` is used to provide the number of total nodes (the number of points in the timeseries) to the downstream arc length calculation.

{{ embed_test('dymos.examples.length_constrained_brachistochrone.doc.test_doc_length_constrained_brachistochrone.TestLengthConstrainedBrachistochrone.test_length_constrained_brachistochrone') }}
