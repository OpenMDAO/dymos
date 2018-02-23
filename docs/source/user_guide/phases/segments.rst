Segments
--------

All phases in |project| are decomposed into one or more *segments* in time.  These segments
serve the following purposes:

- In implicit phases, on each segment each state variable is modeled as a polynomial segment in nondimensional time

The order of the polynomial segment is given by the phase argument `transcription_order`.
In |project| the minimum support transcription order is 3.

- In all phase types, dynamic controls are modeled as a set of discrete values provided at *all* nodes of the segment.


For implicit phases,
state-time histories within a segment are modelled as a Lagrange polynomial.  Continuity in state
value may be enforced via linear constraints at the segment boundaries (the default behavior) or
by choosing specifying a *compressed* transcription whereby the state value at a segment boundary
is provided as a single value.  In practice, the default uncompressed transcription is often superior
since it yields more sparsity in the partial derivatives of the model.
