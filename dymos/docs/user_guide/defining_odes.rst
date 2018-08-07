=============
Defining ODEs
=============

Optimal control problems contain ordinary differential equations (ODE), or less frequently,
differential algebraic equations (DAE) that dictate the evolution of the state of the system.
Typically this evolution occurs in time, and the ODE represents equations of motion (EOM).  
The equations of motion can define a variety of systems, not just mechanical ones.
In other fields they are sometimes referred to as *process equations* or
*plants*.

.. math::

  \dot{\bar{x}} = f_{ode}(\bar{x},t,\bar{u},\bar{d})


To represent EOM, |project| uses a standard OpenMDAO System (a Group or Component).  This System
takes some set of variables as input and computes outputs that include the time-derivatives of
the state variables :math:`\bar{x}`.

ODE Options
-----------

|project| needs to know how state, time, and control variables are to be connected to the System,
and needs to know which outputs to use as state time-derivatives.  To do so, the System class
is wrapped with decorators that declare the time, state, and parameter metadata.

For example, lets consider the ODE for the simple Brachistochrone problem.  In this problem
we have three state variables (`x`, `y`, and `v`) and two parameters that can potentially
be used as controls or connected to external sources (`theta` and `g`).

.. code-block:: python

    from openmdao.api import ExplicitComponent
    from dymos import declare_time, declare_state, declare_parameter

    @declare_time(units='s')
    @declare_state('x', rate_source='xdot', units='m')
    @declare_state('y', rate_source='ydot', units='m')
    @declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
    @declare_parameter('theta', targets=['theta'], units='rad')
    @declare_parameter('g', units='m/s**2', targets=['g'])
    class BrachistochroneODE(ExplicitComponent):

        ...

The options for `declare_time` are as follows:

.. embed-options::
    dymos.ode_options
    _ForDocs
    time_options

For `declare_state`, the following options are available:

.. embed-options::
    dymos.ode_options
    _ForDocs
    state_options

And finally, the following options exist for parameters:

.. embed-options::
    dymos.ode_options
    _ForDocs
    parameter_options


Defining an ODE System
----------------------

In addition to specifying the ODE Options, a system used as an ODE is required to have a metadata
entry called `num_nodes`.  When evaluating the dynamics, these systems will receive time, states,
controls, and other inputs as *vectorized* values, where item in the vector represents the variable
value at a discrete time in the trajectory.

The nodes are discretization or collocation locations in the polynomials which represent
each segment.  The number of nodes in a given phase (to be evaluated by the ODE system) is determined
by the number of segments in the phase and the polynomial order in each segment.  When |project| instantiates
the ODE system it provides the total number of nodes at which evaluation is required to the ODE system.
Thus, at a minimum, the `initialize` method of components for an ODE system typically look something
like this:

The inputs and outputs of the system are expected to provide a scalar or dimensioned
value *at each node*.  Vectorization of the component via numpy adds a significant performance increase
compared to using a for loop to cycle through calculations at each node.  It's important to remember
that vectorized data is going to be coming in, this is especially important for defining partials.
From the perspective of the ODE system, the outputs at some time `t` only depend on the values
of the input variables at time `t`.  When the output variables are scalar at any given time, this
results in components whose Jacobian matrices are diagonal.  This large degree of sparsity leads
to computational advantages when using sparse-aware optimizers like SNOPT.  Users should declare
the partial derivatives of their components to be sparse (by specifying nonzero rows and columns)
whenever possible.

.. code-block:: python

    class MyODEComponent(ExplicitComponent):

        def initialize(self):
            self.metadata.declare('num_nodes', types=int)


For example, if `MyODEComponent` is to compute the linear function :math:`y = a * x + b` then the
setup, compute, and compute partials methods might look like this:

.. code-block:: console

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_input('a', shape=(nn,), units='m')
        self.add_input('x', shape=(nn,), units='1/s')
        self.add_input('b', shape=(nn,), units='m/s')

        self.add_output('y', shape=(nn,), units='m/s')

        r = c = np.arange(nn)
        self.declare_partials(of='y', wrt='a', rows=r, cols=c)
        self.declare_partials(of='y', wrt='x', rows=r, cols=c)
        self.declare_partials(of='y', wrt='b', rows=r, cols=c, val=1.0)

    def compute(self, inputs, outputs):
        a = inputs['a']
        x = inputs['x']
        b = inputs['b']

        outputs['y'] = a * x + b

    def compute_partials(self, inputs, outputs, partials):
        a = inputs['a']
        x = inputs['x']
        b = inputs['b']

        partials['y', 'a'] = x
        partials['y', 'x'] = a

A few things to note here.  We can use the `shape` or `val` argument of `add_input` and `add_output`
to dimension each variable.  In this case each variable is assumed to be a scalar at each point in
time (each node).  We use the `rows` and `cols` arguments of `declare_partials` to provide the sparsity.
Here using `arange(nn)` for both gives us a diagonal jacobian with `nn` rows and `nn` columns.  Since
the number of nonzero values in the jacobian is `nn`, we only need to provide `nn` values in the
`compute_partials` method.  It will automatically fill them into the sparse jacobian matrix, in
row-major order.

In this example, the partial of `y` with respect to `b` is linear, so we can simply provide it in
the `declare_partials` call rather than reassigning it every time `compute_partials` is called.
The provided scalar value of `1.0` is broadcast to all `nn` values of the Jacobian matrix.

Dimensioned Inputs and Outputs
------------------------------

The above example assumes all inputs and outputs are scalar at each node.  Sometimes the user may
encounter a situation in which the inputs and/or outputs are vectors, matrices, or tensors at
each node.  In this case the dimension of the variable is `num_nodes`, with the dimension of the
variable at a single node filling out the remaining indices. A 3-vector is thus dimensioned
`(num_nodes, 3)`, while a 3 x 3 matrix would be sized `(num_nodes, 3, 3)`.
