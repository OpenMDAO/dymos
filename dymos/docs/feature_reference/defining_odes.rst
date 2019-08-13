=============
Defining ODEs
=============

Optimal control problems contain ordinary differential equations (ODE) or
differential algebraic equations (DAE) that dictate the evolution of the state of the system.
Typically this evolution occurs in time, and the ODE represents equations of motion (EOM).  
The equations of motion can define a variety of systems, not just mechanical ones.
In other fields they are sometimes referred to as *process equations*.

.. math::

  \dot{\bar{x}} = f_{ode}(\bar{x},t,\bar{u},\bar{d})


To represent EOM, Dymos uses a standard OpenMDAO System (a Group or Component).  This System
takes some set of variables as input and computes outputs that include the time-derivatives of
the state variables :math:`\bar{x}`.  The ODE may also be a function of the the
current time :math:`t`.  Finally, the dynamics may be subject to some set of controllable parameters.
In Dymos these are broken into the dynamic controls :math:`\bar{u}` and the static design
parameters :math:`\bar{d}`.

ODE Systems
-----------

In Dymos, the ODE is simply an OpenMDAO system that accepts some inputs and computes some outputs,
among which are the rates of the state variables :math:`\dot{\bar{x}}`.

The Brachistochrone ODE
-----------------------

For example, lets consider the ODE for the simple Brachistochrone problem.  In this problem
we have three state variables (`x`, `y`, and `v`) and two parameters that can potentially
be used as controls or connected to external sources (`theta` and `g`).

.. embed-code::
   examples.brachistochrone.test.test_brachistochrone_quick_start.BrachistochroneODE
   :layout: code

In the following sections, we'll explain the definition of this ODE in a step-by-step fasion.

System Options
--------------

ODE Systems in Dymos need provide values at numerous points in time we call nodes.  For performance
reasons, it's best if it can do so using vectorized mathematical operations to compute the values
simultaneously rather than using the loop to perform the calculation at each node.  Different
optimal control transcriptions will need to have computed ODE values at different numbers of nodes,
so each ODE system in Dymos is required to support the option `num_nodes`, which is defined
in the `initialize` method.

ODE system may define initial options as well.  Since these options in OpenMDAO are typically
provided as arguments to the instantiation of the ODE system, the user has the ability to provide
additional input keyword arguments using the `ode_init_kwargs` option on Phase.

Variables of the Optimal Control Problem
----------------------------------------

Dymos needs to know how state, time, and control variables are to be connected to the System,
and needs to know which outputs to use as state time-derivatives.

Time
^^^^

The following time options can be set via the `set_time_options` method of Phase:

.. embed-options::
    dymos.phase.options
    _ForDocs
    time_options

States
^^^^^^

States have the following options set via the `add_state` method of Phase:

.. embed-options::
    dymos.phase.options
    _ForDocs
    state_options

Controls
^^^^^^^^

Inputs to the ODE which are to be dynamic controls are added via the `add_control` method of
Phase.  The available options are as follows:

.. embed-options::
    dymos.phase.options
    _ForDocs
    control_options

Design Parameters
^^^^^^^^^^^^^^^^^

Inputs to the ODE which can be design parameters are added via the `add_design_parameter` method of
Phase.  The available options are as follows:

.. embed-options::
    dymos.phase.options
    _ForDocs
    design_parameter_options

Input Parameters
^^^^^^^^^^^^^^^^

Inputs to the ODE which can be input parameters are added via the `add_input_parameter` method of
Phase.
They are similar to design parameters, but the phase cannot treat them as design
variables to the optimization problem.
The available options are as follows:

.. embed-options::
    dymos.phase.options
    _ForDocs
    input_parameter_options

Using decorators to associate time, control, and parameter options with the ODE system
--------------------------------------------------------------------------------------

Some properties of the variables associated with the ODE function are dependent on the particular
optimization problem.
For example, bounds on times, states, and controls will be problem-dependent.
Other options, such as the `rate_source` of state variables or the target of time, states, or controllable parameters are a function of the ODE itself.
Therefore it can sometimes be convenient to associate those properties with the ODE class itself.
To allow this, Dymos provides decorators for ODEs which assign *default* values of these properties at the ODE level.
These values can be overridden using the `set_time_options`, `add_state`, `add_control`, `add_design_parameter` or `add_input_parameter` methods on Phase.

.. note::

    The units of variables need not match the units used inside the ODE system, it only needs to be compatible.
    These are the default units in which the variable will be provided at the Phase level.

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

Vectorizing the ODE
-------------------

In addition to specifying the ODE Options, a system used as an ODE is required to have a metadata
entry called `num_nodes`.  When evaluating the dynamics, these systems will receive time, states,
controls, and other inputs as *vectorized* values, where item in the vector represents the variable
value at a discrete time in the trajectory.

The nodes are discretization or collocation locations in the polynomials which represent
each segment.  The number of nodes in a given phase (to be evaluated by the ODE system) is determined
by the number of segments in the phase and the polynomial order in each segment.  When Dymos instantiates
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

    class MyODESystem(ExplicitComponent):

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

Non-Vector Inputs
-----------------

Declaring inputs as vectors means that they have the potential to be used either as design parameters or as dynamic controls which can assume a different value at each node.
For some quantities, such as gravitational acceleration in the Brachistochrone example, we can assume that the value will never need to be dynamic.
To accommodate this, design and input parameters can be declared static with the argument `dynamic=False`.
This prevents Dymos from "fanning out" the static value to the *n* nodes in the ODE system.
If a parameter is declared static in the `declare_parameter` decorator, it cannot be used as a dynamic control.
