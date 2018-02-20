The Brachistochrone Problem
===========================

Suppose a bead is allowed to slide without friction along a rigid wire between
two points, accelerated only by gravity.

Given the location of the two endpoints, find the shape of the rigid wire that
minimizes the time to travel along the wire.

In this example, we pose the problem by letting our states be the horizontal position component,
vertical position component, and speed of the bead as a function of time.  The equations of
motion can then be written as:

.. math::

    \dot{x}(t) = v \cdot \sin \theta

    \dot{y}(t) = v \cdot \cos \theta

    \dot{v}(t) = -g \cdot \cos \theta

Variable :math:`\theta` represents the angle of the wire, where an angle of :math:`\theta=0`
means the bead is traveling straight down.

With some knowledge of classical optimal control theory, one can show that the solution
to the optimal control is such that:

..math::

    \frac{v}{\sin \theta} = K

where K is some constant dependent upon the constraints of the problem. Our model outputs this
parameter as a check of the validity of the result.