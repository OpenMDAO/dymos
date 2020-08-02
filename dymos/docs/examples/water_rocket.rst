.. _water_rocket:

============
Water Rocket
============

Author: Bernardo Bahia Monteiro <bbahia@umich.edu>

In this example, we will optimize a water rocket for range and height at the
apogee, using design variables that are easily modifiable just before launch:
the empty mass, the initial water volume and the launch angle.  This example
builds on :ref:`multi-phase-cannonball`.

Nomenclature
------------

====================   =======================================
:math:`v_\text{out}`   water exit speed at the nozzle
--------------------   ---------------------------------------
:math:`A_\text{out}`   nozzle area
--------------------   ---------------------------------------
:math:`V_w`            water volume in the rocket
--------------------   ---------------------------------------
:math:`p`              pressure in the rocket
--------------------   ---------------------------------------
:math:`p_a`            ambient pressure
--------------------   ---------------------------------------
:math:`\dot{\,}`       time derivative
--------------------   ---------------------------------------
:math:`k`              polytropic constant
--------------------   ---------------------------------------
:math:`V_b`            internal volume of the rocket
--------------------   ---------------------------------------
:math:`\rho_w`         water density
--------------------   ---------------------------------------
:math:`T`              thrust
--------------------   ---------------------------------------
:math:`q`              dynamic pressure
--------------------   ---------------------------------------
:math:`S`              cross sectional area
--------------------   ---------------------------------------
:math:`(\cdot)_0`      value of :math:`(\cdot)` at :math:`t=0`
--------------------   ---------------------------------------
:math:`t`              time
====================   =======================================


Problem Formulation
-------------------
A natural objective function for a water rocket is the maximum height achieved
by the rocket during flight, or the horizontal distance it travels, i.e. its
range. The design of a water rocket is somewhat constrained by the soda bottle
used as its engine. This means that the volume available for water and air
is fixed, the initial launch pressure is limited by the bottle's strength
(since the pressure is directly related to the energy available for the rocket,
it is easy to see that it should be as high as possible) and the nozzle throat
area is also fixed. Given these manufacturing constraints, the design variables
we are left with are the empty mass (it can be easily changed through adding
ballast), the water volume at the launch, and the launch angle.
With this considerations in mind, a natural formulation for the water rocket problem is

.. math::

    \text{maximize}   &\quad \text{range or height} \\
    \text{w.r.t.}     &\quad \text{empty mass, initial water volume, launch angle, trajectory} \\
    \text{subject to} &\quad \text{flight dynamics} \\
                      &\quad \text{fluid dynamics inside the rocket} \\
                      &\quad 0 < \text{initial water volume} < \text{volume of bottle} \\
                      &\quad 0^\circ < \text{launch angle} < 90^\circ \\
                      &\quad 0 < \text{empty mass}


Model
-----
The water rocket model is divided into three basic components: a *water engine*,
responsible for modelling the fluid dynamics inside the rocket and returning
its thrust;  the *aerodynamics*, responsible for calculating the atmospheric drag
of the rocket; and the *equations of motion*, responsible for propagating the
rocket's trajectory in time, using Newton's laws and the forces provided by the
other two components.

In order to integrate these three basic components, some additional interfacing
components are necessary: an atmospheric model to provide values of ambient
pressure for the water engine and air density to the calculation of the dynamic
pressure for the aerodynamic model, and a component that calculates the
instantaneous mass of the rocket by summing the water mass with the rocket's
empty mass.  The high level layout of this model is shown in below.

.. figure:: water_rocket_overview.svg
    :alt: overview

    N2 diagram for the water rocket model

`atmos`, `kinetic_energy`, `dynamic_pressure`, `aero` and `eom` are the same
models used in :ref:`multi-phase-cannonball`.  The remaining components are
discussed below.

.. warning::
    The `eom` component has a singularity in the flight path angle derivative
    when the flight speed is zero.  This happens because the rotational
    dynamics are not modelled.  This can cause convergence problems if the
    launch velocity is set to zero or the launch angle is set to
    :math:`90^\circ`

.. note::
    Since the range of altitudes achieved by the water rocket is very small
    (100m), the air density and pressure are practically constant, thus the use
    of an atmospheric model is not necessary.  However, using it makes it
    easier to reuse code from :ref:`multi-phase-cannonball`.


Water engine
------------
The water engine is modelled by assuming that the air expansion in the rocket
follows an adiabatic process and the water flow is incompressible and inviscid,
i.e.  it follows Bernoulli's equation. We also make the following simplifying
assumptions:

#. The thrust developed after the water is depleted is negligible
#. The area inside the bottle is much smaller than the nozzle area
#. The inertial forces do not affect the fluid dynamics inside the bottle

This simplified modelling can be found in :cite:`Prusa2000`.  A more rigorous
formulation, which drops all these simplifying assumptions can be found in
:cite:`Wheeler2002,Gommes2010,BarrioPerotti2010`.

The first assumption leads to an underestimation of the rocket performance,
since the air left in the bottle after it is out of water is known to generate
appreciable thrust :cite:`Thorncroft2009`.  This simplified model, however,
produces physically meaningful results.

There are two states in this dynamical model, the water volume in the rocket
:math:`V_w` and the gauge pressure inside the rocket :math:`p`.  The
constitutive equations and the N2 diagram showing the model organization are
shown below.

.. table:: Constitutive equations of the water engine model

    ===================    ============================================================
    Component              Equation
    ===================    ============================================================
    water_exhaust_speed    :math:`v_\text{out} = \sqrt{2(p-p_a)/\rho_w}`
    -------------------    ------------------------------------------------------------
    water_flow_rate        :math:`\dot{V}_w = -v_\text{out} A_\text{out}`
    -------------------    ------------------------------------------------------------
    pressure_rate          :math:`\dot{p} = kp\frac{\dot{V_w}}{(V_b-V_w)}`
    -------------------    ------------------------------------------------------------
    water_thrust           :math:`T = (\rho_w v_\text{out})(v_\text{out}A_\text{out})`
    ===================    ============================================================

.. figure:: water_rocket_waterengine.svg
    :alt: water_engine

    N2 diagram for the water engine group

The `_MassAdder` component calculates the rocket's instantaneous mass by
summing the water mass with the rockets empty mass, i.e.

.. math::
    m = m_\text{empty}+\rho_w V_w

.. literalinclude:: ../../examples/water_rocket/water_engine_comp.py

Now these components are joined in a single group

.. literalinclude:: ../../examples/water_rocket/water_propulsion_ode.py


Phases
------
The flight of the water rocket is split in three distinct phases:
propelled ascent, ballistic ascent and ballistic descent.
If the simplification of no thrust without water were lifted,
there would be an extra "air propelled ascent" phase between
the propelled ascent and ballistic ascent phases.

**Propelled ascent:** is the flight phase where the rocket still has water
inside, and hence it is producing thrust. The thrust is given by the water
engine model, and fed into the flight dynamic equations. It starts at launch
and finishes when the water is depleted, i.e. :math:`V_w=0`.

.. literalinclude:: ../../examples/water_rocket/phases.py
    :pyobject: new_propelled_ascent_phase

**Ballistic ascent:** is the flight phase where the rocket is ascending
(:math:`\gamma>0`) but produces no thrust. This phase begins at the end of the
propelled ascent phase and ends at the apogee, defined by :math:`\gamma=0`.

.. literalinclude:: ../../examples/water_rocket/phases.py
    :pyobject: new_ballistic_ascent_phase

**Descent:** is the phase where the rocket is descending without thrust. It
begins at the end of the ballistic ascent phase and ends with ground impact,
i.e. :math:`h=0`.

.. literalinclude:: ../../examples/water_rocket/phases.py
    :pyobject: new_descent_phase

Model parameters
----------------
The model requires a few constant parameters.
The values used are shown in the following table.

.. table:: Values for parameters in the water rocket model

    ====================  ===================  ==============  ===============================================
    Parameter             Value                Unit            Reference
    ====================  ===================  ==============  ===============================================
    :math:`C_D`           0.3450                -              :cite:`BarrioPerotti2009`
    :math:`S`             :math:`\pi 106^2/4`  :math:`mm^2`    :cite:`BarrioPerotti2009`
    :math:`k`             1.2                   -              :cite:`Thorncroft2009,Fischer2020,Romanelli2013`
    :math:`A_\text{out}`  :math:`\pi22^2/4`    :math:`mm^2`    :cite:`aircommand_nozzle`
    :math:`V_b`           2                    L               -
    :math:`\rho_w`        1000                 :math:`kg/m^3`  -
    :math:`p_0`           6.5                  bar             -
    :math:`v_0`           0.1                  :math:`m/s`     -
    :math:`h_0`           0                    :math:`m`       -
    :math:`r_0`           0                    :math:`m`       -
    ====================  ===================  ==============  ===============================================

Values for the bottle volume :math:`V_b`, its cross-sectional area :math:`S`
and the nozzle area :math:`A_\text{out}` are determined by the soda bottle that
makes the rocket primary structure, and thus are not easily modifiable by the
designer.  The polytropic coefficient :math:`k` is a function of the moist air
characteristics inside the rocket.  The initial speed :math:`v_0` must be set
to a value higher than zero, otherwise the flight dynamic equations become
singular.  This issue arises from the angular dynamics of the rocket not being
modelled.  The drag coefficient :math:`C_D` is sensitive to the aerodynamic
design, but can be optimized by a single discipline analysis.  The initial
pressure :math:`p_0` should be maximized in order to obtain the maximum range
or height for the rocket.  It is limited by the structural properties of the
bottle, which are modifiable by the designer, since the bottle needs to be
available commercially.  Finally, the starting point of the rocket is set to
the origin.


Puttting it all together
------------------------
The different phases must be combined in a single trajectory,
and linked in a sequence.
Here we also define the design variables.

.. literalinclude:: ../../examples/water_rocket/phases.py
    :pyobject: new_water_rocket_trajectory


Optimizing for height
---------------------
.. embed-code::
    dymos.examples.water_rocket.doc.test_doc_water_rocket.TestWaterRocketForDocs.test_water_rocket_height_for_docs
    :layout: code, output, plot


Optimizing for range
--------------------
.. embed-code::
    dymos.examples.water_rocket.doc.test_doc_water_rocket.TestWaterRocketForDocs.test_water_rocket_range_for_docs
    :layout: code, output, plot


Accessing the results
---------------------
.. literalinclude:: ../../examples/water_rocket/doc/test_doc_water_rocket.py
    :pyobject: summarize_results

.. literalinclude:: ../../examples/water_rocket/doc/test_doc_water_rocket.py
    :pyobject: plot_propelled_ascent

.. literalinclude:: ../../examples/water_rocket/doc/test_doc_water_rocket.py
    :pyobject: plot_states

.. literalinclude:: ../../examples/water_rocket/doc/test_doc_water_rocket.py
    :pyobject: plot_trajectory


References
------------
.. bibliography:: water_rocket_refs.bib
    :style: unsrt
