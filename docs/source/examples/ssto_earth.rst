=================
SSTO Earth Launch
=================

This example builds on the SSTO Lunar Ascent example by adding an atmospheric drag model.  This
example is based on the "Time-Optimal Launch of a Titan II" example given in Appendix B of
[Longuski2016]_.

-------------------
Run Script
-------------------

.. embed-code::
    dymos.examples.ssto.test.test_ex_ssto_earth.TestExampleSSTOEarth.test_simulate_plot
    :layout: code, output, plot

--------------------------------
Component and Group Definitions
--------------------------------

ex_ssto_earth.py
----------------------
.. embed-code::
    ../dymos/examples/ssto/ex_ssto_earth.py
    :layout: code

launch_vehicle_2d_eom_comp.py
------------------------------
.. embed-code::
    ../dymos/examples/ssto/launch_vehicle_2d_eom_comp.py
    :layout: code

log_atmosphere_comp.py
------------------------
.. embed-code::
    ../dymos/examples/ssto/log_atmosphere_comp.py
    :layout: code

launch_vehicle_ode.py
----------------------
.. embed-code::
    ../dymos/examples/ssto/launch_vehicle_ode.py
    :layout: code

References
----------
.. [Longuski2016] Longuski, James M., José J. Guzmán, and John E. Prussing. Optimal control with aerospace applications. Springer, 2016.





