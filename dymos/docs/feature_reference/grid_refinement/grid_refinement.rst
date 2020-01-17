===============
Grid Refinement
===============

Grid refinement is carried out via ph-refinement as proposed by Patterson, Hager, and Rao for Radau pseudospectral
methods [PattersonHagerRao2015]_. A similar method was shown for a Gauss-Lobatto transcription by Lei et al
[LeiLiuLiYeShao2017]_.

Refinement is carried out iteratively in four steps:

#. The error in each phase is computed for each phase in the solved problem.
#. The errors are compared to the tolerance to determine which phases require refinement.
#. Phases are refined via ph-refinement and the new required grid is computed.
#. The problem is re-solved on the new grid and the process repeats until the error tolerance is met for all phases.

.. toctree::
    :maxdepth: 2
    :titlesonly:

    error_estimate
    ph_adaptive


References
^^^^^^^^^^
.. [PattersonHagerRao2015] Michael A. Patterson and William W. Hager and Anil V. Rao. "A ph Mesh Refinement Method for Optimal Control." Optimal Control Applications and Methods 36.4 (2015): 398-421
.. [LeiLiuLiYeShao2017] Humin Lei and Tao Liu and Deng Li and Jikun Ye and Lei Shao. " Adaptive Mesh Iteration Method for Trajectory Optimization Based on Hermite-Pseudospectral Direct Transcription." Mathematical Problems in Engineering (2017)