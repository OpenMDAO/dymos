---
title: 'Dymos: Optimal Control of Multidisciplinary Systems'
tags:
  - Python
  - OpenMDAO
  - optimal control
  - trajectory optimization
  - multidisciplinary optimization
authors:
  - name: Rob Falck
    orcid: 0000-0001-9864-4928
    affiliation: 1
  - name: Justin S. Gray
    affiliation: 1
  - name: Kaushik Ponnapalli
    affiliation: 2
affiliations:
 - name: NASA Glenn Research Center.  21000 Brookpark Rd, Cleveland, OH 44135.
   index: 1
-  name: Vantage Partners LLC. 3000 Aerospace Parkway, Brook Park, OH 44142.
-  index: 2
date: 26 June 2020
bibliography: paper.bib

# Summary

Dymos is a Python package for solving optimal control problems within
the OpenMDAO [@Gray2019a] software framework.  While there are a number of software
packages available for finding optimal control solutions for various applications,
Dymos was developed to efficiently solve problems wherein the system dynamics
may include expensive, implicit calculations.  By leveraging the OpenMDAO software
package's approach to calculating derivatives for gradient-based optimization,
Dymos can provide significant improvements in performance.



# Acknowledgements

Dymos was developed with funding from NASA's Transformative Tools and Technologies ($T^3$) Project.

# References
