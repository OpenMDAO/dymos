from __future__ import print_function, absolute_import, division

import os
import unittest

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DenseJacobian,\
    CSCJacobian, CSRJacobian, DirectSolver

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
import dymos.examples.brachistochrone.ex_brachistochrone as ex_brachistochrone


class TestBrachistochroneExample(unittest.TestCase):

    def test(self):
        p = Problem(model=Group())

        phase = Phase('glm',
                      ode_class=BrachistochroneODE,
                      num_segments=3,
                      formulation='solver-based',
                      method_name='ForwardEuler')

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=False)
        phase.set_state_options('y', fix_initial=True, fix_final=False)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_boundary_constraint('x', loc='initial', equals=0.)
        phase.add_boundary_constraint('x', loc='final', equals=10.)
        phase.add_boundary_constraint('y', loc='initial', equals=10.)
        phase.add_boundary_constraint('y', loc='final', equals=5.)
        phase.add_boundary_constraint('v', loc='initial', equals=0.)

        phase.add_control('theta', units='deg', dynamic=True,
                          rate_continuity=True, lower=0.01, upper=179.9)

        phase.add_control('g', units='m/s**2', dynamic=False, opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.setup(force_alloc_complex=False)
        p.set_solver_print(level=-1)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10])
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5])
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9])
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5])

        p.run_model()

    # def test_show_plots(self):
    #     transcription = 'radau-ps'
    #     ex_brachistochrone.SHOW_PLOTS = True
    #     p = ex_brachistochrone.brachistochrone_min_time(transcription=transcription)
