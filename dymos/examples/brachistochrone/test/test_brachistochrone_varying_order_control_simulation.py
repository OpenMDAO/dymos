from __future__ import print_function, absolute_import, division

import unittest
from numpy.testing import assert_almost_equal

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

import openmdao.api as om
import dymos as dm
from openmdao.utils.general_utils import set_pyoptsparse_opt
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT', fallback=True)


class TestBrachistochroneVaryingOrderControlSimulation(unittest.TestCase):

    def test_radau(self):
        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        t = dm.GaussLobatto(num_segments=20,
                            order=[3, 5]*10,
                            compressed=True)

        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        units=BrachistochroneODE.states['x']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)
        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        units=BrachistochroneODE.states['y']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)
        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        targets=BrachistochroneODE.states['v']['targets'],
                        units=BrachistochroneODE.states['v']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', targets=BrachistochroneODE.parameters['theta']['targets'],
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_input_parameter('g', targets=BrachistochroneODE.parameters['g']['targets'],
                                  units='m/s**2', val=9.80665)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.input_parameters:g'] = 9.80665

        p.run_driver()

        phase.simulate()
