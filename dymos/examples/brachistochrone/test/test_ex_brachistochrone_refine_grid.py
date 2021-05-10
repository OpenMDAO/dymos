import os
import unittest
from numpy.testing import assert_almost_equal

import openmdao.api as om

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.testing_utils import use_tempdirs
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT', fallback=True)


@use_tempdirs
class TestBrachistochroneRefineGrid(unittest.TestCase):

    def make_problem(self, transcription='radau-ps', num_segments=5, transcription_order=3, compressed=True):

        self.p = p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        if transcription == 'gauss-lobatto':
            t = dm.GaussLobatto(num_segments=num_segments,
                                order=transcription_order,
                                compressed=compressed)
        elif transcription == 'radau-ps':
            t = dm.Radau(num_segments=num_segments,
                         order=transcription_order,
                         compressed=compressed)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)

        phase.set_refine_options(refine=True, tol=1.0E-6)

        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj0', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False)
        phase.add_state('y', fix_initial=True, fix_final=False)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=['unconnected_inputs'])

        p['traj0.phase0.t_initial'] = 0.0
        p['traj0.phase0.t_duration'] = 2.0

        p['traj0.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj0.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj0.phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['traj0.phase0.controls:theta'] = phase.interp('theta', [5, 100])

        p['traj0.phase0.parameters:g'] = 9.80665

        return p

    def run_asserts(self, p):

        t_initial = p.get_val('traj0.phase0.timeseries.time')[0]
        tf = p.get_val('traj0.phase0.timeseries.time')[-1]

        x0 = p.get_val('traj0.phase0.timeseries.states:x')[0]
        xf = p.get_val('traj0.phase0.timeseries.states:x')[-1]

        y0 = p.get_val('traj0.phase0.timeseries.states:y')[0]
        yf = p.get_val('traj0.phase0.timeseries.states:y')[-1]

        v0 = p.get_val('traj0.phase0.timeseries.states:v')[0]
        vf = p.get_val('traj0.phase0.timeseries.states:v')[-1]

        g = p.get_val('traj0.phase0.timeseries.parameters:g')[0]

        thetaf = p.get_val('traj0.phase0.timeseries.controls:theta')[-1]

        assert_almost_equal(t_initial, 0.0)
        assert_almost_equal(x0, 0.0)
        assert_almost_equal(y0, 10.0)
        assert_almost_equal(v0, 0.0)

        assert_almost_equal(tf, 1.8016, decimal=4)
        assert_almost_equal(xf, 10.0, decimal=3)
        assert_almost_equal(yf, 5.0, decimal=3)
        assert_almost_equal(vf, 9.902, decimal=3)
        assert_almost_equal(g, 9.80665, decimal=3)

        assert_almost_equal(thetaf, 100.12, decimal=0)

    def test_refine_brachistochrone_radau_compressed(self):
        p = self.make_problem(transcription='radau-ps', num_segments=5, transcription_order=3, compressed=True)
        dm.run_problem(p)
        self.run_asserts(self.p)

    def test_refine_brachistochrone_gauss_lobatto_compressed(self):
        p = self.make_problem(transcription='gauss-lobatto', num_segments=5, transcription_order=3, compressed=True)
        dm.run_problem(p)
        self.run_asserts(self.p)
