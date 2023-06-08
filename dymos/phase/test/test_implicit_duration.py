import openmdao.api as om
import dymos as dm
import numpy as np

import unittest
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse


class ODEComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('h', shape=nn, units='m')
        self.add_input('v', shape=nn, units='m/s')
        self.add_output('hdot', shape=nn, units='m/s')
        self.add_output('vdot', shape=nn, units='m/s**2')

        self.declare_partials(of='hdot', wrt='v', rows=np.arange(nn), cols=np.arange(nn), val=1.0)

    def compute(self, inputs, outputs):
        outputs['hdot'] = inputs['v']
        outputs['vdot'] = -9.80665

@use_tempdirs
class TestImplicitDuration(unittest.TestCase):

    @staticmethod
    def _make_problem(tx):
        p = om.Problem()

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=ODEComp, transcription=tx)
        traj.add_phase('phase', phase)
        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, units='s')
        phase.add_state('h', rate_source='hdot', fix_initial=True, units='m', solve_segments='forward')
        phase.add_state('v', rate_source='vdot', fix_initial=True, units='m/s', solve_segments='forward')

        phase.add_objective('time', loc='final')

        phase.add_duration_balance('h', val=0.0, units='m')

        phase.set_simulate_options(rtol=1.0E-9, atol=1.0E-9)

        p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
        p.driver.declare_coloring(tol=1.0E-12)

        p.setup()

        p.set_val('traj.phase.t_initial', 0)
        p.set_val('traj.phase.t_duration', 1)
        p.set_val('traj.phase.states:h', phase.interp('h', [30, 0]))
        p.set_val('traj.phase.states:v', phase.interp('v', [0, -10]))

        return p

    @require_pyoptsparse(optimizer='IPOPT')
    def test_implicit_duration_radau(self):
        tx = dm.Radau(num_segments=12, order=3, solve_segments=False)

        p = self._make_problem(tx)

        dm.run_problem(p, run_driver=False, simulate=False)

        assert_near_equal(p.get_val('traj.phase.timeseries.time')[-1], 2.4735192, tolerance=1E-6)
        assert_near_equal(p.get_val('traj.phase.timeseries.states:h')[-1], 0.0, tolerance=1E-6)

