import os
import unittest

import matplotlib
import openmdao.api as om
import matplotlib.pyplot as plt
import dymos as dm

from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

matplotlib.use('Agg')
plt.style.use('ggplot')


@use_tempdirs
class TestBrachistochroneExprPathConstraint(unittest.TestCase):

    def _make_problem(self, tx):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=tx)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                          units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)

        phase.add_boundary_constraint('x', loc='final', equals=10.0)
        phase.add_path_constraint('pc = y-x/2-1', lower=0.0)

        phase.add_parameter('g', opt=False, units='m/s**2', val=9.80665, shape=None, include_timeseries=True)

        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interp('x', [0, 10])
        p['phase0.states:y'] = phase.interp('y', [10, 5])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p[f'phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 9.80665
        return p

    def test_brachistochrone_expr_path_constraint(self):
        prob = self._make_problem(tx=dm.Radau(num_segments=5, order=3, compressed=True))
        prob.run_driver()
        yf = prob.get_val('phase0.timeseries.y')[-1]

        assert_near_equal(yf, 6)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
