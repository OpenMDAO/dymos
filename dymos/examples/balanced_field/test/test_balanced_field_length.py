from packaging.version import Version
import unittest

import numpy as np

import openmdao
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.assert_utils import assert_near_equal

import dymos as dm
from dymos.examples.balanced_field.balanced_field_ode import BalancedFieldODEComp
from dymos.examples.balanced_field.balanced_field_length import make_balanced_field_length_problem
from dymos.utils.misc import om_version


@use_tempdirs
class TestBalancedFieldLengthRestart(unittest.TestCase):

    @require_pyoptsparse(optimizer='IPOPT')
    @unittest.skipUnless(Version(openmdao.__version__) > Version("3.23"),
                         reason='Test requires OpenMDAO 3.23.0 or later.')
    def test_make_plots(self):
        p = make_balanced_field_length_problem(ode_class=BalancedFieldODEComp, tx=dm.Radau(num_segments=3))
        dm.run_problem(p, run_driver=True, simulate=True, make_plots=True)

    @require_pyoptsparse(optimizer='IPOPT')
    @unittest.skipUnless(Version(openmdao.__version__) > Version("3.23"),
                         reason='Test requires OpenMDAO 3.23.0 or later.')
    def test_restart_from_sol(self):
        p = make_balanced_field_length_problem(ode_class=BalancedFieldODEComp, tx=dm.Radau(num_segments=3))
        dm.run_problem(p, run_driver=True, simulate=False)

        sol_db = 'dymos_solution.db'
        if om_version()[0] > (3, 34, 2):
            sol_db = p.get_outputs_dir() / sol_db

        sol_results = om.CaseReader(sol_db).get_case('final')

        assert_near_equal(sol_results.get_val('traj.climb.timeseries.r')[-1], 2197, tolerance=0.01)

        dm.run_problem(p, run_driver=True, simulate=True, restart=sol_db)

        sol_db = 'dymos_solution.db'
        sim_db = 'dymos_simulation.db'
        if om_version()[0] > (3, 34, 2):
            sol_db = p.get_outputs_dir() / sol_db
            sim_db = p.model.traj.sim_prob.get_outputs_dir() / sim_db

        sol_results = om.CaseReader(sol_db).get_case('final')
        sim_results = om.CaseReader(sim_db).get_case('final')

        assert_near_equal(sol_results.get_val('traj.climb.timeseries.r')[-1], 2197, tolerance=0.01)
        assert_near_equal(sim_results.get_val('traj.climb.timeseries.r')[-1], 2197, tolerance=0.01)

        assert_near_equal(sol_results.get_val('traj.rto.timeseries.r')[-1], 2197, tolerance=0.01)
        assert_near_equal(sim_results.get_val('traj.rto.timeseries.r')[-1], 2197, tolerance=0.01)

    @require_pyoptsparse(optimizer='IPOPT')
    @unittest.skipUnless(Version(openmdao.__version__) > Version("3.23"),
                         reason='Test requires OpenMDAO 3.23.0 or later.')
    def test_restart_from_sim(self):
        p = make_balanced_field_length_problem(ode_class=BalancedFieldODEComp, tx=dm.Radau(num_segments=3))
        dm.run_problem(p, run_driver=True, simulate=True)

        sol_db = 'dymos_solution.db'
        sim_db = 'dymos_simulation.db'
        if om_version()[0] > (3, 34, 2):
            sol_db = p.get_outputs_dir() / sol_db
            sim_db = p.model.traj.sim_prob.get_outputs_dir() / sim_db

        sol_results = om.CaseReader(sol_db).get_case('final')

        assert_near_equal(sol_results.get_val('traj.climb.timeseries.r')[-1], 2197, tolerance=0.01)

        dm.run_problem(p, run_driver=True, simulate=True, restart=sim_db)

        sol_results = om.CaseReader(sol_db).get_case('final')
        sim_results = om.CaseReader(sim_db).get_case('final')

        assert_near_equal(sol_results.get_val('traj.climb.timeseries.r')[-1], 2197, tolerance=0.01)
        assert_near_equal(sim_results.get_val('traj.climb.timeseries.r')[-1], 2197, tolerance=0.01)

        assert_near_equal(sol_results.get_val('traj.rto.timeseries.r')[-1], 2197, tolerance=0.01)
        assert_near_equal(sim_results.get_val('traj.rto.timeseries.r')[-1], 2197, tolerance=0.01)


@use_tempdirs
class TestBalancedFieldLengthDefaultValues(unittest.TestCase):

    @require_pyoptsparse(optimizer='IPOPT')
    def test_default_vals_stick(self):
        """
        Make the balanced field problem without any set_val calls after setup.
        """
        p = make_balanced_field_length_problem(ode_class=BalancedFieldODEComp, tx=dm.GaussLobatto(num_segments=3))

        p.run_model()

        assert_near_equal(p.get_val('traj.rotate.t_initial'), 35)
        assert_near_equal(p.get_val('traj.rotate.t_duration'), 5)
        assert_near_equal(p.get_val('traj.rotate.controls:alpha'), np.array([[0, 0]]).T)
        assert_near_equal(p.get_val('traj.climb.controls:alpha', units='deg'),
                          p.model.traj.phases.climb.interp('', [5, 5], nodes='control_input'))
        assert_near_equal(p.get_val('traj.climb.states:gam', units='deg'),
                          p.model.traj.phases.climb.interp(ys=[0.0, 5.0], nodes='state_input'))
        assert_near_equal(p.get_val('traj.climb.states:h', units='ft'),
                          p.model.traj.phases.climb.interp(ys=[0.0, 35.0], nodes='state_input'))
        assert_near_equal(p.get_val('traj.v1_to_vr.parameters:alpha'), 0.0)
