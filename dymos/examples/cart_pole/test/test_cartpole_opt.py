import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos.examples.cart_pole.cartpole_dynamics import CartPoleDynamics


@use_tempdirs
class TestCartPoleOptimization(unittest.TestCase):
    @require_pyoptsparse(optimizer="SNOPT")
    def test_optimization(self):

        p = om.Problem()

        # --- instantiate trajectory and phase, setup transcription ---
        traj = dm.Trajectory()
        p.model.add_subsystem("traj", traj)
        phase = dm.Phase(
            transcription=dm.GaussLobatto(num_segments=40, order=3, compressed=True, solve_segments=False),
            ode_class=CartPoleDynamics,
        )
        # NOTE: set solve_segments=True to do solver-based shooting
        traj.add_phase("phase", phase)

        # --- set state and control variables ---
        phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=2.0, units="s")
        # declare state variables. You can also set lower/upper bounds and scalings here.
        phase.add_state("x", fix_initial=True, lower=-2, upper=2, rate_source="x_dot", shape=(1,), ref=1, defect_ref=1, units="m")
        phase.add_state("x_dot", fix_initial=True, rate_source="x_dotdot", shape=(1,), ref=1, defect_ref=1, units="m/s")
        phase.add_state("theta", fix_initial=True, rate_source="theta_dot", shape=(1,), ref=1, defect_ref=1, units="rad")
        phase.add_state("theta_dot", fix_initial=True, rate_source="theta_dotdot", shape=(1,), ref=1, defect_ref=1, units="rad/s")
        phase.add_state(
            "energy", fix_initial=True, rate_source="e_dot", shape=(1,), ref=1, defect_ref=1, units="N**2*s"
        )  # integration of force**2. This does not have the energy unit, but I call it "energy" anyway.

        # declare control inputs
        phase.add_control("f", fix_initial=False, rate_continuity=False, lower=-20, upper=20, shape=(1,), ref=10, units="N")

        # add cart-pole parameters (set static_target=True because these params are not time-depencent)
        phase.add_parameter("m_cart", val=1.0, units="kg", static_target=True)
        phase.add_parameter("m_pole", val=0.3, units="kg", static_target=True)
        phase.add_parameter("l_pole", val=0.5, units="m", static_target=True)

        # --- set terminal constraint ---
        # alternatively, you can impose those by setting `fix_final=True` in phase.add_state()
        phase.add_boundary_constraint("x", loc="final", equals=1, ref=1.0, units="m")  # final horizontal displacement
        phase.add_boundary_constraint("theta", loc="final", equals=np.pi, ref=1.0, units="rad")  # final pole angle
        phase.add_boundary_constraint("x_dot", loc="final", equals=0, ref=1.0, units="m/s")  # 0 velocity at the and
        phase.add_boundary_constraint("theta_dot", loc="final", equals=0, ref=0.1, units="rad/s")  # 0 angular velocity at the end
        phase.add_boundary_constraint("f", loc="final", equals=0, ref=1.0, units="N")  # 0 force at the end

        # --- set objective function ---
        # we minimize the integral of force**2.
        phase.add_objective("energy", loc="final", ref=100)

        # --- configure optimizer ---
        p.driver = om.pyOptSparseDriver()
        p.driver.options["optimizer"] = "SNOPT"
        p.driver.options["print_results"] = False
        # SNOPT options
        p.driver.opt_settings["Function precision"] = 1e-14
        p.driver.opt_settings["Major optimality tolerance"] = 1e-7
        p.driver.opt_settings["Major feasibility tolerance"] = 1e-7
        p.driver.opt_settings["Major iterations limit"] = 100
        p.driver.opt_settings["Minor iterations limit"] = 100_000_000
        p.driver.opt_settings["Iterations limit"] = 10000
        p.driver.opt_settings["Hessian full memory"] = 1
        p.driver.opt_settings["Hessian frequency"] = 100
        p.driver.opt_settings["Verify level"] = -1

        # declare total derivative coloring to accelerate the UDE linear solves
        p.driver.declare_coloring()

        p.setup(check=False)

        # --- set initial guess ---
        # The initial condition of cart-pole (i.e., state values at time 0) is set here
        # because we set `fix_initial=True` when declaring the states.
        p.set_val("traj.phase.t_initial", 0.0)  # set initial time to 0.
        # linearly interpolate the states between initial and terminal conditions.
        p.set_val("traj.phase.states:x", phase.interp(ys=[0.0, 1.0], nodes="state_input"), units="m")
        p.set_val("traj.phase.states:x_dot", phase.interp(ys=[0.0, 0.0], nodes="state_input"), units="m/s")
        p.set_val("traj.phase.states:theta", phase.interp(ys=[0.0, np.pi], nodes="state_input"), units="rad")
        p.set_val("traj.phase.states:theta_dot", phase.interp(ys=[0.0, 0.0], nodes="state_input"), units="rad/s")
        p.set_val("traj.phase.states:energy", phase.interp(ys=[0.0, 100.0], nodes="state_input"))  # start "energy" from 0.
        p.set_val("traj.phase.controls:f", phase.interp(ys=[1.0, 0.0], nodes="control_input"), units="N")

        # --- run optimization ---
        dm.run_problem(p, run_driver=True, simulate=False, simulate_kwargs={"method": "Radau", "times_per_seg": 10})

        # --- check outputs ---
        # objective value
        obj = p.get_val("traj.phase.states:energy", units="N**2*s")[-1]
        assert_near_equal(obj, 58.83924863, tolerance=1e-3)


if __name__ == "___main__":
    unittest.main()
