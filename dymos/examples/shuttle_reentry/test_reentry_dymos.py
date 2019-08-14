import unittest
import matplotlib.pyplot as plt
from openmdao.api import Problem, Group, pyOptSparseDriver
from openmdao.utils.assert_utils import assert_rel_error
from dymos import Trajectory, GaussLobatto, Phase, Radau
from shuttle_ode import ShuttleODE
import numpy as np

class TestReentry(unittest.TestCase):

    def test_reentry_radau_dymos(self):

        p = Problem(model=Group())
        p.driver = pyOptSparseDriver()
        p.driver.declare_coloring()

        traj = p.model.add_subsystem("traj", Trajectory())
        phase0 = traj.add_phase("phase0", Phase(ode_class=ShuttleODE, transcription=Radau(num_segments=50, order=3)))

        phase0.set_time_options(fix_initial=True, units="s", duration_ref=200)
        phase0.set_state_options("h", fix_initial=True, fix_final=True, units="ft", rate_source="hdot", targets=["h"], lower=0, ref0=75000, ref=300000, defect_ref=1000)
        phase0.set_state_options("gamma", fix_initial=True, fix_final=True, units="rad", rate_source="gammadot", targets=["gamma"], lower=-89.*np.pi/180, upper=89.*np.pi/180)
        phase0.set_state_options("phi", fix_initial=True, fix_final=False, units="rad", rate_source="phidot", lower=0, upper=89.*np.pi/180)
        phase0.set_state_options("psi", fix_initial=True, fix_final=False, units="rad", rate_source="psidot", targets=["psi"], lower=0, upper=90.*np.pi/180)
        phase0.set_state_options("theta", fix_initial=True, fix_final=False, units="rad", rate_source="thetadot", targets=["theta"], lower=-89.*np.pi/180, upper=89.*np.pi/180)
        phase0.set_state_options("v", fix_initial=True, fix_final=True, units="ft/s", rate_source="vdot", targets=["v"], lower=0, ref0=2500, ref=25000)
        phase0.add_control("alpha", units="rad", opt=True, lower=-np.pi/2, upper=np.pi/2, targets=["alpha"])
        phase0.add_control("beta", units="rad", opt=True, lower=-89*np.pi/180, upper=1*np.pi/180, targets=["beta"])
        phase0.add_path_constraint("q", lower=0, upper=70, units="Btu/ft**2/s", ref=70)
        phase0.add_objective("theta", loc="final", ref=-0.01)

        p.driver.options["optimizer"] = 'SNOPT'
        p.driver.opt_settings["iSumm"] = 6

        p.setup(check=True)

        p.set_val("traj.phase0.states:h", phase0.interpolate(ys=[260000, 80000], nodes="state_input"), units="ft")
        p.set_val("traj.phase0.states:gamma", phase0.interpolate(ys=[-1*np.pi/180, -5*np.pi/180], nodes="state_input"), units="rad")
        p.set_val("traj.phase0.states:phi", phase0.interpolate(ys=[0, 75*np.pi/180], nodes="state_input"), units="rad")
        p.set_val("traj.phase0.states:psi", phase0.interpolate(ys=[90*np.pi/180, 10*np.pi/180], nodes="state_input"), units="rad")
        p.set_val("traj.phase0.states:theta", phase0.interpolate(ys=[0, 25*np.pi/180], nodes="state_input"), units="rad")
        p.set_val("traj.phase0.states:v", phase0.interpolate(ys=[25600, 2500], nodes="state_input"), units="ft/s")
        p.set_val("traj.phase0.t_initial", 0, units="s")
        p.set_val("traj.phase0.t_duration", 2000, units="s")
        p.set_val("traj.phase0.controls:alpha", phase0.interpolate(ys=[17.4*np.pi/180, 17.4*np.pi/180], nodes="control_input"), units="rad")
        p.set_val("traj.phase0.controls:beta", phase0.interpolate(ys=[-75*np.pi/180, 0*np.pi/180], nodes="control_input"), units="rad")

        p.run_driver()

        print(p.get_val("traj.phase0.timeseries.time")[-1])
        print(p.get_val("traj.phase0.timeseries.states:theta")[-1])
        assert_rel_error(self, p.get_val("traj.phase0.timeseries.time")[-1], 2181.90371131, tolerance=1e-3)
        assert_rel_error(self, p.get_val("traj.phase0.timeseries.states:theta")[-1], .53440626, tolerance=1e-3)

    def test_reentry_gauss_lobatto_dymos(self):

        p = Problem(model=Group())
        p.driver = pyOptSparseDriver()
        p.driver.declare_coloring()

        traj = p.model.add_subsystem("traj", Trajectory())
        phase0 = traj.add_phase("phase0", Phase(ode_class=ShuttleODE, transcription=GaussLobatto(num_segments=50, order=3)))

        phase0.set_time_options(fix_initial=True, units="s", duration_ref=200)
        phase0.set_state_options("h", fix_initial=True, fix_final=True, units="ft", rate_source="hdot", targets=["h"], lower=0, ref0=75000, ref=300000, defect_ref=1000)
        phase0.set_state_options("gamma", fix_initial=True, fix_final=True, units="rad", rate_source="gammadot", targets=["gamma"], lower=-89.*np.pi/180, upper=89.*np.pi/180)
        phase0.set_state_options("phi", fix_initial=True, fix_final=False, units="rad", rate_source="phidot", lower=0, upper=89.*np.pi/180)
        phase0.set_state_options("psi", fix_initial=True, fix_final=False, units="rad", rate_source="psidot", targets=["psi"], lower=0, upper=90.*np.pi/180)
        phase0.set_state_options("theta", fix_initial=True, fix_final=False, units="rad", rate_source="thetadot", targets=["theta"], lower=-89.*np.pi/180, upper=89.*np.pi/180)
        phase0.set_state_options("v", fix_initial=True, fix_final=True, units="ft/s", rate_source="vdot", targets=["v"], lower=0, ref0=2500, ref=25000)
        phase0.add_control("alpha", units="rad", opt=True, lower=-np.pi/2, upper=np.pi/2, targets=["alpha"])
        phase0.add_control("beta", units="rad", opt=True, lower=-89*np.pi/180, upper=1*np.pi/180, targets=["beta"])
        phase0.add_path_constraint("q", lower=0, upper=70, units="Btu/ft**2/s", ref=70)
        phase0.add_objective("theta", loc="final", ref=-0.01)

        p.driver.options["optimizer"] = 'SNOPT'
        p.driver.opt_settings["iSumm"] = 6

        p.setup(check=True)

        p.set_val("traj.phase0.states:h", phase0.interpolate(ys=[260000, 80000], nodes="state_input"), units="ft")
        p.set_val("traj.phase0.states:gamma", phase0.interpolate(ys=[-1*np.pi/180, -5*np.pi/180], nodes="state_input"), units="rad")
        p.set_val("traj.phase0.states:phi", phase0.interpolate(ys=[0, 75*np.pi/180], nodes="state_input"), units="rad")
        p.set_val("traj.phase0.states:psi", phase0.interpolate(ys=[90*np.pi/180, 10*np.pi/180], nodes="state_input"), units="rad")
        p.set_val("traj.phase0.states:theta", phase0.interpolate(ys=[0, 25*np.pi/180], nodes="state_input"), units="rad")
        p.set_val("traj.phase0.states:v", phase0.interpolate(ys=[25600, 2500], nodes="state_input"), units="ft/s")
        p.set_val("traj.phase0.t_initial", 0, units="s")
        p.set_val("traj.phase0.t_duration", 2000, units="s")
        p.set_val("traj.phase0.controls:alpha", phase0.interpolate(ys=[17.4*np.pi/180, 17.4*np.pi/180], nodes="control_input"), units="rad")
        p.set_val("traj.phase0.controls:beta", phase0.interpolate(ys=[-75*np.pi/180, 0*np.pi/180], nodes="control_input"), units="rad")

        p.run_driver()

        print(p.get_val("traj.phase0.timeseries.time")[-1])
        print(p.get_val("traj.phase0.timeseries.states:theta")[-1])
        assert_rel_error(self, p.get_val("traj.phase0.timeseries.time")[-1], 2181.88191719, tolerance=1e-3)
        assert_rel_error(self, p.get_val("traj.phase0.timeseries.states:theta")[-1], .53440955, tolerance=1e-3)

if __name__ == "___main__":
    unittest.main()