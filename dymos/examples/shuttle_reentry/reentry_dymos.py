import numpy as np 
import matplotlib.pyplot as plt 
from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver
from dymos import Trajectory, GaussLobatto, Phase, Radau
from shuttle_ode import ShuttleODE

# Create the problem
prob = Problem(model=Group())

# Create the trajectory
traj = prob.model.add_subsystem("traj", Trajectory())

# Create the phase and add it to the trajectory
phase0 = Phase(ode_class=ShuttleODE, transcription=Radau(num_segments=50, order=3))
traj.add_phase(name="phase0", phase=phase0)

# Fix the initial time and manually scale time
phase0.set_time_options(fix_initial=True, units="s", duration_ref=200)

# Fix initial and necessary final states and manually scale them
phase0.set_state_options("h", fix_initial=True, fix_final=True, units="ft", rate_source="hdot", targets=["h"], lower=0, ref0=75000, ref=300000, defect_ref=1000)
phase0.set_state_options("gamma", fix_initial=True, fix_final=True, units="rad", rate_source="gammadot", targets=["gamma"], lower=-89.*np.pi/180, upper=89.*np.pi/180)
phase0.set_state_options("phi", fix_initial=True, fix_final=False, units="rad", rate_source="phidot", lower=0, upper=89.*np.pi/180)
phase0.set_state_options("psi", fix_initial=True, fix_final=False, units="rad", rate_source="psidot", targets=["psi"], lower=0, upper=90.*np.pi/180)
phase0.set_state_options("theta", fix_initial=True, fix_final=False, units="rad", rate_source="thetadot", targets=["theta"], lower=-89.*np.pi/180, upper=89.*np.pi/180)
phase0.set_state_options("v", fix_initial=True, fix_final=True, units="ft/s", rate_source="vdot", targets=["v"], lower=0, ref0=2500, ref=25000)

# Add necessary controls, enforce bounds on them, and connect them to variables in the ode
phase0.add_control("alpha", units="rad", opt=True, lower=-np.pi/2, upper=np.pi/2, targets=["alpha"])
phase0.add_control("beta", units="rad", opt=True, lower=-89*np.pi/180, upper=1*np.pi/180, targets=["beta"])

# Constrain the maximum leading edge heating and scale it
phase0.add_path_constraint("q", lower=0, upper=70, units="Btu/ft**2/s", ref=70)

# Maximize the crossrange (latitude) and scale it
phase0.add_objective("theta", loc="final", ref=-0.01)

# Add the pyOptSparseDriver to the problem and allow it to use coloring
prob.driver = pyOptSparseDriver()
prob.driver.declare_coloring()

# Set the optimizer
prob.driver.options["optimizer"] = 'SNOPT'
prob.driver.opt_settings["iSumm"] = 6

# Set up the problem
prob.setup(check=True)

# Set the initial guesses for states, controls, and time values
prob.set_val("traj.phase0.states:h", phase0.interpolate(ys=[260000, 80000], nodes="state_input"), units="ft")
prob.set_val("traj.phase0.states:gamma", phase0.interpolate(ys=[-1*np.pi/180, -5*np.pi/180], nodes="state_input"), units="rad")
prob.set_val("traj.phase0.states:phi", phase0.interpolate(ys=[0, 75*np.pi/180], nodes="state_input"), units="rad")
prob.set_val("traj.phase0.states:psi", phase0.interpolate(ys=[90*np.pi/180, 10*np.pi/180], nodes="state_input"), units="rad")
prob.set_val("traj.phase0.states:theta", phase0.interpolate(ys=[0, 25*np.pi/180], nodes="state_input"), units="rad")
prob.set_val("traj.phase0.states:v", phase0.interpolate(ys=[25600, 2500], nodes="state_input"), units="ft/s")
prob.set_val("traj.phase0.t_initial", 0, units="s")
prob.set_val("traj.phase0.t_duration", 2000, units="s")
prob.set_val("traj.phase0.controls:alpha", phase0.interpolate(ys=[17.4*np.pi/180, 17.4*np.pi/180], nodes="control_input"), units="rad")
prob.set_val("traj.phase0.controls:beta", phase0.interpolate(ys=[-75*np.pi/180, 0*np.pi/180], nodes="control_input"), units="rad")

# Run the driver
prob.run_driver()

# Output the results
print("\nTotal time is: ", prob.get_val("traj.phase0.timeseries.time")[-1], "s\n")
print("Final crossrange is: ", prob.get_val("traj.phase0.timeseries.states:theta")[-1]*180/np.pi, "degrees\n")

# Run the simulation to check if the model is physically valid
sim_out = traj.simulate()

# Plot the results
plt.figure(0)
plt.plot(prob.get_val("traj.phase0.timeseries.time", units="s"), prob.get_val("traj.phase0.timeseries.controls:alpha", units="deg"), "ro", label="Solution")
plt.plot(sim_out.get_val("traj.phase0.timeseries.time", units="s"), sim_out.get_val("traj.phase0.timeseries.controls:alpha", units="deg"), "b-", label="Simulation")
plt.title("Angle of Attack over Time")
plt.xlabel("Time (s)")
plt.ylabel("Angle of Attack (degrees)")
plt.legend()

plt.figure(1)
plt.plot(prob.get_val("traj.phase0.timeseries.time", units="s"), prob.get_val("traj.phase0.timeseries.controls:beta", units="deg"), "ro", label="Solution")
plt.plot(sim_out.get_val("traj.phase0.timeseries.time", units="s"), sim_out.get_val("traj.phase0.timeseries.controls:beta", units="deg"), "b-", label="Simulation")
plt.title("Bank Angle over Time")
plt.xlabel("Time points")
plt.ylabel("Bank Angle (degrees)")
plt.legend()

plt.figure(2)
plt.plot(prob.get_val("traj.phase0.timeseries.time", units="s"), prob.get_val("traj.phase0.timeseries.states:h", units="ft"), "ro", label="Solution")
plt.plot(sim_out.get_val("traj.phase0.timeseries.time", units="s"), sim_out.get_val("traj.phase0.timeseries.states:h", units="ft"), "b-", label="Simulation")
plt.title("Altitude over Time")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (feet)")
plt.legend()

plt.figure(3)
plt.plot(prob.get_val("traj.phase0.timeseries.time", units="s"), prob.get_val("traj.phase0.timeseries.states:gamma", units="deg"), "ro", label="Solution")
plt.plot(sim_out.get_val("traj.phase0.timeseries.time", units="s"), sim_out.get_val("traj.phase0.timeseries.states:gamma", units="deg"), "b-", label="Simulation")
plt.title("Flight Path Angle over Time")
plt.xlabel("Time (s)")
plt.ylabel("Flight Path Angle (degrees)")
plt.legend()

plt.figure(4)
plt.plot(prob.get_val("traj.phase0.timeseries.time", units="s"), prob.get_val("traj.phase0.timeseries.states:phi", units="deg"), "ro", label="Solution")
plt.plot(sim_out.get_val("traj.phase0.timeseries.time", units="s"), sim_out.get_val("traj.phase0.timeseries.states:phi", units="deg"), "b-", label="Simulation")
plt.title("Longitude over Time")
plt.xlabel("Time (s)")
plt.ylabel("Longitudinal Angle (degrees)")
plt.legend()

plt.figure(5)
plt.plot(prob.get_val("traj.phase0.timeseries.time", units="s"), prob.get_val("traj.phase0.timeseries.states:psi", units="deg"), "ro", label="Solution")
plt.plot(sim_out.get_val("traj.phase0.timeseries.time", units="s"), sim_out.get_val("traj.phase0.timeseries.states:psi", units="deg"), "b-", label="Simulation")
plt.title("Azimuth over Time")
plt.xlabel("Time (s)")
plt.ylabel("Azimuthal Angle (degrees)")
plt.legend()

plt.figure(6)
plt.plot(prob.get_val("traj.phase0.timeseries.time", units="s"), prob.get_val("traj.phase0.timeseries.states:theta", units="deg"), "ro", label="Solution")
plt.plot(sim_out.get_val("traj.phase0.timeseries.time", units="s"), sim_out.get_val("traj.phase0.timeseries.states:theta", units="deg"), "b-", label="Simulation")
plt.title("Latitude over Time")
plt.xlabel("Time (s)")
plt.ylabel("Latitudinal Angle (degrees)")
plt.legend()

plt.figure(7)
plt.plot(prob.get_val("traj.phase0.timeseries.time", units="s"), prob.get_val("traj.phase0.timeseries.states:v"), "ro", label="Solution")
plt.plot(sim_out.get_val("traj.phase0.timeseries.time", units="s"), sim_out.get_val("traj.phase0.timeseries.states:v"), "b-", label="Simulation")
plt.title("Velocity over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (ft/s)")
plt.legend()

plt.figure(8)
plt.plot(prob.get_val("traj.phase0.timeseries.time", units="s"),
         prob.get_val("traj.phase0.timeseries.q"), "ro", label="Solution")
plt.plot(sim_out.get_val("traj.phase0.timeseries.time", units="s"),
         sim_out.get_val("traj.phase0.timeseries.q"), "b-", label="Simulation")
plt.title("Heating rate over Time")
plt.xlabel("Time points")
plt.ylabel("Heating Rate (BTU/ft**2/s")

plt.show()