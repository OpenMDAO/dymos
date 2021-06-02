import warnings

import openmdao.api as om
from dymos.trajectory.trajectory import Trajectory
from dymos.load_case import load_case
from dymos.visualization.timeseries_plots import timeseries_plots

from .grid_refinement.refinement import _refine_iter


def run_problem(problem, refine_method='hp', refine_iteration_limit=0, run_driver=True,
                simulate=False, restart=None,
                solution_record_file='dymos_solution.db',
                simulation_record_file='dymos_simulation.db',
                make_plots=False,
                plot_dir="plots"
                ):
    """
    A Dymos-specific interface to execute an OpenMDAO problem containing Dymos Trajectories or
    Phases.  This function can iteratively call run_driver to perform grid refinement, and
    automatically call simulate following a run to check the validity of a result.

    Parameters
    ----------
    problem : om.Problem
        The OpenMDAO problem object to be run.
    refine_method : String
        The choice of refinement algorithm to use for grid refinement
    refine_iteration_limit : int
        The number of passes through the grid refinement algorithm to be made.
    run_driver : bool
        If True, run the driver (optimize the problem), otherwise just run the model one time.
    simulate : bool
        If True, perform a simulation of Trajectories found in the Problem after the driver
        has been run and grid refinement is complete.
    restart : str
        If not None, the path to a CaseRecorder file used to load in recording from a previous run.
    make_plots : bool
        If True, automatically generate plots of all timeseries outputs.
    solution_record_file : String
        Path to case recorder file use to store results from solution.
    simulation_record_file : String
        Path to case recorder file use to store results from simulation.
    plot_dir : str
        Path to directory for plot files.
    """
    if restart is not None:
        case = om.CaseReader(restart).get_case('final')

    if solution_record_file not in [rec._filepath for rec in iter(problem._rec_mgr)]:
        recorder = om.SqliteRecorder(solution_record_file)
        problem.add_recorder(recorder)
        # record_inputs is needed to capture potential input parameters that aren't connected
        problem.recording_options['record_inputs'] = True
        # record_outputs is need to capture the timeseries outputs
        problem.recording_options['record_outputs'] = True

    problem.final_setup()

    if restart is not None:
        load_case(problem, case)

    if run_driver:
        problem.run_driver()
        _refine_iter(problem, refine_iteration_limit, refine_method)
    else:
        problem.run_model()
        if refine_iteration_limit > 0:
            warnings.warn("Refinement not performed. Set run_driver to True to perform refinement.")

    problem.record('final')  # save case for potential restart
    problem.cleanup()

    if simulate:
        for subsys in problem.model.system_iter(include_self=True, recurse=True):
            if isinstance(subsys, Trajectory):
                subsys.simulate(record_file=simulation_record_file)

    if make_plots:
        if simulate:
            timeseries_plots(solution_record_file, simulation_record_file=simulation_record_file,
                             plot_dir=plot_dir)
        else:
            timeseries_plots(solution_record_file, simulation_record_file=None,
                             plot_dir=plot_dir)
