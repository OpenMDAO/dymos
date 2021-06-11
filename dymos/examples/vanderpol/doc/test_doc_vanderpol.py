import os
import unittest

from dymos.utils.doc_utils import save_for_docs
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.mpi import MPI


@use_tempdirs
class TestVanderpolForDocs(unittest.TestCase):
    def tearDown(self):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out', 'SNOPT_summary.out']:
            if os.path.exists(filename):
                os.remove(filename)

    @save_for_docs
    def test_vanderpol_for_docs_simulation(self):
        import dymos as dm
        from dymos.examples.vanderpol.vanderpol_dymos import vanderpol

        # Create the Dymos problem instance
        p = vanderpol(transcription='gauss-lobatto', num_segments=75)

        dm.run_problem(p, run_driver=False, simulate=True, make_plots=True)

    @save_for_docs
    def test_vanderpol_for_docs_optimize(self):
        import dymos as dm
        from dymos.examples.vanderpol.vanderpol_dymos import vanderpol

        # Create the Dymos problem instance
        p = vanderpol(transcription='gauss-lobatto', num_segments=75,
                      transcription_order=3, compressed=True, optimizer='SLSQP')

        dm.run_problem(p, simulate=True, make_plots=True)

    @save_for_docs
    def test_vanderpol_for_docs_optimize_refine(self):
        import dymos as dm
        from dymos.examples.vanderpol.vanderpol_dymos import vanderpol
        from openmdao.utils.assert_utils import assert_near_equal

        # Create the Dymos problem instance
        p = vanderpol(transcription='gauss-lobatto', num_segments=15,
                      transcription_order=3, compressed=True, optimizer='SLSQP')

        # Enable grid refinement and find optimal control solution to stop oscillation
        p.model.traj.phases.phase0.set_refine_options(refine=True)

        dm.run_problem(p, refine_iteration_limit=10, simulate=True, make_plots=True)

        assert_near_equal(p.get_val('traj.phase0.states:x0')[-1, ...], 0.0)
        assert_near_equal(p.get_val('traj.phase0.states:x1')[-1, ...], 0.0)
        assert_near_equal(p.get_val('traj.phase0.states:J')[-1, ...], 5.2808, tolerance=1.0E-3)
        assert_near_equal(p.get_val('traj.phase0.controls:u')[-1, ...], 0.0, tolerance=1.0E-3)


@unittest.skipUnless(MPI, "MPI is required.")
@save_for_docs
@use_tempdirs
class TestVanderpolDelayMPI(unittest.TestCase):
    N_PROCS = 2

    def test_vanderpol_delay_mpi(self):
        import openmdao.api as om
        import dymos as dm
        from dymos.examples.vanderpol.vanderpol_ode import VanderpolODE
        from openmdao.utils.assert_utils import assert_near_equal

        DELAY = 0.005

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        # define a Trajectory object and add to model
        traj = dm.Trajectory()
        p.model.add_subsystem('traj', subsys=traj)

        t = dm.Radau(num_segments=30, order=3)

        # define a Phase as specified above and add to Phase
        phase = dm.Phase(ode_class=VanderpolODE, transcription=t,
                         ode_init_kwargs={'delay': DELAY, 'distrib': True})
        traj.add_phase(name='phase0', phase=phase)

        t_final = 15
        phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=t_final, units='s')

        # set the State time options
        phase.add_state('x0', fix_initial=False, fix_final=False,
                        rate_source='x0dot',
                        units='V/s',
                        targets='x0')  # target required because x0 is an input
        phase.add_state('x1', fix_initial=False, fix_final=False,
                        rate_source='x1dot',
                        units='V',
                        targets='x1')  # target required because x1 is an input
        phase.add_state('J', fix_initial=False, fix_final=False,
                        rate_source='Jdot',
                        units=None)

        # define the control
        phase.add_control(name='u', units=None, lower=-0.75, upper=1.0, continuity=True,
                          rate_continuity=True,
                          targets='u')  # target required because u is an input

        # add constraints
        phase.add_boundary_constraint('x0', loc='initial', equals=1.0)
        phase.add_boundary_constraint('x1', loc='initial', equals=1.0)
        phase.add_boundary_constraint('J', loc='initial', equals=0.0)

        phase.add_boundary_constraint('x0', loc='final', equals=0.0)
        phase.add_boundary_constraint('x1', loc='final', equals=0.0)

        # define objective to minimize
        phase.add_objective('J', loc='final')

        # setup the problem
        p.setup(check=True)

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = t_final

        # add a linearly interpolated initial guess for the state and control curves
        p['traj.phase0.states:x0'] = phase.interp('x0', [1, 0])
        p['traj.phase0.states:x1'] = phase.interp('x1', [1, 0])
        p['traj.phase0.states:J'] = phase.interp('J', [0, 1])
        p['traj.phase0.controls:u'] = phase.interp('u', [-0.75, -0.75])

        dm.run_problem(p, run_driver=True, simulate=False)

        assert_near_equal(p.get_val('traj.phase0.states:x0')[-1, ...], 0.0)
        assert_near_equal(p.get_val('traj.phase0.states:x1')[-1, ...], 0.0)
        assert_near_equal(p.get_val('traj.phase0.states:J')[-1, ...], 5.2808, tolerance=1.0E-3)
        assert_near_equal(p.get_val('traj.phase0.controls:u')[-1, ...], 0.0, tolerance=1.0E-3)
