import os
import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from dymos.utils.doc_utils import save_for_docs
from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestVanderpolForDocs(unittest.TestCase):
    def tearDown(self):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out', 'SNOPT_summary.out']:
            if os.path.exists(filename):
                os.remove(filename)

    @save_for_docs
    def test_vanderpol_for_docs_simulation(self):
        from dymos.examples.plotting import plot_results
        from dymos.examples.vanderpol.vanderpol_dymos import vanderpol

        # Create the Dymos problem instance
        p = vanderpol(transcription='gauss-lobatto', num_segments=75)

        # Run the problem (simulate only)
        p.run_model()

        # check validity by using scipy.integrate.solve_ivp to integrate the solution
        exp_out = p.model.traj.simulate()

        # Display the results
        plot_results([('traj.phase0.timeseries.time',
                       'traj.phase0.timeseries.states:x1',
                       'time (s)',
                       'x1 (V)'),
                     ('traj.phase0.timeseries.time',
                      'traj.phase0.timeseries.states:x0',
                      'time (s)',
                      'x0 (V/s)'),
                      ('traj.phase0.timeseries.states:x0',
                       'traj.phase0.timeseries.states:x1',
                       'x0 vs x1',
                       'x0 vs x1'),
                     ('traj.phase0.timeseries.time',
                      'traj.phase0.timeseries.controls:u',
                      'time (s)',
                      'control u'),
                      ],
                     title='Van Der Pol Simulation',
                     p_sol=p, p_sim=exp_out)

        plt.show()

    @save_for_docs
    def test_vanderpol_for_docs_optimize(self):
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.vanderpol.vanderpol_dymos import vanderpol

        # Create the Dymos problem instance
        p = vanderpol(transcription='gauss-lobatto', num_segments=75,
                      transcription_order=3, compressed=True, optimizer='SLSQP')

        # Find optimal control solution to stop oscillation
        dm.run_problem(p)

        # check validity by using scipy.integrate.solve_ivp to integrate the solution
        exp_out = p.model.traj.simulate()

        # Display the results
        plot_results([('traj.phase0.timeseries.time',
                       'traj.phase0.timeseries.states:x1',
                       'time (s)',
                       'x1 (V)'),
                     ('traj.phase0.timeseries.time',
                      'traj.phase0.timeseries.states:x0',
                      'time (s)',
                      'x0 (V/s)'),
                      ('traj.phase0.timeseries.states:x0',
                       'traj.phase0.timeseries.states:x1',
                       'x0 vs x1',
                       'x0 vs x1'),
                     ('traj.phase0.timeseries.time',
                      'traj.phase0.timeseries.controls:u',
                      'time (s)',
                      'control u'),
                      ],
                     title='Van Der Pol Optimization',
                     p_sol=p, p_sim=exp_out)

        plt.show()

    @save_for_docs
    def test_vanderpol_for_docs_optimize_refine(self):
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.vanderpol.vanderpol_dymos import vanderpol

        # Create the Dymos problem instance
        p = vanderpol(transcription='gauss-lobatto', num_segments=15,
                      transcription_order=3, compressed=True, optimizer='SLSQP')

        # Enable grid refinement and find optimal control solution to stop oscillation
        p.model.traj.phases.phase0.set_refine_options(refine=True)
        dm.run_problem(p, refine_iteration_limit=10)

        # check validity by using scipy.integrate.solve_ivp to integrate the solution
        exp_out = p.model.traj.simulate()

        # Display the results
        plot_results([('traj.phase0.timeseries.time',
                       'traj.phase0.timeseries.states:x1',
                       'time (s)',
                       'x1 (V)'),
                     ('traj.phase0.timeseries.time',
                      'traj.phase0.timeseries.states:x0',
                      'time (s)',
                      'x0 (V/s)'),
                      ('traj.phase0.timeseries.states:x0',
                       'traj.phase0.timeseries.states:x1',
                       'x0 vs x1',
                       'x0 vs x1'),
                     ('traj.phase0.timeseries.time',
                      'traj.phase0.timeseries.controls:u',
                      'time (s)',
                      'control u'),
                      ],
                     title='Van Der Pol Optimization with Grid Refinement',
                     p_sol=p, p_sim=exp_out)

        plt.show()
