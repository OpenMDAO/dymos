import unittest

import numpy as np
import openmdao.api as om
from dymos.utils.doc_utils import save_for_docs
from openmdao.utils.testing_utils import use_tempdirs


class BrachistochroneArclengthODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')
        self.add_input('theta', val=np.zeros(nn), desc='angle of wire', units='rad')
        self.add_output('Sdot', val=np.zeros(nn), desc='rate of change of arclength', units='m/s')

        # Setup partials
        arange = np.arange(nn)

        self.declare_partials(of='Sdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='Sdot', wrt='theta', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        v = inputs['v']
        outputs['Sdot'] = np.sqrt(1.0 + (1.0/np.tan(theta))**2) * v * np.sin(theta)

    def compute_partials(self, inputs, jacobian):
        theta = inputs['theta']
        v = inputs['v']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        tan_theta = np.tan(theta)
        cot_theta = 1.0 / tan_theta
        csc_theta = 1.0 / sin_theta

        jacobian['Sdot', 'v'] = sin_theta * np.sqrt(1.0 + cot_theta**2)
        jacobian['Sdot', 'theta'] = v * (cos_theta * (cot_theta**2 + 1) - cot_theta * csc_theta) / \
            (np.sqrt(1 + cot_theta**2))


@use_tempdirs
class TestBrachistochroneTandemPhases(unittest.TestCase):

    @save_for_docs
    def test_brachistochrone_tandem_phases(self):
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        import numpy as np
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import openmdao.api as om
        import dymos as dm

        from openmdao.utils.assert_utils import assert_near_equal

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        # The transcription of the first phase
        tx0 = dm.GaussLobatto(num_segments=10, order=3, compressed=False)

        # The transcription for the second phase (and the secondary timeseries outputs from the first phase)
        tx1 = dm.Radau(num_segments=20, order=9, compressed=False)

        #
        # First Phase: Integrate the standard brachistochrone ODE
        #
        phase0 = dm.Phase(ode_class=BrachistochroneODE, transcription=tx0)

        p.model.add_subsystem('phase0', phase0)

        phase0.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase0.add_state('x', fix_initial=True, fix_final=False)

        phase0.add_state('y', fix_initial=True, fix_final=False)

        phase0.add_state('v', fix_initial=True, fix_final=False)

        phase0.add_control('theta', continuity=True, rate_continuity=True,
                           units='deg', lower=0.01, upper=179.9)

        phase0.add_parameter('g', units='m/s**2', val=9.80665)

        phase0.add_boundary_constraint('x', loc='final', equals=10)
        phase0.add_boundary_constraint('y', loc='final', equals=5)

        # Add alternative timeseries output to provide control inputs for the next phase
        phase0.add_timeseries('timeseries2', transcription=tx1, subset='control_input')

        #
        # Second Phase: Integration of ArcLength
        #
        phase1 = dm.Phase(ode_class=BrachistochroneArclengthODE, transcription=tx1)

        p.model.add_subsystem('phase1', phase1)

        phase1.set_time_options(fix_initial=True, input_duration=True)

        phase1.add_state('S', fix_initial=True, fix_final=False,
                         rate_source='Sdot', units='m')

        phase1.add_control('theta', opt=False, units='deg', targets='theta')
        phase1.add_control('v', opt=False, units='m/s', targets='v')

        #
        # Connect the two phases
        #
        p.model.connect('phase0.t_duration', 'phase1.t_duration')

        p.model.connect('phase0.timeseries2.controls:theta', 'phase1.controls:theta')
        p.model.connect('phase0.timeseries2.states:v', 'phase1.controls:v')

        # Minimize arclength at the end of the second phase
        phase1.add_objective('S', loc='final', ref=1)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase0.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase0.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase0.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase0.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.parameters:g'] = 9.80665

        p['phase1.states:S'] = 0.0

        dm.run_problem(p)

        expected = np.sqrt((10-0)**2 + (10 - 5)**2)
        assert_near_equal(p.get_val('phase1.timeseries.states:S')[-1], expected, tolerance=1.0E-3)

        fig, (ax0, ax1) = plt.subplots(2, 1)
        fig.tight_layout()
        ax0.plot(p.get_val('phase0.timeseries.states:x'), p.get_val('phase0.timeseries.states:y'), '.')
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')
        ax1.plot(p.get_val('phase1.timeseries.time'), p.get_val('phase1.timeseries.states:S'), '+')
        ax1.set_xlabel('t (s)')
        ax1.set_ylabel('S (m)')
        plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
