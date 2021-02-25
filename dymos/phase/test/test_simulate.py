import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm


class MainPhase(dm.Phase):

    def initialize(self):
        super(MainPhase, self).initialize()

    def setup(self):
        self.options['ode_class'] = TestODE
        super(MainPhase, self).setup()


class TestODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('Sink',
                           om.ExecComp('sink = chord[0]', sink={'value': 0.0, 'units': None},
                                       chord={'value': np.zeros(4), 'units': 'm'}),
                           promotes_inputs=['chord'])

        self.add_subsystem('calc', om.ExecComp('Out = Thrust * 2',
                                               Out={'value': np.zeros(nn), 'units': 'N'},
                                               Thrust={'value': np.zeros(nn), 'units': 'N'}),
                           promotes_inputs=['Thrust'],
                           promotes_outputs=['Out'])


@use_tempdirs
class TestSimulateShapedParams(unittest.TestCase):

    def test_shaped_params(self):

        main_tx = dm.Radau(num_segments=1, order=3, compressed=False)

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        des_vars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes_outputs=['*'])
        des_vars.add_output('chord', val=4 * np.ones(4), units='inch')

        hop0 = dm.Trajectory()
        p.model.add_subsystem('hop0', hop0)
        main_phase = hop0.add_phase(name='main_phase',
                                    phase=MainPhase(transcription=main_tx))

        main_phase.set_time_options(fix_initial=True, fix_duration=True, units='s')

        main_phase.add_parameter('chord', targets='chord', shape=(4,), units='inch',
                                 dynamic=False)
        p.model.connect('chord', 'hop0.main_phase.parameters:chord')

        main_phase.add_state('impulse', fix_initial=True, fix_final=False, units='N*s',
                             rate_source='Out',
                             solve_segments=False)

        main_phase.add_polynomial_control('Thrust', units='N',
                                          targets='Thrust',
                                          lower=-3450, upper=-500,
                                          order=5, opt=True)

        main_phase.add_objective('impulse', loc='final', ref=-1)

        p.setup(mode='auto', check=['unconnected_inputs'], force_alloc_complex=True)

        p['hop0.main_phase.t_initial'] = 0.0
        p['hop0.main_phase.t_duration'] = 10
        p['hop0.main_phase.polynomial_controls:Thrust'][:, 0] = -3400
        p['hop0.main_phase.states:impulse'] = main_phase.interpolate(ys=[0, 0], nodes='state_input')

        p.run_driver()

        try:
            hop0.simulate()
        except:
            self.fail('Simulate did not correctly complete.')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
