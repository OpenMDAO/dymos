import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal

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
                           om.ExecComp('sink = chord[0] + chord[1] + chord[2] + chord[3]',
                                       sink={'val': 0.0, 'units': None},
                                       chord={'val': np.zeros(4), 'units': 'm',
                                              'tags': ['dymos.static_target']}),
                           promotes_inputs=['chord'])

        self.add_subsystem('calc', om.ExecComp('Out = Thrust * 2',
                                               Out={'val': np.zeros(nn), 'units': 'N'},
                                               Thrust={'val': np.zeros(nn), 'units': 'N'}),
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

        main_phase.add_parameter('chord', units='inch')
        p.model.connect('chord', 'hop0.main_phase.parameters:chord')

        main_phase.add_state('impulse', fix_initial=True, fix_final=False, units='N*s',
                             rate_source='Out',
                             solve_segments=False)

        main_phase.add_control('Thrust', units='N',
                               targets='Thrust',
                               lower=-3450, upper=-500,
                               order=5, opt=True, control_type='polynomial')

        main_phase.add_objective('impulse', loc='final', ref=-1)

        p.setup(mode='auto', check=['unconnected_inputs'], force_alloc_complex=True)

        p.set_val('hop0.main_phase.t_initial', 0.0)
        p.set_val('hop0.main_phase.t_duration', 10)
        p.set_val('hop0.main_phase.controls:Thrust', val=-3400, indices=om.slicer[:, 0])
        p.set_val('hop0.main_phase.states:impulse',  main_phase.interp('impulse', [0, 0]))

        p.run_driver()

        assert_near_equal(p.get_val('hop0.main_phase.timeseries.impulse')[-1, 0], -7836.66666, tolerance=1.0E-4)

        try:
            hop0.simulate()
        except Exception:
            self.fail('Simulate did not correctly complete.')

    def test_shaped_traj_params(self):

        main_tx = dm.Radau(num_segments=1, order=3, compressed=False)

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        hop0 = dm.Trajectory()
        p.model.add_subsystem('hop0', hop0)
        main_phase = hop0.add_phase(name='main_phase',
                                    phase=MainPhase(transcription=main_tx))

        main_phase.set_time_options(fix_initial=True, fix_duration=True, units='s')

        hop0.add_parameter('chord', units='inch', targets={'main_phase': ['chord']})

        main_phase.add_state('impulse', fix_initial=True, fix_final=False, units='N*s',
                             rate_source='Out',
                             solve_segments=False)

        main_phase.add_control('Thrust', units='N',
                               targets='Thrust',
                               lower=-3450, upper=-500,
                               order=5, opt=True, control_type='polynomial')

        main_phase.add_objective('impulse', loc='final', ref=-1)

        p.setup(mode='auto', check=['unconnected_inputs'], force_alloc_complex=True)

        p.set_val('hop0.main_phase.t_initial', 0.0)
        p.set_val('hop0.main_phase.t_duration', 10)
        p.set_val('hop0.main_phase.controls:Thrust', val=-3400, indices=om.slicer[:, 0])
        p.set_val('hop0.main_phase.states:impulse',  main_phase.interp('impulse', [0, 0]))

        p.run_driver()

        assert_near_equal(p.get_val('hop0.main_phase.timeseries.impulse')[-1, 0], -7836.66666, tolerance=1.0E-4)

        try:
            hop0.simulate()
        except Exception:
            self.fail('Simulate did not correctly complete.')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
