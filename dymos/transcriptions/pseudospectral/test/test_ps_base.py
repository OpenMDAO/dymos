import unittest

import numpy as np

import openmdao.api as om
import dymos as dm
from dymos import Trajectory, GaussLobatto, Phase, Radau


class crtbp_ode(om.ExplicitComponent):
    """
    ODE for the circular restricted three-body problem (CRTBP)
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('system', default='earth-moon', values=['earth-moon', 'sun-earth', 'jupiter-europa'],
                             desc='primary and secondary masses to be considered')

    def setup(self):
        nn = self.options['num_nodes']
        mu_val = 123.0

        self.add_input('mu', val=mu_val, desc='gravitational parameter for the specified CRTBP system')
        self.add_input('x', val=np.ones(nn), desc='x-position in rotating frame')
        self.add_input('y', val=np.ones(nn), desc='y-position in rotating frame')
        self.add_input('z', val=np.ones(nn), desc='z-position in rotating frame')
        self.add_input('x_dot', val=np.ones(nn), desc='x-velocity in rotating frame')
        self.add_input('y_dot', val=np.ones(nn), desc='y-velocity in rotating frame')
        self.add_input('z_dot', val=np.ones(nn), desc='z-velocity in rotating frame')

        self.add_output('vx', val=np.ones(nn), desc='computed velocity in the rotating frame')
        self.add_output('vy', val=np.ones(nn), desc='computed velocity in the rotating frame')
        self.add_output('vz', val=np.ones(nn), desc='computed velocity in the rotating frame')
        self.add_output('vx_dot', val=np.ones(nn), desc='computed acceleration in the rotating frame')
        self.add_output('vy_dot', val=np.ones(nn), desc='computed acceleration in the rotating frame')
        self.add_output('vz_dot', val=np.ones(nn), desc='computed acceleration in the rotating frame')


def make_problem(transcription=GaussLobatto, num_segments=10, order=3, compressed=True):
    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()

    traj = p.model.add_subsystem('traj', Trajectory())
    phase = traj.add_phase('phase', Phase(ode_class=crtbp_ode, transcription=transcription(num_segments=num_segments,
                                                                                           order=order,
                                                                                           compressed=compressed)))

    phase.set_time_options(fix_initial=True, fix_duration=True)
    phase.add_state('x', rate_source='vx', )
    phase.add_state('y', rate_source='vz', fix_initial=True)
    phase.add_state('z', rate_source='vz')
    phase.add_state('x_dot', rate_source='vx_dot', fix_initial=True, units=None)
    phase.add_state('y_dot', rate_source='vy_dot', units=None)
    phase.add_state('z_dot', rate_source='vz_dot', fix_initial=True, units=None)

    p.model.add_subsystem('x_periodic_bc', om.ExecComp('bc_defect=final-initial'))
    p.model.connect('traj.phase.timeseries.states:x', 'x_periodic_bc.initial', src_indices=0)
    p.model.connect('traj.phase.timeseries.states:x', 'x_periodic_bc.final', src_indices=-1)

    p.model.add_constraint('x_periodic_bc.bc_defect', equals=0)

    p.model.add_subsystem('z_periodic_bc', om.ExecComp('bc_defect=final-initial'))
    p.model.connect('traj.phase.timeseries.states:z', 'z_periodic_bc.initial', src_indices=0)
    p.model.connect('traj.phase.timeseries.states:z', 'z_periodic_bc.final', src_indices=-1)

    p.model.add_constraint('z_periodic_bc.bc_defect', equals=0)

    p.model.add_subsystem('vy_periodic_bc', om.ExecComp('bc_defect=final-initial'))
    p.model.connect('traj.phase.timeseries.states:y_dot', 'vy_periodic_bc.initial', src_indices=0)
    p.model.connect('traj.phase.timeseries.states:y_dot', 'vy_periodic_bc.final', src_indices=-1)

    p.model.add_constraint('vy_periodic_bc.bc_defect', equals=0)

    phase.add_objective('time', loc='final')

    p.setup(check=True)
    return p


class TestPseudospectralBase(unittest.TestCase):

    def test_add_state_units_none(self):
        p = make_problem(transcription=GaussLobatto, num_segments=10, order=3, compressed=True)
        p.run_model()

        io_meta = p.model.traj.phases.phase.timeseries.get_io_metadata(iotypes=('output'), get_remote=True)
        self.assertEqual(io_meta['states:x']['units'], None)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
