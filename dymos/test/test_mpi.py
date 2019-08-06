from __future__ import print_function, division, absolute_import

import matplotlib
import matplotlib.pyplot as plt

import copy
import time

import openmdao.api as om
import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class BrachistochroneODETestComp(BrachistochroneODE):

    def __init__(self, setup_delay=0.0, compute_delay=0.0, comp_partials_delay=0.0, **kwargs):
        super(BrachistochroneODETestComp, self).__init__(**kwargs)
        self._setup_delay = setup_delay
        self._compute_delay = compute_delay
        self._comp_partials_delay = comp_partials_delay

    def setup(self):
        if self._setup_delay > 0.0:
            time.sleep(self._setup_delay)
        super(BrachistochroneODETestComp, self).setup()

    def compute(self, inputs, outputs):
        if self._compute_delay > 0.0:
            time.sleep(self._compute_delay)
        super(BrachistochroneODETestComp, self).compute(inputs, outputs)

    def compute_partials(self, inputs, jacobian):
        if self._comp_partials_delay > 0.0:
            time.sleep(self._comp_partials_delay)
        super(BrachistochroneODETestComp, self).compute_partials(inputs, jacobian)


@dm.declare_time(units='s')
@dm.declare_state('x', rate_source='xdot', units='m')
@dm.declare_state('y', rate_source='ydot', units='m')
@dm.declare_state('v', rate_source='vdot', targets='v', units='m/s')
@dm.declare_parameter('theta', targets='theta', units='rad')
@dm.declare_parameter('g', units='m/s**2', targets='g')
class BrachistochroneGroup(om.Group):
    """
    An ODE class used for testing of MPI issues in dymos.
    """

    def __init__(self, setup_delay=0.0, compute_delay=0.0, comp_partials_delay=0.0, **kwargs):
        super(BrachistochroneGroup, self).__init__(**kwargs)
        self._setup_delay = setup_delay
        self._compute_delay = compute_delay
        self._comp_partials_delay = comp_partials_delay

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):

        nn = self.options['num_nodes']

        demux_comp = self.add_subsystem('demux', om.DemuxComp(vec_size=nn),
                                        promotes_inputs=['v','g','theta'])

        demux_comp.add_var('v', val=0.0, shape=(nn,), units='m/s')
        demux_comp.add_var('g', val=9.80665, shape=(nn,), units='m/s/s')
        demux_comp.add_var('theta', val=0.0, shape=(nn,), units='rad')

        points = self.add_subsystem('points', om.ParallelGroup(), promotes=['*'])

        for i in range(nn):
            point_name = f'point_{i}'

            points.add_subsystem(point_name,
                                 BrachistochroneODETestComp(setup_delay=self._setup_delay,
                                                            compute_delay=self._compute_delay,
                                                            comp_partials_delay=self._comp_partials_delay,
                                                            num_nodes=1))

            self.connect(f'demux.v_{i}', f'{point_name}.v')
            self.connect(f'demux.g_{i}', f'{point_name}.g')
            self.connect(f'demux.theta_{i}', f'{point_name}.theta')

            self.connect(f'{point_name}.xdot', f'mux.xdot_{i}')
            self.connect(f'{point_name}.ydot', f'mux.ydot_{i}')
            self.connect(f'{point_name}.vdot', f'mux.vdot_{i}')
            self.connect(f'{point_name}.check', f'mux.check_{i}')

        mux_comp = self.add_subsystem(name='mux', subsys=om.MuxComp(vec_size=nn),
                                      promotes_outputs=['xdot', 'ydot', 'vdot', 'check'])
        mux_comp.add_var('xdot', shape=(1,), units='m/s')
        mux_comp.add_var('ydot', shape=(1,), units='m/s')
        mux_comp.add_var('vdot', shape=(1,), units='m/s/s')
        mux_comp.add_var('check', shape=(1,), units='m/s')


def brachistochrone_min_time(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                             compressed=False, optimizer='SLSQP', show_plots=False):
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.declare_coloring()

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=num_segments,
                            order=transcription_order,
                            compressed=compressed)
    elif transcription == 'radau-ps':
        t = dm.Radau(num_segments=num_segments,
                     order=transcription_order,
                     compressed=compressed)
    elif transcription == 'runge-kutta':
        t = dm.RungeKutta(num_segments=num_segments,
                          order=transcription_order,
                          compressed=compressed)

    kwargs = {
        'setup_delay': 0.01,
        'compute_delay': 0.01,
        'comp_partials_delay': 0.01,
    }
    phase = dm.Phase(ode_class=BrachistochroneGroup, transcription=t, ode_init_kwargs=kwargs)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

    phase.set_state_options('x', fix_initial=True, fix_final=False, solve_segments=False, rate_source='xdot', units='m')
    phase.set_state_options('y', fix_initial=True, fix_final=False, solve_segments=False, rate_source='ydot', units='m')
    phase.set_state_options('v', fix_initial=True, fix_final=False, solve_segments=False, rate_source='vdot', units='m/s')

    phase.add_control('theta', continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_input_parameter('g', units='m/s**2', val=9.80665)

    phase.add_timeseries('timeseries2',
                         transcription=dm.Radau(num_segments=num_segments*5,
                                                order=transcription_order,
                                                compressed=compressed),
                         subset='control_input')

    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)
    # Minimize time at the end of the phase
    phase.add_objective('time_phase', loc='final', scaler=10)

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 2.0

    p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
    p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
    p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
    p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
    p['phase0.input_parameters:g'] = 9.80665

    p.run_driver()

    return p


if __name__ == "__main__":
    import os
    from openmdao.devtools.debug import profiling
    from openmdao.utils.general_utils import do_nothing_context
    from openmdao.utils.mpi import MPI

    if MPI:
        rank = MPI.COMM_WORLD.rank
    else:
        rank = 0

    do_profile = int(os.environ.get('PROFILE', 0))

    with profiling('prof_%d.out' % rank) if do_profile else do_nothing_context():
        p = brachistochrone_min_time(transcription='gauss-lobatto', num_segments=8,
                                     transcription_order=3,
                                     compressed=False, optimizer='SNOPT', show_plots=True)

    # Plot results
    if False:
        phase = p.model.phase0
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.time_phase')
        y_imp = p.get_val('phase0.timeseries.controls:theta')

        x_exp = exp_out.get_val('phase0.timeseries.time_phase')
        y_exp = exp_out.get_val('phase0.timeseries.controls:theta')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('time (s)')
        ax.set_ylabel('theta (rad)')
        ax.grid(True)
        ax.legend(loc='lower right')

        plt.show()
