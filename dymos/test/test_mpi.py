from __future__ import print_function, division, absolute_import

import matplotlib
import matplotlib.pyplot as plt

import copy
import time

import numpy as np

import openmdao.api as om
import dymos as dm


@dm.declare_time(units='s')
@dm.declare_state('x', rate_source='xdot', units='m')
@dm.declare_state('y', rate_source='ydot', units='m')
@dm.declare_state('v', rate_source='vdot', targets='v', units='m/s')
@dm.declare_parameter('theta', targets='theta', units='rad')
@dm.declare_parameter('g', units='m/s**2', targets='g')
class BrachComp(om.ExplicitComponent):

    def __init__(self, setup_delay=0.0, compute_delay=0.0, comp_partials_delay=0.0, **kwargs):
        super(BrachComp, self).__init__(**kwargs)
        self._setup_delay = setup_delay
        self._compute_delay = compute_delay
        self._comp_partials_delay = comp_partials_delay
        print("Delay for setup/compute/compute_partials =", setup_delay, compute_delay, comp_partials_delay)

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        if self._setup_delay > 0.0:
            time.sleep(self._setup_delay)

        nn = self.options['num_nodes']

        print(self.pathname, "num_nodes =", nn)

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

        self.add_input('g', val=9.80665 * np.ones(nn), desc='grav. acceleration', units='m/s/s')

        self.add_input('theta', val=np.zeros(nn), desc='angle of wire', units='rad')

        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s')

        self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s')

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2')

        self.add_output('check', val=np.zeros(nn), desc='check solution: v/sin(theta) = constant',
                        units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange)
        self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='check', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='check', wrt='theta', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        if self._compute_delay > 0.0:
            time.sleep(self._compute_delay)
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g * cos_theta
        outputs['xdot'] = v * sin_theta
        outputs['ydot'] = -v * cos_theta
        outputs['check'] = v / sin_theta

    def compute_partials(self, inputs, jacobian):
        if self._comp_partials_delay > 0.0:
            time.sleep(self._comp_partials_delay)
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        jacobian['vdot', 'g'] = cos_theta
        jacobian['vdot', 'theta'] = -g * sin_theta

        jacobian['xdot', 'v'] = sin_theta
        jacobian['xdot', 'theta'] = v * cos_theta

        jacobian['ydot', 'v'] = -cos_theta
        jacobian['ydot', 'theta'] = v * sin_theta

        jacobian['check', 'v'] = 1 / sin_theta
        jacobian['check', 'theta'] = -v * cos_theta / sin_theta**2


@dm.declare_time(units='s')
@dm.declare_state('x', rate_source='xdot', units='m')
@dm.declare_state('y', rate_source='ydot', units='m')
@dm.declare_state('v', rate_source='vdot', targets='v', units='m/s')
@dm.declare_parameter('theta', targets='theta', units='rad')
@dm.declare_parameter('g', units='m/s**2', targets='g')
class BrachGroup(om.Group):
    """
    An ODE class used for testing of MPI issues in dymos.
    """

    def __init__(self, setup_delay=0.0, compute_delay=0.0, comp_partials_delay=0.0, **kwargs):
        super(BrachGroup, self).__init__(**kwargs)
        self._setup_delay = setup_delay
        self._compute_delay = compute_delay
        self._comp_partials_delay = comp_partials_delay
        print("Delay for setup/compute/compute_partials =", setup_delay, compute_delay, comp_partials_delay)

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):

        nn = self.options['num_nodes']
        print(self.pathname, "num_nodes =", nn)

        demux_comp = self.add_subsystem('demux', om.DemuxComp(vec_size=nn),
                                        promotes_inputs=['v','g','theta'])

        demux_comp.add_var('v', val=0.0, shape=(nn,), units='m/s')
        demux_comp.add_var('g', val=9.80665, shape=(nn,), units='m/s/s')
        demux_comp.add_var('theta', val=0.0, shape=(nn,), units='rad')

        points = self.add_subsystem('points', om.ParallelGroup(), promotes=['*'])

        for i in range(nn):
            point_name = f'point_{i}'

            points.add_subsystem(point_name,
                                 BrachComp(setup_delay=self._setup_delay,
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

        # points.linear_solver = om.LinearBlockJac()
        # points.nonlinear_solver = om.NonlinearBlockJac()


def brachistochrone_min_time(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                             compressed=False, optimizer='SLSQP', ode=BrachComp,
                             ode_kwargs={}):
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

    phase = dm.Phase(ode_class=ode, transcription=t, ode_init_kwargs=ode_kwargs)

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

    profile = os.environ.get('PROFILE', '')

    ode = os.environ.get('ODE', 'BrachComp')
    kwargs = {
        'setup_delay': 1e-3,
        'compute_delay': 1e-3,
        'comp_partials_delay': 1e-3,
    }

    # num_nodes
    with profiling(profile + '_%d.out' % rank) if profile else do_nothing_context():
        p = brachistochrone_min_time(transcription='gauss-lobatto', num_segments=10,
                                     transcription_order=3,
                                     compressed=False, optimizer='SNOPT',
                                     ode=globals()[ode], ode_kwargs=kwargs)

    # Plot results
    do_plots = int(os.environ.get('PLOT', 0))
    if do_plots:
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
