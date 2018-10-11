"""
Define the NonlinearRK class.

Custom nonlinear solver that time-steps through the integration of an ODE using rk4
"""
from collections import OrderedDict

from six import iteritems

import numpy as np

from openmdao.core.system import System
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.solver import NonlinearSolver
from openmdao.utils.mpi import multi_proc_fail_check

from dymos.utils.simulation import ScipyODEIntegrator
from dymos.utils.simulation.components.state_rate_collector_comp import StateRateCollectorComp
from dymos.utils.rk_methods import rk_methods
from dymos.utils.misc import get_rate_units


def _single_rk4_step(f, h, t0, y0, u, u_rate, u_rate2):
    """takes a single RK4 step in time"""
    size_y = len(y0)
    K_ = np.zeros((size_y, 4))

    if u.size > 0:
        K0 = K_[:, 0] = h*f(t0, y0, u[0, ...], u_rate[..., 0], u_rate2[..., 0])
        K1 = K_[:, 1] = h*f(t0 + h / 2., y0 + K0 / 2, u[..., 1], u_rate[..., 1], u_rate2[..., 1])
        K2 = K_[:, 2] = h*f(t0 + h / 2., y0 + K1 / 2, u[..., 2], u_rate[..., 2], u_rate2[..., 2])
        K_[:, 3] = h*f(t0 + h, y0 + K2, u[..., 3], u_rate[..., 3], u_rate2[..., 3])
    else:
        K0 = K_[:, 0] = h*f(t0, y0, None, None, None)
        K1 = K_[:, 1] = h*f(t0 + h / 2., y0 + K0 / 2, None, None, None)
        K2 = K_[:, 2] = h*f(t0 + h / 2., y0 + K1 / 2, None, None, None)
        K_[:, 3] = h*f(t0 + h, y0 + K2, None, None, None)

    next_y = y0 + np.sum(K_*np.array([1, 2, 2, 1]))/6.
    return next_y, K_


class NonlinearRK(NonlinearSolver):

    SOLVER = 'NL: RK'

    def _setup_ode_interface(self):
        self.ode_interface = Problem(model=Group())

        ops = self._system.options

        ode_class = ops['ode_class']
        time_options = ops['time_options']
        state_options = ops['state_options']
        control_options = ops['control_options']
        design_parameter_options = ops['design_parameter_options']
        input_parameter_options = ops['input_parameter_options']
        ode_init_kwargs = ops['ode_init_kwargs']
        num_stages = 4

        self.ode_options = ode_class.ode_options

        # Get the state vector.  This isn't necessarily ordered
        # so just pick the default ordering and go with it.
        self.state_options = OrderedDict()
        self.time_options = time_options
        self.control_options = OrderedDict()
        self.design_parameter_options = design_parameter_options
        self.input_parameter_options = input_parameter_options

        # Make a duplicate copy of state options that includes the position of each state
        # in the state vector
        pos = 0
        for state, options in iteritems(state_options):
            self.state_options[state] = {'rate_source': options['rate_source'],
                                         'pos': pos,
                                         'shape': options['shape'],
                                         'size': np.prod(options['shape']),
                                         'units': options['units'],
                                         'targets': options['targets']}
            pos += self.state_options[state]['size']

        self._state_vec = np.zeros(pos, dtype=float)
        self._state_rate_vec = np.zeros(pos, dtype=float)

        # Make a duplicate copy of the control options that includes the position of each control
        # in the control vector
        pos = 0
        for control_name, options in iteritems(control_options):
            self.control_options[control_name] = {'pos': pos,
                                                  'shape': options['shape'],
                                                  'size': np.prod(options['shape']),
                                                  'units': options['units'],
                                                  'targets': options['targets'],
                                                  'rate_param': options['rate_param'],
                                                  'rate2_param': options['rate2_param']}
            pos += self.control_options[control_name]['size']

        self._control_vec = np.zeros((pos, num_stages), dtype=float)
        self._control_rate_vec = np.zeros((pos, num_stages), dtype=float)
        self._control_rate2_vec = np.zeros((pos, num_stages), dtype=float)

        # The Time Comp
        ode_time_units = self.ode_options._time_options['units']
        ode_time_targets = self.ode_options._time_options['targets']
        self.ode_interface.model.add_subsystem('time_input',
                                               IndepVarComp('time', val=0.0, units=ode_time_units),
                                               promotes_outputs=['time'])

        if ode_time_targets is not None:
            self.ode_interface.model.connect('time', ['ode.{0}'.format(tgt) for tgt in ode_time_targets])

        # The States Comp
        indep = IndepVarComp()
        for name, options in iteritems(self.state_options):
            indep.add_output('states:{0}'.format(name),
                             shape=(1, np.prod(options['shape'])),
                             units=options['units'])
            if options['targets'] is not None:
                self.ode_interface.model.connect('states:{0}'.format(name),
                                                 ['ode.{0}'.format(tgt) for tgt in options['targets']])
            self.ode_interface.model.connect('ode.{0}'.format(options['rate_source']),
                                    'state_rate_collector.state_rates_in:{0}_rate'.format(name))

        self.ode_interface.model.add_subsystem('indep_states', subsys=indep,
                                               promotes_outputs=['*'])

        # The Controls comp
        if self.control_options:
            indep = IndepVarComp()
            for name, options in iteritems(self.control_options):
                ode_targets = self.ode_options._parameters[name]['targets']
                indep.add_output('control_values:{0}'.format(name),
                                 shape=(1, np.prod(options['shape'])),
                                 units=options['units'])
                indep.add_output('control_rates:{0}_rate'.format(name),
                                 shape=(1, np.prod(options['shape'])),
                                 units=get_rate_units(options['units'], time_options['units']))
                indep.add_output('control_rates:{0}_rate2'.format(name),
                                 shape=(1, np.prod(options['shape'])),
                                 units=get_rate_units(options['units'], time_options['units'],
                                                      deriv=2))
                if ode_targets:
                    self.ode_interface.model.connect('control_values:{0}'.format(name),
                                                     ['ode.{0}'.format(tgt) for tgt in ode_targets])
                if options['rate_param']:
                    ode_targets = self.ode_options._parameters[options['rate_param']]['targets']
                    self.ode_interface.model.connect('control_rates:{0}_rate'.format(name),
                                                     ['ode.{0}'.format(tgt) for tgt in ode_targets])
                if options['rate2_param']:
                    ode_targets = self.ode_options._parameters[options['rate2_param']]['targets']
                    self.ode_interface.model.connect('control_rates:{0}_rate2'.format(name),
                                                     ['ode.{0}'.format(tgt) for tgt in ode_targets])

            self.ode_interface.model.add_subsystem('controls_comp', subsys=indep,
                                                   promotes_outputs=['*'])

        # The Design parameters comp
        if self.design_parameter_options:
            indep = IndepVarComp()
            for name, options in iteritems(self.design_parameter_options):
                ode_targets = self.ode_options._parameters[name]['targets']
                indep.add_output('design_parameters:{0}'.format(name),
                                 shape=(1, np.prod(options['shape'])),
                                 units=options['units'])
                if ode_targets:
                    self.ode_interface.model.connect('design_parameters:{0}'.format(name),
                                                     ['ode.{0}'.format(tgt) for tgt in ode_targets])
            self.ode_interface.model.add_subsystem('input_param_comp', subsys=indep,
                                                   promotes_outputs=['*'])

        # The Input parameters comp
        if self.input_parameter_options:
            indep = IndepVarComp()
            for name, options in iteritems(self.input_parameter_options):
                ode_targets = self.ode_options._parameters[name]['targets']
                indep.add_output('input_parameters:{0}'.format(name),
                                 shape=(1, np.prod(options['shape'])),
                                 units=options['units'])
                if ode_targets:
                    self.ode_interface.model.connect('input_parameters:{0}'.format(name),
                                                     ['ode.{0}'.format(tgt) for tgt in ode_targets])
            self.ode_interface.model.add_subsystem('input_param_comp', subsys=indep,
                                                   promotes_outputs=['*'])

        # The ODE system
        self.ode_interface.model.add_subsystem('ode', subsys=ode_class(num_nodes=1, **ode_init_kwargs))

        # The state rate collector comp
        self.ode_interface.model.add_subsystem('state_rate_collector',
                                               StateRateCollectorComp(state_options=self.state_options,
                                                                 time_units=time_options['units']))

        # Setup the ode_interface
        self.ode_interface.setup(check=False)

    def _f_ode(self, t, x, u, u_rate, u_rate2):
        """
        The function interface used by _single_rk_step

        Parameters
        ----------
        t : float
            The current time, t.
        x : array_like
            The flattened state vector at time t.
        u : array_like
            The flattened control vector at time t.
        u_rate : array_like
            The flattened control rate vector at time t.
        u_rate2 : array_like
            The flattened control rate2 (second derivative) vector at time t.

        Returns
        -------
        xdot : np.array
            The 1D vector of state time-derivatives.

        """
        self.ode_interface['time'] = t

        # Unpack the state vector and set the values in the ode_interface accordingly
        for state_name, state_options in iteritems(self.state_options):
            pos = state_options['pos']
            size = state_options['size']
            units = state_options['units']
            self.ode_interface.set_val('states:{0}'.format(state_name), x[pos:pos + size], units=units)

        # Unpack the control vector and set the values in the ode_interface accordingly
        for control_name, control_options in iteritems(self.control_options):
            pos = control_options['pos']
            size = control_options['size']
            units = control_options['units']
            # print(u_rate[pos:pos + size])
            # print(u_rate2[pos:pos + size])
            self.ode_interface.set_val('control_values:{0}'.format(control_name), u[pos:pos + size], units=units)
            self.ode_interface.set_val('control_rates:{0}_rate'.format(control_name), u_rate[pos:pos + size], units=get_rate_units(units, self.time_options['units']))
            self.ode_interface.set_val('control_rates:{0}_rate2'.format(control_name), u_rate2[pos:pos + size], units=get_rate_units(units, self.time_options['units'], deriv=2))
            # print(t, u[pos:pos + size], self.ode_interface.get_val('control_values:{0}'.format(control_name)))

        # Execute the ODE interface to compute the rates
        self.ode_interface.run_model()

        # Pack the state rate vector from the outputs of ode_interface
        for state_name, state_options in iteritems(self.state_options):
            pos = state_options['pos']
            size = state_options['size']
            self._state_rate_vec[pos:pos + size] = \
                np.ravel(self.ode_interface['state_rate_collector.'
                                            'state_rates:{0}_rate'.format(state_name)])
        return self._state_rate_vec

    def _setup_solvers(self, system, depth):
        super(NonlinearRK, self)._setup_solvers(system, depth)

        self._setup_ode_interface()

    def solve(self):
        """
        Run the solver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        system = self._system

        for isub, subsys in enumerate(system._subsystems_myproc):
            system._transfer('nonlinear', 'fwd', isub)
            subsys._solve_nonlinear()

        method = system.options['method']
        num_stages = rk_methods[method]['num_stages']
        ode_options = system.options['ode_class'].ode_options

        f = self._f_ode

        num_steps = system.options['num_steps']
        seg_t0, seg_tf = system._inputs['seg_t0_tf']
        h = ((seg_tf - seg_t0) / num_steps) * np.ones(num_steps)
        t = np.linspace(seg_t0, seg_tf, num_steps + 1)

        y_i = self._state_vec
        u_i = self._control_vec
        ui_rate = self._control_rate_vec
        ui_rate2 = self._control_rate2_vec
        y0 = {}
        y = {}

        for state_name, options in iteritems(self.state_options):
            y0[state_name] = system._inputs['initial_states:{0}'.format(state_name)]
            y[state_name] = system._outputs['step_states:{0}'.format(state_name)]
            y[state_name][0, ...] = y0[state_name]

        for param_name, options in iteritems(self.design_parameter_options):
            units = options['units']

            if param_name in ode_options._parameters:
                ode_tgt = ode_options._parameters[param_name]['targets'][0]
                val = system._inputs['stage_ode.{0}'.format(ode_tgt)][0, ...]
                self.ode_interface.set_val('design_parameters:{0}'.format(param_name), val, units)

        for param_name, options in iteritems(self.input_parameter_options):
            units = options['units']
            val = system._inputs['input_parameters:{0}'.format(param_name)]
            self.ode_interface.set_val('input_parameters:{0}'.format(param_name), val, units)

        for i in range(num_steps):
            # Pack the state vector
            # y_i = self.ode_wrap._pack_state_vec(y, index=i)
            for state_name, state_options in iteritems(self.state_options):
                pos = state_options['pos']
                size = state_options['size']
                y_i[pos:pos + size] = np.ravel(y[state_name][i])

            # Pack the control vector
            for control_name, options in iteritems(self.control_options):
                pos = options['pos']
                size = options['size']
                units = options['units']
                val = system._outputs['stage_control_values:{0}'.format(control_name)][i, ...]
                u_i[pos: pos + size] = np.ravel(val)
                val = system._outputs['stage_control_rates:{0}_rate'.format(control_name)][i, ...]
                ui_rate[pos: pos + size] = np.ravel(val)
                val = system._outputs['stage_control_rates:{0}_rate2'.format(control_name)][i, ...]
                ui_rate2[pos: pos + size] = np.ravel(val)

            yn, Kn = _single_rk4_step(f, h[i], t[i], y_i, u_i, ui_rate, ui_rate2)

            # Unpack the output state vector and k vector
            for state_name, options in iteritems(self.state_options):
                pos = options['pos']
                size = options['size']
                y[state_name][i + 1, ...] = yn[pos:pos + size]
                system._outputs['k:{0}'.format(state_name)][i] = \
                    Kn[pos:pos + size].reshape((num_stages, size))

        # TODO: optionally check the residual values to ensure the RK was stable
        #       only do this optionally, because it will require an additional
        #       call to ODE which might be expensive

        #TODO: optionally have one more _solve_nonlinear to make sure the whole
        with Recording('NLRunOnce', 0, self) as rec:
            # If this is a parallel group, transfer all at once then run each subsystem.
            if len(system._subsystems_myproc) != len(system._subsystems_allprocs):
                system._transfer('nonlinear', 'fwd')

                with multi_proc_fail_check(system.comm):
                    for subsys in system._subsystems_myproc:
                        subsys._solve_nonlinear()

                system._check_reconf_update()

            # If this is not a parallel group, transfer for each subsystem just prior to running it.
            else:
                for isub, subsys in enumerate(system._subsystems_myproc):
                    system._transfer('nonlinear', 'fwd', isub)
                    subsys._solve_nonlinear()
                    system._check_reconf_update()

            system._apply_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0

        return False, 0.0, 0.0
