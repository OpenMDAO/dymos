"""
Define the NonlinearRK class.

Custom nonlinear solver that time-steps through the integration of an ODE using rk4
"""
from collections import OrderedDict

from six import iteritems

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.solver import NonlinearSolver
from openmdao.utils.mpi import multi_proc_fail_check

from dymos.phases.simulation.state_rate_collector_comp import StateRateCollectorComp
from dymos.utils.rk_methods import rk_methods
from dymos.utils.misc import get_rate_units


def _single_rk4_step2(ode_interface, h, init_time, init_states, controls, control_rates,
                      control_rates2):
    """
    Using the given ODE interface to the ODE system, take a single step in time of step length h

    Parameters
    ----------
    ode_interface
    h
    init_time
    init_states
    controls
    control_rates
    control_rates2

    Returns
    -------
    """
    num_stages = rk_methods['rk4']['num_stages']
    c = rk_methods['rk4']['c']
    A = rk_methods['rk4']['A']
    b = rk_methods['rk4']['b']

    k_stage = {}
    y_stage = {}
    ydot = {}
    yf = {}
    for name, val in iteritems(init_states):
        k_stage[name] = np.zeros([num_stages] + (list(val.shape)))
        y_stage[name] = np.zeros([num_stages] + (list(val.shape)))
        ydot[name] = np.zeros([num_stages] + (list(val.shape)))
        yf[name] = np.zeros_like(val)

    for istage in range(num_stages):

        # Set time
        t = init_time + h * c[istage]
        ode_interface.set_val('time', t)

        # Set the state values in the ode_interface accordingly
        for state_name in y_stage:
            y_stage[state_name][istage, ...] = init_states[state_name]
            y_stage[state_name][...] += np.dot(A, k_stage[state_name])
            ode_interface.set_val('states:{0}'.format(state_name), y_stage[state_name][istage, ...])

        # Unpack the control vector and set the values in the ode_interface accordingly
        for control_name, val in iteritems(controls):
            ode_interface.set_val('control_values:{0}'.format(control_name),
                                  val[istage, ...])
            ode_interface.set_val('control_rates:{0}_rate'.format(control_name),
                                  control_rates[control_name][istage, ...])
            ode_interface.set_val('control_rates:{0}_rate2'.format(control_name),
                                  control_rates2[control_name][istage, ...])

        # Run the ODE
        ode_interface.run_model()

        # Extract state rates and compute k for each state, and advance each state
        rate_template = 'state_rate_collector.state_rates:{0}_rate'
        for name in init_states:
            # Pack the state rate vector from the outputs of ode_interface
            ydot[name][istage, ...] = ode_interface[rate_template.format(name)]
            k_stage[name][istage, ...] = h * ydot[name][istage, ...]

    # Advance states to the end of the step
    for name in init_states:
        yf[name] = init_states[name] + np.dot(b, k_stage[name])

    return k_stage, yf


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
        ode_iface = self.ode_interface

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
        ode_iface.model.add_subsystem('time_input',
                                      IndepVarComp('time', val=0.0, units=ode_time_units),
                                      promotes_outputs=['time'])

        if ode_time_targets is not None:
            ode_iface.model.connect('time', ['ode.{0}'.format(tgt) for tgt in ode_time_targets])

        # The States Comp
        indep = IndepVarComp()
        for name, options in iteritems(self.state_options):
            indep.add_output('states:{0}'.format(name),
                             shape=(1, np.prod(options['shape'])),
                             units=options['units'])
            if options['targets'] is not None:
                ode_iface.model.connect('states:{0}'.format(name),
                                        ['ode.{0}'.format(tgt) for tgt in options['targets']])

                rate_path = self._get_rate_source_path(name)
                ode_iface.model.connect(rate_path,
                                        'state_rate_collector.state_rates_in:{0}_rate'.format(name))

        ode_iface.model.add_subsystem('indep_states', subsys=indep, promotes_outputs=['*'])

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
                    ode_iface.model.connect('control_values:{0}'.format(name),
                                            ['ode.{0}'.format(tgt) for tgt in ode_targets])
                if options['rate_param']:
                    ode_targets = self.ode_options._parameters[options['rate_param']]['targets']
                    ode_iface.model.connect('control_rates:{0}_rate'.format(name),
                                            ['ode.{0}'.format(tgt) for tgt in ode_targets])
                if options['rate2_param']:
                    ode_targets = self.ode_options._parameters[options['rate2_param']]['targets']
                    ode_iface.model.connect('control_rates:{0}_rate2'.format(name),
                                            ['ode.{0}'.format(tgt) for tgt in ode_targets])

            ode_iface.model.add_subsystem('controls_comp', subsys=indep, promotes_outputs=['*'])

        # The Design parameters comp
        if self.design_parameter_options:
            indep = IndepVarComp()
            for name, options in iteritems(self.design_parameter_options):
                ode_targets = self.ode_options._parameters[name]['targets']
                indep.add_output('design_parameters:{0}'.format(name),
                                 shape=(1, np.prod(options['shape'])),
                                 units=options['units'])
                if ode_targets:
                    ode_iface.model.connect('design_parameters:{0}'.format(name),
                                            ['ode.{0}'.format(tgt) for tgt in ode_targets])
            ode_iface.model.add_subsystem('design_param_comp', subsys=indep, promotes_outputs=['*'])

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
            ode_iface.model.add_subsystem('input_param_comp', subsys=indep, promotes_outputs=['*'])

        # The ODE system
        ode_iface.model.add_subsystem('ode', subsys=ode_class(num_nodes=1, **ode_init_kwargs))

        # The state rate collector comp
        state_rate_col_comp = StateRateCollectorComp(state_options=self.state_options,
                                                     time_units=time_options['units'])
        self.ode_interface.model.add_subsystem('state_rate_collector', subsys=state_rate_col_comp)

        # Setup the ode_interface
        self.ode_interface.setup(check=False)

    def _setup_solvers(self, system, depth):
        super(NonlinearRK, self)._setup_solvers(system, depth)

        self._setup_ode_interface()

    def _get_rate_source_path(self, state_var):
        var = self.state_options[state_var]['rate_source']

        if var == 'time':
            rate_path = 'time'
        elif var == 'time_phase':
            var_type = 'time_phase'
        elif var in self.state_options:
            rate_path = 'state'
        elif var in self.control_options:
            rate_path = 'control_values:{0}'.format(var)
        elif var in self.design_parameter_options:
            rate_path = 'design_parameters:{0}'.format(var)
        elif var in self.input_parameter_options:
            rate_path = 'input_parameters:{0}'.format(var)
        elif var.endswith('_rate'):
            if var[:-5] in self.control_options:
                rate_path = 'control_rates:{0}'.format(var)
        elif var.endswith('_rate2'):
            if var[:-6] in self.control_options:
                rate_path = 'control_rates:{0}'.format(var)
        else:
            rate_path = 'ode.{0}'.format(var)

        return rate_path

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

        ode_options = system.options['ode_class'].ode_options

        num_steps = system.options['num_steps']
        seg_t0, seg_tf = system._inputs['seg_t0_tf']
        h = ((seg_tf - seg_t0) / num_steps) * np.ones(num_steps)
        t = np.linspace(seg_t0, seg_tf, num_steps + 1)
        y0 = {}
        u = {}
        u_rate = {}
        u_rate2 = {}

        for state_name, options in iteritems(self.state_options):
            y0[state_name] = system._inputs['initial_states:{0}'.format(state_name)]
            system._outputs['step_states:{0}'.format(state_name)][0, ...] = y0[state_name]

        # Set the design parameter values, only need to do this once for the entire segment
        for param_name, options in iteritems(self.design_parameter_options):
            units = options['units']
            if param_name in ode_options._parameters:
                ode_tgt = ode_options._parameters[param_name]['targets'][0]
                val = system._inputs['stage_ode.{0}'.format(ode_tgt)][0, ...]
                self.ode_interface.set_val('design_parameters:{0}'.format(param_name), val, units)

        # Set the input parameter values, only need to do this once for the entire segment
        for param_name, options in iteritems(self.input_parameter_options):
            units = options['units']
            val = system._inputs['input_parameters:{0}'.format(param_name)]
            self.ode_interface.set_val('input_parameters:{0}'.format(param_name), val, units)

        for i in range(num_steps):
            # Pack the control vector
            for control_name, options in iteritems(self.control_options):
                u[control_name] = \
                    system._outputs['stage_control_values:{0}'.format(control_name)][i, ...]
                u_rate[control_name] = \
                    system._outputs['stage_control_rates:{0}_rate'.format(control_name)][i, ...]
                u_rate2[control_name] = \
                    system._outputs['stage_control_rates:{0}_rate2'.format(control_name)][i, ...]

            k_stage, yn = _single_rk4_step2(self.ode_interface, h[i], t[i], y0, u, u_rate, u_rate2)

            # Unpack the output state vector and k vector
            for state_name, options in iteritems(self.state_options):
                system._outputs['step_states:{0}'.format(state_name)][i + 1, ...] = yn[state_name]
                y0[state_name] = yn[state_name]
                system._outputs['k:{0}'.format(state_name)][i, ...] = k_stage[state_name]

        # TODO: optionally check the residual values to ensure the RK was stable
        #       only do this optionally, because it will require an additional
        #       call to ODE which might be expensive

        # TODO: optionally have one more _solve_nonlinear to make sure the whole
        with Recording('NonlinearRK', 0, self) as rec:
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
