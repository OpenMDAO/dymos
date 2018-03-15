from __future__ import print_function, division, absolute_import

import sys
from collections import OrderedDict
from six import iteritems, string_types

import numpy as np
from scipy.integrate import ode

from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent, AnalysisError
from openmdao.utils.units import convert_units, valid_units

from ..utils.misc import get_rate_units


class ProgressBarObserver(object):
    """
    A simple observer that outputs a progress bar based on the current and
    final simulation time.

    Parameters
    ----------
    ode_integrator : ODEIntegrator
        The ODEIntegrator instance that will be calling this observer.
    out_stream : file-like
        The stream to which output will be written.
    """
    def __init__(self, ode_integrator, t0, tf, out_stream=sys.stdout):
        self._prob = ode_integrator.prob
        self.t0 = t0
        self.tf = tf
        self.out_stream = out_stream

    def __call__(self, t, y, prob):
        t0 = self.t0
        tf = self.tf
        print('Simulation time: {0:6.3f} of {1:6.3f} ({2:6.3f}%)'.format(t, tf, 100*(t-t0)/(tf-t0)),
              file=self.out_stream)


class StdOutObserver(object):
    """
    The default observer callable for use by RHSIntegrator.integrate_times.

    This observer provides column output of the model variables when called.

    Parameters
    ----------
    ode_integrator : ODEIntegrator
        The ODEIntegrator instance that will be calling this observer.
    out_stream : file-like
        The stream to which column-output will be written.
    """

    def __init__(self, ode_integrator, out_stream=sys.stdout):
        self._prob = ode_integrator.prob
        self.out_stream = out_stream
        self.fmt = ''
        self._first = True
        self._output_order = None

    def __call__(self, t, y, prob):
        out_stream = sys.stdout
        outputs = dict(self._prob.model.list_outputs(units=True, shape=True, out_stream=None))

        if self._first:

            for output_name in outputs:
                outputs[output_name]['prom_name'] = self._prob.model._var_abs2prom['output'][
                    output_name]

            output_order = list(outputs.keys())
            insert_at = 0

            # Move time to front of list
            output_order.insert(insert_at, output_order.pop(output_order.index('time_input.time')))
            insert_at += 1

            # Move states after time
            states = sorted([s for s in output_order if s.startswith('indep_states.states:')])
            for i, name in enumerate(states):
                output_order.insert(insert_at, output_order.pop(output_order.index(name)))
                insert_at += 1

            # Move controls and control rates after states
            controls = sorted([s for s in output_order if s.startswith('indep_controls.controls:')])
            for j, name in enumerate(controls):
                output_order.insert(insert_at, output_order.pop(output_order.index(name)))
                insert_at += 1

            control_rates = sorted(
                [s for s in output_order if s.startswith('indep_controls.control_rates:')])
            for k, name in enumerate(control_rates):
                output_order.insert(insert_at, output_order.pop(output_order.index(name)))
                insert_at += 1

            # Remove state_rate_collector since it is redundant
            output_order = [o for o in output_order if not o.startswith('state_rate_collector.')]

            self._output_order = output_order

            header_names = [outputs[o]['prom_name'] for o in output_order]
            header_units = [outputs[o]['units'] for o in output_order]

            max_width = max([len(outputs[o]['prom_name']) for o in outputs]) + 4
            header_fmt = ('{:>' + str(max_width) + 's}') * len(header_names)
            self.fmt = ('{:' + str(max_width) + '.6f}') * len(output_order)
            print(header_fmt.format(*header_names), file=out_stream)
            print(header_fmt.format(*header_units), file=out_stream)
            self._first = False
        vals = [np.ravel(self._prob[var])[0] for var in self._output_order]
        print(self.fmt.format(*vals), file=out_stream)


class SimulationResults(object):
    """
    SimulationResults is returned by phase.simulate.  It's primary
    purpose is to hold the dictionary of results from the integration
    and to provide a `get_values` interface that is equivalent to that
    in Phase (except that it has no knowledge of nodes).
    """
    def __init__(self, state_options, control_options):
        """

        Parameters
        ----------
        phase : dymos.Phase object
            The phase being simulated.  Phase is passed on initialization of
            SimulationResults so that it can gather knowledge of time units,
            state options, control options, and ODE outputs.
        """
        self.state_options = state_options
        self.control_options = control_options
        self.outputs = {}
        self.units = {}

    def get_values(self, var, units=None):

        if units is not None and not valid_units(units):
            raise ValueError('{0} is not a valid set of units.'.format(units))

        if var == 'time':
            output_path = 'time'

        elif var in self.state_options:
            output_path = 'states:{0}'.format(var)

        elif var in self.control_options and self.control_options[var]['opt']:
            output_path = 'controls:{0}'.format(var)

        elif var in self.control_options and not self.control_options[var]['opt']:
            # TODO: make a test for this, haven't experimented with this yet.
            output_path = 'controls:{0}'.format(var)

        elif var.endswith('_rate') and var[:-5] in self.control_options:
            output_path = 'control_rates:{0}'.format(var)

        elif var.endswith('_rate2') and var[:-6] in self.control_options:
            output_path = 'control_rates:{0}'.format(var)

        else:
            output_path = 'ode.{0}'.format(var)

        output = convert_units(self.outputs[output_path]['value'],
                               self.outputs[output_path]['units'],
                               units)

        return output


class ControlInterpolationComp(ExplicitComponent):
    """
    Provides the interpolated value and rate of a control variable during explicit integration.

    For each control handled by ControlInterpolationComp, the user must provide an object
    with methods `eval(t)` and `eval_deriv(t)` which return the interpolated value and
    derivative of the control at time `t`, respectively.
    """
    def initialize(self):
        self.metadata.declare('time_units', default='s', allow_none=True, types=string_types,
                              desc='Units of time')
        self.metadata.declare('control_options', types=dict,
                              desc='Dictionary of options for the dynamic controls')
        self.interpolants = {}

    def setup(self):
        time_units = self.metadata['time_units']

        self.add_input('time', val=1.0, units=time_units)

        for control_name, options in iteritems(self.metadata['control_options']):
            shape = options['shape']
            units = options['units']
            rate_units = get_rate_units(units, time_units, deriv=1)
            rate2_units = get_rate_units(units, time_units, deriv=2)

            self.add_output('controls:{0}'.format(control_name), shape=shape, units=units)

            self.add_output('control_rates:{0}_rate'.format(control_name), shape=shape,
                            units=rate_units)

            self.add_output('control_rates:{0}_rate2'.format(control_name), shape=shape,
                            units=rate2_units)

    def compute(self, inputs, outputs):
        time = inputs['time']

        for name in self.metadata['control_options']:
            if name not in self.interpolants:
                raise(ValueError('No interpolant has been specified for {0}'.format(name)))

            outputs['controls:{0}'.format(name)] = self.interpolants[name].eval(time)

            outputs['control_rates:{0}_rate'.format(name)] = \
                self.interpolants[name].eval_deriv(time)

            outputs['control_rates:{0}_rate2'.format(name)] = \
                self.interpolants[name].eval_deriv(time, der=2)


class StateRateCollectorComp(ExplicitComponent):
    """
    Collects the state rates and outputs them in the units specified in the state options.
    For explicit integration this is necessary when the output providing the state rate has
    different units than those defined in the state_options/time_options.
    """
    def initialize(self):
        self.metadata.declare(
            'state_options', types=dict,
            desc='Dictionary of options for the ODE state variables.')
        self.metadata.declare(
            'time_units', default=None, allow_none=True, types=string_types,
            desc='Units of time')

        # Save the names of the dynamic controls/parameters
        self._input_names = {}
        self._output_names = {}

    def setup(self):
        state_options = self.metadata['state_options']
        time_units = self.metadata['time_units']

        for name, options in iteritems(state_options):
            self._input_names[name] = 'state_rates_in:{0}_rate'.format(name)
            self._output_names[name] = 'state_rates:{0}_rate'.format(name)
            shape = options['shape']
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            self.add_input(self._input_names[name], val=np.ones(shape), units=rate_units)
            self.add_output(self._output_names[name], shape=shape, units=rate_units)

    def compute(self, inputs, outputs):
        state_options = self.metadata['state_options']

        for name, options in iteritems(state_options):
            outputs[self._output_names[name]] = inputs[self._input_names[name]]


class ODEIntegrator(object):
    """
    Given a system class, create a callable object with the same signature as that required
    by scipy.integrate.ode::

        f(t, x, *args)

    Internally, this is accomplished by constructing an OpenMDAO problem using the ODE with
    a single node.  The interface populates the values of the time, states, and controls,
    and then calls `run_model()` on the problem.  The state rates generated by the ODE
    are then returned back to scipy ode, which continues the integration.

    Parameters
    ----------
    ode_class : OpenMDAO system with ode_options attribute
        The ODE system to be numerically integrated.

    """
    def __init__(self, ode_class, time_options, state_options, control_options,
                 ode_init_kwargs=None):
        self.prob = Problem(model=Group())

        self.ode = ode

        # The ODE System
        self.prob.model.add_subsystem('ode', subsys=ode_class(num_nodes=1, **ode_init_kwargs))
        self.ode_options = ode_class.ode_options

        # Get the state vector.  This isn't necessarily ordered
        # so just pick the default ordering and go with it.
        self.state_options = OrderedDict()
        self.time_options = time_options
        self.control_options = control_options

        pos = 0

        for state, options in iteritems(self.ode_options._states):
            self.state_options[state] = {'rate_source': options['rate_source'],
                                         'pos': pos,
                                         'shape': options['shape'],
                                         'size': np.prod(options['shape']),
                                         'units': options['units'],
                                         'targets': options['targets']}
            pos += self.state_options[state]['size']

        self._state_vec = np.zeros(pos, dtype=float)
        self._state_rate_vec = np.zeros(pos, dtype=float)

        # The Time Comp
        self.prob.model.add_subsystem('time_input',
                                      IndepVarComp('time',
                                                   val=0.0,
                                                   units=self.ode_options._time_options['units']),
                                      promotes_outputs=['time'])

        if self.ode_options._time_options['targets'] is not None:
            self.prob.model.connect('time',
                                    ['ode.{0}'.format(tgt) for tgt in
                                     self.ode_options._time_options['targets']])

        # The States Comp
        indep = IndepVarComp()
        for name, options in iteritems(self.state_options):
            indep.add_output('states:{0}'.format(name),
                             shape=(1, np.prod(options['shape'])),
                             units=options['units'])
            if options['targets'] is not None:
                self.prob.model.connect('states:{0}'.format(name),
                                        ['ode.{0}'.format(tgt) for tgt in options['targets']])
            self.prob.model.connect('ode.{0}'.format(options['rate_source']),
                                    'state_rate_collector.state_rates_in:{0}_rate'.format(name))

        self.prob.model.add_subsystem('indep_states', subsys=indep,
                                      promotes_outputs=['*'])

        # The Control interpolation comp
        time_units = self.ode_options._time_options['units']
        self._interp_comp = ControlInterpolationComp(time_units=time_units,
                                                     control_options=control_options)

        # The state rate collector comp
        self.prob.model.add_subsystem('state_rate_collector',
                                      StateRateCollectorComp(state_options=self.state_options,
                                                             time_units=time_options['units']))

        # Flag that is set to true if has_controls is called
        self._has_dynamic_controls = False

    def setup(self, check=False, mode='fwd'):
        """
        Call setup on the ODEIntegrator's problem instance.

        Parameters
        ----------
        check : bool
            If True, run setup on the problem instance with checks enabled.
            Default is False.

        """
        model = self.prob.model
        if self.control_options:
            model.add_subsystem('indep_controls', self._interp_comp, promotes_outputs=['*'])

            model.set_order(['time_input', 'indep_states', 'indep_controls', 'ode',
                             'state_rate_collector'])
            model.connect('time', ['indep_controls.time'])

            for name, options in iteritems(self.control_options):
                if name in self.ode_options._dynamic_parameters:
                    targets = self.ode_options._dynamic_parameters[name]['targets']
                    model.connect('controls:{0}'.format(name),
                                  ['ode.{0}'.format(tgt) for tgt in targets])
                if options['rate_param']:
                    rate_param = options['rate_param']
                    rate_targets = self.ode_options._dynamic_parameters[rate_param]['targets']
                    model.connect('control_rates:{0}_rate'.format(name),
                                  ['ode.{0}'.format(tgt) for tgt in rate_targets])
                if options['rate2_param']:
                    rate2_param = options['rate2_param']
                    rate2_targets = self.ode_options._dynamic_parameters[rate2_param]['targets']
                    model.connect('control_rates:{0}_rate2'.format(name),
                                  ['ode.{0}'.format(tgt) for tgt in rate2_targets])
        else:
            model.set_order(['time_input', 'indep_states', 'ode', 'state_rate_collector'])

        self.prob.setup(check=check, mode=mode)

    def set_interpolant(self, name, interpolant):
        """
        Set the interpolator to be used for the control of the given name.

        Parameters
        ----------
        name : str
            The name of the control whose interpolant is being specified.
        interpolant : object
            An object that provides interpolation for the control as a function of time.
            The object must have methods `eval(t)` which returns the interpolated value
            of the control at time t, and `eval_deriv(t)` which returns the interpolated
            value of the first time-derivative of the control at time t.
        """
        self._interp_comp.interpolants[name] = interpolant

    def _unpack_state_vec(self, x):
        """
        Given the state vector in 1D, extract the values corresponding to
        each state into the ode integrators problem states.

        Parameters
        ----------
        x : np.array
            The 1D state vector.

        Returns
        -------
        None

        """
        for state_name, state_options in self.state_options.items():
            pos = state_options['pos']
            size = state_options['size']
            self.prob['states:{0}'.format(state_name)][0, ...] = x[pos:pos+size]

    def _pack_state_rate_vec(self):
        """
        Pack the state rates into a 1D vector for use by scipy odeint.

        Returns
        -------
        dXdt: np.array
            The 1D state-rate vector.

        """
        for state_name, state_options in self.state_options.items():
            pos = state_options['pos']
            size = state_options['size']
            self._state_rate_vec[pos:pos+size] = \
                np.ravel(self.prob['state_rate_collector.'
                                   'state_rates:{0}_rate'.format(state_name)])
        return self._state_rate_vec

    def _pack_state_vec(self, x_dict):
        """
        Pack the state into a 1D vector for use by scipy.integrate.ode.

        Returns
        -------
        x: np.array
            The 1D state vector.

        """
        self._state_vec[:] = 0.0
        for state_name, state_options in self.state_options.items():
            pos = state_options['pos']
            size = state_options['size']
            self._state_vec[pos:pos+size] = np.ravel(x_dict[state_name])
        return self._state_vec

    def _f_ode(self, t, x, *args):
        """
        The function interface used by scipy.ode

        Parameters
        ----------
        t : float
            The current time, t.
        x : np.array
            The 1D state vector.

        Returns
        -------
        xdot : np.array
            The 1D vector of state time-derivatives.

        """
        self.prob['time'] = t
        self._unpack_state_vec(x)
        self.prob.run_model()
        xdot = self._pack_state_rate_vec()
        return xdot

    def integrate_times(self, x0_dict, times,
                        integrator='vode', integrator_params=None,
                        observer=None):
        """
        Integrate the RHS with the given initial state, and record the values at the
        specified times.

        Parameters
        ----------
        x0_dict : dict
            A dictionary keyed by state variable name that contains the
            initial values for the states.  If absent the initial value is
            assumed to be zero.
        times : sequence
            The sequence of times at which output for the integration is desired.
        integrator : str
            The integrator to be used by scipy.ode.  This is one of:
            vode, zvode, lsoda, dopri5, or dopri853.
        integrator_params : dict
            Parameters specific to the chosen integrator.  See the Scipy
            documentation for details.
        observer : callable, str, or None
            A callable function to be called at the specified timesteps in
            `integrate_times`.  This can be used to record the integrated trajectory.
            If 'default', a StdOutObserver will be used, which outputs all variables
            in the model to standard output by default.
            If None, no observer will be called.

        Returns
        -------
        dict
            A dictionary of variables in the RHS and their values at the given times.

        """

        # Prepare the observer
        if observer == 'stdout':
            _observer = StdOutObserver(self)
        elif observer == 'progress-bar':
            _observer = ProgressBarObserver(self, out_stream=sys.stdout, t0=times[0], tf=times[-1])
        else:
            _observer = observer

        int_params = integrator_params if integrator_params else {}

        solver = ode(self._f_ode)

        x0 = self._pack_state_vec(x0_dict)

        solver.set_integrator(integrator, **int_params)
        solver.set_initial_value(x0, times[0])

        delta_times = np.diff(times)

        # Run the Model once to get the initial values of all variables
        self._f_ode(solver.t, solver.y)

        # Prepare the output dictionary
        results = SimulationResults(state_options=self.state_options,
                                    control_options=self.control_options)

        model_outputs = self.prob.model.list_outputs(units=True, shape=True, out_stream=None)

        for output_name, options in model_outputs:
            prom_name = self.prob.model._var_abs2prom['output'][output_name]
            results.outputs[prom_name] = {}
            results.outputs[prom_name]['value'] = np.atleast_2d(self.prob[prom_name]).copy()
            results.outputs[prom_name]['units'] = options['units']

        if _observer:
            _observer(solver.t, solver.y, self.prob)

        terminate = False
        for dt in delta_times:

            try:
                solver.integrate(solver.t+dt)
                self._f_ode(solver.t, solver.y)
            except AnalysisError:
                terminate = True

            for var in results.outputs:
                results.outputs[var]['value'] = np.concatenate((results.outputs[var]['value'],
                                                                np.atleast_2d(self.prob[var])),
                                                               axis=0)
            if _observer:
                _observer(solver.t, solver.y, self.prob)

            if terminate:
                break

        return results
