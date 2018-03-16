from __future__ import print_function, division, absolute_import

import sys

import numpy as np


class StdOutObserver(object):
    """
    The default observer callable for use by RHSIntegrator.integrate_times.

    This observer provides column output of the model variables when called.

    Parameters
    ----------
    ode_integrator : ScipyODEIntegrator
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
