import numpy as np

import openmdao.api as om

from dymos.transcriptions.grid_data import GridData
from dymos.utils.misc import get_rate_units
from dymos._options import options as dymos_options
from dymos.utils.lgl import lgl
from dymos.utils.lgr import lgr
from dymos.utils.birkhoff import birkhoff_matrices


class BirkhoffCollocationComp(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of state names/options for the phase')

        self.options.declare(
            'time_units', default=None, allow_none=True, types=str,
            desc='Units of time')

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine shape and units.
        """
        gd = self.options['grid_data']
        num_nodes = gd.subset_num_nodes['col']
        time_units = self.options['time_units']
        state_options = self.options['state_options']

        self.add_input('dt_dstau', units=time_units, shape=(num_nodes,))
        self.var_names = var_names = {}
        for state_name in state_options:
            var_names[state_name] = {
                'f_value': f'f_value:{state_name}',
                'f_computed': f'f_computed:{state_name}',
                'state_value': f'state_value:{state_name}',
                'state_defect': f'state_defects:{state_name}',
                'state_rate_defect': f'state_rate_defects:{state_name}',
                'initial_state_rate_defect': f'initial_state_rate_defects:{state_name}',
                'final_state_defect': f'final_state_defects:{state_name}'
            }

        for state_name, options in state_options.items():
            shape = options['shape']
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            var_names = self.var_names[state_name]

            self.add_input(
                name=var_names['f_value'],
                shape=(num_nodes,) + shape,
                desc=f'Estimated derivative of state {state_name} at the collocation nodes',
                units=rate_units)

            self.add_input(
                name=var_names['f_computed'],
                shape=(num_nodes,) + shape,
                desc=f'Computed derivative of state {state_name} at the collocation nodes',
                units=rate_units)

            self.add_input(
                name=var_names['state_value'],
                shape=(num_nodes+1,) + shape,
                units=units
            )

            self.add_output(
                name=var_names['state_defect'],
                shape=((num_nodes-1),) + shape,
                units=units
            )

            self.add_output(
                name=var_names['state_rate_defect'],
                shape=(num_nodes,) + shape,
                units=rate_units
            )

            # self.add_output(
            #     name=var_names['initial_state_rate_defect'],
            #     shape=shape,
            #     units=rate_units
            # )

            self.add_output(
                name=var_names['final_state_defect'],
                shape=shape,
                units=units
            )

            if 'defect_ref' in options and options['defect_ref'] is not None:
                defect_ref = options['defect_ref']
            elif 'defect_scaler' in options and options['defect_scaler'] is not None:
                defect_ref = np.divide(1.0, options['defect_scaler'])
            else:
                if 'ref' in options and options['ref'] is not None:
                    defect_ref = options['ref']
                elif 'scaler' in options and options['scaler'] is not None:
                    defect_ref = np.divide(1.0, options['scaler'])
                else:
                    defect_ref = 1.0

            if not np.isscalar(defect_ref):
                defect_ref = np.asarray(defect_ref)
                if defect_ref.shape == shape:
                    defect_ref = np.tile(defect_ref.flatten(), num_nodes)
                else:
                    raise ValueError('array-valued scaler/ref must length equal to state-size')

            if not options['solve_segments']:
                self.add_constraint(name=var_names['state_defect'],
                                    equals=0.0,
                                    ref=defect_ref)

                self.add_constraint(name=var_names['state_rate_defect'],
                                    equals=0.0,
                                    ref=defect_ref)
                #
                # self.add_constraint(name=var_names['initial_state_rate_defect'],
                #                     equals=0.0,
                #                     ref=defect_ref)

                self.add_constraint(name=var_names['final_state_defect'],
                                    equals=0.0,
                                    ref=defect_ref)

        self.declare_partials(of='*', wrt='*', method='fd')

        # Setup partials
        # for state_name, options in state_options.items():
        #     shape = options['shape']
        #     size = np.prod(shape)
        #
        #     r1 = np.arange((num_nodes-1) * size)
        #     r2 = np.arange(size)
        #
        #     c1 = np.arange(size, num_nodes*size)
        #     c2 = np.repeat(np.arange(1, num_nodes), size)
        #
        #     var_names = self.var_names[state_name]
        #
        #     self.declare_partials(of=var_names['state_defect'],
        #                           wrt=var_names['state_value'],
        #                           rows=r1, cols=c1,
        #                           val=1.0)
        #
        #     self.declare_partials(of=var_names['state_defect'],
        #                           wrt=var_names['f_value'],
        #                           rows=r1, cols=r1,
        #                           val=-1.0)
        #
        #     self.declare_partials(of=var_names['state_rate_defect'],
        #                           wrt=var_names['f_value'],
        #                           rows=r1, cols=r1)
        #
        #     self.declare_partials(of=var_names['state_rate_defect'],
        #                           wrt=var_names['f_computed'],
        #                           rows=r1, cols=c1)
        #
        #     self.declare_partials(of=var_names['state_rate_defect'],
        #                           wrt='dt_dstau',
        #                           rows=r1, cols=c2)
        #
        #     self.declare_partials(of=var_names['initial_state_rate_defect'],
        #                           wrt=var_names['f_value'],
        #                           rows=r2, cols=r2)
        #
        #     self.declare_partials(of=var_names['initial_state_rate_defect'],
        #                           wrt=var_names['state_value'],
        #                           rows=r2, cols=r2)
        #
        #     self.declare_partials(of=var_names['initial_state_rate_defect'],
        #                           wrt=var_names['f_computed'],
        #                           rows=r2, cols=r2)
        #
        #     self.declare_partials(of=var_names['initial_state_rate_defect'],
        #                           wrt='dt_dstau',
        #                           rows=r2, cols=np.zeros(size))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        state_options = self.options['state_options']
        gd = self.options['grid_data']
        num_nodes = gd.transcription_order[0]
        dt_dstau = inputs['dt_dstau']
        if gd.transcription == 'gauss-lobatto':
            tau, w = lgl(num_nodes)
        elif gd.transcription == 'radau-ps':
            tau, w = lgr(num_nodes)
        else:
            raise ValueError('invalid transcription')

        B, Bd = birkhoff_matrices(tau, w)

        for state_name in state_options:
            var_names = self.var_names[state_name]
            # state value: value of the state design variable
            # state approx: value of the state as computed using the integration matrix
            # f_value: value of the state rate design variable
            # f_computed: output from rhs_all

            state_value = inputs[var_names['state_value']]
            f_value = inputs[var_names['f_value']]
            f_computed = inputs[var_names['f_computed']]

            f_initial_approx = Bd[0, 1:] @ f_value[1:] + state_value[0] * Bd[0, 0]

            state_approx = np.zeros((num_nodes-1, 1))
            for i in range(0, num_nodes - 1):
                state_approx[i] = state_value[0] * B[i+1, 0]
                for j in range(1, num_nodes):
                    state_approx[i] += B[i+1, j] * f_value[j]*dt_dstau[j]

            outputs[var_names['state_defect']] = (state_value[1:num_nodes] - state_approx).T
            outputs[var_names['state_rate_defect']] = (f_value.T * dt_dstau - f_computed.T).T
            # outputs[var_names['initial_state_rate_defect']] = ((f_initial_approx - f_value[0]).T * dt_dstau[0]).T
            outputs[var_names['final_state_defect']] = state_value[0] + np.dot(w, f_value.ravel() * dt_dstau)\
                                                       - state_value[-1]

    # def compute_partials(self, inputs, partials):
    #     """
    #     Compute sub-jacobian parts. The model is assumed to be in an unscaled state.
    #
    #     Parameters
    #     ----------
    #     inputs : Vector
    #         Unscaled, dimensional input variables read via inputs[key].
    #     partials : Jacobian
    #         Subjac components written to partials[output_name, input_name].
    #     """
    #     dt_dstau = inputs['dt_dstau']
    #     for state_name, options in self.options['state_options'].items():
    #         size = np.prod(options['shape'])
    #         var_names = self.var_names[state_name]
    #         f_value = inputs[var_names['f_value']]
    #         f_computed = inputs[var_names['f_computed']]
    #         f_initial_approx = inputs[var_names['f_initial_approx']]
    #
    #         k1 = np.repeat(dt_dstau[1:], size)
    #         k2 = np.repeat(dt_dstau[0], size)
    #
    #         partials[var_names['state_rate_defect'], var_names['f_value']] = k1
    #         partials[var_names['state_rate_defect'], var_names['f_computed']] = -k1
    #         partials[var_names['state_rate_defect'], 'dt_dstau'] = (f_value - f_computed[1:]).ravel()
    #
    #         partials[var_names['initial_state_rate_defect'], var_names['f_initial_approx']] = k2
    #         partials[var_names['initial_state_rate_defect'], var_names['f_computed']] = -k2
    #         partials[var_names['initial_state_rate_defect'], 'dt_dstau'] = (f_initial_approx - f_computed[0]).ravel()
