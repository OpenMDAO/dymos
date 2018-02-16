from __future__ import division, print_function, absolute_import

from six import iteritems
import numpy as np

from .optimizer_based_phase_base import OptimizerBasedPhaseBase


class RadauPseudospectralPhase(OptimizerBasedPhaseBase):
    """
    RadauPseudospectralPhase implements Legendre-Gauss-Radau
    pseudospectral transcription for solving optimal control problems.

    Attributes
    ----------
    self.time_options : dict of TimeOptionsDictionary
        A dictionary of options for time (integration variable) in the phase.

    self.state_options : dict of StateOptionsDictionary
        A dictionary of options for the RHS states in the Phase.

    self.control_options : dict of ControlOptionsDictionary
        A dictionary of options for the controls in the Phase.

    self._ode_controls : dict of ControlOptionsDictionary
        A dictionary of the default options for controllable inputs of the Phase RHS

    """
    def __init__(self, **kwargs):
        super(RadauPseudospectralPhase, self).__init__(**kwargs)

    def initialize(self, **kwargs):
        super(RadauPseudospectralPhase, self).initialize(**kwargs)
        self.metadata['transcription'] = 'radau-ps'

    def _setup_time(self):
        comps = super(RadauPseudospectralPhase, self)._setup_time()

        if self.time_options['targets']:
            self.connect('time',
                         ['rhs_all.{0}'.format(t) for t in self.time_options['targets']],
                         src_indices=self.grid_data.subset_node_indices['disc'])
        return comps

    def _setup_controls(self):
        super(RadauPseudospectralPhase, self)._setup_controls()

        ode = self.metadata['ode_function']

        for name, options in iteritems(self.control_options):

            if options['dynamic']:
                map_indices_to_all = self.grid_data.input_maps['dynamic_control_to_all']
            else:
                map_indices_to_all = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)

            if options['opt']:
                control_src_name = 'controls:{0}'.format(name)
            else:
                control_src_name = 'controls:{0}_out'.format(name)

            if name in ode._dynamic_parameters:
                targets = ode._dynamic_parameters[name]['targets']
                self.connect(control_src_name,
                             ['rhs_all.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_all)

            if options['rate_param']:
                targets = ode._dynamic_parameters[options['rate_param']]['targets']
                self.connect('control_rates:{0}_rate'.format(name),
                             ['rhs_all.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_all)

            if options['rate2_param']:
                targets = ode._dynamic_parameters[options['rate2_param']]['targets']
                self.connect('control_rates:{0}_rate2'.format(name),
                             ['rhs_all.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_all)

    def _setup_rhs(self):
        super(RadauPseudospectralPhase, self)._setup_rhs()

        ode = self.metadata['ode_function']
        grid_data = self.grid_data
        num_input_nodes = self.grid_data.num_state_input_nodes

        map_input_indices_to_disc = self.grid_data.input_maps['state_to_disc']

        kwargs = ode._system_init_kwargs
        self.add_subsystem('rhs_all',
                           subsys=ode._system_class(num_nodes=grid_data.subset_num_nodes['disc'],
                                                    **kwargs))

        for name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])

            src_idxs_mat = np.reshape(np.arange(size * num_input_nodes, dtype=int),
                                      (num_input_nodes, size), order='C')

            src_idxs = src_idxs_mat[map_input_indices_to_disc, :]

            if size == 1:
                """ Flat state variable is passed as 1D data."""
                src_idxs = src_idxs.ravel()

            if options['targets']:
                self.connect(
                    'states:{0}'.format(name),
                    ['rhs_all.{0}'.format(tgt) for tgt in options['targets']],
                    src_indices=src_idxs, flat_src_indices=True)

    def _setup_defects(self):
        super(RadauPseudospectralPhase, self)._setup_defects()
        grid_data = self.grid_data

        for name, options in iteritems(self.state_options):

            self.connect(
                'state_interp.staterate_col:{0}'.format(name),
                'collocation_constraint.f_approx:{0}'.format(name))

            self.connect('rhs_all.{0}'.format(options['rate_source']),
                         'collocation_constraint.f_computed:{0}'.format(name),
                         src_indices=grid_data.subset_node_indices['col'])

    def get_values(self, var, nodes='all'):
        """
        Retrieve the values of the given variable at the given
        subset of nodes.

        Parameters
        ----------
        var : str
            The variable whose values are to be returned.  This may be
            the name 'time', the name of a state, control, or parameter,
            or the path to a variable in the ODEFunction of the phase.
        nodes : str
            The name of a node subset, one of 'disc', 'col', or 'all'.
            The default is 'all'.

        Returns
        -------
        ndarray
            An array of the values at the requested node subset.  The
            node index is the first dimension of the ndarray.
        """
        gd = self.grid_data
        disc_node_idxs = gd.subset_node_indices['disc']

        var_type = self._classify_var(var)

        if var_type == 'time':
            output = np.zeros((self.grid_data.num_nodes, 1))
            time_comp = self.time
            output[:, 0] = time_comp._outputs[var]

        elif var_type == 'state':
            output = np.zeros((gd.num_nodes,) + self.state_options[var]['shape'])
            state_comp = self.indep_states
            state_disc_values = state_comp._outputs['states:{0}'.format(var)]
            output[disc_node_idxs] = state_disc_values[gd.input_maps['state_to_disc']]

        elif var_type == 'indep_control':
            control_comp = self.indep_controls
            if self.control_options[var]['dynamic']:
                output = control_comp._outputs['controls:{0}'.format(var)]
            else:
                val = control_comp._outputs['controls:{0}'.format(var)]
                output = np.repeat(val, gd.num_nodes, axis=0)

        elif var_type == 'input_control':
            control_input_comp = self.input_controls
            if self.control_options[var]['dynamic']:
                output = control_input_comp._outputs['controls:{0}_out'.format(var)]
            else:
                val = control_input_comp._outputs['controls:{0}_out'.format(var)]
                output = np.repeat(val, gd.num_nodes, axis=0)

        elif var_type == 'control_rate':
            control_rate_comp = self.control_rate_comp
            output = control_rate_comp._outputs['control_rates:{0}'.format(var)]

        elif var_type == 'control_rate2':
            control_rate_comp = self.control_rate_comp
            output = control_rate_comp._outputs['control_rates:{0}'.format(var)]

        elif var_type == 'rhs':
            rhs_all = self.rhs_all

            rhs_all_outputs = rhs_all.list_outputs(out_stream=None)

            prom2abs_all = rhs_all._var_allprocs_prom2abs_list

            # Is var in prom2abs_disc['output']?
            abs_path_all = prom2abs_all['output'][var][0]

            output = dict(rhs_all_outputs)[abs_path_all]['value']

        return output[gd.subset_node_indices[nodes]]
