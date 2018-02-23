from __future__ import division, print_function, absolute_import

from six import iteritems

import numpy as np

from .optimizer_based_phase_base import OptimizerBasedPhaseBase
from ..components import GaussLobattoPathConstraintComp
from ...utils.misc import get_rate_units


class GaussLobattoPhase(OptimizerBasedPhaseBase):
    """
    GaussLobattoPhase implements GaussLobatto transcription
    for solving optimal control problems.

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
        super(GaussLobattoPhase, self).__init__(**kwargs)

    def initialize(self, **kwargs):
        super(GaussLobattoPhase, self).initialize(**kwargs)
        self.metadata['transcription'] = 'gauss-lobatto'

    def _setup_time(self):
        comps = super(GaussLobattoPhase, self)._setup_time()

        if self.time_options['targets']:
            tgts = self.time_options['targets']
            self.connect('time',
                         ['rhs_col.{0}'.format(t) for t in tgts],
                         src_indices=self.grid_data.subset_node_indices['col'])
            self.connect('time',
                         ['rhs_disc.{0}'.format(t) for t in tgts],
                         src_indices=self.grid_data.subset_node_indices['disc'])

        return comps

    def _setup_controls(self):
        num_dynamic = super(GaussLobattoPhase, self)._setup_controls()
        grid_data = self.grid_data

        for name, options in iteritems(self.control_options):
            if options['dynamic']:
                map_indices_to_all = grid_data.input_maps['dynamic_control_to_all']
                map_indices_to_disc = map_indices_to_all[grid_data.subset_node_indices['disc']]
                map_indices_to_col = map_indices_to_all[grid_data.subset_node_indices['col']]
            else:
                map_indices_to_disc = np.zeros(grid_data.subset_num_nodes['disc'], dtype=int)
                map_indices_to_col = np.zeros(grid_data.subset_num_nodes['col'], dtype=int)

            if options['opt']:
                control_src_name = 'controls:{0}'.format(name)
            else:
                control_src_name = 'controls:{0}_out'.format(name)

            if name in self.ode_options._dynamic_parameters:
                targets = self.ode_options._dynamic_parameters[name]['targets']
                self.connect(control_src_name,
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_disc)

                self.connect(control_src_name,
                             ['rhs_col.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_col)

            if options['rate_param']:
                targets = self.ode_options._dynamic_parameters[options['rate_param']]['targets']

                self.connect('control_rates:{0}_rate'.format(name),
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_disc)

                self.connect('control_rates:{0}_rate'.format(name),
                             ['rhs_col.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_col)

            if options['rate2_param']:
                targets = self.ode_options._dynamic_parameters[options['rate2_param']]['targets']

                self.connect('control_rates:{0}_rate2'.format(name),
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_disc)

                self.connect('control_rates:{0}_rate2'.format(name),
                             ['rhs_col.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_col)

        return num_dynamic

    def _setup_path_constraints(self):
        """
        Add a path constraint component if necessary and issue appropriate connections as
        part of the setup stack.
        """
        path_comp = None
        gd = self.grid_data

        if self._path_constaints:
            path_comp = GaussLobattoPathConstraintComp(grid_data=gd)
            self.add_subsystem('path_constraints', subsys=path_comp)

        for var, options in iteritems(self._path_constaints):
            con_units = options.get('units', None)
            con_name = options['constraint_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = self._classify_var(var)

            if var_type == 'time':
                options['shape'] = (1,)
                options['units'] = self.time_options['units'] if con_units is None else con_units
                options['linear'] = True
                self.connect(src_name='time',
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))
            elif var_type == 'state':
                state_shape = self.state_options[var]['shape']
                state_units = self.state_options[var]['units']
                options['shape'] = state_shape
                options['units'] = state_units if con_units is None else con_units
                options['linear'] = False
                self.connect(src_name='states:{0}'.format(var),
                             tgt_name='path_constraints.disc_values:{0}'.format(con_name),
                             src_indices=gd.input_maps['state_to_disc'])
                self.connect(src_name='state_interp.state_col:{0}'.format(var),
                             tgt_name='path_constraints.col_values:{0}'.format(con_name))

            elif var_type == 'indep_control':
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'controls:{0}'.format(var)

                if self.control_options[var]['dynamic']:
                    ctrl_src_indices_all = gd.input_maps['dynamic_control_to_all']
                else:
                    ctrl_src_indices_all = np.zeros(gd.subset_num_nodes['all'], dtype=int)

                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name),
                             src_indices=ctrl_src_indices_all)

            elif var_type == 'input_control':
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'input_controls:{0}_out'.format(var)

                if self.control_options[var]['dynamic']:
                    ctrl_src_indices_all = gd.input_maps['dynamic_control_to_all']
                else:
                    ctrl_src_indices_all = np.zeros(gd.subset_num_nodes['all'], dtype=int)

                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name),
                             src_indices=ctrl_src_indices_all)

            elif var_type == 'control_rate':
                control_name = var[:-5]
                control_shape = self.control_options[control_name]['shape']
                control_units = self.control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate2':
                control_name = var[:-6]
                control_shape = self.control_options[control_name]['shape']
                control_units = self.control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate2'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            else:
                # Failed to find variable, assume it is in the RHS
                options['linear'] = False
                self.connect(src_name='rhs_disc.{0}'.format(var),
                             tgt_name='path_constraints.disc_values:{0}'.format(con_name))
                self.connect(src_name='rhs_col.{0}'.format(var),
                             tgt_name='path_constraints.col_values:{0}'.format(con_name))

            kwargs = options.copy()
            if var_type == 'control_rate':
                kwargs['units'] = get_rate_units(options['units'],
                                                 self.time_options['units'],
                                                 deriv=1)
            elif var_type == 'control_rate2':
                kwargs['units'] = get_rate_units(options['units'],
                                                 self.time_options['units'],
                                                 deriv=2)
            kwargs.pop('constraint_name', None)
            path_comp._add_path_constraint(con_name, var_type, **kwargs)

    def _setup_rhs(self):
        super(GaussLobattoPhase, self)._setup_rhs()

        grid_data = self.grid_data
        ODEClass = self.metadata['ode_class']
        num_input_nodes = self.grid_data.num_state_input_nodes

        kwargs = self.metadata['ode_init_kwargs']
        rhs_disc = ODEClass(num_nodes=grid_data.subset_num_nodes['disc'], **kwargs)
        rhs_col = ODEClass(num_nodes=grid_data.subset_num_nodes['col'], **kwargs)

        map_input_indices_to_disc = self.grid_data.input_maps['state_to_disc']

        self.add_subsystem('rhs_disc', rhs_disc)
        self.add_subsystem('rhs_col', rhs_col)

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
                    ['rhs_disc.{0}'.format(tgt) for tgt in options['targets']],
                    src_indices=src_idxs, flat_src_indices=True)
                self.connect(
                    'state_interp.state_col:{0}'.format(name),
                    ['rhs_col.{0}'.format(tgt) for tgt in options['targets']])

            self.connect(
                'rhs_disc.{0}'.format(options['rate_source']),
                'state_interp.staterate_disc:{0}'.format(name))

    def _setup_defects(self):
        super(GaussLobattoPhase, self)._setup_defects()

        for name, options in iteritems(self.state_options):

            self.connect(
                'state_interp.staterate_col:%s' % name,
                'collocation_constraint.f_approx:%s' % name)

            self.connect(
                'rhs_col.%s' % options['rate_source'],
                'collocation_constraint.f_computed:%s' % name)

    def get_values(self, var, nodes='all'):
        """
        Retrieve the values of the given variable at the given
        subset of nodes.

        Parameters
        ----------
        var : str
            The variable whose values are to be returned.  This may be
            the name 'time', the name of a state, control, or parameter,
            or the path to a variable in the ODE system of the phase.
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
        col_node_idxs = gd.subset_node_indices['col']

        var_type = self._classify_var(var)

        if var_type == 'time':
            output = np.zeros((self.grid_data.num_nodes, 1))
            time_comp = self.time
            output[:, 0] = time_comp._outputs[var]

        elif var_type == 'state':
            output = np.zeros((gd.num_nodes,) + self.state_options[var]['shape'])
            state_comp = self.indep_states
            state_interp_comp = self.state_interp
            state_disc_vals = state_comp._outputs['states:{0}'.format(var)]
            output[disc_node_idxs] = state_disc_vals[gd.input_maps['state_to_disc']]
            output[col_node_idxs] = state_interp_comp._outputs['state_col:{0}'.format(var)]

        elif var_type == 'indep_control':
            control_comp = self.indep_controls
            if self.control_options[var]['dynamic']:
                output = control_comp._outputs['controls:{0}'.format(var)]
            else:
                val = control_comp._outputs['controls:{0}'.format(var)]
                output = np.repeat(val, gd.num_nodes, axis=0)

        elif var_type == 'input_control':
            parameter_comp = self.input_controls
            if self.control_options[var]['dynamic']:
                output = parameter_comp._outputs['controls:{0}_out'.format(var)]
            else:
                val = parameter_comp._outputs['controls:{0}_out'.format(var)]
                output = np.repeat(val, gd.num_nodes, axis=0)

        elif var_type == 'control_rate':
            control_rate_comp = self.control_rate_comp
            output = control_rate_comp._outputs['control_rates:{0}'.format(var)]

        elif var_type == 'control_rate2':
            control_rate_comp = self.control_rate_comp
            output = control_rate_comp._outputs['control_rates:{0}'.format(var)]

        elif var_type == 'rhs':
            rhs_disc = self.rhs_disc
            rhs_col = self.rhs_col

            rhs_disc_outputs = rhs_disc.list_outputs(out_stream=None)
            rhs_col_outputs = rhs_col.list_outputs(out_stream=None)

            prom2abs_disc = rhs_disc._var_allprocs_prom2abs_list
            prom2abs_col = rhs_col._var_allprocs_prom2abs_list

            # Is var in prom2abs_disc['output']?
            abs_path_disc = prom2abs_disc['output'][var][0]
            abs_path_col = prom2abs_col['output'][var][0]

            disc_vals = dict(rhs_disc_outputs)[abs_path_disc]['value']
            col_vals = dict(rhs_col_outputs)[abs_path_col]['value']

            output = np.zeros((gd.num_nodes,) + disc_vals.shape[1:])

            output[gd.subset_node_indices['disc']] = disc_vals
            output[gd.subset_node_indices['col']] = col_vals

        return output[gd.subset_node_indices[nodes]]
