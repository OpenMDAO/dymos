from __future__ import division, print_function, absolute_import

from six import iteritems

import numpy as np

from openmdao.utils.units import convert_units, valid_units

from .optimizer_based_phase_base import OptimizerBasedPhaseBase
from .components import ControlEndpointDefectComp
from ..components import RadauPathConstraintComp
from ...utils.misc import get_rate_units


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
        self.options['transcription'] = 'radau-ps'

    def _setup_time(self):
        comps = super(RadauPseudospectralPhase, self)._setup_time()

        if self.time_options['targets']:
            self.connect('time',
                         ['rhs_all.{0}'.format(t) for t in self.time_options['targets']],
                         src_indices=self.grid_data.subset_node_indices['all'])
        return comps

    def _setup_controls(self):
        super(RadauPseudospectralPhase, self)._setup_controls()

        added_defect_constraint = False

        for name, options in iteritems(self.control_options):

            map_indices_to_all = self.grid_data.input_maps['dynamic_control_input_to_disc']
            if options['opt']:
                if not added_defect_constraint:
                    def_comp = ControlEndpointDefectComp(grid_data=self.grid_data,
                                                         control_options=self.control_options)
                    self.add_subsystem('control_defect_comp', subsys=def_comp,
                                       promotes_outputs=['*'])
                    added_defect_constraint = True
                self.connect('controls:{0}'.format(name),
                             'control_defect_comp.controls:{0}'.format(name),
                             src_indices=map_indices_to_all)

            if options['opt']:
                control_src_name = 'controls:{0}'.format(name)
            else:
                control_src_name = 'controls:{0}_out'.format(name)

            if name in self.ode_options._parameters:
                targets = self.ode_options._parameters[name]['targets']
                self.connect(control_src_name,
                             ['rhs_all.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_all)

            if options['rate_param']:
                targets = self.ode_options._parameters[options['rate_param']]['targets']
                self.connect('control_rates:{0}_rate'.format(name),
                             ['rhs_all.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_all)

            if options['rate2_param']:
                targets = self.ode_options._parameters[options['rate2_param']]['targets']
                self.connect('control_rates:{0}_rate2'.format(name),
                             ['rhs_all.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_all)

    def _setup_design_parameters(self):
        super(RadauPseudospectralPhase, self)._setup_design_parameters()

        for name, options in iteritems(self.design_parameter_options):

            map_indices_to_all = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)

            if options['opt']:
                src_name = 'design_parameters:{0}'.format(name)
            else:
                src_name = 'design_parameters:{0}_out'.format(name)

            if name in self.ode_options._parameters:
                targets = self.ode_options._parameters[name]['targets']
                self.connect(src_name,
                             ['rhs_all.{0}'.format(t) for t in targets],
                             src_indices=map_indices_to_all)

    def _setup_path_constraints(self):
        """
        Add a path constraint component if necessary and issue appropriate connections as
        part of the setup stack.
        """
        path_comp = None
        gd = self.grid_data
        time_units = self.time_options['units']

        if self._path_constraints:
            path_comp = RadauPathConstraintComp(grid_data=gd)
            self.add_subsystem('path_constraints', subsys=path_comp)

        for var, options in iteritems(self._path_constraints):
            con_units = options.get('units', None)
            con_name = options['constraint_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = self._classify_var(var)

            if var_type == 'time':
                options['shape'] = (1,)
                options['units'] = time_units if con_units is None else con_units
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
                             tgt_name='path_constraints.all_values:{0}'.format(con_name),
                             src_indices=gd.input_maps['state_input_to_disc'])

            elif var_type == 'indep_control':
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'controls:{0}'.format(var)

                if self.control_options[var]['dynamic']:
                    ctrl_src_indices_all = gd.input_maps['dynamic_control_input_to_disc']
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
                    ctrl_src_indices_all = gd.input_maps['dynamic_control_input_to_disc']
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
                options['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate2':
                control_name = var[:-6]
                control_shape = self.control_options[control_name]['shape']
                control_units = self.control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate2'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            else:
                # Failed to find variable, assume it is in the RHS
                options['linear'] = False
                self.connect(src_name='rhs_all.{0}'.format(var),
                             tgt_name='path_constraints.all_values:{0}'.format(con_name),
                             src_indices=gd.subset_node_indices['all'])

            kwargs = options.copy()
            kwargs.pop('constraint_name', None)
            path_comp._add_path_constraint(con_name, var_type, **kwargs)

    def _setup_rhs(self):
        super(RadauPseudospectralPhase, self)._setup_rhs()

        ODEClass = self.options['ode_class']
        grid_data = self.grid_data
        num_input_nodes = self.grid_data.num_state_input_nodes

        map_input_indices_to_disc = self.grid_data.input_maps['state_input_to_disc']

        kwargs = self.options['ode_init_kwargs']
        self.add_subsystem('rhs_all',
                           subsys=ODEClass(num_nodes=grid_data.subset_num_nodes['all'],
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

    def add_objective(self, name, loc='final', index=None, shape=(1,), ref=None, ref0=None,
                      adder=None, scaler=None, parallel_deriv_color=None,
                      vectorize_derivs=False):
        """
        Allows the user to add an objective in the phase.  If name is not a state,
        control, control rate, or 'time', then this is assumed to be the path of the variable
        to be constrained in the RHS.

        Parameters
        ----------
        name : str
            Name of the objective variable.  This should be one of 'time', a state or control
            variable, or the path to an output from the top level of the RHS.
        loc : str
            Where in the phase the objective is to be evaluated.  Valid
            options are 'initial' and 'final'.  The default is 'final'.
        index : int, optional
            If variable is an array at each point in time, this indicates which index is to be
            used as the objective, assuming C-ordered flattening.
        shape : int, optional
            The shape of the objective variable, at a point in time
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        parallel_deriv_color : string
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        vectorize_derivs : bool
            If True, vectorize derivative calculations.
        """
        var_type = self._classify_var(name)

        # Determine the path to the variable
        if var_type == 'time':
            obj_path = 'time'
        elif var_type == 'state':
            obj_path = 'states:{0}'.format(name)
        elif var_type == 'indep_control':
            obj_path = 'controls:{0}'.format(name)
        elif var_type == 'input_control':
            obj_path = 'controls:{0}'.format(name)
        elif var_type == 'control_rate':
            control_name = name[:-5]
            obj_path = 'control_rates:{0}_rate'.format(control_name)
        elif var_type == 'control_rate2':
            control_name = name[:-6]
            obj_path = 'control_rates:{0}_rate2'.format(control_name)
        else:
            # Failed to find variable, assume it is in the RHS
            obj_path = 'rhs_all.{0}'.format(name)

        pdc = parallel_deriv_color
        super(RadauPseudospectralPhase, self)._add_objective(obj_path, loc=loc, index=index,
                                                             shape=shape, ref=ref, ref0=ref0,
                                                             adder=adder, scaler=scaler,
                                                             parallel_deriv_color=pdc,
                                                             vectorize_derivs=vectorize_derivs)

    def get_values(self, var, nodes=None, units=None):
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
            The name of the node subset.
        units : str
            The units in which the values should be expressed.  Must be compatible
            with the corresponding units inside the phase.

        Returns
        -------
        ndarray
            An array of the values at the requested node subset.  The
            node index is the first dimension of the ndarray.
        """
        if nodes is None:
            nodes = 'all'

        gd = self.grid_data

        var_type = self._classify_var(var)

        op = dict(self.list_outputs(explicit=True, values=True, units=True, shape=True,
                                    out_stream=None))

        if units is not None:
            if not valid_units(units):
                raise ValueError('Units {0} is not a valid units identifier'.format(units))

        var_prefix = '{0}.'.format(self.pathname) if self.pathname else ''

        path_map = {'time': 'time.{0}',
                    'state': 'indep_states.states:{0}',
                    'indep_control': 'indep_controls.controls:{0}',
                    'input_control': 'input_controls.controls:{0}_out',
                    'indep_design_parameter': 'indep_design_params.design_parameters:{0}',
                    'input_design_parameter': 'input_design_params.design_parameters:{0}_out',
                    'control_rate': 'control_rate_comp.control_rates:{0}',
                    'control_rate2': 'control_rate_comp.control_rates:{0}',
                    'rhs': 'rhs_all.{0}'}

        if var_type == 'state':
            var_path = var_prefix + path_map[var_type].format(var)
            output_units = op[var_path]['units']
            output_value = convert_units(op[var_path]['value'][gd.input_maps['state_input_to_disc'],
                                                               ...], output_units, units)

        elif var_type in ('input_control', 'indep_control'):
            var_path = var_prefix + path_map[var_type].format(var)
            output_units = op[var_path]['units']

            if self.control_options[var]['dynamic']:
                vals = op[var_path]['value'][gd.input_maps['dynamic_control_input_to_disc'], ...]
                output_value = convert_units(vals, output_units, units)
            else:
                output_value = convert_units(op[var_path]['value'], output_units, units)
                output_value = np.repeat(output_value, gd.num_nodes, axis=0)

        elif var_type in ('input_design_parameter', 'indep_design_parameter'):
            var_path = var_prefix + path_map[var_type].format(var)
            output_units = op[var_path]['units']

            output_value = convert_units(op[var_path]['value'], output_units, units)
            output_value = np.repeat(output_value, gd.num_nodes, axis=0)

        elif var_type == 'rhs':
            rhs_all_outputs = dict(self.rhs_all.list_outputs(out_stream=None, values=True,
                                                             shape=True, units=True))
            prom2abs_all = self.rhs_all._var_allprocs_prom2abs_list
            abs_path_all = prom2abs_all['output'][var][0]
            output_value = rhs_all_outputs[abs_path_all]['value']
            output_units = rhs_all_outputs[abs_path_all]['units']
            output_value = convert_units(output_value, output_units, units)
        else:
            var_path = var_prefix + path_map[var_type].format(var)
            output_units = op[var_path]['units']
            output_value = convert_units(op[var_path]['value'], output_units, units)

        # Always return a column vector
        if len(output_value.shape) == 1:
            output_value = np.reshape(output_value, (gd.num_nodes, 1))

        return output_value[gd.subset_node_indices[nodes], ...]
