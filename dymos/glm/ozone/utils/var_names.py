# F Y y_old y_new Y_in Y_out
# initial_conditions ode_inputs ode_states ode_outputs
# ode_initial_conditions params states outputs


def get_name(var_type, state_name, i_step=None, i_stage=None, j_stage=None):
    name = '{}:{}'.format(var_type, state_name)

    if j_stage is not None:
        name = 'stage{}_{}'.format(j_stage, name)

    if i_stage is not None:
        name = 'stage{}_{}'.format(i_stage, name)

    if i_step is not None:
        name = 'step{}_{}'.format(i_step, name)

    return name
