import numpy as np
import time

from openmdao.api import Problem, ScipyOptimizeDriver, IndepVarComp

from dymos.glm.ozone.ode_integrator import ODEIntegrator
from dymos.glm.ozone.utils.suppress_printing import nostdout
from dymos.glm.ozone.methods_list import get_method


def run_integration(num_times, t0, t1, initial_conditions, ode_system_class, formulation, method_name):
    try:
        exact_solution = ode_system_class().get_exact_solution(initial_conditions, t0, t1)
    except AttributeError:
        raise NotImplementedError('{0} does not implement get_exact_solution'.format(ode_system_class))

    times = np.linspace(t0, t1, num_times)

    integrator = ODEIntegrator(ode_system_class, formulation, method_name,
                               times=times, initial_conditions=initial_conditions)
    prob = Problem(integrator)

    if formulation == 'optimizer-based':
        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        integrator.add_subsystem('dummy_comp', IndepVarComp('dummy_var', val=1.0))
        integrator.add_objective('dummy_comp.dummy_var')

    with nostdout():
        prob.setup()
        runtime0 = time.time()
        prob.run_driver()
        runtime1 = time.time()

    runtime = runtime1 - runtime0
    errors = {}
    for key in exact_solution:
        errors[key] = np.linalg.norm(prob['state:%s' % key][-1] - exact_solution[key])

    return runtime, errors


def compute_runtimes(num_times_vector, t0, t1,
        ode_function, formulation, method_name, initial_conditions):
    num = len(num_times_vector)

    step_sizes_vector = np.zeros(num)
    runtimes_vector = np.zeros(num)

    for ind, num_times in enumerate(num_times_vector):
        runtime, errors = run_integration(
            num_times, t0, t1, initial_conditions, ode_function, formulation, method_name)

        step_sizes_vector[ind] = (t1 - t0) / (num_times - 1)
        runtimes_vector[ind] = runtime

    return step_sizes_vector, runtimes_vector


def compute_ideal_runtimes(step_sizes_vector, runtimes_vector):
    ideal_step_sizes_vector = np.array([
        step_sizes_vector[0],
        step_sizes_vector[-1],
    ])

    ideal_runtimes = np.array([
        runtimes_vector[0],
        runtimes_vector[0] * (step_sizes_vector[0] / step_sizes_vector[-1]),
    ])

    return ideal_step_sizes_vector, ideal_runtimes


def compute_convergence_order(num_times_vector, t0, t1, state_name,
                              ode_class, formulation, method_name, initial_conditions):
    num = len(num_times_vector)

    step_sizes_vector = np.zeros(num)
    errors_vector = np.zeros(num)
    orders_vector = np.zeros(num)

    ode_instance = ode_class(num_nodes=5)

    for ind, num_times in enumerate(num_times_vector):
        times = np.linspace(t0, t1, num_times)

        integrator = ODEIntegrator(ode_class, formulation, method_name,
                                   times=times, initial_conditions=initial_conditions)
        prob = Problem(integrator)

        with nostdout():
            prob.setup()
            prob.run_driver()

        approx_y = prob['state:%s' % state_name][-1][0]
        true_y = ode_instance.get_exact_solution(initial_conditions, t0, t1)[state_name]

        errors_vector[ind] = np.linalg.norm(approx_y - true_y)
        step_sizes_vector[ind] = (t1 - t0) / (num_times - 1)

    errors0 = errors_vector[:-1]
    errors1 = errors_vector[1:]

    step_sizes0 = step_sizes_vector[:-1]
    step_sizes1 = step_sizes_vector[1:]

    orders_vector = np.log( errors1 / errors0 ) / np.log( step_sizes1 / step_sizes0 )

    ideal_order = get_method(method_name).order

    return errors_vector, step_sizes_vector, orders_vector, ideal_order


def compute_ideal_error(step_sizes_vector, errors_vector, ideal_order):
    ideal_step_sizes_vector = np.array([
        step_sizes_vector[0],
        step_sizes_vector[-1],
    ])

    ideal_errors_vector = np.array([
        errors_vector[0],
        errors_vector[0] * ( step_sizes_vector[-1] / step_sizes_vector[0] ) ** ideal_order,
    ])

    return ideal_step_sizes_vector, ideal_errors_vector
