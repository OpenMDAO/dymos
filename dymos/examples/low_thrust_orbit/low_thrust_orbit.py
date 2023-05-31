import openmdao.api as om
import dymos as dm
import jax.numpy as jnp
import jax
import numpy as np
import sys

from functools import partial


class LowThrustOrbitODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('g0', types=float, default=32.174, desc='standard acceleration due to gravity (ft/sec^2)')
        self.options.declare('mu', types=float, default=3.986004418e14, desc='graviational parameter (m^3/sec^2)')
        self.options.declare('Re', types=float, default=6378100.0, desc='radius of the Earth (m)')
        self.options.declare('J2', types=float, default=1082.639e-6)
        # self.options.declare('J3', types=float, default=-2.565e-6)
        # self.options.declare('J4', types=float, default=-1.608e-6)

    def setup(self):
        nn = self.options['num_nodes']
        # method = 'fd'
        method = 'exact'

        # Inputs
        self.add_input('p', shape=nn, desc='semi-parameter', units='m')
        self.add_input('f', shape=nn, desc='x component of eccentricity', units='unitless')
        self.add_input('g', shape=nn, desc='y component of eccentricity', units='unitless')
        self.add_input('h', shape=nn, desc='x component of the node vector', units='unitless')
        self.add_input('k', shape=nn, desc='y component of the node vector', units='unitless')
        self.add_input('L', shape=nn, desc='true longitude', units='rad')
        self.add_input('m', shape=nn, desc='mass of spacecraft', units='kg')

        self.add_input('u_r', shape=nn, desc='percent thrust in radial direction', units='unitless')
        self.add_input('u_theta', shape=nn, desc='percent thrust in tangential direction', units='unitless')
        self.add_input('u_h', shape=nn, desc='thrust in normal direction', units='unitless')
        
        self.add_input('tau', shape=nn, desc='throttle factor', units='unitless')
        
        self.add_input('T', shape=1, desc='maximum thrust', units='N', tags=['dymos.static_target'])
        self.add_input('Isp', shape=1, desc='specific impulse', units='s', tags=['dymos.static_target'])

        # Outputs
        self.add_output('p_dot', shape=nn, desc='time derivative of semi-parameter', units='m/s')
        self.add_output('f_dot', shape=nn, desc='time derivative of x component of eccentricity', units='unitless/s')
        self.add_output('g_dot', shape=nn, desc='time derivative of y component of eccentricity', units='unitless/s')
        self.add_output('h_dot', shape=nn, desc='time derivative of x component of the node vector', units='unitless/s')
        self.add_output('k_dot', shape=nn, desc='time derivative of y component of the node vector', units='unitless/s')
        self.add_output('L_dot', shape=nn, desc='time derivative of true longitude', units='rad/s')
        self.add_output('m_dot', shape=nn, desc='time derivative of mass', units='kg/s')

        # Setup partials
        # self.declare_partials(of='*', wrt='*', method='fd')

        ar = np.arange(nn, dtype=int)
        # NOTE these are the partials that each vectorized input hasin common
        self.declare_partials(of=['p_dot', 'f_dot', 'g_dot', 'h_dot', 'k_dot', 'L_dot'],
                              wrt=['p', 'f', 'g', 'L', 'm', 'tau'],
                              method=method, rows=ar, cols=ar)
        self.declare_partials(of=['p_dot', 'f_dot', 'g_dot', 'h_dot', 'k_dot', 'L_dot'], wrt=['T'],
                              method=method)
        
        self.declare_partials(of='p_dot', wrt=['u_theta'], method=method, rows=ar, cols=ar)
        self.declare_partials(of=['f_dot', 'g_dot'], wrt=['h', 'k', 'u_r', 'u_theta', 'u_h'], method=method, rows=ar, cols=ar)
        self.declare_partials(of=['h_dot', 'k_dot', 'L_dot'], wrt=['h', 'k', 'u_h'], method=method, rows=ar, cols=ar)

        self.declare_partials(of='m_dot', wrt=['tau'], rows=ar, cols=ar, method=method)
        self.declare_partials(of='m_dot', wrt=['Isp', 'T'], method=method)
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_partials_jacfwd(self, *inputs):
        deriv_func = jax.jacfwd(self.compute_primal, argnums=range(13))
        derivs = deriv_func(*inputs)
        
        reduced_derivs = []
        
        for of in derivs:
            reduced_derivs.append([])
            for wrt in of:
                if len(wrt[0]) > 1:
                    reduced_derivs[-1].append(jnp.diagonal(wrt))
                else:
                    reduced_derivs[-1].append(wrt)

        return reduced_derivs

    @partial(jax.jit, static_argnums=(0,))
    def compute_primal(self, *inputs):
        # g0 = self.options['g0']
        mu = self.options['mu']
        # Re = self.options['Re']
        # J2 = self.options['J2']
        # J3 = self.options['J3']
        # J4 = self.options['J4']
        
        inp_dict = {name: val for name, val in zip(self._var_rel_names['input'], inputs)}
        
        p = inp_dict['p']
        f = inp_dict['f']
        g = inp_dict['g']
        h = inp_dict['h']
        k = inp_dict['k']
        L = inp_dict['L']
        m = inp_dict['m']
        u_r = inp_dict['u_r']
        u_theta = inp_dict['u_theta']
        u_h = inp_dict['u_h']
        tau = inp_dict['tau']
        T = inp_dict['T']
        Isp = inp_dict['Isp']
        
        sinL = jnp.sin(L)
        cosL = jnp.cos(L)
        
        q = 1 + f*cosL + g*sinL
        # r_mag = p/q
        # alpha_squared = h**2 - k**2
        chi = jnp.sqrt(h**2 + k**2)
        s_squared = 1 + chi**2


        sqrt_p_mu = jnp.sqrt(p/mu)
        # a_rsw = (g0*T*(1 + 0.01*tau))/w
        a_rsw = (T*(1 + 0.01*tau))/m
        
        p_dot = a_rsw*u_theta*(2*p)/q*sqrt_p_mu
        f_dot = a_rsw*(u_r*sqrt_p_mu)*sinL + u_theta*sqrt_p_mu*(1/q)*((q + 1)*cosL + f) + u_h*-sqrt_p_mu*(g/q)*(h*sinL - k*cosL)
        g_dot = a_rsw*(u_r*-sqrt_p_mu)*cosL + u_theta*sqrt_p_mu*(1/q)*((q + 1)*sinL + g) + u_h*sqrt_p_mu*(f/q)*(h*sinL - k*cosL)
        h_dot = a_rsw*(u_h*sqrt_p_mu)*(s_squared*cosL)/(2*q)
        k_dot = a_rsw*(u_h*sqrt_p_mu)*(s_squared*sinL)/(2*q)
        L_dot = a_rsw*(u_h*sqrt_p_mu)*(1/q)*(h*sinL - k*cosL) + jnp.sqrt(mu*p)*(q/p)**2
        m_dot = -T*(1 + 0.01*tau)/Isp
        
        return p_dot, f_dot, g_dot, h_dot, k_dot, L_dot, m_dot
 
    def compute(self, inputs, outputs):
        outputs['p_dot'], outputs['f_dot'], outputs['g_dot'], outputs['h_dot'], outputs['k_dot'], outputs['L_dot'], outputs['m_dot'] = self.compute_primal(*inputs.values())


    def compute_partials(self, inputs, partials):
        jac_fwd_partials = self._compute_partials_jacfwd(*inputs.values())
        
        for out_name, of_data in zip(self._var_rel_names['output'], jac_fwd_partials):
            for inp_name, wrt_data in zip(self._var_rel_names['input'], of_data):
                abs_inp = self._var_allprocs_prom2abs_list['input'][inp_name][0]
                abs_out = self._var_allprocs_prom2abs_list['output'][out_name][0]
                if (abs_out, abs_inp) not in partials._subjacs_info.keys():
                    continue
                partials[out_name, inp_name] = wrt_data.ravel()
                # print(out_name, inp_name, wrt_data.ravel())

p = om.Problem()

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.opt_settings['Verify level'] = 3
p.driver.opt_settings['iSumm'] = 6
# p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
p.driver.declare_coloring()
# p.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs']

traj = dm.Trajectory()

# T = 4.446618e-3 lb -> 0.019779542235 N
traj.add_parameter('T', val=0.019779542235, units='N', targets={'spiral': ['T']}, opt=False)
traj.add_parameter('Isp', val=450, units='s', targets={'spiral': ['Isp']}, opt=False)

tx = dm.Radau(num_segments=35, order=4, compressed=True) #solve_segments='forward')
spiral = dm.Phase(ode_class=LowThrustOrbitODE, transcription=tx)
spiral = traj.add_phase('spiral', spiral)

spiral.set_time_options(fix_initial=True,# duration_bounds=(60000, 90000),
                        units='s')
# spiral.set_time_options(fix_initial=True, fix_duration=True, units='s')
# NOTE need defect scalars?
spiral.add_state('p', fix_initial=True, rate_source='p_dot', lower=6378100.0, upper=100*6378100.0, ref=6378100.0)
spiral.add_state('f', fix_initial=True, rate_source='f_dot', lower=-1, upper=1)
spiral.add_state('g', fix_initial=True, rate_source='g_dot', lower=-1, upper=1)
spiral.add_state('h', fix_initial=True, rate_source='h_dot', lower=-1, upper=1)
spiral.add_state('k', fix_initial=True, rate_source='k_dot', lower=-1, upper=1)
spiral.add_state('L', fix_initial=True, rate_source='L_dot', lower=0.0, scaler=1e-2, defect_ref=1e-2)
spiral.add_state('m', fix_initial=True, rate_source='m_dot', lower=0.01, upper=1.0)

spiral.add_control('tau', opt=True, rate_continuity=False, rate2_continuity=False,
                #    rate_continuity_scaler=1e-3,
                   units='unitless', lower=-50, upper=0)
spiral.add_control('u_r', opt=True, rate_continuity=False, rate2_continuity=False,
                #    rate_continuity_scaler=1e-3,
                   units='unitless', lower=-1, upper=1)
spiral.add_control('u_theta', opt=True, rate_continuity=False, rate2_continuity=False,
                #    rate_continuity_scaler=1e-3,
                   units='unitless', lower=-1, upper=1)
spiral.add_control('u_h', opt=True, rate_continuity=False, rate2_continuity=False,
                #    rate_continuity_scaler=1e-3,
                   units='unitless', lower=-1, upper=1)

spiral.add_objective('m', loc='final', scaler=-1)
# spiral.add_objective('p', loc='final', ref=6378.137)

spiral.add_boundary_constraint('p', loc='final', equals=12194239.065442713, ref=12194239.065442713)
spiral.add_boundary_constraint('eccentricity = (f**2 + g**2)**0.5', loc='final', equals=0.73550320568829)
spiral.add_boundary_constraint('tan_inclination = (h**2 + k**2)**0.5', loc='final', equals=0.61761258786099)
spiral.add_boundary_constraint('comp_const1 = f*h + g*k', loc='final', equals=0.0)
spiral.add_boundary_constraint('comp_const2 = g*h - k*f', loc='final', upper=0.0)
spiral.add_path_constraint('u_mag = (u_r**2 + u_theta**2 + u_h**2)**0.5', equals=1.0)

p.model.add_subsystem('traj', traj)

p.setup(check=True, force_alloc_complex=True)

# p.set_val('traj.spiral.initial_states:p', 21837080.052835)
# p.set_val('traj.spiral.initial_states:f', 0.0)
# p.set_val('traj.spiral.initial_states:g', 0.0)
# p.set_val('traj.spiral.initial_states:h', -0.25396764647494)
# p.set_val('traj.spiral.initial_states:k', 0)
# p.set_val('traj.spiral.initial_states:L', 0)
# p.set_val('traj.spiral.initial_states:m', 1)

p.set_val('traj.spiral.states:p', 6655942.00010410789)
p.set_val('traj.spiral.states:f', 0.0)
p.set_val('traj.spiral.states:g', 0.0)
p.set_val('traj.spiral.states:h', -0.25396764647494)
p.set_val('traj.spiral.states:k', 0)
p.set_val('traj.spiral.states:L', np.pi)
p.set_val('traj.spiral.states:m', 1)
p.set_val('traj.spiral.t_initial', 0.0)
# p.set_val('traj.spiral.t_duration', 5_000.0)
# p.set_val('traj.spiral.controls:u_theta', 1)

# p.set_val('traj.spiral.t_duration', 90*60)


# p.run_model()
# p.model.list_outputs(print_arrays=True, units=True)

# p.run_model()

dm.run_problem(p, run_driver=True, simulate=True, make_plots=True)
# p.run_model()
# print('\n\n\n')
# p.model.list_outputs(print_arrays=True, units=True)

# p.model.list_outputs(print_arrays=True, units=True)

# with np.printoptions(linewidth=10000):
#     p.check_partials(method='cs', compact_print=True)

with open('orbital_elements_real.txt', 'w') as sys.stdout:
    print('STATES: t p f g h k L m u_r u_theta u_h tau')
    print(f'MAX THRUST: {p.get_val("traj.parameter_vals:T")[0][0]}')
    print(f'ISP: {p.get_val("traj.parameter_vals:Isp")[0][0]}')
    print()
    
    t = p.get_val('traj.spiral.timeseries.time')
    _p = p.get_val('traj.spiral.timeseries.states:p')
    f = p.get_val('traj.spiral.timeseries.states:f')
    g = p.get_val('traj.spiral.timeseries.states:g')
    h = p.get_val('traj.spiral.timeseries.states:h')
    k = p.get_val('traj.spiral.timeseries.states:k')
    L = p.get_val('traj.spiral.timeseries.states:L')
    m = p.get_val('traj.spiral.timeseries.states:m')
    
    for i in range(len(t)):
        print(f'{t[i][0]} {_p[i][0]} {f[i][0]} {g[i][0]} {h[i][0]} {k[i][0]} {L[i][0]} {m[i][0]}')
    