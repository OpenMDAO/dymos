import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys

mu = 3.986004418e14 # m^3/s^2
Re = 6378100.0 # m

def calc_r_v(p, f, g, h, k, L):
    cosL = np.cos(L)
    sinL = np.sin(L)
    mu_p = np.sqrt(mu/p)

    q = 1 + f*cosL + g*sinL
    r = p/q
    alpha_squared = h**2 - k**2
    chi = np.sqrt(h**2 + k**2)
    s_squared = 1 + chi**2
    
    r_vec = np.array([[(r/s_squared)*(cosL + alpha_squared*cosL + 2*h*k*sinL)],
                      [(r/s_squared)*(sinL - alpha_squared*sinL + 2*h*k*cosL)],
                      [(2*r/s_squared)*(h*sinL - k*cosL)]])    
    v_vec = np.array([[(-1/s_squared)*mu_p*(sinL + alpha_squared*sinL - 2*h*k*cosL + g - 2*f*h*k + alpha_squared*g)],
                      [(-1/s_squared)*mu_p*(-cosL + alpha_squared*cosL + 2*h*k*sinL - f + 2*g*h*k + alpha_squared*f)],
                      [(2/s_squared)*mu_p*(h*cosL + k*sinL + f*h + g*k)]])
    
    return r_vec, v_vec


file_name = 'orbital_elements_min_p.txt'
for i in range(1, len(sys.argv), 2):
    if sys.argv[i] == '-f':
        file_name = sys.argv[i+1]

with open(file_name, 'r') as file:
    states = {}
    params = {}
    for line in file:
        line = line.strip()
        
        if line.startswith('STATES:'):
            for key in line.split()[1:]:
                states[key] = []
            continue
        elif line.startswith('MAX THRUST:'):
            params['T'] = float(line.split()[-1])
            continue
        elif line.startswith('ISP:'):
            params['Isp'] = float(line.split()[-1])
            continue
        elif line == '':
            continue
        
        line = line.split()
        for i, key in enumerate(states.keys()):
            states[key].append(float(line[i]))

# r, v = r_and_v_calc(states)

#####################
#           p       f
#   ORBIT   g       h
#           k       L
#               m
#####################
anim_fig = plt.figure(figsize=(12, 6))
# anim_fig.subplot_tool()
# anim_fig.tight_layout(pad=10)

# orbit ax will have:
#   - r timeseries
#   - v direction (quiver)
#   - thrust direction (quiver)
#   - sphere for earth??
orbit_ax = anim_fig.add_subplot(1, 2, 1, projection='3d')
p_ax = anim_fig.add_subplot(3, 4, 3)
fg_ax = anim_fig.add_subplot(3, 4, 4)
hk_ax = anim_fig.add_subplot(3, 4, 7)
L_ax = anim_fig.add_subplot(3, 4, 8)
m_ax = anim_fig.add_subplot(3, 2, 6)

anim_fig.subplots_adjust(left=0.07,
                         bottom=0.09,
                         right=0.95,
                         top=0.9,
                         wspace=0.26,
                         hspace=0.55)

p = states['p']
f = states['f']
g = states['g']
h = states['h']
k = states['k']
L = states['L']
m = states['m']
t = states['t']

r, v = calc_r_v(p[0], f[0], g[0], h[0], k[0], L[0])
r_line = orbit_ax.plot3D(r[0], r[1], r[2], '-o')

u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = Re*np.cos(u) * np.sin(v)
y = Re*np.sin(u) * np.sin(v)
z = Re*np.cos(v)
earth = orbit_ax.plot_surface(x, y, z, color='b')

orbit_ax.set_box_aspect([1,1,1])
orbit_ax.view_init(elev=45, azim=-90)

p_line = p_ax.plot(t[0], p[0])
f_line = fg_ax.plot(t[0], f[0])
g_line = fg_ax.plot(t[0], g[0])
h_line = hk_ax.plot(t[0], h[0])
k_line = hk_ax.plot(t[0], k[0])
L_line = L_ax.plot(t[0], L[0])
m_line = m_ax.plot(t[0], m[0])

# TODO set up titles, labels, and legends


# want to plot:
#   - p and a on same ax
#   - f, g, and e on same ax
#   - h, k, and i on same ax
#   - L, Omega, omega, and nu on same ax
#   - m
element_fig = plt.figure()

# state fig will plot:
#   - velocity dir vs thrust dir
#   - r magnitude (AKA altitude)
#   - v magnitude
#   - perturbing acceleration
#   - thrust
#   - yaw and pitch
state_fig = plt.figure()

