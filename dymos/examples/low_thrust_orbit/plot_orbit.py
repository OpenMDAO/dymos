import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
import numpy as np
import sys

mu = 3.986004418e14 # m^3/s^2
Re = 6378.10 # km
scale = 1000

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

def getNonRepeatedIndices(lst):
    indices = []
    for i in range(1,len(lst)):
        if lst[i] != lst[i-1]:
            indices.append(i)
    
    return indices

t = states['t']
non_repeated = getNonRepeatedIndices(t)

t = [t[i] for i in non_repeated]
p = [states['p'][i] for i in non_repeated]
f = [states['f'][i] for i in non_repeated]
g = [states['g'][i] for i in non_repeated]
h = [states['h'][i] for i in non_repeated]
k = [states['k'][i] for i in non_repeated]
L = [states['L'][i] for i in non_repeated]
m = [states['m'][i] for i in non_repeated]
u_r = [states['u_r'][i] for i in non_repeated]
u_theta = [states['u_theta'][i] for i in non_repeated]
u_h = [states['u_h'][i] for i in non_repeated]
tau = [states['tau'][i] for i in non_repeated]

r, v = calc_r_v(p[0], f[0], g[0], h[0], k[0], L[0])

r_mag = np.linalg.norm(r)
v_mag = np.linalg.norm(v)

rs = [[r[0][0]], [r[1][0]], [r[2][0]]]
r_line, = orbit_ax.plot3D(rs[0], rs[1], rs[2], '-', color='k', zorder=10)

v_pts = [[r[i], r[i] + v[i]/v_mag*scale] for i in range(len(r))]
v_line, = orbit_ax.plot3D(v_pts[0][:], v_pts[1][:], v_pts[2][:], '-', color='red', zorder=10)

u = [u_r[0], u_theta[0], u_h[0]]
T_pts = [[r[i], r[i] + u[i]*scale] for i in range(len(r))]
T_line, = orbit_ax.plot3D(T_pts[0][:], T_pts[1][:], T_pts[2][:], '-', color='orange', zorder=10)


u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = Re*np.cos(u) * np.sin(v)
y = Re*np.sin(u) * np.sin(v)
z = Re*np.cos(v)
earth = orbit_ax.plot_surface(x, y, z, color='b', alpha=1)
earth.set_zorder(10)

orbit_ax.set_box_aspect([1,1,1])
ax_scale = 1.05
orbit_ax.set_xlim(-Re*ax_scale, Re*ax_scale)
orbit_ax.set_ylim(-Re*ax_scale, Re*ax_scale)
orbit_ax.set_zlim(-Re*ax_scale, Re*ax_scale)
orbit_ax.view_init(elev=60, azim=-90)

p_line, = p_ax.plot(t[0], p[0], '-')
f_line, = fg_ax.plot(t[0], f[0], '-')
g_line, = fg_ax.plot(t[0], g[0], '-')
h_line, = hk_ax.plot(t[0], h[0], '-')
k_line, = hk_ax.plot(t[0], k[0], '-')
L_line, = L_ax.plot(t[0], L[0], '-')
m_line, = m_ax.plot(t[0], m[0], '-')

def animate(step):
    r, v = calc_r_v(p[step], f[step], g[step], h[step], k[step], L[step])
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    u = [u_r[step], u_theta[step], u_h[step]]

    rs[0].append(r[0][0])
    rs[1].append(r[1][0])
    rs[2].append(r[2][0])
    
    r_line.set_data_3d((rs[0], rs[1], rs[2]))

    v_pts = [[r[i][0], r[i][0] + v[i][0]/v_mag*scale] for i in range(len(r))]
    v_line.set_data_3d((v_pts[0], v_pts[1], v_pts[2]))

    T_pts = [[r[i][0], r[i][0] + u[i]*scale] for i in range(len(r))]
    T_line.set_data_3d((T_pts[0], T_pts[1], T_pts[2]))
    
    p_line.set_xdata(t[:step])
    p_line.set_ydata(p[:step])
    p_ax.set_xlim(0, t[step]+5)
    p_ax.set_ylim(p[0], p[step]+100)

    return r_line, v_line, T_line, p_line, earth

# TODO set up titles, labels, and legends
ani = animation.FuncAnimation(anim_fig, animate, frames=np.arange(1, len(non_repeated)))
# plt.show()
ani.save('orbit.gif')
# want to plot:
#   - p and a on same ax
#   - f, g, and e on same ax
#   - h, k, and i on same ax
#   - L, Omega, omega, and nu on same ax
#   - m
# element_fig = plt.figure()

# state fig will plot:
#   - velocity dir vs thrust dir
#   - r magnitude (AKA altitude)
#   - v magnitude
#   - perturbing acceleration
#   - thrust
#   - yaw and pitch
# state_fig = plt.figure()

