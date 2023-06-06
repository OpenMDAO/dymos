import matplotlib.pyplot as plt
import numpy as np
from OrbitAnim import OrbitAnim
import sys

from rotation_matrices import R_PQW2IJK, R_PQW2RSW

filename = 'orbital_elements.txt'
if len(sys.argv) > 1 and sys.argv[1] != '':
    filename = sys.argv[1]

orbit = OrbitAnim(filename, animate=True)
orbit.extract_data()
# orbit.set_elev(30)
# orbit.set_azim(-120)
# orbit.run_animation()

states = orbit.states
params = orbit.params

# want to plot:
#   - p and a on same ax
#   - f, g, and e on same ax
#   - h, k, and i on same ax
#   - L, Omega, omega, and nu on same ax
#   - m
element_fig = plt.figure(figsize=(10, 8))

pa_ax = element_fig.add_subplot(321)
fge_ax = element_fig.add_subplot(322)
hki_ax = element_fig.add_subplot(323)
angles_ax = element_fig.add_subplot(324)
m_ax = element_fig.add_subplot(313)

element_fig.subplots_adjust(left=0.12,
                            bottom=0.12,
                            right=0.9,
                            top=0.88,
                            wspace=0.5,
                            hspace=0.5)

t = states['t']
p = states['p']
f = states['f']
g = states['g']
h = states['h']
k = states['k']
L = states['L']
m = states['m']
u_r = states['u_r']
u_theta = states['u_theta']
u_h = states['u_h']
tau = states['tau']

a = np.zeros(len(t))
e = np.zeros(len(t))
i = np.zeros(len(t))
Omega = np.zeros(len(t))
w = np.zeros(len(t))
nu = np.zeros(len(t))

for j in range(len(t)):
    e[j] = np.sqrt(f[j]**2 + g[j]**2)
    a[j] = p[j]/(1 - e[j]**2)
    i[j] = 2*np.arctan(np.sqrt(h[j]**2 + k[j]**2))
    Omega[j] = np.arccos(h[j]/np.tan(i[j]/2))
    Omega[j] = np.nan if np.isnan(Omega[j]) else Omega[j]
    w[j] = np.arccos(f[j]/e[j]) - Omega[j]
    nu[j] = L[j] - Omega[j] - w[j]

pa_ax.plot(t, p, label='p')
pa_ax.plot(t, a, label='a')
pa_ax.legend()
pa_ax.grid()
pa_ax.set_xlabel('t (s)')
pa_ax.set_ylabel('p, a (km)')

fge_ax.plot(t, f, label='f')
fge_ax.plot(t, g, label='g')
fge_ax.plot(t, e, label='e')
fge_ax.grid()
fge_ax.legend()
fge_ax.set_xlabel('t (s)')
fge_ax.set_ylabel('f, g, e (unitless)')

hki_ax.plot(t, h, label='h')
hki_ax.plot(t, k, label='k')
hki_ax.plot(t, i, label='i')
hki_ax.grid()
hki_ax.legend()
hki_ax.set_xlabel('t (s)')
hki_ax.set_ylabel('h, k, i (unitless)')

angles_ax.plot(t, L, label='L')
angles_ax.plot(t, Omega, label=r'$\Omega$')
angles_ax.plot(t, w, label=r'$\omega$')
angles_ax.plot(t, nu, label=r'$\nu$')
angles_ax.legend()
angles_ax.grid()
angles_ax.set_xlabel('t (s)')
angles_ax.set_ylabel(r'L, $\Omega$, $\omega$, $\nu$ (rad)')

m_ax.plot(t, m)
m_ax.grid()
m_ax.set_xlabel('t (s)')
m_ax.set_ylabel('m (kg)')

element_fig.suptitle('Orbital Elements')

# plt.show()
element_fig.savefig('Orbital_Elements.png')

orbit.run_animation()
plt.close()

# state fig will plot:
#   - velocity dir vs thrust dir -> angle btwn velocity and thrust
#   - r magnitude (AKA altitude)
#   - v magnitude
#   - perturbing acceleration
#   - yaw and pitch

state_fig = plt.figure(figsize=(10, 8))

v_T_angle = np.zeros(len(t))
r_mags = np.array(orbit.r_mags)
v_mags = np.array(orbit.v_mags)
vs = np.array(orbit.vs)
us = np.array([u_r, u_h, u_theta])
dT = np.zeros(len(t))
yaw = np.zeros(len(t))
pitch = np.zeros(len(t))

for i in range(len(t)):
    v_T_angle[i] = np.rad2deg(np.arccos(np.dot(vs[:, i], us[:, i])/(np.linalg.norm(vs[:, i])*np.linalg.norm(us[:, i]))))
    dT[i] = (params['T']*(1 + 0.01*tau[i]))/m[i]
    # NOTE the optimizer sometimes cheats, u_r, u_theta, and u_h should not be greater than 1,
    # but if they are slightly over 1 that'll cause some nans to show up
    # other nans seen in pitch are due to the arithmetic yielding a value barely greater than 1
    # which means there might be some other thing to look at
    u_r[i] = 1.0 if u_r[i] > 1.0 else u_r[i]
    u_r[i] = -1.0 if u_r[i] < -1.0 else u_r[i]
    yaw[i] = np.rad2deg(np.arcsin(-u_r[i]))
    u_theta[i] = 1.0 if u_theta[i] > 1.0 else u_theta[i]
    u_theta[i] = -1.0 if u_theta[i] < -1.0 else u_theta[i]
    pitch[i] = np.rad2deg(np.arccos(u_theta[i]/np.cos(np.deg2rad(yaw[i]))))    
        

vT_ax = state_fig.add_subplot(321)
dT_ax = state_fig.add_subplot(322)
dir_ax = state_fig.add_subplot(323)
r_ax = state_fig.add_subplot(324)
v_ax = state_fig.add_subplot(313)
# tau_ax = state_fig.add_subplot(326)

state_fig.subplots_adjust(left=0.125,
                          bottom=0.11,
                          right=0.9,
                          top=0.88,
                          wspace=0.35,
                          hspace=0.36)

vT_ax.plot(t, v_T_angle)
vT_ax.grid()
vT_ax.set_xlabel('t (s)')
vT_ax.set_ylabel('Angle btwn v and T (deg)')

dT_ax.plot(t, dT)
dT_ax.grid()
dT_ax.set_xlabel('t (s)')
dT_ax.set_ylabel(r'$\Delta_T$ $(N)$')

dir_ax.plot(t, yaw, label='yaw')
dir_ax.plot(t, pitch, label='pitch')
dir_ax.legend()
dir_ax.grid()
dir_ax.set_xlabel('t (s)')
dir_ax.set_ylabel('yaw, pitch (deg)')

r_ax.plot(t, r_mags)
r_ax.grid()
r_ax.set_xlabel('t (s)')
r_ax.set_ylabel('Altitude (km)')

v_ax.plot(t, v_mags)
v_ax.grid()
v_ax.set_xlabel('t (s)')
v_ax.set_ylabel('Velocity (m/s)')

# tau_ax.plot(t, tau)
# tau_ax.grid()
# tau_ax.set_xlabel('t (s)')
# tau_ax.set_ylabel('Throttle (unitless)')

state_fig.suptitle('Important States')

state_fig.savefig('State_Plots.png')
