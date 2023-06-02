import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class OrbitAnim:
    def __init__(self, filename, savefile='orbit.gif', animate=True):
        self.filename = filename
        self.savefile = savefile
        self.animate = animate
        
        self.mu = 3.986004418e14 # m^3/s^2
        self.Re = 6378.10 # km
        self.scale = 1000
        self.earth_scale = 0.8
        
        self.ax_scaler = 1.05
        self.elev = 45
        self.azim = -90
    
    def calc_r_v(self, p, f, g, h, k, L):
        cosL = np.cos(L)
        sinL = np.sin(L)
        mu_p = np.sqrt(self.mu/p)

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
    
    def extract_data(self):
       with open(self.filename, 'r') as file:
        self.states = {}
        self.params = {}
        for line in file:
            line = line.strip()
            
            if line.startswith('STATES:'):
                for key in line.split()[1:]:
                    self.states[key] = []
                continue
            elif line.startswith('MAX THRUST:'):
                self.params['T'] = float(line.split()[-1])
                continue
            elif line.startswith('ISP:'):
                self.params['Isp'] = float(line.split()[-1])
                continue
            elif line == '':
                continue
            
            line = line.split()
            for i, key in enumerate(self.states.keys()):
                # sometimes there are repeats in the dataset, don't want to include those
                if key == 't' and float(line[i]) in self.states['t']:
                    break
                self.states[key].append(float(line[i]))
        
    def initialize(self):
        self.fig = plt.figure(figsize=(12, 6))
        
        self.orbit_ax = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.p_ax = self.fig.add_subplot(3, 4, 3)
        self.fg_ax = self.fig.add_subplot(3, 4, 4)
        self.hk_ax = self.fig.add_subplot(3, 4, 7)
        self.L_ax = self.fig.add_subplot(3, 4, 8)
        self.m_ax = self.fig.add_subplot(3, 2, 6)

        self.fig.subplots_adjust(left=0.07,
                                 bottom=0.09,
                                 right=0.95,
                                 top=0.9,
                                 wspace=0.26,
                                 hspace=0.55)

        self.extract_data()
        
        t = self.states['t'][0]
        p = self.states['p'][0]
        f = self.states['f'][0]
        g = self.states['g'][0]
        h = self.states['h'][0]
        k = self.states['k'][0]
        L = self.states['L'][0]
        m = self.states['m'][0]
        u_r = self.states['u_r'][0]
        u_theta = self.states['u_theta'][0]
        u_h = self.states['u_h'][0]
        tau = self.states['tau'][0]
        
        r, v = self.calc_r_v(p, f, g, h, k, L)
        u = [u_r, u_theta, u_h]
        self.rs = [[r[0][0]], [r[1][0]], [r[2][0]]]
        
        self.r_mags = [np.linalg.norm(r)]
        self.v_mags = [np.linalg.norm(v)]
        
        self.r_line, = self.orbit_ax.plot3D(self.rs[0], self.rs[1], self.rs[2], '-', color='k', zorder=5)
        
        v_pts = [[r[i], r[i] + v[i]/self.v_mags[0]*self.scale] for i in range(len(r))]
        self.v_line, = self.orbit_ax.plot3D(v_pts[0], v_pts[1], v_pts[2], '-', color='red', zorder=5)
        
        T_pts = [[r[i], r[i] + u[i]*self.scale] for i in range(len(r))]
        self.T_line, = self.orbit_ax.plot3D(T_pts[0], T_pts[1], T_pts[2], '-', color='orange', zorder=5)
        
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x = self.earth_scale*self.Re*np.cos(u) * np.sin(v)
        y = self.earth_scale*self.Re*np.sin(u) * np.sin(v)
        z = self.earth_scale*self.Re*np.cos(v)
        self.earth = self.orbit_ax.plot_surface(x, y, z, color='b', alpha=1)
        self.earth.set_zorder(10)
        
        self.orbit_ax.set_box_aspect([1,1,1])
        self.orbit_ax.set_xlim(-self.Re*self.ax_scaler, self.Re*self.ax_scaler)
        self.orbit_ax.set_ylim(-self.Re*self.ax_scaler, self.Re*self.ax_scaler)
        self.orbit_ax.set_zlim(-self.Re*self.ax_scaler, self.Re*self.ax_scaler)
        self.orbit_ax.view_init(elev=self.elev, azim=self.azim)
        
        self.p_line, = self.p_ax.plot(t, p, '-')
        self.f_line, = self.fg_ax.plot(t, f, '-')
        self.g_line, = self.fg_ax.plot(t, g, '-')
        self.h_line, = self.hk_ax.plot(t, h, '-')
        self.k_line, = self.hk_ax.plot(t, k, '-')
        self.L_line, = self.L_ax.plot(t, L, '-')
        self.m_line, = self.m_ax.plot(t, m, '-')

    # TODO have a bunch of setters for ax scaling
    
    def set_3D_ax_scaler(self, ax_scaler):
        self.ax_scaler = ax_scaler
    
    def set_elev(self, elev):
        self.elev = elev
    
    def set_azim(self, azim):
        self.azim = azim
    
    def set_earth_scale(self, scale):
        self.earth_scale = scale
    
    def run_animation(self):
        self.initialize()

        t = self.states['t']
        p = self.states['p']
        f = self.states['f']
        g = self.states['g']
        h = self.states['h']
        k = self.states['k']
        L = self.states['L']
        m = self.states['m']
        u_r = self.states['u_r']
        u_theta = self.states['u_theta']
        u_h = self.states['u_h']
        tau = self.states['tau']
        
        def animate(i):
            r, v = self.calc_r_v(p[i], f[i], g[i], h[i], k[i], L[i])
            r_mag = np.linalg.norm(r)
            v_mag = np.linalg.norm(v)
            u = [u_r[i], u_theta[i], u_h[i]]

            self.rs[0].append(r[0][0])
            self.rs[1].append(r[1][0])
            self.rs[2].append(r[2][0])
            
            self.r_line.set_data_3d((self.rs[0], self.rs[1], self.rs[2]))

            v_pts = [[r[j][0], r[j][0] + v[j][0]/v_mag*self.scale] for j in range(len(r))]
            self.v_line.set_data_3d((v_pts[0], v_pts[1], v_pts[2]))

            T_pts = [[r[j][0], r[j][0] + u[j]*self.scale] for j in range(len(r))]
            self.T_line.set_data_3d((T_pts[0], T_pts[1], T_pts[2]))
            
            self.p_line.set_xdata(t[:i])
            self.p_line.set_ydata(p[:i])
            self.p_ax.set_xlim(0, t[i]+5)
            self.p_ax.set_ylim(p[0], p[i])
            
            self.f_line.set_xdata(t[:i])
            self.f_line.set_ydata(f[:i])
            # self.f_line.set_xlim(0, t[i]+5)
            # self.f_line.set_ylim(f[0], f[i])
            
            self.g_line.set_xdata(t[:i])
            self.g_line.set_ydata(g[:i])
            # self.g_line.set_xlim(0, t[i]+5)
            # self.g_line.set_ylim(g[0], g[i])
            
            self.fg_ax.set_xlim(0, t[i]+5)
            self.fg_ax.set_ylim(-1, 1)
            
            self.h_line.set_xdata(t[:i])
            self.h_line.set_ydata(h[:i])
            # self.h_line.set_xlim(0, t[i]+5)
            # self.h_line.set_ylim(h[0], h[i])
            
            self.k_line.set_xdata(t[:i])
            self.k_line.set_ydata(k[:i])
            # self.k_line.set_xlim(0, t[i]+5)
            # self.k_line.set_ylim(k[0], k[i])
            
            self.hk_ax.set_xlim(0, t[i]+5)
            self.hk_ax.set_ylim(-1, 1)
            
            self.L_line.set_xdata(t[:i])
            self.L_line.set_ydata(L[:i])
            self.L_ax.set_xlim(0, t[i]+5)
            self.L_ax.set_ylim(L[0], L[i])
            
            self.m_line.set_xdata(t[:i])
            self.m_line.set_ydata(m[:i])
            self.m_ax.set_xlim(0, t[i]+5)
            self.m_ax.set_ylim(m[i], m[0])

            return self.r_line, self.v_line, self.T_line, self.p_line, self.f_line, self.g_line, self.h_line, self.k_line, self.L_line, self.m_line
        
        self.ani = animation.FuncAnimation(self.fig, animate, frames=np.arange(1,len(t)))

        if self.animate:
            self.ani.save(self.savefile)
        else:
            plt.show()