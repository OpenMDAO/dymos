import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys

def get_r(elements):
    p = elements['p']
    f = elements['f']
    g = elements['g']
    h = elements['h']
    k = elements['k']
    L = elements['L']
    
    q = 1 + f*np.cos(L) + g*np.sin(L)
    r_mag = p/q
    alpha_squared = h**2 - k**2
    chi = np.sqrt(h**2 + k**2)
    s_squared = 1 + chi**2
    
    r = np.array([[(r_mag/s_squared)*(np.cos(L) + alpha_squared*np.cos(L) + 2*h*k*np.sin(L))],
                  [(r_mag/s_squared)*(np.sin(L) - alpha_squared*np.sin(L) + 2*h*k*np.cos(L))],
                  [(2*r/s_squared)*(h*np.sin(L) - k*np.cos(L))]])

file_name = 'orbital_elements.txt'
for i in range(1, len(sys.argv), 2):
    if sys.argv[i] == '-f':
        file_name = sys.argv[i+1]

with open(file_name, 'r') as file:
    states = {}
    for line in enumerate(file):
        line = line.strip()
        
            


fig = plt.figure()

#####################
#           p       f
#   ORBIT   g       h
#           k       L
#               m
#####################

orbit_ax = fig.add_subplot(projection='3d')
p_ax = fig.add_subplot()
f_ax = fig.add_subplot()
g_ax = fig.add_subplot()
h_ax = fig.add_subplot()
k_ax = fig.add_subplot()
L_ax = fig.add_subplot()
m_ax = fig.add_subplot()

