import numpy as np
from numpy import sin as s 
from numpy import cos as c 

def R_PQW2IJK(Omega, w, i):
    R = np.array([[c(Omega)*c(w) + s(Omega)*s(w)*c(i), c(Omega)*c(w) - s(Omega)*c(w)*c(i), s(Omega)*s(i)],
                  [s(Omega)*c(w) + c(Omega)*s(w)*c(i), -s(Omega)*s(w) + c(Omega)*c(w)*c(i), -c(Omega)*s(i)],
                  [s(Omega)*s(i), c(w)*s(i), c(i)]])
    return R

def R_PQW2RSW(L):
    R = np.array([[c(L), -s(L), 0],
                  [s(L), c(L), 0],
                  [0, 0, 1]])
    return R

def Qr(r, v):
    r_mag = np.linalg.norm(r)
    rxv = np.cross(r, v)
    
    i_r = r/r_mag
    i_theta = np.cross(rxv, r)/(np.linalg.norm(rxv)*r_mag)
    i_h = rxv/(np.linalg.norm(rxv))
    
    R = np.array([i_r, i_theta, i_h])
    
    return R

# r = np.array([1.0, 2.0, 3.0])
# v = np.array([1.0, 0.0, 0.0])

# R = Qr(r, v)

# print(R)

# Omega = np.pi
# w = -0.5
# i = 0.5
# L = 4.5

# R = np.linalg.inv(R_PQW2IJK(Omega, w, i)) @ R_PQW2RSW(L)

# print(R)