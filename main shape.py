#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 00:50:40 2020

@author: harshita
"""
import  numpy as np
z=int(1)
a=int(5.29*10**(-11))
def dxy(r,theta,phi):
        rp =2/((81*(3.14)**(0.5))*(z)**(3/2))
        e = (z*r/a)**2*np.exp((-1)*z*r/a*3)
        ang =np.sin(theta)*np.cos(theta)*np.cos(phi)
        psi = rp*e*ang
rl=[]
r_x=[]
r_y=[]
r_z=[]
n=3
l=2
m=1
plot_range = int((4*n+4*l)*a)
for x in range (-plot_range,plot_range):
    for y in range (-plot_range,plot_range):
        for z in range (-plot_range,plot_range):
            r= np.sqrt(x**2+y**2+z**2)
            rl.append(r)
            
             
theta = np.r_[0:np.pi:50j]                  
phi = np.r_[0:2*np.pi:50j]
radius = np.r_[0:1:50j]
x = np.array([r * np.cos(theta) for r in radius])
y = np.array([r * np.sin(theta) for r in radius])
z = np.array([dxy(r,theta, phi) for r in radius])
z2 = np.array([dxy(r,theta, phi) for r in radius])
z2.append(z)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, z2, rstride=1, cstride=1, cmap=cm.jet)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()