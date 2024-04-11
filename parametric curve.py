#%%
import math
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib
#PARAMETRIC CURVE
mpl.rcParams['legend.fontsize']=10
fig= plt.figure(figsize=(10,10))
ax=fig.gca(projection='3d')
theta=np.linspace(-4*np.pi,4*np.pi,100)
z=np.linspace(-2,2,100)
r= z**2 + 1
x=r*np.sin(theta)
y=r*np.cos(theta)
ax.plot(x,y,z,label='parametric curve')
ax.legend()
plt.show()

# %%

def randrange(n,vmin,vmax):
    return(vmax-vmin)*np.random.rand(n)+vmin
fig = plt.figure(figsize=(10,10))
u=np.linspace(0,2*np.pi,100)
v=np.linspace(0,np.pi,100)
x=10*np.outer(np.cos(u),np.sin(v))
y=10*np.outer(np.sin(u),np.sin(v))
z=10*np.outer(np.ones(np.size(u)),np.cos(v))
ax.plot_surface(x,y,z,rstride=4,cstride=4,color='b')
plt.show()


# %%
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make the grid
x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.1),
                      np.arange(-0.8, 1, 0.4))

# Make the direction data for the arrows
u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
     np.sin(np.pi * z))

ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

plt.show()


# %%
