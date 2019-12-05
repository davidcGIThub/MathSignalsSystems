#satelite animation
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from SateliteModel import SateliteModel as sat
from GPS_Reciever import GPS_Reciever as reciever
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from time import time

##### input parameters

num_orbits = 6
satelitesPerOrbit = 4
t = 86400.0 #sec
dt = 1000
time_steps = int(t/dt)

ave_altitude = 20200000.0 #20200 km

### retrieve data ######

satelites = sat(num_orbits,satelitesPerOrbit)
num_satelites = satelites.getNumSatelites()
sat_locs = satelites.getLocations()
satelite_locations = np.zeros((num_satelites,3,time_steps))
visibility_indices = np.zeros((time_steps,num_satelites)).astype(int)
visibility_flag = np.zeros(time_steps)
for i in range(0,time_steps):
    satelite_locations[:,:,i] = sat_locs
    satelites.propogateSatelites(dt)
    sat_locs = satelites.getLocations()
    vis_ind = satelites.getVisibleIndices()
    obs_ind = satelites.getObscuredIndices()
    visibility_indices[i] = (np.concatenate((vis_ind,obs_ind))).astype(int)
    visibility_flag[i] = np.size(vis_ind)

##### create earth ######
radius_earth = 6.371*10**6
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = radius_earth * np.outer(np.cos(u), np.sin(v))
y = radius_earth * np.outer(np.sin(u), np.sin(v))
z = radius_earth * np.outer(np.ones(np.size(u)), np.cos(v))

#create reference point
radius = 500000
ur = np.linspace(0, 2 * np.pi, 100)
vr = np.linspace(0, np.pi, 100)
xr = radius * np.outer(np.cos(u), np.sin(v))
yr = radius * np.outer(np.sin(u), np.sin(v))
zr = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + radius_earth

#create equator
p = np.linspace(0, 2 * np.pi, 100)
ey = radius_earth * np.cos(p)
ez = radius_earth * np.sin(p)
ex = p*0

###### plotting ##########

fig = plt.figure()
ax = p3.Axes3D(fig)
visible_satelite_figs, = ax.plot([],[],[], 'bo',color='b')
obscured_satelite_figs, = ax.plot([],[],[], 'bo', color='r')

#earth
ax.plot_surface(x, y, z, color='g',alpha=0.25)
#reference point
ax.plot_surface(xr, yr, zr, color='k')
#axis of rotation
ax.plot([-3*radius_earth,3*radius_earth], [0,0], [0,0], color='g')
#equator
ax.plot(p,ey,ez, color='g')

for i in range(0,num_satelites):
    ax.plot3D(satelite_locations[i,0,:],satelite_locations[i,1,:],satelite_locations[i,2,:],color='k',alpha=0.25)

# Setting the axes properties
ax.set_xlim3d([-ave_altitude-1000000, ave_altitude+1000000])
ax.set_xlabel('X')

ax.set_ylim3d([-ave_altitude-1000000, ave_altitude+1000000])
ax.set_ylabel('Y')

ax.set_zlim3d([-ave_altitude-1000000, ave_altitude+1000000])
ax.set_zlabel('Z')

# Hide grid lines
ax.grid(False)

# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

def update(num,satelite_locations,visible_satelite_figs,obscured_satelite_figs,visibility_indices,visibility_flag):
    #visible satelites
    marker = int(visibility_flag[num])
    vis_ind = visibility_indices[num,0:marker]
    vis_sat_locs = satelite_locations[vis_ind,:,num].T
    visible_satelite_figs.set_data(vis_sat_locs[:2, :])
    visible_satelite_figs.set_3d_properties(vis_sat_locs[2,:])
    #obscured satelites
    obs_ind = visibility_indices[num,marker:]
    obs_sat_locs = satelite_locations[obs_ind,:,num].T
    obscured_satelite_figs.set_data(obs_sat_locs[:2, :])
    obscured_satelite_figs.set_3d_properties(obs_sat_locs[2,:])

ani = animation.FuncAnimation(fig, update, time_steps, fargs=(satelite_locations, visible_satelite_figs,obscured_satelite_figs,visibility_indices,visibility_flag),
                                 interval=dt, blit=False)
plt.show()
