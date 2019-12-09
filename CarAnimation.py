#CarAnimation
import numpy as np
from UnicycleMotionModel import UnicycleMotionModel as umd
import matplotlib.animation as animation
import matplotlib.pyplot as plt 
from SateliteModel import SateliteModel as sat
from GPS_Reciever import GPS_Reciever as reciever

####### initialization parameters #######
#time
time = 30.0
dt = 0.1
time_steps = int(time/dt)
t = np.linspace(0,time,time_steps)
#satelites
num_orbits = 6
satelitesPerOrbit = 4
reciever_clock_bias = 0.0005
numVisibleSatelites = np.zeros(time_steps)
#pose
radius_earth = 6.371*10**6
x0 = 0.0 # along line of longitude
y0 = 0.0 # along line of latitude
z0 = radius_earth # direction of altitude 
theta0 = 0.0 
pose2D = np.array([x0,y0,theta0])
altitude = z0
#input
v = 10*np.abs(2*np.sin(t))
w = np.abs(np.cos(t/4))
u = np.concatenate((v[:,None],w[:,None]),axis=1)
#plots
animate = False
x_lim = 100
y_lim = 100
size_car = 5.0 #meters
size_gpsReciever = 5

######## initialize models #######
#car
car = umd()
carOutline = car.getFigurePoints(pose2D,size_car)
carOutline_data = np.zeros((3,2,time_steps))
pose2D_data = np.zeros((time_steps,3))
#satelites
satelites = sat(num_orbits,satelitesPerOrbit)
gpsEstimate = np.array([2,4,radius_earth+100])
gpsEstimate_data = np.zeros((time_steps,3))
gps = reciever(gpsEstimate, t[0], reciever_clock_bias) 

#### simulate model ######
for i in range(0,time_steps):
    pose2D_data[i] = pose2D
    carOutline_data[:,:,i] = carOutline
    sat_locs = satelites.getVisibleLocations()
    sat_biases = satelites.getVisibleClockBias()
    coordinates = np.array([pose2D[0],pose2D[1],altitude])
    numVisibleSatelites[i] = satelites.getNumVisible()
    transmission_time = satelites.getTransmissionTime(coordinates, t[0])
    gpsEstimate_data[i] = gps.estimatePosition(t[i],transmission_time,sat_biases,sat_locs)
    pose2D = car.propagate(pose2D,u[i],dt)
    carOutline = car.getFigurePoints(pose2D,size_car)
    satelites.propogateSatelites(dt)

####### Animation ########
if animate == True:
    #initialize figures
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(-x_lim, x_lim), ylim=(-y_lim, y_lim))
    car_fig = plt.Polygon(carOutline_data[:,:,0],fc = 'b')
    gpsReciever_fig, = ax.plot([],[], 'bo', ms=size_gpsReciever,color='g'); 

    ##### show animation ######
    def init():
        ax.add_patch(car_fig)
        return car_fig, gpsReciever_fig

    def update(time_steps,car_fig,carOutline_data,gpsReciever_fig,gpsEstimate_data):
        car_fig.xy = carOutline_data[:,:,time_steps]
        gpsReciever_fig.set_data(gpsEstimate_data[time_steps,0:2])

    ani = animation.FuncAnimation(fig, update, time_steps, fargs=(car_fig,carOutline_data,gpsReciever_fig,gpsEstimate_data),
                                 interval=10, blit=False,init_func = init)
    plt.show()

gpsReciever_x = gpsEstimate_data[:,0]
gpsReciever_y = gpsEstimate_data[:,1]
gpsReciever_z = gpsEstimate_data[:,2]

truth_x = pose2D_data[:,0]
truth_y = pose2D_data[:,1]
truth_z = t*0 + altitude

figure1, ax = plt.subplots()
ax.plot(truth_x,truth_y,color='b',label="truth")
ax.plot(gpsReciever_x,gpsReciever_y,color='g',label="estimate")
ax.legend()
ax.set(xlabel = 'x position (m)')
ax.set(ylabel = 'y position (m)')
true_fig = plt.Polygon(carOutline,fc = 'b')
ax.add_patch(true_fig)
ax.set_title("Vehicle Path and GPS Estimate")
plt.show()

figure2, (ax3) = plt.subplots()
#ax1.plot(t[2:],truth_x[2:], label = 'true')
#ax1.plot(t[2:],gpsReciever_x[2:], label = 'estimate')
#ax1.legend()
#ax1.set(ylabel = 'x position (m)')
#ax2.plot(t[2:],truth_y[2:])
#ax2.plot(t[2:],gpsReciever_y[2:])
#ax2.set(ylabel = 'y position (m)')
ax3.legend()
ax3.plot(t[2:],truth_z[2:],label="truth")
ax3.plot(t[2:],gpsReciever_z[2:],label="estimate")
ax3.legend()
ax3.set(xlabel = 'z position (m)')
ax3.set(ylabel = 'time (sec)')
ax3.set_title("GPS Estimate of Altitude")
#ax4.plot(t,numVisibleSatelites)
#ax4.set(xlabel = 'Number of Visible Satelites')
#ax4.set(ylabel = 'time (sec)')
plt.show()


figure3, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(t[2:],(truth_x-gpsReciever_x)[2:], color = 'r')
ax1.set(ylabel = 'x error (m)')
ax2.plot(t[2:],(truth_y-gpsReciever_y)[2:], color = 'r')
ax2.set(ylabel = 'y error (m)')
ax3.plot(t[2:],(truth_z-gpsReciever_z)[2:], color = 'r')
ax3.set(ylabel = 'z error (m)')
ax1.set_title("GPS Estimate Error")
plt.show()
