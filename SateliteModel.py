#satelite model
import numpy as np

class SateliteModel:


    def __init__(self, num_orbits = 6, 
                 satelitesPerOrbit = 4, 
                 sigma = 1.0, 
                 alpha = 0.001, 
                 quadrants = np.array([1,1,1,1])):
        self.sigma = sigma # noise standard deviation for range(meters)
        self.alpha = alpha # noise standard deviation for clock bias (sec)
        self.num_orbits = num_orbits
        self.quadrants = quadrants #quadrants that are visible to satelites
        self.satelitesPerOrbit = satelitesPerOrbit
        self.num_satelites = self.num_orbits*self.satelitesPerOrbit
        self.satelites = self.generateSatelites() #NX3 matrix whre N number satelites, x,y,z positions
        self.orbits = self.generateOrbits() #NX3 matrix of vectors orthogonal vector pointing to each satelite
        self.tau = self.alpha * np.random.uniform(-1,1,self.num_satelites) #clock bias of satelites
        self.numVisible = 0
        self.numObscured = self.num_satelites - self.numVisible
        self.visibleSatelites = np.array([])
        self.visibleIndices = np.array([])
        self.obscuredSatelites = np.array([])
        self.obscuredIndices = np.array([])
        self.visibleTau = np.array([]) 
        self.setVisibleSatelites()

    def getNumSatelites(self):
        return self.num_satelites

    def getLocations(self):
        return self.satelites

    def getVisibleLocations(self):
        if(self.numVisible == 0):
            return np.array([])
        return self.visibleSatelites

    def getVisibleIndices(self):
        return self.visibleIndices
    
    def getObscuredLocations(self):
        if(self.numVisible == 0):
            return np.array([])
        return self.visibleSatelites

    def getObscuredIndices(self):
        return self.obscuredIndices

    def getClockBias(self):
        return self.tau

    def getVisibleClockBias(self):
        if(self.numVisible == 0):
            return np.array([])
        return self.visibleTau

    def getOrbits(self):
        return self.orbits

    def propogateClockBias(self):
        self.tau += .000001*np.random.randn()
        self.tau[np.abs(self.tau) > self.alpha] = 0

    def generateSateliteOrbitGroup(self,k):
        sig = 1000.0 #devation from average altitude 1 km
        ave_altitude = 20200000.0 #20200 km
        sat_group = np.zeros((self.satelitesPerOrbit,3))
        for i in range(0,self.satelitesPerOrbit):
            thetax = i*(2.0/self.satelitesPerOrbit)*np.pi + k*(1/self.satelitesPerOrbit)*np.pi/2
            Rx = np.array([[1 , 0 , 0],
                            [0 , np.cos(thetax) , -np.sin(thetax)],
                            [0 , np.sin(thetax) , np.cos(thetax)]])
            r = ave_altitude + sig * np.random.randn()
            sat_group[i] = np.dot(Rx, np.array([0,0,1])*r)
        return sat_group.T

    def generateSatelites(self):
        spacing_angle = 360/self.num_orbits
        thetay = 55 * np.pi/180.0
        Ry = np.array([[np.cos(thetay) , 0 , np.sin(thetay)],
                       [0 , 1 , 0],
                       [-np.sin(thetay), 0 , np.cos(thetay)] ])
        points = np.zeros((self.num_satelites, 3))
        for i in range(0,self.num_orbits):
            sat_group = self.generateSateliteOrbitGroup(i)
            sat_group = np.dot(Ry , sat_group)
            thetax = i*spacing_angle*np.pi/180.0
            Rx = np.array([[1 , 0 , 0],
                            [0 , np.cos(thetax) , -np.sin(thetax)],
                            [0 , np.sin(thetax) , np.cos(thetax)]])
            sat_group = np.dot(Rx , sat_group)
            for k in range(0,self.satelitesPerOrbit):
                points[self.satelitesPerOrbit*i+k] = (sat_group[:,k]).flatten()
        return points

        #returns the orbit, angular speed and direction of each satelite (rad/sec)
    def generateOrbits(self):
        spacing_angle = 360/self.num_orbits
        thetay = 55 * np.pi/180.0
        Ry = np.array([[np.cos(thetay) , 0 , np.sin(thetay)],
                       [0 , 1 , 0],
                       [-np.sin(thetay), 0 , np.cos(thetay)] ])
        orbits = np.zeros((self.num_satelites, 3))
        for i in range(0,self.num_orbits):
            orbit = np.array([1,0,0])
            orbit = np.dot(Ry , orbit)
            thetax = i*spacing_angle*np.pi/180.0
            Rx = np.array([[1 , 0 , 0],
                            [0 , np.cos(thetax) , -np.sin(thetax)],
                            [0 , np.sin(thetax) , np.cos(thetax)]])
            orbit = np.dot(Rx , orbit)
            for k in range(0,self.satelitesPerOrbit):
                orbits[self.satelitesPerOrbit*i+k] = orbit
        return orbits

    #set satelites visible where quadrants q1,q2,q3,q4 are true (xy quadrants). Not visible where z < 0
    def setVisibleSatelites(self):
        radius_earth = 6.371*10**6
        q1 = self.quadrants[0]
        q2 = self.quadrants[1]
        q3 = self.quadrants[2]
        q4 = self.quadrants[3]
        truth_bin = np.zeros(self.num_satelites)
        x = self.satelites[:,0]
        y = self.satelites[:,1]
        z = self.satelites[:,2]
        if(q1):
            truth_bin[ np.all([x >= 0, y >= 0],axis=0) ] = 1
        if(q2):
            truth_bin[np.all([x <= 0, y >= 0],axis=0)] = 1
        if(q3):
            truth_bin[np.all([x <= 0, y <= 0],axis=0)] = 1
        if(q4):
            truth_bin[np.all([x >= 0, y <= 0],axis=0)] = 1
        truth_bin[z < radius_earth] = 0
        self.visibleIndices = np.where(truth_bin > 0)[0]
        self.obscuredIndices = np.where(truth_bin == 0)[0]
        self.numVisible = np.size(self.visibleIndices)
        self.numObscured = np.size(self.obscuredIndices)
        self.visibleSatelites = self.satelites[self.visibleIndices,:]
        self.obscuredSatelites = self.satelites[self.obscuredIndices,:]
        self.visibleTau = self.tau[self.visibleIndices]

    #rotates the satelites around the center of the grid map
    def propogateSatelites(self,dt):
        self.propogateClockBias()
        sec_day = 86400.0
        revolutions_day = 1.0
        omega = 2 * np.pi * revolutions_day / sec_day #rad/sec
        theta = omega * dt
        #rodrigues rotation formula
        I = self.satelites[:,1]*self.orbits[:,2] - self.satelites[:,2]*self.orbits[:,1]
        J = -(self.satelites[:,0]*self.orbits[:,2] - self.satelites[:,2]*self.orbits[:,0])
        K = self.satelites[:,0]*self.orbits[:,1] - self.satelites[:,1]*self.orbits[:,0]
        cross_result = np.concatenate( (I[:,None],J[:,None],K[:,None]) , 1)
        dot_result = np.sum(self.satelites*self.orbits,axis=1)[:,None]
        self.satelites = self.satelites*np.cos(theta) + cross_result*np.sin(theta) + self.orbits * np.tile(dot_result,(1,3)) * (1 - np.cos(theta))
        self.setVisibleSatelites()

    #returns tuple with an array of distance to each satelite
    def getTransmissionTime(self, reference, t_reciever):
    #reference is a 1x3 vector containing the cartesian coordinates of the reference
        if(self.numVisible == 0):
            return np.inf
        c = 299792458.0 #speed of light (m/s)
        true_Range = np.sqrt((reference[0] - self.visibleSatelites[:,0])**2 + \
                        (reference[1] - self.visibleSatelites[:,1])**2 + \
                        (reference[2] - self.visibleSatelites[:,2])**2)
        #add noise to measurements
        Range = true_Range + self.sigma * np.random.randn((self.numVisible))
        t_transmission = t_reciever - Range/c #time the transmission was sent with noise
        T_transmission = t_transmission + self.visibleTau #time transition was sent with noise and clock bias
        return T_transmission

'''
    def generateRandomSatelites(self):
        ave_altitude = 20200000.0 #20200 km
        sig = 1000.0 #devation from average altitude 1 km
        R = np.zeros(self.num_satelites) + ave_altitude + sig * np.random.randn(self.num_satelites)
        x = np.random.uniform(-1,1,(self.num_satelites)) * R
        y = np.sqrt(R**2 - x**2) * np.random.uniform(-1,1,(self.num_satelites))
        z = np.sqrt(R**2 - x**2 - y**2) * np.random.choice([-1,1],(self.num_satelites))
        return np.array([x,y,z]).T

    #returns the orbit, angular speed and direction of each satelite (rad/sec)
    def generateRandomOrbits(self):
        v = np.zeros((self.num_satelites,3))
        for i in range(0,self.num_satelites):
             k = np.random.randn(3) 
             v[i] = np.cross(k,self.satelites[i])
        v = v / np.tile( np.linalg.norm(v,axis=1)[:,None] , (1,3))
        return v
'''