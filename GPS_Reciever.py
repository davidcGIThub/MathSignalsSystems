#GPS_Reciever
import numpy as np


class GPS_Reciever:

    #cartesian coordinates of the reciever
    def __init__(self, coordinates, current_time, clock_bias):
        self.coordinates = coordinates
        self.current_time = current_time
        self.clock_bias = clock_bias

 
    #update the new coordinates (belief)
    def updateCoordinates(self, newCoordinates):
        self.coordinates = newCoordinates

    def updateTime(self, time):
        self.current_time = time

    def updateClockBias(self, newBias):
        self.clock_bias = newBias

    #returns the estimated position of the reference using least squares method for solving the trilateration equations
    #if solution is underdetermined, returns array of possible solutions
    def estimatePosition(self,time, t_transmission, tau_transmission, satelites): 
        num_satelites = np.size(satelites,0)
        if(num_satelites == 0):
            print("No satelites visible")
            return self.coordinates, self.clock_bias
        if(np.size(t_transmission) != num_satelites or np.size(tau_transmission) != num_satelites):
            print("Error size of transmission, and number of satelites not equal")
        c = 299792458.0 #speed of light (m/s)
        P = c*(time - t_transmission) #psuedorange
        row = np.sqrt((satelites[:,0] - self.coordinates[0])**2 + (satelites[:,1] - self.coordinates[1])**2 + \
                      (satelites[:,2] - self.coordinates[2])**2)
        Pc = row + c * (self.clock_bias - tau_transmission)
        dpdx = (self.coordinates[0] - satelites[:,0]) / row
        dpdy = (self.coordinates[1] - satelites[:,1]) / row
        dpdz = (self.coordinates[2] - satelites[:,2]) / row
        dpdtau = np.repeat(c,num_satelites)
        #least squares coeficient calculation
        x = (P - Pc)[:,None]
        A = np.concatenate((dpdx[:,None] , dpdy[:,None] , dpdz[:,None] , dpdtau[:,None]),axis=1)
        C = np.dot(np.linalg.pinv(A),x)
        state = np.concatenate(  (self.coordinates , np.array([self.clock_bias]) )  ,  axis=0  )
        state_est = C + state[:,None]
        est_coordinates = state_est[0:3].flatten()
        est_clock_bias = state_est[3][0]
        self.updateTime(time)
        self.updateClockBias(est_clock_bias)
        self.updateCoordinates(est_coordinates)
        return est_coordinates

    def pinvSVD(self, A):
        U, Sigma, V = np.linalg.svd(A)
        