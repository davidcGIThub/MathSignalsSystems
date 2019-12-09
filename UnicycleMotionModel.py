import numpy as np

class UnicycleMotionModel:

    def __init__(self, a=1.0):
        self.a = a # noise characteristics

    def propagate(self, x, u, dt):
        new_x = np.copy(x)
        new_x[0] += u[0]*np.cos(x[2])*dt
        new_x[1] += u[0]*np.sin(x[2])*dt
        new_x[2] += u[1]*dt
        return new_x

    def sample(self, x, u, dt):
        noisy_u = np.copy(u)
        noisy_u[0] += np.random.randn()*(self.a[0]*u[0]**2+self.a[1]*u[1])
        noisy_u[1] += np.random.randn()*(self.a[2]*u[0]**2+self.a[3]*u[1])
        noisy_u[2] += np.random.randn()*(self.a[4]*u[0]**2+self.a[5]*u[1])
        return self.propagate(x, noisy_u, dt)

    def getFigurePoints(self,X,fig_size):
        x = X[0]
        y = X[1]
        theta = X[2]
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        xy = np.array([[-1, 1, -1],
                       [.5, 0, -0.5]])*fig_size
        xy = np.dot(R,xy)
        xy = xy + np.array([[x],[y]])
        return np.transpose(xy)

    def dgdx(self, x, u, dt):
        return np.array([[1, 0, -u[0]*np.sin(x[2])*dt]
                         [0, 1,  u[0]*np.cos(x[2])*dt]
                         [0, 0,          1           ]])

    def dgdu(self, x, u, dt):
        return np.array([[np.cos(x[2])*dt,  0]
                         [np.sin(x[2])*dt,  0]
                         [              0, dt]])
