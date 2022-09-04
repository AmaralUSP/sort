import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import utils
import random
from time import time

measurement = np.array([[[-393.66], [300.4]], [[-375.93], [301.78]], [[-351.04], [295.1]], [[-328.96], [305.19]], [[-299.35], [301.06]], [[-273.36], [302.05]], [[-245.89], [300]], [[-222.58], [303.57]], [[-198.03], [296.33]], [[-174.17], [297.65]], [[-146.32], [297.41]], [[-123.72], [299.61]],	[[-103.47], [299.6]], [[-78.23], [302.39]], [[-52.63], [295.04]], [[-23.34], [300.09]], [[25.96], [294.72]], [[49.72], [298.61]], [[76.94], [294.64]], [[95.38], [284.88]], [[119.83], [272.82]], [[144.01], [264.93]], [[161.84], [251.46]], [[180.56], [241.27]], [[201.42], [222.98]], [[222.62], [203.73]], [[239.4], [184.1]], [[252.51], [166.12]], [[266.26], [138.71]], [[271.75], [119.71]],	[[277.4], [100.41]], [[294.12], [79.76]], [[301.23], [50.62]],	[[291.8], [32.99]], [[299.89], [2.14]]])

class KalmanObject(object):

    # _uncert matriz covariancia da predicao
    # Q incerteza do processo (modelo)
    # (measurement - (self.__H @ self.__state)) inovacao (i)
    # R incerteza da observacao
    # H matriz de transformacao de estados

    def __init__(self, sigma, convariance_uncert, init_pos, id):
        self.frames_without_dets = 0
        self.unique_id = id
        self.d_t = 0.033      
        
        self.cov_q = np.array([0.01, 0.01, 0.01, 0.001, 0.05, 0.05, 0.01])

        self.__Q = np.eye(7, dtype=float) * self.cov_q
        
        self.__R = np.eye(4, dtype=float)
        self.__R[2:,2:] *= 10.
                            # x  y  s  r  vx vy ds  
        self.__H = np.array([[1, 0, 0, 0, 0, 0, 0],  # x
                             [0, 1, 0, 0, 0, 0, 0],  # y 
                             [0, 0, 1, 0, 0, 0, 0],  # s
                             [0, 0, 0, 1, 0, 0, 0]]) # r

        # Update
        self.__state = np.zeros((7,1), dtype=float)
        self.__state[:4] = init_pos
        self.__uncert = np.zeros((7, 7), dtype=float)

        self.__R[0][0] = 9
        self.__R[1][1] = 9

        np.fill_diagonal(self.__uncert, convariance_uncert)
        
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        self.color = (r,g,b)

    def predict(self):

                             # x  y  s  r  vx vy ds 
        jacobiana = np.array([[1, 0, 0, 0, self.d_t, 0, 0],     # x  = x_0 + vx * delta_t
                              [0, 1, 0, 0, 0, self.d_t, 0],     # y  = y_0 + vy * delta_t
                              [0, 0, 1, 0, 0, 0, 1],            # s  = s_0 + ds
                              [0, 0, 0, 1, 0, 0, 0],            # r  = r_0
                              [1/self.d_t, 0, 0, 0, 0, 0, 0],   # vx = (x - x_0) / delta_t
                              [0, 1/self.d_t, 0, 0, 0, 0, 0],   # vy = (y - y_0) / delta_t
                              [0, 0, 0, 0, 0, 0, 1]],           # ds = ds
                              dtype=float) 

        x = self.__state[0] + self.__state[4] * self.d_t
        y = self.__state[1] + self.__state[5] * self.d_t
        s = self.__state[2] + self.__state[6]
        r = self.__state[3]
        vx = (x - self.__state[0])/self.d_t
        vy = (y - self.__state[1])/self.d_t
        ds = self.__state[6]

        self.__state = np.array([x,y,s,r,vx,vy,ds]).reshape((7,1))
        self.__uncert = jacobiana @ self.__uncert @ jacobiana.T + self.__Q 

        self.frames_without_dets += 1

    def is_valid(self, time_without_dets):
        return self.frames_without_dets < time_without_dets and self.getState()[2] > 2500

    def update(self, measurement):
        innovation = measurement - self.__H @ self.__state
        cov_innovation = self.__H @ self.__uncert @ self.__H.T + self.__R
        k_gain = self.__uncert @ self.__H.T @ inv(cov_innovation)
        self.__state = self.__state + k_gain @ innovation
        self.__uncert = (np.eye(7, dtype=float) - (k_gain @ self.__H)) @ self.__uncert

        self.frames_without_dets = 0

    def getState(self):
        return self.__state

    def getId(self):
        return self.unique_id

def main():
    x = measurement[:,0]
    y = measurement[:,1]
    predicted = []
    updated = []

    init_pos = [-100, 100, 100, -100] 
    init_pos = utils.convert_bbox_to_z(init_pos)
    KO = KalmanObject(2, 500, init_pos)

    x = np.arange(100)
    y = 0.5*x**2
    
    for xi, yi in zip(x,y):    
        KO.predict()
        predicted.append(KO.getState())

        obs = [xi-100, yi+100, xi+100, yi-100] 
        obs = utils.convert_bbox_to_z(obs)

        KO.update(obs)
        updated.append(KO.getState())

        KO.frame_count += 1

    predicted = np.array(predicted)
    x_p = predicted[2:,0]
    y_p = predicted[2:,3]

    updated = np.array(updated)
    x_u = updated[2:,0]
    y_u = updated[2:,3]

    plt.plot(x, y)
    plt.plot(x_u, y_u)
    plt.plot(x_p, y_p)

    plt.show()

if __name__ == "__main__":
    main()