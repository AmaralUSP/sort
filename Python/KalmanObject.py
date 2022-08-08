import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import utils

measurement = np.array([[[-393.66], [300.4]], [[-375.93], [301.78]], [[-351.04], [295.1]], [[-328.96], [305.19]], [[-299.35], [301.06]], [[-273.36], [302.05]], [[-245.89], [300]], [[-222.58], [303.57]], [[-198.03], [296.33]], [[-174.17], [297.65]], [[-146.32], [297.41]], [[-123.72], [299.61]],	[[-103.47], [299.6]], [[-78.23], [302.39]], [[-52.63], [295.04]], [[-23.34], [300.09]], [[25.96], [294.72]], [[49.72], [298.61]], [[76.94], [294.64]], [[95.38], [284.88]], [[119.83], [272.82]], [[144.01], [264.93]], [[161.84], [251.46]], [[180.56], [241.27]], [[201.42], [222.98]], [[222.62], [203.73]], [[239.4], [184.1]], [[252.51], [166.12]], [[266.26], [138.71]], [[271.75], [119.71]],	[[277.4], [100.41]], [[294.12], [79.76]], [[301.23], [50.62]],	[[291.8], [32.99]], [[299.89], [2.14]]])

class KalmanObject(object):
    #  Helper
    id = 0
    frame_count = 0

    # _uncert matriz covariancia da predicap
    # F modelo de movimento
    # Q incerteza do processo (modelo)
    # (measurement - (self.__H @ self.__state)) inovacao (i)
    # R incerteza da observacao
    # H matriz de transformacao de estados

    def __init__(self, sigma, convariance_uncert, init_pos):
        self.unique_id = self.id
                            # u  v  s  r  du dv ds 
        self.__F = np.array([[1, 0, 0, 0, 1, 0, 0],    # u -> x 
                             [0, 1, 0, 0, 0, 1, 0],    # v -> y
                             [0, 0, 1, 0, 0, 0, 1],    # s
                             [0, 0, 0, 1, 0, 0, 0],    # r
                             [0, 0, 0, 0, 1, 0, 0],    # du -> dx
                             [0, 0, 0, 0, 0, 1, 0],    # dv -> dy
                             [0, 0, 0, 0, 0, 0, 1]],   # ds
                             dtype=float) 

        self.__Q = np.eye(7, dtype=float)
        self.__Q[-1,-1] *= 0.01
        self.__Q[4:,4:] *= 0.01 

        self.__R = np.eye(4, dtype=float)
        self.__R[2:,2:] *= 10.

        self.__H = np.zeros((4, 7), dtype=float)

        # Update
        self.__k_gain = np.zeros((7, 4), dtype=float)
        self.__state = np.zeros((7,1), dtype=float)
        self.__state[:4] = init_pos
        self.__uncert = np.zeros((7, 7), dtype=float)

        self.__R[0][0] = 9
        self.__R[1][1] = 9
        self.__H[0][0] = 1
        self.__H[1][3] = 1

        np.fill_diagonal(self.__uncert, convariance_uncert)

        self.id = self.id+1

    def predict(self):
        # print(f"x {self.__state[0]} y {self.__state[1]} s {self.__state[2]} r {self.__state[3]} x` {self.__state[4]} y` {self.__state[5]} s` {self.__state[6]}")
        self.__state = self.__F @ self.__state
        # print(f"dps x {self.__state[0]} y {self.__state[1]} s {self.__state[2]} r {self.__state[3]} x` {self.__state[4]} y` {self.__state[5]} s` {self.__state[6]}")
        self.__uncert = self.__F @ self.__uncert @ self.__F.T + self.__Q 

    def update(self, measurement):
        self.__k_gain = self.__uncert @ self.__H.T @ inv((self.__H @ self.__uncert @ self.__H.T) + self.__R)
        # print(f"x {self.__state[0]} y {self.__state[1]} s {self.__state[2]} r {self.__state[3]} x` {self.__state[4]} y` {self.__state[5]} s` {self.__state[6]}")
        self.__state = self.__state + self.__k_gain @ (measurement - (self.__H @ self.__state))
        # print(f"dps x {self.__state[0]} y {self.__state[1]} s {self.__state[2]} r {self.__state[3]} x` {self.__state[4]} y` {self.__state[5]} s` {self.__state[6]}")
        self.__uncert = (np.eye(7, dtype=float) - (self.__k_gain @ self.__H)) @ self.__uncert

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