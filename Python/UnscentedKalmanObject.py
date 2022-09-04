from asyncio import FastChildWatcher
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import utils
import random
from scipy.linalg import sqrtm

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
        self.n_states = 7
        self.n_sig = 1 + self.n_states * 2
        self.alpha = 1e-3
        self.beta = 2
        self.k = 3 - self.n_states

        self.lambd = pow(self.alpha, 2) * (self.n_states + self.k) - self.n_states

        self.__state = np.zeros((7,1), dtype=float)
        self.__state[:4] = init_pos

        self.covar_weights = np.zeros(self.n_sig)
        self.mean_weights = np.zeros(self.n_sig)

        self.covar_weights[0] = (self.lambd / (self.n_states + self.lambd)) + (1 - pow(self.alpha, 2) + self.beta)
        self.mean_weights[0] = (self.lambd / (self.n_states + self.lambd))

        for i in range(1, self.n_sig):
            self.covar_weights[i] = 1 / (2*(self.n_states + self.lambd))
            self.mean_weights[i] = 1 / (2*(self.n_states + self.lambd))
        
        self.P = np.zeros((7, 7), dtype=float)
        np.fill_diagonal(self.P, convariance_uncert)

        self.sigmas = self.calculate_sigma()

        self.cov_q = np.array([0.01, 0.01, 0.01, 0.001, 0.05, 0.05, 0.01])

        self.__Q = np.eye(7, dtype=float) * self.cov_q

        self.__R = np.eye(4, dtype=float)
        self.__R[2:,2:] *= 10.

        self.__R[0][0] = 9
        self.__R[1][1] = 9
        
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        self.color = (r,g,b)

    def predict(self): 
        temp_sig = [self.model(s) for s in self.sigmas]
        temp_sig = np.array(temp_sig)
        predicted_mean = np.zeros_like(self.__state)
        predicted_cov = np.zeros_like(self.P)

        predicted_mean = np.sum(np.expand_dims(self.mean_weights, axis=1) * temp_sig, axis=0)

        for i in range(0, self.n_sig):
            helper = np.expand_dims(temp_sig[i] - predicted_mean, axis=1)
            predicted_cov += self.covar_weights[i] * helper @ helper.T

        predicted_cov += self.__Q
            
        self.__state = np.expand_dims(predicted_mean, axis=1)
        self.P = predicted_cov
        self.sigmas = temp_sig

        self.frames_without_dets += 1

    def is_valid(self, time_without_dets):
        return self.frames_without_dets < time_without_dets and self.getState()[2] > 2500

    def update(self, measurement):
        temp_observation = [self.observation_model(s) for s in self.sigmas]
        temp_observation = np.asarray(temp_observation)
        observation_cov = np.zeros_like(self.__R)
        cross_correlation = np.zeros((7,4))

        mean_observation = np.sum(np.expand_dims(self.mean_weights, axis=1) * temp_observation, axis=0)

        for i in range(0, self.n_sig):
            helper = np.expand_dims(temp_observation[i] - mean_observation, axis=1)
            observation_cov += self.covar_weights[i] * helper @ helper.T

        observation_cov += self.__R

        for i in range(0, self.n_sig):
            helper = np.expand_dims(self.sigmas[i], axis=1) - self.__state
            helper_2 = np.expand_dims(temp_observation[i] - mean_observation, axis=1)
            cross_correlation += self.covar_weights[i] * helper @ helper_2.T

        kalman_gain = cross_correlation @ np.linalg.inv(observation_cov)

        innovation = measurement - np.expand_dims(mean_observation, axis=1)

        self.__state = self.__state + kalman_gain @ innovation
        self.sigmas = self.calculate_sigma()
        # self.P = (np.eye(self.n_states) - cross_correlation @ kalman_gain.T) @ self.P
        self.P -= kalman_gain @ (observation_cov @ kalman_gain.T)

        self.frames_without_dets = 0

    def calculate_sigma(self):
        sigmas = np.zeros((self.n_sig, self.n_states))
        plau = (self.n_states+self.lambd) * self.P
        
        sqrt_matrix = sqrtm(plau)
        
        sigmas[0] = np.squeeze(self.__state)
        for x in range(self.n_states):
            sigmas[1+x] = np.squeeze(self.__state) + sqrt_matrix[x]
            sigmas[1+x+self.n_states] = np.squeeze(self.__state) - sqrt_matrix[x]

        return sigmas

    def observation_model(self, z):
        ret = np.zeros((4,))
        
        ret[0] = z[0]
        ret[1] = z[1]
        ret[2] = z[2]
        ret[3] = z[3]

        return ret 

    def model(self, q):
        ret = np.zeros_like(q)

        ret[0] = q[0] + q[4] * self.d_t
        ret[1] = q[1] + q[5] * self.d_t
        ret[2] = q[2] + q[6]
        ret[3] = q[3]
        ret[4] = (ret[0] - q[0])/self.d_t
        ret[5] = (ret[1] - q[1])/self.d_t
        ret[6] = q[6]

        return ret

    def getState(self):
        return self.__state

    def getId(self):
        return self.unique_id

def main():
    measurement = np.array([[[-393.66], [300.4]], [[-375.93], [301.78]], [[-351.04], [295.1]], [[-328.96], [305.19]], [[-299.35], [301.06]], [[-273.36], [302.05]], [[-245.89], [300]], [[-222.58], [303.57]], [[-198.03], [296.33]], [[-174.17], [297.65]], [[-146.32], [297.41]], [[-123.72], [299.61]],	[[-103.47], [299.6]], [[-78.23], [302.39]], [[-52.63], [295.04]], [[-23.34], [300.09]], [[25.96], [294.72]], [[49.72], [298.61]], [[76.94], [294.64]], [[95.38], [284.88]], [[119.83], [272.82]], [[144.01], [264.93]], [[161.84], [251.46]], [[180.56], [241.27]], [[201.42], [222.98]], [[222.62], [203.73]], [[239.4], [184.1]], [[252.51], [166.12]], [[266.26], [138.71]], [[271.75], [119.71]],	[[277.4], [100.41]], [[294.12], [79.76]], [[301.23], [50.62]],	[[291.8], [32.99]], [[299.89], [2.14]]])

    x = measurement[:,0]
    y = measurement[:,1]
    predicted = []
    updated = []

    init_pos = [-100, 100, 100, -100] 
    init_pos = utils.convert_bbox_to_z(init_pos)
    new_obj = KalmanObject(500, 10, init_pos, 1)

    x = np.arange(100)
    y = 0.5*x**2
    
    for xi, yi in zip(x,y):    
        new_obj.predict()
        predicted.append(new_obj.getState())

        obs = [xi-100, yi+100, xi+100, yi-100] 
        obs = utils.convert_bbox_to_z(obs)

        new_obj.update(obs)
        updated.append(new_obj.getState())

    predicted = np.array(predicted)
    x_p = predicted[2:,0]
    y_p = predicted[2:,1]

    updated = np.array(updated)
    x_u = updated[2:,0]
    y_u = updated[2:,1]

    plt.plot(x, y, '.')
    plt.plot(x_u, y_u, '*')
    plt.plot(x_p, y_p, '^')

    plt.show()

if __name__ == "__main__":
    main()