import numpy as np 
import cv2
import math

THICKNESS = 2

class BoundingBox:
    
    thickness = THICKNESS

    def __init__(self, x0, y0, h, w):
        # __edges = np.zeros((6, 2), dtype=float)
        self.__edges = np.array([int(x0), int(y0), int(x0+w), int(y0+h)], dtype=float)
        self.__center = np.array([(self.__edges[0] + self.__edges[2])/2, (self.__edges[1] + self.__edges[3])/2], dtype=float)
        self.__color = (255%h, 255%w, 255%(w+h))
