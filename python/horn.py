import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2
from gradhorn import gradhorn

def horn(I1, I2, alpha, N):
    I1 = cv2.imread(I1, 0).astype(np.float32)
    I2 = cv2.imread(I2, 0).astype(np.float32)
    Ix, Iy, It = gradhorn(I1, I2)
    A = np.matrix([[1, 2, 1],
                   [2, 0, 2],
                   [1, 2, 1]]) / 12

    u = np.zeros(I1.shape)
    v = np.zeros(I1.shape)

    for k in range(0, N - 1):
        u = convolve2d(u, A, "same")
        v = convolve2d(v, A, "same")

        tmp = np.divide(Ix * u + Iy * v + It, alpha + Ix**2 + Iy**2)

        u = u - Ix * tmp
        v = v - Iy * tmp

    w = np.zeros([I1.shape[0], I1.shape[1], 2])
    w[:, :, 0] = u
    w[:, :, 1] = v
    return w

