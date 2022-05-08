import matplotlib.pyplot as plt
import cv2
import numpy as np
from gradhorn import gradhorn

from scipy.ndimage.filters import gaussian_filter

def gaussian_kernel(shape, sigma=1):
    kernel = np.zeros(shape)
    kernel[shape[0]//2, shape[1]//2] = 1    
    return gaussian_filter(kernel, sigma)

def lucas_gauss(I1, I2, n, sigma):
    Ix, Iy, It = gradhorn(I1, I2)
    center = int(n / 2)
    u = np.zeros(I1.shape)
    v = np.zeros(I1.shape)

    # Normalize I1 and I2 ?
    
    for i in range(center, I1.shape[0] - center):
        for j in range(center, I1.shape[1] - center):
            # Extracting pixels  windows X and Y from Ix and Iy
           
                   
            X =  Ix[i - center:i + center + 1, j - center:j + center + 1]
            Y =  Iy[i - center:i + center + 1, j - center:j + center + 1]
            B = -It[i - center:i + center + 1, j - center:j + center + 1]
            X = (X * gaussian_kernel(X.shape, sigma)).flatten()
            Y = (Y * gaussian_kernel(Y.shape, sigma)).flatten()
            B = (B * gaussian_kernel(B.shape, sigma)).flatten()
            
            A = np.vstack((X, Y)).T
            w = np.linalg.pinv(A.T @ A) @ A.T @ B

            u[i, j] = w[0]
            v[i, j] = w[1]

    w = np.zeros([I1.shape[0], I1.shape[1], 2])
    w[:, :, 0] = u
    w[:, :, 1] = v
    return w

