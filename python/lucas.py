import matplotlib.pyplot as plt
import cv2
import numpy as np
from gradhorn import gradhorn

def lucas(I1, I2, n):
    I1 = cv2.imread(I1, 0).astype(np.float32)
    I2 = cv2.imread(I2, 0).astype(np.float32)
    Ix, Iy, It = gradhorn(I1, I2)

    center = int(n / 2)
    u = np.zeros(I1.shape)
    v = np.zeros(I1.shape)

    # Normalize I1 and I2 ?

    for i in range(center, I1.shape[0] - center):
        for j in range(center, I1.shape[1] - center):

            X =  Ix[i - center:i + center + 1, j - center:j + center + 1].flatten()
            Y =  Iy[i - center:i + center + 1, j - center:j + center + 1].flatten()
            B = -It[i - center:i + center + 1, j - center:j + center + 1].flatten()
            A = np.vstack((X, Y)).T
            w = np.linalg.pinv(A.T @ A) @ A.T @ B

            u[i, j] = w[0]
            v[i, j] = w[1]

    w = np.zeros([I1.shape[0], I1.shape[1], 2])
    w[:, :, 0] = u
    w[:, :, 1] = v
    return w


