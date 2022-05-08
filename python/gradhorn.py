import numpy as np
from scipy.signal import convolve2d

def gradhorn(I1, I2):
    # kernel for computing d/dx
    kX = np.array([[-1, 1], [-1, 1]]) * 0.25
    # kernel for computing d/dy
    kY = np.array([[-1, -1], [1, 1]]) * 0.25
    kT = np.ones((2, 2)) * 0.25

    Ix = convolve2d(I1, kX, "same") + convolve2d(I2, kX, "same")
    Iy = convolve2d(I1, kY, "same") + convolve2d(I2, kY, "same")
    It = convolve2d(I1, kT, "same") + convolve2d(I2, -kT, "same")
    return Ix, Iy, It