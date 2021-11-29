from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from helpers import rgb2gray
import os


# Apply Sobel Filter using the convolution operation Note that in this case I have used the filter to have a maximum
# magnitude of 2, but it can also be changed to other numbers for aggressive edge extraction For eg
# [-1,0,1], [-5,0,5], [-1,0,1]

def SobelFilter(image: np.ndarray, direction: str) -> np.ndarray:
    if direction == 'x':
        gx = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
        res = ndimage.convolve(image, gx)
        return res
    if direction == 'y':
        gy = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
        res = ndimage.convolve(image, gy)
        return res


# Normalize the pixel array, so that values are <= 1
def Normalize(image: np.ndarray) -> np.ndarray:
    # img = np.multiply(img, 255 / np.max(img))
    image = image / np.max(image)
    return image


# Double threshold Hysteresis. Note that I have used a very slow iterative approach for ease of understanding,
# a faster implementation using recursion can be done instead This recursive approach would recurse through every
# strong edge and find all connected weak edges

def DoThreshHyst(image: np.ndarray, ht: float = 0.2, lt: float = 0.15) -> np.ndarray:
    high_threshold_ratio = ht
    low_threshold_ratio = lt

    g_sup = np.copy(image)

    h = int(g_sup.shape[0])
    w = int(g_sup.shape[1])

    high_threshold = np.max(g_sup) * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    x = 0.1
    old_x = 0

    # The while loop is used so that the loop will keep executing till the number of strong edges do not change,
    # i.e all weak edges connected to strong edges have been found
    while old_x != x:
        old_x = x
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if g_sup[i, j] > high_threshold:
                    g_sup[i, j] = 1
                elif g_sup[i, j] < low_threshold:
                    g_sup[i, j] = 0
                else:
                    if ((g_sup[i - 1, j - 1] > high_threshold) or
                            (g_sup[i - 1, j] > high_threshold) or
                            (g_sup[i - 1, j + 1] > high_threshold) or
                            (g_sup[i, j - 1] > high_threshold) or
                            (g_sup[i, j + 1] > high_threshold) or
                            (g_sup[i + 1, j - 1] > high_threshold) or
                            (g_sup[i + 1, j] > high_threshold) or
                            (g_sup[i + 1, j + 1] > high_threshold)):
                        g_sup[i, j] = 1

        x = np.sum(g_sup == 1)

    # This is done to remove/clean all the weak edges which are not connected to strong
    # edges
    g_sup = (g_sup == 1) * g_sup

    return g_sup


if __name__ == '__main__':
    for file in os.listdir("../data/"):
        if "Gura_Portitei_Scara" in file:
            # Read image
            img = plt.imread("../data/{}".format(file))
            # Convert to grayscale
            img = rgb2gray(img)
            _img = SobelFilter(image=img, direction='x')
            # Normalize
            _img = Normalize(_img)
            # Double threshold hysteresis
            _img = DoThreshHyst(_img)
            plt.imsave("../data/{}_edges.jpg".format(file), _img)
