# Canny edge detection

from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from helpers import rgb2gray


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


if __name__ == '__main__':
    # Read image
    img = plt.imread("../data/test.jpg")
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    img = rgb2gray(img)
    _img = SobelFilter(image=img, direction='x')
    plt.imshow(_img)
    plt.show()
    plt.imsave("../data/test_sobel.jpg", _img)
