from edge_detection import Normalize, DoThreshHyst, SobelFilter
import matplotlib.pyplot as plt
from helpers import rgb2gray
import numpy as np


def create_blue_pool_filter_010(image: np.ndarray) -> np.ndarray:
    # Custom image with a single block of white pixels for pool
    f = np.copy(image)
    w, h = f.shape
    for i in range(w):
        for j in range(h):
            if 180 < i < 200 and 232 < j < 248:
                f[i, j] = 1
            else:
                continue
    return f


if __name__ == '__main__':
    # Read image
    img = plt.imread('../data/Gura_Portitei_Scara_010.jpg')
    # Convert to grayscale
    img = rgb2gray(img)
    _img = SobelFilter(image=img, direction='x')

    # Normalize
    _img = Normalize(_img)

    # Double threshold hysteresis
    _img = DoThreshHyst(_img)
    plt.imshow(_img)
    plt.show()

    # Create blue pool filter
    flt = create_blue_pool_filter_010(image=_img)

    # Apply filter


