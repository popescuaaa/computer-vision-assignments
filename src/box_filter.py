# Image box filter
import numpy as np
import matplotlib.pyplot as plt
from helpers import rgb2gray
from typing import Tuple


def box_filter(image: np.ndarray,
               kernel_shape: Tuple[int, int],
               stride: Tuple[int, int]) -> np.ndarray:
    """
    image is a numpy.ndarray with shape (Hi, Wi) containing the
    input image.
    kernel_shape is a tuple of (kh, kw) containing the kernel shape.
    stride is a tuple of (sh, sw) containing the strides for the
    convolution.
    Returns: a numpy.ndarray containing the convolved image.
    """
    # Get image dimensions
    H, W = image.shape

    # Get kernel dimensions
    kh, kw = kernel_shape
    sh, sw = stride

    # Get output dimensions
    H_out = int((H - kh) / sh + 1)
    W_out = int((W - kw) / sw + 1)

    # Initialize output image
    out = np.zeros((H_out, W_out))

    # Convolve image
    for i in range(H_out):
        for j in range(W_out):
            out[i, j] = np.sum(image[i * sh:i * sh + kh, j * sw:j * sw + kw])
    return out


# tst box filter with image
if __name__ == "__main__":
    img = plt.imread("../data/test.jpg")
    img = rgb2gray(img)
    _img = box_filter(image=img, kernel_shape=(15, 15), stride=(1, 1))
    plt.imshow(_img)
    plt.show()
    plt.imsave("../data/test3.jpg", _img)
