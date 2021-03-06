from numpy import mgrid, square, exp, pi, zeros, ravel, dot, uint8, ndarray
from itertools import product
import matplotlib.pyplot as plt
from helpers import rgb2gray


def gaussian_filter_kernel(size: int, sigma: float) -> ndarray:
    center = size // 2
    x, y = mgrid[0 - center: size - center, 0 - center: size - center]
    gk = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
    return gk


def gaussian_filter(image: ndarray, size: int, sigma: float) -> ndarray:
    height, width = image.shape[0], image.shape[1]
    dst_height = height - size + 1
    dst_width = width - size + 1

    image_array = zeros((dst_height * dst_width, size * size))
    row = 0
    for i, j in product(range(dst_height), range(dst_width)):
        window = ravel(image[i: i + size, j: j + size])
        image_array[row, :] = window
        row += 1

    gaussian_kernel = gaussian_filter_kernel(size, sigma)
    filter_array = ravel(gaussian_kernel)

    dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

    return dst


if __name__ == '__main__':
    img = plt.imread("../data/Gura_Portitei_Scara_040.jpg")
    img = rgb2gray(img)
    _img = gaussian_filter(image=img, size=15, sigma=1)
    plt.imshow(_img)
    plt.show()
    plt.imsave("../data/Gura_Portitei_Scara_040_gaussian.jpg", _img)
