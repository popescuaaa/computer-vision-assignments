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

    samples = [flt, _img]
    sample_images = np.concatenate([image.reshape((1, image.shape[0], image.shape[1])) for image in samples], axis=0)

    print(sample_images.shape)
    plt.figure(1)
    for i in range(sample_images.shape[0]):
        plt.subplot(2, 2, i + 1)
        plt.imshow(sample_images[i, ...])
        plt.axis("off")
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.show()

    # determine per-pixel variability, std() over all images
    variability = sample_images.std(axis=0)
    print(variability.shape)
    plt.imshow(variability, interpolation="nearest", origin="lower")
    plt.show()

    plt.figure(2)
    # determine bounding box
    thresholds = [1.00]
    colors = ["r", "b", "g"]
    for threshold, color in zip(thresholds, colors):  # variability.mean()
        non_empty_columns = np.where(variability.min(axis=0) < threshold)[0]
        non_empty_rows = np.where(variability.min(axis=1) < threshold)[0]
        boundingBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

        # plot and print boundingBox
        bb = boundingBox
        plt.plot([bb[2], bb[3], bb[3], bb[2], bb[2]],
                 [bb[0], bb[0], bb[1], bb[1], bb[0]],
                 "%s-" % color,
                 label="threshold %s" % threshold)

    plt.xlim(0, variability.shape[1])
    plt.ylim(variability.shape[0], 0)
    plt.legend()
    plt.show()