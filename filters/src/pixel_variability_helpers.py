"""
    One method to find objects that are in the same position in images is to
    calculate the standard deviation of the pixel values in the image.
    This is done by using the numpy.std() function.
    This method is called "pixel_variability"
"""
import numpy as np
import matplotlib.pyplot as plt


def pixel_variability(image: np.ndarray, flt: np.ndarray) -> np.ndarray:
    """
        Calculates the standard deviation of the pixel values in the image.
        This is done by using the numpy.std() function.
        This method is called "pixel_variability"
    """
    samples = [flt, image]
    sample_images = np.concatenate(
        [image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) for image in samples], axis=0)

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
    plt.imshow(variability)
    plt.show()

    plt.figure(2)
    # determine bounding box
    thresholds = [5, 10, 20]
    colors = ["r", "b", "g"]
    for threshold, color in zip(thresholds, colors):
        non_empty_columns = np.where(variability.max(axis=0) < threshold)[0]
        non_empty_rows = np.where(variability.max(axis=1) < threshold)[0]
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
