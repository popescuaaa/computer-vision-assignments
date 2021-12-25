import os
from cv2 import cv2
import matplotlib.pyplot as plt

BASE_PATH = './frames'
FRAMES_FOLDERS = os.listdir(BASE_PATH)


if __name__ == '__main__':
    # Load images
    for folder in FRAMES_FOLDERS:
        logger.info('Currently working on {}'.format(folder))
        frames = os.listdir('{}/{}'.format(BASE_PATH, folder))
        images = []

        count = 10

        for image_name in frames:
            count -= 1
            if count == 0:
                break

            image = cv2.imread('./frames/{}/{}'.format(folder, image_name))
            image = cv2.resize(image, (0, 0), None, 0.2, 0.2)
            images.append(image)


        stitcher = cv2.Stitcher.create()
        status, result = stitcher.stitch(images)
        plt.imshow(result)
        plt.show()

        if status == cv2.STITCHER_OK:
            print("aljsdlkajlkjas")

        break