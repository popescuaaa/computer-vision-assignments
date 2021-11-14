import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2

if __name__ == '__main__':
    # Read in the image
    image = cv2.imread('../data/Gura_Portitei_Scara_020.jpg', cv2.IMREAD_COLOR)

    # 1. Change color to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Produce a binary image for finding contours

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create a binary threshold image
    retval, binary = cv2.threshold(gray.copy(), 225, 255, cv2.THRESH_BINARY_INV)

    # 3.  Find and draw the contours
    # Find contours from threshold, binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Draw all contours on a copy of the original image
    contours_image = np.copy(image)
    contours_image = cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 3)

    # 5. Show the image
    plt.imshow(contours_image)
    plt.show()
    plt.imsave('../data/contours.png', contours_image)

    # 6. Find average color of each contour
    cnts = contours
    for cnt in cnts:
        if cv2.contourArea(cnt) > 50:  # filter small contours
            x, y, w, h = cv2.boundingRect(cnt)  # offsets - with this you get 'mask'

            # Pool
            if w > 20 and h > 20:  # filter medium contours as the pool is pretty large in images
                # get average color of contour
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                mean_color = cv2.mean(image, mask=mask)

                if mean_color[2] > mean_color[1] and mean_color[2] > mean_color[0]:
                    print(mean_color)
                    # draw rectangle around contour
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for cnt in cnts:
        if 400 < cv2.contourArea(cnt) < 500:  # filter small contours
            x, y, w, h = cv2.boundingRect(cnt)  # offsets - with this you get 'mask'

            # get average color of contour
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            mean_color = cv2.mean(image, mask=mask)

            if mean_color[0] > mean_color[1] and mean_color[0] > mean_color[2] and mean_color[0] > 100:  # redish
                print(mean_color)
                # draw rectangle around contour
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 7. Show the image
    plt.imshow(image)
    plt.show()

    # 8. Save the image
    plt.imsave('../data/contours_with_rectangles_pool_and_helipad.png', image)
