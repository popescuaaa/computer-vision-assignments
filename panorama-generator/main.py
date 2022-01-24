import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils
import features
import os
import stitch

def convertResult(img):
    '''
    Because of your images which were loaded by opencv, 
    in order to display the correct output with matplotlib, 
    you need to reduce the range of your floating point image from [0,255] to [0,1] 
    and converting the image from BGR to RGB:
    
    '''
    img = np.array(img,dtype=float)/float(255)
    img = img[:,:,::-1]
    return img


BASE_PATH = './frames'
FRAMES_FOLDERS = os.listdir(BASE_PATH)

if __name__ == '__main__':
    for folder in list(filter(lambda f: '.DS' not in f, FRAMES_FOLDERS)):
        
        print('We are currently working on the following folder: {}'.format(folder))

        # Load images
        list_images = utils.loadImages('{}/{}'.format(BASE_PATH, folder), 10, 10)

        # Extract keypoints and descriptors using sift
        kf = []
        for img in list_images:
            k,f = features.findAndDescribeFeatures(img, opt='SIFT')
            kf.append((img, k, f))

        # Draw keypoints
        imgs_kp = []
        for e in kf:
            img, k, f = e
            img_kp = features.drawKeypoints(img, k)
            imgs_kp.append(img_kp)

        plt_img = np.concatenate(imgs_kp, axis=1)
        plt.figure(figsize=(15,15))
        plt.imshow(convertResult(plt_img))
        plt.show()

        # Matching features using BruteForce 
        mat = features.matchFeatures(kf[0][2], kf[1][2], ratio = 2, opt = 'BF')

        # Computing Homography matrix and mask
        H, matMask = features.generateHomography(kf[0][0], kf[1][0])

        # Draw matches
        img=features.drawMatches(kf[0][0], kf[0][1] ,kf[1][0], kf[1][1], mat, matMask)
        plt.figure(figsize=(15,15))
        plt.imshow(convertResult(img))
        plt.show()
        
        # Generate panorama
        pano, non_blend, left_side, right_side = stitch.warpTwoImages(list_images[1], list_images[0], True)
        plt.figure(figsize=(15,15))
        plt.imshow(convertResult(pano))
        plt.show()
