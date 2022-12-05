###################################
# Author: Arif Shariar Rahman
# Email:  1705095@ugrad.cse.buet.ac.bd
# Author: Sadia Saman
# Email:  1705102@ugrad.cse.buet.ac.bd
# This is the code for Mean Squared Error
# algorithm of Mean Squared Error:
# 1. Take two images as input
# 2. Compute the difference between the two images
# 3. Compute the square of the difference
# 4. Compute the mean of the square of the difference
###################################

import numpy as np
import cv2
from numpy import ndarray


def mse(image_1: np.ndarray, image_2: np.ndarray) -> ndarray:
    err = np.sum((image_1.astype("float") - image_2.astype("float")) ** 2)
    err /= float(image_1.shape[0] * image_1.shape[1])
    return err


if __name__ == "__main__":
    image1 = cv2.imread("D:\\Pycharm\\ImageEnhancementFactor\\data\\010.jpg")
    image2 = cv2.imread("D:\\Pycharm\\ImageEnhancementFactor\\data\\010_hef.jpg")
    image3 = cv2.imread("D:\\Pycharm\\ImageEnhancementFactor\\data\\010_UM.jpg")
    image4 = cv2.imread("D:\\Pycharm\\ImageEnhancementFactor\\data\\010_clahe.jpg")
    print("original and hef: " + str(mse(image1, image2)))
    print("original and um: " + str(mse(image1, image3)))
    print("original and clahe: " + str(mse(image1, image4)))
