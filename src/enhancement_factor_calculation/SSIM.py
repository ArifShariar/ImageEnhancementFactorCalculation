from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt



def compare_ssim(original_image, another_image):
    # convert the images to grayscale
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    another_image = cv2.cvtColor(another_image, cv2.COLOR_BGR2GRAY)
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = ssim(original_image, another_image, full=True)
    return score


if __name__ == "__main__":
    image1 = cv2.imread("D:\\Pycharm\\ImageEnhancementFactor\\data\\010.jpg")
    image2 = cv2.imread("D:\\Pycharm\\ImageEnhancementFactor\\data\\010_hef.jpg")
    image3 = cv2.imread("D:\\Pycharm\\ImageEnhancementFactor\\data\\010_UM.jpg")
    image4 = cv2.imread("D:\\Pycharm\\ImageEnhancementFactor\\data\\010_clahe.jpg")

    print(compare_ssim(image1, image2))
    print(compare_ssim(image1, image3))
    print(compare_ssim(image1, image4))

