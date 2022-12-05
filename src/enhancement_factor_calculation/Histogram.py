import matplotlib.pyplot as plt
import numpy as np

# plot histogram of image
from PIL import Image


def plot_histogram(image, title, save_path):
    plt.title(title)
    plt.hist(image.flat, range=(0, 255))
    plt.savefig(save_path)
    plt.show()


def calculate_histogram(image):
    """
    This function calculates the histogram of the image
    :param image: the image to be calculated
    :return: the histogram
    """
    image = image.convert("L")
    image = np.asarray(image)
    h = [0] * 256
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            h[image[i, j]] += 1
    return h


def calculate_cdf(hist, bins):
    """
    This function calculates the cumulative distribution function
    :param hist: frequencies of each pixel
    :param bins: pixels
    :return: cdf
    """
    # probability of each pixel
    pixel_probability = hist / hist.sum()

    # cumulative distribution function
    cdf = np.cumsum(pixel_probability)

    cdf_normalized = cdf * 255

    return cdf_normalized


if __name__ == "__main__":
    # open image
    # image = cv2.imread("D:\\Pycharm\\ImageEnhancementFactor\\data\\010.jpg", 0)
    # plot_histogram(image, "Original", "D:\\Pycharm\\ImageEnhancementFactor\\data\\010_original_histogram.jpg")
    # image2 = cv2.imread("D:\\Pycharm\\ImageEnhancementFactor\\data\\010_hef.jpg", 0)
    # plot_histogram(image2, "HEF", "D:\\Pycharm\\ImageEnhancementFactor\\data\\010_hef_hist.jpg")
    image1 = Image.open("D:\\Pycharm\\ImageEnhancementFactor\\data\\010.jpg")
    h1 = calculate_histogram(image1)
    print(h1)

    image2 = Image.open("D:\\Pycharm\\ImageEnhancementFactor\\data\\010_hef.jpg")
    h2 = calculate_histogram(image2)
    print(h2)

    image3 = Image.open("D:\\Pycharm\\ImageEnhancementFactor\\data\\010_UM.jpg")
    h3 = calculate_histogram(image3)
    print(h3)

    image4 = Image.open("D:\\Pycharm\\ImageEnhancementFactor\\data\\010_clahe.jpg")
    h4 = calculate_histogram(image4)
    print(h4)

    # plot the histogram
    plt.plot(h1)
    plt.legend(["original"])
    plt.grid()
    plt.show()

    plt.plot(h2)
    plt.legend(["hef"])
    plt.grid()
    plt.show()

    plt.plot(h3)
    plt.legend(["UM"])
    plt.grid()
    plt.show()

    plt.plot(h4)
    plt.legend(["clahe"])
    plt.grid()
    plt.show()

    plt.plot(h1)
    plt.plot(h2)
    plt.plot(h3)
    plt.plot(h4)
    plt.legend(["original", "hef", "UM", "clahe"])
    plt.grid()
    plt.show()
