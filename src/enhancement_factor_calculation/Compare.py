import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt

import MeanSquaredError
import SSIM

original_image_dir = "D:\\Pycharm\\ImageEnhancementFactor\\data\\enhanced\\"
um_image_dir = "D:\\Pycharm\\ImageEnhancementFactor\\data\\enhanced\\um_out\\"
hef_image_dir = "D:\\Pycharm\\ImageEnhancementFactor\\data\\enhanced\\hef_out\\"
clahe_image_dir = "D:\\Pycharm\\ImageEnhancementFactor\\data\\enhanced\\clahe_out\\"

list_of_files = os.listdir(original_image_dir)
list_of_um_files = os.listdir(um_image_dir)
list_of_hef_files = os.listdir(hef_image_dir)
list_of_clahe_files = os.listdir(clahe_image_dir)

original_files = [file for file in list_of_files if file.endswith(".jpg")]
um_files = [file for file in list_of_um_files if file.endswith(".jpg")]
hef_files = [file for file in list_of_hef_files if file.endswith(".jpg")]
clahe_files = [file for file in list_of_clahe_files if file.endswith(".jpg")]


def plot_ssim():
    original = []
    ssim_um = []
    ssim_hef = []
    ssim_clahe = []
    for i in original_files:
        file_name = i[:-4]
        original_image = imageio.imread(original_image_dir + i)
        um_image = imageio.imread(um_image_dir + file_name + "_UM.jpg")
        hef_image = imageio.imread(hef_image_dir + file_name + "_HEF.jpg")
        clahe_image = imageio.imread(clahe_image_dir + file_name + "_CLAHE.jpg")

        original.append(SSIM.ssim(original_image, original_image))
        ssim_um.append(SSIM.ssim(original_image, um_image))
        ssim_hef.append(SSIM.ssim(original_image, hef_image))
        ssim_clahe.append(SSIM.ssim(original_image, clahe_image))

    print("Original SSIM: " + str(sum(original) / len(original)))
    print("SSIM UM: " + str(sum(ssim_um) / len(ssim_um)))
    print("SSIM HEF: " + str(sum(ssim_hef) / len(ssim_hef)))
    print("SSIM CLAHE: " + str(sum(ssim_clahe) / len(ssim_clahe)))

    plt.plot(original, label="Original")
    plt.plot(ssim_um, label="UM")
    plt.plot(ssim_hef, label="HEF")
    plt.plot(ssim_clahe, label="CLAHE")
    plt.legend()
    plt.grid()
    plt.show()


def plot_mse():
    original = []
    mse_um = []
    mse_hef = []
    mse_clahe = []
    for i in original_files:
        file_name = i[:-4]
        original_image = imageio.imread(original_image_dir + i)
        um_image = imageio.imread(um_image_dir + file_name + "_UM.jpg")
        hef_image = imageio.imread(hef_image_dir + file_name + "_HEF.jpg")
        clahe_image = imageio.imread(clahe_image_dir + file_name + "_CLAHE.jpg")

        original.append(MeanSquaredError.mse(original_image, original_image))
        mse_um.append(MeanSquaredError.mse(original_image, um_image))
        mse_hef.append(MeanSquaredError.mse(original_image, hef_image))
        mse_clahe.append(MeanSquaredError.mse(original_image, clahe_image))

    print("Original MSE: " + str(sum(original) / len(original)))
    print("MSE UM: " + str(sum(mse_um) / len(mse_um)))
    print("MSE HEF: " + str(sum(mse_hef) / len(mse_hef)))
    print("MSE CLAHE: " + str(sum(mse_clahe) / len(mse_clahe)))

    plt.plot(original, label="Original")
    plt.plot(mse_um, label="UM")
    plt.plot(mse_hef, label="HEF")
    plt.plot(mse_clahe, label="CLAHE")

    # move legend to top right
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # plot_ssim()
    plot_mse()
