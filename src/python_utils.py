import cv2
import numpy as np

from skimage.metrics import structural_similarity as skt_ssim 
import matplotlib.pyplot as plt
import re


def read_image(path: str):
    return cv2.imread(path)

def ssim(img1, img2) -> float:
    """
    Structural Simularity Index
    """
    return skt_ssim(img1, img2, channel_axis=2)

def ssd(img1, img2):
    """Computing the sum of squared differences (SSD) between two images."""
    return np.sum((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32))**2)

def ncc(img1, img2) -> float:
    """
    Normalized Cross Corelation
    """
    return cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)

metrics = {
    "ssd": ssd,
    "ncc": ncc,
    "ssim": ssim,
}

dataset = ["Art", "Books", "Dolls", "Laundry", "Moebius", "Reindeer"]
algorithms = ["naive", "dp", "sgbm"]
window_size = [1, 3, 5]

data_dir = "../results/"

def bar_plots_metrics(metric, ax):
    output_dict = {}
    for name in dataset:
        output_dict[name] = {}
        for algo in algorithms:
            output_dict[name][algo] = {}
            for size in window_size:
                metric_func = metrics[metric]
                image_name = "{}_window_{}_{}.png".format(name, size, algo)
                gt_img = read_image(data_dir + f"{name}/" + f"{name}_gt.png")
                calc_img = read_image(data_dir + f"{name}/" + image_name)
                out_value = float(metric_func(gt_img, calc_img))
                output_dict[name][algo][size] = out_value

    for id, name in enumerate(dataset):
        first_index = int(id/3)

        x1 = [1, 3, 5]
        y1 = [output_dict[name]['naive'][1], output_dict[name]['naive'][3], output_dict[name]['naive'][5]]

        x2 = [1, 3, 5]
        y2 = [output_dict[name]['dp'][1], output_dict[name]['dp'][3], output_dict[name]['dp'][5]]

        x3 = [1, 3, 5]
        y3 = [output_dict[name]['sgbm'][1], output_dict[name]['sgbm'][3], output_dict[name]['sgbm'][5]]

        width = np.min(np.diff(x3))/5

        ax[first_index][id % 3].bar(x1-width, y1, label="Naive", color='b', width=0.4)
        ax[first_index][id % 3].bar(x2, y2, label="DP", color='g', width=0.4)
        ax[first_index][id % 3].bar(x3+width, y3, label="SGBM", color='r', width=0.4)
        ax[first_index][id % 3].plot()

        ax[first_index][id % 3].set_xlabel("window size")
        ax[first_index][id % 3].set_ylabel(metric.upper())
        ax[first_index][id % 3].set_title("{} Image Middlebury".format(name))
        ax[first_index][id % 3].legend()

def bar_plots_time(ax):
    def numbers_from_file(file):
        """From a list of integers in a file, creates a list of tuples"""
        with open(file, 'r') as f:
            return([float(x) for x in re.findall(r'[\d]*[.][\d]+', f.read())])

    processing_time_dict = {}
    for name in dataset:
        processing_time_dict[name] = {}
        for size in window_size:
            processing_time_dict[name][size] = {}
            file_name = "{}_window_{}_processing_time.txt".format(name, size)
            time_arr = numbers_from_file(data_dir + f"{name}/" + file_name)
            processing_time_dict[name][size] = time_arr

    for id, name in enumerate(dataset):
        first_index = int(id/3)

        x1 = [1, 3, 5]
        y1 = [processing_time_dict[name][1][0], processing_time_dict[name][3][0], processing_time_dict[name][5][0]]

        x2 = [1, 3, 5]
        y2 = [processing_time_dict[name][1][1], processing_time_dict[name][3][1], processing_time_dict[name][5][1]]

        x3 = [1, 3, 5]
        y3 = [processing_time_dict[name][1][2] * 100, processing_time_dict[name][3][2] * 100, processing_time_dict[name][5][2] * 100]

        width = np.min(np.diff(x3))/5

        ax[first_index][id % 3].bar(x1-width, y1, label="Naive", color='b', width=0.4)
        ax[first_index][id % 3].bar(x2, y2, label="DP", color='g', width=0.4)
        ax[first_index][id % 3].bar(x3+width, y3, label="SGBM * 100", color='r', width=0.4)
        ax[first_index][id % 3].plot()

        ax[first_index][id % 3].set_xlabel("window size")
        ax[first_index][id % 3].set_ylabel("Time (seconds)")
        ax[first_index][id % 3].set_title("{} Image Middlebury".format(name))
        ax[first_index][id % 3].legend()

def line_plot_lambda(metric):
    lambda_val = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]
    dataset = ["Art", "Books", "Dolls"]
    dp_lambda_dict = {}
    for data in dataset:
        dp_lambda_dict[data] = []
        for val in lambda_val:
            metric_func = metrics[metric]
            image_name = data_dir + f"{data}/tune_lambda/" + "{}_w1_l{}_dp.png".format(data, val)
            gt_img = read_image(data_dir + f"{data}/" + f"{data}_gt.png")
            calc_img = read_image(image_name)
            out_value = float(metric_func(gt_img, calc_img))
            dp_lambda_dict[data].append(out_value)

    x  = lambda_val
    y1 = dp_lambda_dict["Art"]
    y2 = dp_lambda_dict["Books"]
    y3 = dp_lambda_dict["Dolls"]
    plt.plot(x, y1, label="Art Image")
    plt.plot(x, y2, label="Books Image")
    plt.plot(x, y3, label="Dolls Image")
    plt.plot()

    plt.xlabel("Lambda")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs Lambda for DP Algorithm")
    plt.legend(loc="center right")

    return x, [y1, y2, y3]

def diff_image(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (score, diff) = skt_ssim(img1_gray, img2_gray, full=True)

    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    filled_after = img2.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(img1)
    ax[0].set_title("DP Disparity Image")
    ax[1].imshow(img2)
    ax[1].set_title("Ground Truth Disparity Image")
    ax[2].imshow(filled_after)
    ax[2].set_title("Diff Image (Green shows missing parts in Image 2 w.r.t Image 1)")

def disp_image(img1, img2, img3):
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(img1)
    ax[0].set_title("Naive Disparity")
    ax[1].imshow(img2)
    ax[1].set_title("Dynamic Programming Disparity")
    ax[2].imshow(img3)
    ax[2].set_title("SGBM Disparity")