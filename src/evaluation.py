import cv2
import numpy as np

import matplotlib.pyplot as plt
from python_utils import *

metrics = {
    "ssd": ssd,
    "ncc": ncc,
    "ssim": ssim,
}

def main():
    img1 = cv2.imread('../results/Books/Books_window_1_naive.png')
    img2 = cv2.imread('../results/Books/Books_window_1_dp.png')
    img3 = cv2.imread('../results/Books/Books_window_1_sgbm.png')
    disp_image(img1, img2, img3)

    for metric in metrics:
        fig, ax = plt.subplots(2, 3, figsize=(15, 15))
        fig.suptitle(f"{metric.upper()} vs window_size for 6 pairs and 3 algorithms")
        bar_plots_metrics(metric, ax)

    fig, ax = plt.subplots(2, 3, figsize=(15, 15))
    fig.suptitle("Processing Time (s) vs window_size for 6 pairs and 3 algorithms")
    bar_plots_time(ax)

    fig = plt.figure()
    x, y = line_plot_lambda("ssd")
    print("Optimal lambda for min SSD for Art Image: ", x[np.argmin(y[0])])
    print("Optimal lambda for min SSD for Books Image: ", x[np.argmin(y[1])])
    print("Optimal lambda for min SSD for Dolls Image: ", x[np.argmin(y[2])])

    fig = plt.figure()
    x, y = line_plot_lambda("ssim")
    print("Optimal lambda for max SSIM for Art Image: ", x[np.argmax(y[0])])
    print("Optimal lambda for max SSIM for Books Image: ", x[np.argmax(y[1])])
    print("Optimal lambda for max SSIM for Dolls Image: ", x[np.argmax(y[2])])

    # Load images
    img1 = cv2.imread('../results/Art/Art_window_1_sgbm.png')
    img2 = cv2.imread('../results/Art/Art_gt.png')

    diff_image(img1, img2)

    plt.show()

if __name__ == "__main__":
    main()

