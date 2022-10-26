#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>
#include "main.h"
#include <thread>

int main(int argc, char** argv) {

  ////////////////
  // Parameters //
  ////////////////

  // camera setup parameters
  const double focal_length = 3740;
  const double baseline = 0.160;

  // stereo estimation parameters
  const int dmin = 200;
  const int window_size = 1;
  const double lambda = 150;

  ///////////////////////////
  // Commandline arguments //
  ///////////////////////////

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " IMAGE1 IMAGE2 OUTPUT_FILE" << std::endl;
    return 1;
  }

  cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  const std::string output_file = argv[3];

  if (!image1.data) {
    std::cerr << "No image1 data" << std::endl;
    return EXIT_FAILURE;
  }

  if (!image2.data) {
    std::cerr << "No image2 data" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "------------------ Parameters -------------------" << std::endl;
  std::cout << "focal_length = " << focal_length << std::endl;
  std::cout << "baseline = " << baseline << std::endl;
  std::cout << "window_size = " << window_size << std::endl;
  std::cout << "disparity added due to image cropping = " << dmin << std::endl;
  std::cout << "output filename = " << argv[3] << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  int height = image1.size().height;
  int width = image1.size().width;
  double matching_time;
  const int min_disparity = 0;
  const int num_disparities = 64;

  ////////////////////
  // Reconstruction //
  ////////////////////

  // Naive disparity image
  //cv::Mat naive_disparities = cv::Mat::zeros(height - window_size, width - window_size, CV_8UC1);
  cv::Mat naive_disparities = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat dp_disparities = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat sgbm_disparities = cv::Mat::zeros(height, width, CV_8UC1);

  std::stringstream outTime;
  outTime << output_file << "_processing_time.txt";
  std::ofstream outfileTime(outTime.str());
  
  // std::stringstream out2;
  // out2 << "disp1.png";
  // naive_disparities = cv::imread(out2.str(), cv::IMREAD_GRAYSCALE);

  matching_time = (double)cv::getTickCount();
  StereoEstimation_Naive(
    window_size, dmin, height, width,
    image1, image2,
    naive_disparities);
  matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
  outfileTime << "Naive: " << matching_time << " seconds" << std::endl;
  
  matching_time = (double)cv::getTickCount();
  StereoEstimation_DP(
    window_size, dmin, lambda, height, width,
    image1, image2,
    dp_disparities);
  matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
  outfileTime << "Dynamic Programming: " << matching_time << " seconds" << std::endl;

  matching_time = (double)cv::getTickCount();
  StereoEstimation_SGBM(
    min_disparity, num_disparities, window_size, 
    height, width, image1, image2,
    sgbm_disparities);
  matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
  outfileTime << "SGBM: " << matching_time << " seconds" << std::endl;

  // save / display images
  std::stringstream out_naive;
  out_naive << output_file << "_naive.png";
  cv::imwrite(out_naive.str(), naive_disparities);

  cv::namedWindow("Naive", cv::WINDOW_AUTOSIZE);
  cv::imshow("Naive", naive_disparities);

  std::stringstream out_dp;
  out_dp << output_file << "_dp.png";
  cv::imwrite(out_dp.str(), dp_disparities);

  cv::namedWindow("DP", cv::WINDOW_AUTOSIZE);
  cv::imshow("DP", dp_disparities);

  std::stringstream out_sgbm;
  out_sgbm << output_file << "_sgbm.png";
  cv::imwrite(out_sgbm.str(), sgbm_disparities);

  cv::namedWindow("SGBM", cv::WINDOW_AUTOSIZE);
  cv::imshow("SGBM", sgbm_disparities);

  cv::waitKey(0);

  return 0;
}

void StereoEstimation_Naive(
  const int& window_size,
  const int& dmin,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities)
{
  int half_window_size = window_size / 2;

  for (int i = half_window_size; i < height - half_window_size; ++i) {

    std::cout
      << "Calculating disparities for the naive approach... "
      << std::ceil(((i - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
      << std::flush;

    for (int j = half_window_size; j < width - half_window_size; ++j) {
      int min_ssd = INT_MAX;
      int disparity = 0;

      for (int d = -j + half_window_size; d < width - j - half_window_size; ++d) {
        int ssd = 0;
        for (int u = -half_window_size; u <= half_window_size; ++u) {
          for (int v = -half_window_size; v <= half_window_size; ++v) {
            int difference = image1.at<uchar>(u + i, v + j) - image2.at<uchar>(u + i, v + j + d);
            ssd += difference * difference;
          }
        }

        if (ssd < min_ssd) {
          min_ssd = ssd;
          disparity = d;
        }
      }

      naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) = std::abs(disparity);
    }
  }

  std::cout << "Calculating disparities for the naive approach... Done.\r" << std::flush;
  std::cout << std::endl;
}


void StereoEstimation_DP(
  const int& window_size,
  const int& dmin,
  const double& lambda,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities)
{
  int half_window_size = window_size / 2;
  // for each row (scanline)
  for (int y_0 = half_window_size; y_0 < height - half_window_size; ++y_0) {

    std::cout
      << "Calculating disparities for the dp approach... "
      << std::ceil(((y_0 - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
      << std::flush;

    // dissimilarity
    cv::Mat dissim = cv::Mat::zeros(width, width, CV_32FC1);
    
    for (int i = half_window_size; i < width - half_window_size; ++i) { // left image
      for (int j = half_window_size; j < width - half_window_size; ++j) { // right image
        float sum = 0;
        for (int u = -half_window_size; u <= half_window_size; ++u) {
          for (int v = -half_window_size; v <= half_window_size; ++v) {
            float i1 = static_cast<float>(image1.at<uchar>(y_0 + v, i + u));
            float i2 = static_cast<float>(image2.at<uchar>(y_0 + v, j + u));
            sum += std::abs(i1 - i2); // SAD
            // sum += (i1 - i2); * (i1 - i2); // SSD
          }
        }
        dissim.at<float>(i,j) = sum;
      }
    }
    // but can you save some computations here? (TODO)

    // allocate C,M
    cv::Mat C = cv::Mat::zeros(width, width, CV_32FC1);
    cv::Mat M = cv::Mat::zeros(width, width, CV_8UC1); // match 0; left-occ 1, right-occ 2

    // populate C,M
    for (int i = 0; i < width; ++i) {
      C.at<float>(0, i) = i * lambda;
      C.at<float>(i, 0) = i * lambda; 
    }

    for (int i = 1; i < width; ++i) {
      for (int j = 1; j < width; ++j) {
        double min1 = C.at<float>(i - 1, j - 1) + dissim.at<float>(i,j);
        double min2 = C.at<float>(i - 1, j ) + lambda;
        double min3 = C.at<float>(i, j - 1) + lambda;
        double min = std::min(
          (min1), 
          std::min(
            (min2), 
            (min3)
          )
        );
        C.at<float>(i, j) = min;
        // std::cout << min1 << std::endl;
        // std::cout << min2 << std::endl;
        // std::cout << min3 << std::endl;
        // std::cout << min << std::endl;
        if (min1 == min) {
          M.at<uchar>(i, j) = 0;
          // std::cout << "0 value" << std::endl;
        }
        else if (min2 == min) {
          M.at<uchar>(i, j) = 1;
          // std::cout << "1 value" << std::endl;
        }
        else if (min3 == min) {
          M.at<uchar>(i, j) = 2;
          // std::cout << "2 value" << std::endl;
        }
      }
    }

    int i = width - 1;
    int j = width - 1;
    while (i > 0 && j > 0) {
      int disparity = 0;
      if (M.at<uchar>(i, j) == 0) {
        // std::cout << j << std::endl;
        disparity = j - i;
        i--;
        j--;
      }
      else if (M.at<uchar>(i, j) == 1) {
        i--;
      }
      else if (M.at<uchar>(i, j) == 2) {
        j--;
      }
      dp_disparities.at<uchar>(y_0 - half_window_size, i) = std::abs(disparity);
    }
  }

  std::cout << "Calculating disparities for the naive approach... Done.\r" << std::flush;
  std::cout << std::endl;
}


void StereoEstimation_SGBM(
  const int& min_disparity,
  const int& num_disparities,
  const int& block_size,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& sgbm_disparities)
{
  std::cout
      << "Calculating disparities for the SGBM approach... "
      << std::flush;
  cv::Mat imgDisparity16S = cv::Mat(height, width, CV_16S);
  cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(0, 64, block_size);
  stereo->compute(image1, image2, imgDisparity16S);
  double minVal, maxVal;
  cv::minMaxLoc( imgDisparity16S, &minVal, &maxVal);
  imgDisparity16S.convertTo(sgbm_disparities, CV_8UC1, 255/(maxVal - minVal));
  std::cout << "Calculating disparities for the SGBM approach... Done.\r" << std::flush;
  std::cout << std::endl;
}
