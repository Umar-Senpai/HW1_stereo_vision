#pragma once

void StereoEstimation_Naive(
  const int& window_size,
  const int& dmin,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities);

void StereoEstimation_DP(
  const int& window_size,
  const int& dmin,
  const double& lambda,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities);

void StereoEstimation_SGBM(
  const int& min_disparity,
  const int& num_disparities,
  const int& block_size,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& sgbm_disparities);

void Disparity2PointCloud_PCL(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length);

void Display_PointCloud(
  const std::string& output_file);
