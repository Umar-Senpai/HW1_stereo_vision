# Stereo Vision: Match and Display
This repository contains the solution to homework 1 for 3D Sensing and Sensor Fusion course at ELTE, Budapest. It implements stereo matching techniques and 3D displays using Point Cloud Library. 

## Subtask 1: Algorithms
The main C++ file in `/src` named `main.cpp` implements three algorithms for generating disparity maps between a stereo pair of images. The implemented algorithms are:

* Naive Stereo Matching
* Dynamic Programming
* OpenCV's StereoSGBM

The algorithms can be run on a pair of image using the following command
```
./OpenCV_naive_stereo Image1 Image2 output
```
The C++ file will apply the three algorithms on the provided images and generate four files as a result. An example format of these files is shown:
```
output_naive.png
output_dp.png
output_sgbm.png
output_processing_time.txt
```
The first three are disparity images in `PNG` format. The last one is a text file containing the processing time of each algorithm in seconds. 
Different parameters such as window_size, lambda etc. are tunable and can be changed for fine tuning in the main function. 

## Subtask 2: 3D Display
Another C++ file in `/src` named `disparity2pc.cpp` implements functions to generate a point cloud from the disparity images and to visualize these point clouds. Using the Point Cloud Library in C++, it performs normals computation on point cloud and triangulation of the point cloud. We need the Point Cloud Library to run this file. A simple tutorial to install PCL library is given here: https://pcl.readthedocs.io/projects/tutorials/en/master/compiling_pcl_posix.html

After the library is compiled and installed, we can run the file on an image using the following command
```
./Disparity_2_point_cloud Disparity_Image output
```
By taking the disparity image, it first converts the image to point cloud and then visualizes simple point cloud along with normal and triangulation. In the end, it will generate three files. An example format of these files is shown:
```
output.xyz
output.pcd
output_mesh.obj
```
The `output.xyz` and `output.xyz` files are the same, and contains `XYZ` coordinates of point cloud. The `output_mesh.obj` file contains the polygon mesh generated by triangulation. It can be viewed in `MeshLab` or other software. `PCL` also provides visualization tools and running the above code will automatically display point cloud, normal and triangulation in a separate windows. You can press R to center the axis and click and move cursor to change view of the point cloud. Different parameters such as search radius for normal and triangulation can be changed and fine tuned. 

## Subtask 3: Evaluation
In addition to the above visualizations by PCL library, we will be using python and matplotlib to display our results. These results include displaying metric (SSD, SSIM, NCC) values on different pair of images using the three algorithms. We are also plotting processing times of the three algorithms. In order to find optimal lambda for DP algorithm, we are plotting SSD/SSIM vs lambda values. In the end, some difference images are shown by comparing a few samples with their respective ground truth. The python script can be run using the following command
```
python3 evaluation.py
```
The `evaluation.py` script also uses `python_utils.py` which are just utility functions separated out for readability. The results are discussed below:
## Evaluation Results
### Algorithm results
We have run the algorithm on 6 image pairs from Middlebury 2005 dataset. The ground truth disparity image for Books is shown below:
<p align="center">
  <img src="https://user-images.githubusercontent.com/31202659/197873034-547e1dcd-e7a7-4ddd-9869-608b2cf26307.png" width="400" height="400" />
</p>
And the results obtained by algorithms on Books pair with window size 1 are shown below:

![Screenshot from 2022-10-26 00-19-19](https://user-images.githubusercontent.com/31202659/197873631-990e9354-b3cf-4e38-8cd5-c9a7cc6285d6.png)

This is just a sample output from one image pair and one window size. We have evaulate 6 image pair with 3 different window sizes and 3 different metrics (SSD, SSIM, NCC) and 3 different algorithms (Naive, DP, SGBM). The results for each metrics are shown in a separate plot.

#### Difference Image
To better visualize these results, we have generated a difference image between the DP algorithm and ground truth for the Art Image pair. The results are displayed below:

![Screenshot from 2022-10-26 02-30-29](https://user-images.githubusercontent.com/31202659/197894351-99517e2d-42b3-4731-ad91-986d1d737079.png)

#### SSD
The plot below shows results for SSD. Each subplot belongs to one image pair and in each subplot, we have a bar represting each algorithm. 

![Figure_1](https://user-images.githubusercontent.com/31202659/197872144-6e5d0ad3-aec6-4586-b34c-41a2f39f9f1a.png)

We can see that with increasing window_size, the SSD increases for almost all cases. Meaning that the error is increasing by increasing window_size.

#### SSIM
The SSIM or Structural Similarity Index Measure gives a value of 1 for perfect match. Below we can observer SSIM for each image pair.

![Figure_3](https://user-images.githubusercontent.com/31202659/197872150-4d7606e3-c131-45c4-9de9-09080a503048.png)

For SSIM, in case of naive approach, increasing window_size matches the image closer to the ground truth as SSIM is increasing. But for DP approach, lower window_size seems to perform better. 

#### NCC
The normalized cross corelation results are shown below:

![Figure_2](https://user-images.githubusercontent.com/31202659/197872148-2bf1dbe0-9b31-4667-aa75-fbec5fd870e1.png)

The trend followed by NCC is almost the same as SSIM but the change is small between window sizes. 

The OpenCV's SGBM approach is not much affected by the window_size as evident from the above plots.

#### Processing Time
Next, we take a look at the processing time of different algorithms with different window sizes. The processing time can be affected by multiple things e.g., some of the results shown below were run using WSL on Windows with low-power mode. Others were run on High-Performance mode. Some are run on Ubuntu.

![Figure_4](https://user-images.githubusercontent.com/31202659/197872153-ae7100fa-8692-4b07-8e59-e0f40fb57ab4.png)

But the general trend is the same that with increasing window_size, we see a good amount of jump in processing time. The SGBM is not much affected and is relatively very fast.

#### Optimal Lambda
In order to figure out the optimal lambda for the dynamic programming approach, as we have seen that smaller window size generates much better results on DP approach, we will run DP algorithm with `window_size=1` for different lambda values. The trend for SSD.

<p align="center">
  <img src="https://user-images.githubusercontent.com/31202659/197872157-19107503-eb9f-4810-a747-dfb46d337a37.png" width="500" height="300" />
</p>

The optimal lambda value for different image pairs lie between 30-60. Let's change the metric and use SSIM to figure out the optimal lambda. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/31202659/197872161-e3693157-7c9d-498b-8620-c907c371d40c.png" width="500" height="300" />
</p>

In this case, the optimal lambda value lies between 50-90.

#### 3D Displays

We have generated some screenshots of 3D displays with DP algorithms for Art image pair and window size of 1. The DP algorithm is much better as it has less noise:

Downsampled DP Point cloud             |  Normal Estimation | Triangulated surface
:-------------------------:|:-------------------------:|:-------------------------:
![Screenshot from 2022-10-26 02-40-33](https://user-images.githubusercontent.com/31202659/197895652-2134ae20-0f3f-4606-b9ef-3241ffa3463a.png)  |  ![Screenshot from 2022-10-26 02-40-58](https://user-images.githubusercontent.com/31202659/197895658-50429a17-5d82-42dc-b603-dba65dfdecdc.png) | ![Screenshot from 2022-10-26 02-41-25](https://user-images.githubusercontent.com/31202659/197895659-d483b5ac-deff-46d8-9da2-01d9d5e59b37.png)

We can tune the radius search size for normal estimation and triagulation.

### Results Folder
In the repository, the results folder contain all the generated disparity images for metric evaluations. It also contains some other results. The name of the files display the algorithm and window size/lambda value used for each output. 


