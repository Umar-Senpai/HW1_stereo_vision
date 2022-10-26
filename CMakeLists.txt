cmake_minimum_required(VERSION 3.9)

project( stereo_estimation )

###############
# packages #
###############

find_package( OpenCV REQUIRED )
find_package(PCL 1.2 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenMP)

###############
# executables #
###############

add_executable( OpenCV_naive_stereo src/main.cpp )
target_link_libraries( OpenCV_naive_stereo ${OpenCV_LIBS})

add_executable( Disparity_2_point_cloud src/disparity2pc.cpp )
target_link_libraries( Disparity_2_point_cloud ${OpenCV_LIBS} ${PCL_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(OpenCV_naive_stereo OpenMP::OpenMP_CXX)
endif()