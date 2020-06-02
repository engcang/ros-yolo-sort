# SORT, Simple Online Realtime Tracking of multiple object
  + Original github [here](https://github.com/abewley/sort)
  + C++ Window version github [here](https://github.com/mcximing/sort-cpp)

<br>

### Implemented version
  + Original version + (OpenCV instead of matplotlib), **69.3/15.9FPS** (w, w/o display) on NVIDIA Jetson TX2
  + C++ Ubuntu version => ***200 times faster*** than Python(original) version, **2718/4790FPS** (w, w/o display) on NVIDIA Jetson TX2
  + ROS version implemented

<br>

## Installation
#### Please install *pip packages on Python 3*
  + Original version : referred [here](https://github.com/abewley/sort)
    + running the code using **Detected bounding boxes from [Faster RCNN](https://github.com/ShaoqingRen/faster_rcnn)**
  ~~~shell
  # Dependencies
  $ python3 -m pip install -r requirements.txt
  # To run the tracker with the provided detections:
  $ cd path/to/sort
  $ python3 sort.py
  ~~~
  + To display the results you need to Download the [2D MOT 2015 benchmark dataset](https://motchallenge.net/data/2D_MOT_2015/#download)
    + running the code using **Detected bounding boxes from [Faster RCNN](https://github.com/ShaoqingRen/faster_rcnn)**
    + using *matplotlib* library, slow
  ~~~shell
  $ cd <sort_directory>
  # Create a symbolic link to the dataset
  $ ln -s /path/to/MOT2015_challenge/data/2DMOT2015 mot_benchmark
  # Run the demo with the --display flag
  $ python3 sort.py --display
  ~~~
  + Python using OpenCV verison
  ~~~shell
  # need same dataset as above
  $ git clone https://github.com/engcang/ros-yolov3-sort.git
  $ cd ros-yolov3-sort/SORT/python_original
  $ python3 sort-opencv.py --display
  ~~~
  + C++ Windows version : refer [here](https://github.com/mcximing/sort-cpp)
  + C++ Ubuntu version
  ~~~shell
  $ git clone https://github.com/engcang/ros-yolov3-sort.git
  $ cd ros-yolov3-sort/SORT/cpp_ubuntu/build
  $ cmake .. && make
  $ cd .. 
  $ ./sort_cpp_node 1 (with OpenCV image)
  $ ./sort_cpp_node 0 (without image)
  ~~~
  
<br>
  
## To run ROS version

## Result
