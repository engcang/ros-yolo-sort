# SORT, Simple Online Realtime Tracking of multiple object
  + Original github [here](https://github.com/abewley/sort)
  + C++ Window version github [here](https://github.com/mcximing/sort-cpp)

<br>

### Implemented version
  + Original version + (OpenCV instead of matplotlib), **69.3/15.9FPS** (w, w/o display) on NVIDIA Jetson TX2
  + C++ Ubuntu version => ***200 times faster*** than Python(original) version, **2718/4790FPS** (w, w/o display) on NVIDIA Jetson TX2
    + Just few lines were edited from *Windows version*
  + **ROS** version implemented - only ***Python version yet***

<br>

  
## To run ROS version
+ Check the *topic Names* and tune *max_age* and *min_hits* in sort_ROS_python folder's **ros-sort.py**
+ and simply run
~~~shell
  $ mv ros-sort.py ~/your_workspace/src/package_PATH/scripts/ && chmod +x ros-sort.py
  $ rosrun package_name ros-sort.py _/display:=True 
  # ROS parameters : _/display (for image publish) _/max_age:=200 (max age for track)
  # _/min_hits:=1 (min hits for track) _/img_topic:=/usb_cam/image_raw (topic name)
  # _/iou_threshold:=0.2 (IoU threshold for ID matching)
~~~
+ ROS params can be set using .launch file

<br>

## Installation
#### Please install *pip packages on Python 3*
  + Original version : referred [here](https://github.com/abewley/sort)
    + running the code using **Pre-Detected bounding boxes from [Faster RCNN](https://github.com/ShaoqingRen/faster_rcnn)**
  ~~~shell
  # Dependencies
  $ python3 -m pip install -r requirements.txt
  # To run the tracker with the provided detections:
  $ cd path/to/sort
  $ python3 sort.py
  ~~~
  + To display the results you need to Download the [2D MOT 2015 benchmark dataset](https://motchallenge.net/data/2D_MOT_2015/#download)
    + using *matplotlib* library, slow
  ~~~shell
  $ cd <sort_directory>
  # Create a symbolic link to the dataset
  $ ln -s /path/to/MOT2015_challenge/data/2DMOT2015 mot_benchmark
  # Run the demo with the --display flag
  $ python3 sort.py --display
  ~~~
  
  <br>
  
  + Python using OpenCV verison
  ~~~shell
  # need same dataset as above
  $ git clone https://github.com/engcang/ros-yolov3-sort.git
  $ cd ros-yolov3-sort/SORT/python_original
  # Create a symbolic link to the dataset
  $ ln -s /path/to/MOT2015_challenge/data/2DMOT2015 mot_benchmark
  $ python3 sort-opencv.py --display
  ~~~
  
  <br>
  
  + C++ Windows version : refer [here](https://github.com/mcximing/sort-cpp)
  
  <br>
  
  + C++ Ubuntu version
    + running the code using **Pre-Detected bounding boxes from [Faster RCNN](https://github.com/ShaoqingRen/faster_rcnn)**
  ~~~shell
  $ git clone https://github.com/engcang/ros-yolov3-sort.git
  $ cd ros-yolov3-sort/SORT/cpp_ubuntu/build
  $ cmake .. && make
  $ cd .. 
  $ ln -s /path/to/MOT2015_challenge/data/2DMOT2015 mot_benchmark
  $ ./sort_cpp_node 1 (with OpenCV image)
  $ ./sort_cpp_node 0 (without image)
  ~~~
  
<br>


## Result
+ on MOT benchmark dataset
  + Python version **TX2 Video [here](https://youtu.be/MYbjjg_Mics)**
  + **Xavier NX Video [here](https://youtu.be/iruvwU7yveA)**
  <p align="center">
  <img src="https://github.com/engcang/ros-yolov3-sort/blob/master/SORT_and_ROS_ver/python.JPG" width="600"/>
  </p>
  
  + Cpp version **TX2 Video [here](https://youtu.be/vkucBw3mQ7Y)**
  + **Xavier NX Video [here](https://youtu.be/xKaU3FE9PoI)**
  <p align="center">
  <img src="https://github.com/engcang/ros-yolov3-sort/blob/master/SORT_and_ROS_ver/cpp.JPG" width="600"/>
  </p>
