### Original repo : upto [v4 : here](https://github.com/tom13133/darknet_ros) upto [v3 : here](https://github.com/leggedrobotics/darknet_ros)
+ Just explanation for installation

<br>

### Installation
+ build OpenCV with CUDA as [here](https://github.com/engcang/vins-application#-opencv-with-cuda--necessary-for-gpu-version-1)
  + **Recommend version is 3.4.0** because darknet has to use C API with OpenCV [refer](https://github.com/pjreddie/darknet/issues/551)
  + or **Patch as [here](https://github.com/opencv/opencv/issues/10963)** to use other version 
    + should **comment** the /usr/local/include/opencv2/highgui/highgui_c.h line 139 [as here](https://stackoverflow.com/questions/48611228/yolo-compilation-with-opencv-1-fails)

<br>

+ build cv_bridge as I did for VINS-Fusion [here](https://github.com/engcang/vins-application#-installation-1)
~~~shell
$ cd ~/catkin_ws/src && git clone https://github.com/ros-perception/vision_opencv
# since ROS Noetic is added, we have to checkout to melodic tree
$ cd vision_opencv && git checkout origin/melodic
$ gedit vision_opencv/cv_bridge/CMakeLists.txt
~~~
+ Edit OpenCV PATHS in CMakeLists and include cmake file
~~~txt
#when error, try both lines
#find_package(OpenCV 3 REQUIRED PATHS /usr/local/share/OpenCV NO_DEFAULT_PATH
find_package(OpenCV 3 HINTS /usr/local/share/OpenCV NO_DEFAULT_PATH
  COMPONENTS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
  CONFIG
)
include(/usr/local/share/OpenCV/OpenCVConfig.cmake) #under catkin_python_setup()
~~~
~~~shell
$ cd .. && catkin build cv_bridge
~~~

<br>
<br>

+ Get and build Darknet_ROS version from upto [v4 : here](https://github.com/tom13133/darknet_ros) (not recommended) upto v3 [here](https://github.com/leggedrobotics/darknet_ros)
~~~shell
$ cd catkin_workspace/src
$ git clone https://github.com/tom13133/darknet_ros
$ cd darknet_ros/ && git submodule update --init --recursive
$ cd ~/catkin_workspace
# before build, check (-O3 -gencode arch=compute_<version>,code=sm_<version>) part in darknet_ros/darknet_ros/CMakeLists.txt if you use CUDA
# ex) 72 for GTX1650
$ catkin build darknet_ros -DCMAKE_BUILD_TYPE=Release
~~~

### Running
+ To run, need cfg files from [darknet homepage](https://github.com/pjreddie/darknet/tree/master/cfg)
+ need weights file
~~~shell
$ wget https://pjreddie.com/media/files/yolov3-tiny.weights
~~~
+ and use the proper .yaml file and .launch files as attached in this repo

~~~shell
$ roslaunch darknet_ros yolov3tiny.launch
~~~

<br>

### Results
+ with [USB-Camera ROS driver](http://wiki.ros.org/usb_cam), Logitech c930e Video clip [here](https://youtu.be/nfPVkNXSs-A)
  <p align="center">
  <img src="https://github.com/engcang/ros-yolov3-sort/blob/master/YOLO-ROS/YOLO%20V3_screenshot_20.06.2020.png" width="600"/>
  </p>
