### Original repo : [here](https://github.com/leggedrobotics/darknet_ros)
+ Just explanation for installation

<br>

### Installation
+ build OpenCV with CUDA as [here](https://github.com/engcang/vins-application#-opencv-with-cuda--necessary-for-gpu-version-1)
  + **Recommend version is 3.4.0** because darknet has to use C API with OpenCV [refer](https://github.com/pjreddie/darknet/issues/551)
  + or **Patch as [here](https://github.com/opencv/opencv/issues/10963)** to use other version 
    + should comment the /usr/local/include/opencv2/highgui/highgui_c.h line 119 [as here](https://stackoverflow.com/questions/48611228/yolo-compilation-with-opencv-1-fails)
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

+ Get and build Darknet_ROS version from [here](https://github.com/leggedrobotics/darknet_ros)
~~~shell
$ cd catkin_workspace/src
$ git clone --recursive git@github.com:leggedrobotics/darknet_ros.git
$ cd ../
# before build, check (-O3 -gencode arch=compute_<version>,code=sm_<version>) part in darknet_ros/darknet_ros/CMakeLists.txt if you use CUDA
$ catkin build darknet_ros -DCMAKE_BUILD_TYPE=Release
~~~
+ or it also can be done using https:// address
~~~shell
$ cd catkin_workspace/src
$ git clone https://github.com/leggedrobotics/darknet_ros
$ cd darknet_ros/ && git submodule update --init --recursive
$ cd ~/catkin_workspace
# before build, check (-O3 -gencode arch=compute_<version>,code=sm_<version>) part in darknet_ros/darknet_ros/CMakeLists.txt if you use CUDA
$ catkin build darknet_ros -DCMAKE_BUILD_TYPE=Release
~~~
