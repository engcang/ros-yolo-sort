# YOLOv3-application
+ Darknet [homepage](https://pjreddie.com/darknet/)
+ Darknet [Github](https://github.com/pjreddie/darknet)
#### ● GPU monitor from Jetsonhacks for *Jetson boards* [here](https://github.com/jetsonhacks/gpuGraphTX)

<br><br>

#### ● OpenCV version >= 3.4 is needed to run YOLO v3
### ● OpenCV build
+ For **Jetson Xavier NX** board -> just build same as Desktop PC as [here](https://github.com/engcang/vins-application#-opencv-with-cuda--necessary-for-gpu-version-1)
+ Jetsonhacks build OpenCV on TX2 [here](https://github.com/jetsonhacks/buildOpenCVTX2)
  + before run the .sh script file, please refer [this file change](https://github.com/jetsonhacks/buildOpenCVTX2/pull/34/files) ***For JetPack >= 4.3***

+ To use **onboard camera of TX2 Development Kit, use this OpenCV build [here](https://github.com/Alro10/OpenCVTX2)**
  + Just using here instead of Jetsonhacks is totally fine
pkg-config --modversion opencv
pkg-config --libs --cflags opencv
  
+ After built OpenCV, we may remove OpenCV source folder to save disk
  + But if you want to delete the built OpenCV, you should go to OpenCV source folder and ***$ sudo make uninstall***
~~~shell
  $ cd buildOpenCVTX2 && ./removeOpenCVSource.sh #if used Jetsonhacks' script
  $ rm -r ~/opencv  #or just manually removing the folder is fine.
~~~

<br><br>

### ● Installation
+ Clone and make
~~~shell
  $ cd 
  $ git clone https://github.com/pjreddie/darknet
  $ gedit Makefile # => Edit first 3 lines if you want to use them (OPENCV=1 is needed to watch GUI result)
  $ make
~~~
+ Download weights from homepage
~~~shell
  $ cd ~/darknet
  $ wget https://pjreddie.com/media/files/yolov3.weights
  $ wget https://pjreddie.com/media/files/yolov3-tiny.weights #for tiny (much faster, less accurate)
~~~

<br><br>

### ● Execution

+ Using on Test data (Image)
~~~shell
  $ ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg #or any other image files
   # -> yolov3 will assume memory a lot.
  $ ./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg  #V3 tiny
~~~
+ Using on Test data (Video, Live)
~~~shell
  $ ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg #or any other image files
   # -> yolov3 will assume memory a lot.
  $ ./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg  #V3 tiny
~~~
+ Using onboard camera of TX2 development kit (Live), *tiny*
~~~shell
  $ ./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
~~~
+ Using USB camera on TX2 (Live), *tiny*
~~~shell
  $ ./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -c 1 # 1 is camera number, as onboard camera is 0, usb camera is 1
  $ ./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights /dev/video1 #same here
  # if not using development kit, instead carrier board, usb camera will be camera0
~~~

<br><br>


### ● Trouble shooting
+ *not such file of directory tegra/libGL.so* when building **OpenCV**
  + Change the script file before run it, [here](https://github.com/jetsonhacks/buildOpenCVTX2/pull/34/files)
+ *make[2]: *** No rule to make target '/usr/lib/aarch64-linux-gnu/libGL.so', needed by 'lib/libopencv_cudev.so.3.4.1'. Stop.*
  + **OpenCV** was not built well.
  + or for TX2, did not build **OpenCV** manually yet -> If you want to use pre-installed OpenCV from Jetpack, 
+ *nvcc not found* -> After OpenCV, when building **YOLOv3**
  ~~~shell
    $ echo "export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}" >> ~/.bashrc
    $ echo "export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
    # or type those exports into ~/.bashrc manually
    $ source ~/.bashrc
  ~~~
+ *No such file lib....* when **execute**
~~~shell
  $ sudo ldconfig
~~~
  
<br><br>

### ● Result

<br>

+ **Result [Video on Xavier NX](https://youtu.be/Rqkp7XEiQqU)
+ **Result [video on TX2 using USB cam](https://youtu.be/w3Em89Z58og)** with ***default cfg, default weights, default trained model***
  + on the monitor playing youtube [video for detection](https://www.youtube.com/watch?v=wqctLW0Hb_0&feature=youtu.be)
  <p align="center">
  <img src="yolo.png" width="600"/>
  </p>

<br>

+ on Test image with ***default cfg, default weights, default trained model***
  <p align="center">
  <img src="tiny.png" width="600"/>
  </p>
  <p align="center">
  <img src="dog.png" width="600"/>
  </p>



---



### Original repo - upto [v4 : here](https://github.com/tom13133/darknet_ros), upto [v3 : here](https://github.com/leggedrobotics/darknet_ros)
+ Just explanation for installation

<br>

### Installation
+ build OpenCV with CUDA as [here](https://github.com/engcang/vins-application#-opencv-with-cuda--necessary-for-gpu-version-1)
  + **User OpenCV version 3.4.0** because darknet has to use C API with OpenCV [refer](https://github.com/pjreddie/darknet/issues/551)
  + **(Recommended)** or **Patch as [here](https://github.com/opencv/opencv/issues/10963)** to use other version 
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

+ Get and build Darknet_ROS version from upto [v4 : here](https://github.com/tom13133/darknet_ros) upto v3 [here](https://github.com/leggedrobotics/darknet_ros)
~~~shell
$ cd catkin_workspace/src
$ git clone https://github.com/leggedrobotics/darknet_ros # up to v3
$ git clone https://github.com/tom13133/darknet_ros # up to v4
$ cd darknet_ros/ && git submodule update --init --recursive
$ cd ~/catkin_workspace
# before build, check (-O3 -gencode arch=compute_<version>,code=sm_<version>) part in darknet_ros/darknet_ros/CMakeLists.txt if you use CUDA
# ex) 75 for GTX1650
$ catkin build darknet_ros -DCMAKE_BUILD_TYPE=Release
~~~

### Running
+ To run, need cfg files from [darknet homepage](https://github.com/AlexeyAB/darknet/tree/master/cfg)
+ need weights file
~~~shell
$ wget https://pjreddie.com/media/files/yolov3-tiny.weights
# or download at the site : https://github.com/AlexeyAB/darknet/releases
~~~
+ and use the proper .yaml file and .launch files as attached in this repo

~~~shell
$ roslaunch darknet_ros yolov3tiny.launch
$ roslaunch darknet_ros yolov4tiny.launch
~~~

<br>

### Results
+ with [USB-Camera ROS driver](http://wiki.ros.org/usb_cam), Logitech c930e Video clip [here](https://youtu.be/nfPVkNXSs-A)
  <p align="center">
  <img src="yolo_v3_capture_20200620.png" width="600"/>
  </p>
