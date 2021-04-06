# YOLO and ROS version installation, implementation, application, and comparison
+ Darknet version: [recent github](https://github.com/AlexeyAB/darknet), [homepage](https://pjreddie.com/darknet/), [old github](https://github.com/pjreddie/darknet)
+ OpenCV(DNN) version: [original code gist](https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49)
  + [OpenCV-dnn benchmark and discuss](https://github.com/AlexeyAB/darknet/issues/6245), [OpenCV-dnn benchmark](https://github.com/AlexeyAB/darknet/issues/6067)
+ OpenVINO version - not OpenVINO built binary but OpenVINO enabled OpenCV(DNN) version
+ tensorRT(tkDNN) version - [github](https://github.com/ceccocats/tkDNN)
+ other versions: [other versions links in original github](https://github.com/AlexeyAB/darknet#yolo-v4-in-other-frameworks)

| YOLO<br>v3 / v3-tiny<br>v4 / v4-tiny|                        Remark                    |     CPU version | CPU<br>openMP<br>AVX | GPU<br>    support |
|:-----------------------------------:|:------------------------------------------------:|:---------------:|:--------------------:|:------------------:|
|               Darknet               |                     .c codes                     |        O        |           O          |          O         |
|              OpenCV-dnn             |               OpenCV ver from 4.4.0              |        O        |           -          |          -         |
|      OpenCV-dnn<br>+ CUDA/cuDNN     |                 OpenCV ver from 4.4.0            |       utilzed   |           -          |          O         |
|         OpenCV-dnn<br>+ OpenVINO    | Intel only, <br>prebuilt OpenCV<br>from OpenVINO |          O      |           -          |          -         |
|           TensorRT(tkDNN)           |                     need GPU                     |     utlized     |           -          |          O         |

<br>

# Index
## 1. [Results](#1-results-1)

## 2. Prerequisites
#### ● [CUDA / cuDNN](#-cuda--cudnn-1)
#### ● [OpenCV](#-opencv-1)
#### ● [OpenCV with CUDA / cuDNN](#-opencv-with-cuda--cudnn-1)
#### ● OpenCV with OpenVINO manual build: not recommended, [direct link](https://github.com/opencv/opencv/wiki/Intel's-Deep-Learning-Inference-Engine-backend)
  + OpenVINO's prebuilt binary OpenCV is recommended instead. Refer installation below
#### ● [cv_bridge](#-cv_bridge-opencv---ros-bridge): OpenCV - ROS bridge, should be built when OpenCV is manually built
#### ● [tensorRT](#-tensorrt-1)
#### ● [OpenVINO](#-openvino-1)

## 3. Installation
#### ● Darknet ver.
#### ● OpenCV(DNN) ver.
#### ● OpenVINO ver.
#### ● tensorRT(tkDNN) ver.

## 4. Installation for ROS version
#### ● Darknet ver.
#### ● OpenCV(DNN) ver.
#### ● OpenVINO ver.
#### ● tensorRT(tkDNN) ver.

---

<br><br><br><br>

# 1. Results
#### ● Tested on [2015 MOT dataset](https://motchallenge.net/data/MOT15/)
#### ● on i9-10900k+GTX Titan X(pascal) / i9-10900k+RTX 3080 / Intel NUC10i7FNH (i7-10710U) / Jetson TX2 / Jetson NX
#### ● GPU monitor from Jetsonhacks for *Jetson boards* [here](https://github.com/jetsonhacks/gpuGraphTX)
## ● Youtube videos: [Playlist of all results](https://www.youtube.com/playlist?list=PLvgPHeVm_WqIUHg7iu0g73-yaS08kv6-5)
+ text
<a href="http://www.youtube.com/watch?feature=player_embedded&v=MYbjjg_Mics" target="_blank"><img src="http://img.youtube.com/vi/MYbjjg_Mics/0.jpg" alt="IMAGE ALT TEXT" width="320" border="10" /></a>

<br><br><br>

# 2. Prerequisites
### ● CUDA / cuDNN

---

<br>

### ● OpenCV

---

<br>

### ● OpenCV with CUDA / cuDNN

---

<br>

### ● cv_bridge: OpenCV - ROS bridge

---

<br>

### ● tensorRT

---

<br>

### ● OpenVINO
[OpenVINO download](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html)

---

<br><br><br>

# 3. Installation
### ● Darknet ver.

<details><summary>[CLICK To See]</summary>
  
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

</details>

---

<br>

### ● OpenCV(DNN) ver.

---

<br>

### ● OpenVINO ver.

---

<br>

### ● tensorRT(tkDNN) ver.

---

<br><br><br>

# 4. Installation for ROS version
### ● Darknet ver.
<details><summary>[CLICK To See]</summary>
  
#### original repo - upto [v4 : here](https://github.com/tom13133/darknet_ros), upto [v3 : here](https://github.com/leggedrobotics/darknet_ros)
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
+ with Logitech c930e Video clip [here](https://youtu.be/nfPVkNXSs-A)
<p align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=nfPVkNXSs-A" target="_blank"><img src="yolo_v3_capture_20200620.png" alt="IMAGE ALT TEXT" width="320" border="10" /></a></p>

</details>

---

<br>

### ● OpenCV(DNN) ver.


---

<br>

### ● OpenVINO ver.

---

<br>

### ● tensorRT(tkDNN) ver.

