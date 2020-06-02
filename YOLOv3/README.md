# YOLOv3-application
+ Darknet [homepage](https://pjreddie.com/darknet/)
+ Darknet [Github](https://github.com/pjreddie/darknet)
#### ● GPU monitor from Jetsonhacks for *Jetson boards* [here](https://github.com/jetsonhacks/gpuGraphTX)

<br><br>

#### ● OpenCV version >= 3.4 is needed to run YOLO v3
### ● OpenCV build
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

+ **Result [video 1](https://youtu.be/w3Em89Z58og)** with ***default cfg, default weights, default trained model***
  + on the monitor playing youtube [video for detection](https://www.youtube.com/watch?v=wqctLW0Hb_0&feature=youtu.be)
  <p align="center">
  <img src="https://github.com/engcang/YOLOv3-application/blob/master/yolo.png" width="600"/>
  </p>

<br>

+ on Test image with ***default cfg, default weights, default trained model***
  <p align="center">
  <img src="https://github.com/engcang/YOLOv3-application/blob/master/tiny.png" width="600"/>
  </p>
  <p align="center">
  <img src="https://github.com/engcang/YOLOv3-application/blob/master/dog.png" width="600"/>
  </p>
