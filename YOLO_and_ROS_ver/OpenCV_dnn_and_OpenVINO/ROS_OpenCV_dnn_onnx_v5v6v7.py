#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon April 19 20:58:03 2021
@author: mason
"""

''' import libraries '''
import cv2
import time
import rospy
import signal
import sys
import numpy as np

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

def signal_handler(signal, frame): # ctrl + c -> exit program
        print('You pressed Ctrl+C!')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


class cv_yolo_ros():
    def __init__(self):
        rospy.init_node('cv_yolo_ros_node', anonymous=True)
        self.flag = False
        self.inference_rate = rospy.get_param("/inference_rate", 30)
        self.inference_img_size = rospy.get_param("/inference_img_size", 640)
        self.img_in_topic = rospy.get_param("/img_in_topic", "/kitti/camera_color_left/image_raw")
        self.img_out = rospy.get_param("/img_out", True)
        self.img_out_topic = rospy.get_param("/img_out_topic", "/detected")

        self.confidence_threshold = rospy.get_param("/confidence_threshold", 0.98)
        self.nms_threshold = rospy.get_param("/nms_threshold", 0.25)

        self.class_file = rospy.get_param("/class_file", "classes.txt")
        self.onnx_file = rospy.get_param("/onnx_file", "yolov7-tiny_640x640.onnx")
        self.backend = rospy.get_param("/backend", cv2.dnn.DNN_BACKEND_CUDA)
    ### cv2.dnn.DNN_BACKEND_CUDA for GPU, 
    ### cv2.dnn.DNN_BACKEND_OPENCV for CPU
    ### cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE for OpenVINO
        self.target = rospy.get_param("/target", cv2.dnn.DNN_TARGET_CUDA)
    ### Either DNN_TARGET_CUDA_FP16 or DNN_TARGET_CUDA must be enabled for GPU
    ### cv2.dnn.DNN_TARGET_CPU for CPU or OpenVINO

        self.net=cv2.dnn.readNet(self.onnx_file)
        self.net.setPreferableBackend(self.backend)
        self.net.setPreferableTarget(self.target)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.img_subscriber = rospy.Subscriber(self.img_in_topic, Image, self.img_callback)
        self.img_publisher = rospy.Publisher(self.img_out_topic, Image, queue_size=1)
        
        self.bridge = CvBridge()
        self.rate = rospy.Rate(self.inference_rate)

        self.class_names = []
        with open(self.class_file, "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

    def img_callback(self, msg):
        self.img_cb_in = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.flag = True



if __name__=='__main__':
    avg_FPS=0; count=0; total_fps=0;
    cyr=cv_yolo_ros()
    COLORS = np.random.uniform(0, 255, size=(len(cyr.class_names), 3))
    while 1:
        try:
            if cyr.flag:
                start = time.time()
                frame = cyr.img_cb_in #temporal backup
                height, width, channels = frame.shape
                x_factor = width / float(cyr.inference_img_size)
                y_factor =  height / float(cyr.inference_img_size)
                blob = cv2.dnn.blobFromImage(frame, 0.003921569, (cyr.inference_img_size, cyr.inference_img_size), (0, 0, 0), swapRB=True, crop=False)
                cyr.net.setInput(blob)
                outputs = cyr.net.forward(cyr.output_layers)
                end = time.time()

                FPS = 1 / (end - start)
                total_fps = total_fps + FPS; count=count+1;
                avg_FPS = total_fps / float(count)

                start_postprocessing = time.time()
                class_ids = []
                confidences = []
                boxes = []
                for out in outputs[0]:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > cyr.confidence_threshold:
                            # Object detected
                            center_x = int(detection[0] * x_factor)
                            center_y = int(detection[1] * y_factor)
                            w = int(detection[2] * x_factor)
                            h = int(detection[3] * y_factor)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, cyr.confidence_threshold, cyr.nms_threshold)

                result_class_ids = []
                result_confidences = []
                result_boxes = []
                for i in indexes:
                    result_confidences.append(confidences[i[0]])
                    result_class_ids.append(class_ids[i[0]])
                    result_boxes.append(boxes[i[0]])
                end_postprocessing = time.time()

                start_drawing = time.time()
                for (classid, score, box) in zip(result_class_ids, result_confidences, result_boxes):
                    color = COLORS[int(classid) % len(COLORS)]
                    label = "%s : %f" % (cyr.class_names[classid], score)
                    cv2.rectangle(frame, box, color, 2)
                    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                end_drawing = time.time()

                fps_label = "avg FPS: %.2f FPS: %.2f (excluding postprocessing %.2fms drawing %.2fms)" % (avg_FPS, 1 / (end - start), (end_postprocessing - start_postprocessing)*1000.0, (end_drawing - start_drawing) * 1000.0)
                cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 127), 1)
                img=cyr.bridge.cv2_to_imgmsg(frame, "bgr8")
                img.header.stamp = rospy.Time.now()
                cyr.img_publisher.publish(img)
            cyr.rate.sleep()
        except (rospy.ROSInterruptException, SystemExit, KeyboardInterrupt) :
            print(avg_FPS)
            sys.exit(0)
