import cv2
import time
import numpy as np

CONFIDENCE_THRESHOLD = 0.99
NMS_THRESHOLD = 0.05

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

vc = cv2.VideoCapture("MOT_19201080_30hz.mp4")

net = cv2.dnn.readNet("yolov7-tiny_640x640.onnx")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #must be enabled for GPU
   ### Either DNN_TARGET_CUDA_FP16 or DNN_TARGET_CUDA must be enabled for GPU
   ### DNN_TARGET_CUDA shows better perf. (default for most CNN)
   ### DNN_TARGET_CUDA_FP16 shows faster, but only supported for recent GPUs
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16) 
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) #must be enabled for CPU
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) #must be enabled for CPU

#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE) #OpenVINO
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) #must be enabled for CPU
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

avg_FPS=0; count=0; total_fps=0;
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        break
    height, width, channels = frame.shape
    x_factor = width / 640.0
    y_factor =  height / 640.0

    start = time.time()
    blob = cv2.dnn.blobFromImage(frame, 0.003921569, (640, 640), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
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
            if confidence > CONFIDENCE_THRESHOLD:
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
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

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
        label = "%s : %f" % (class_names[classid], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    end_drawing = time.time()
    
    fps_label = "avg FPS: %.2f FPS: %.2f (excluding postprocessing %.2fms drawing %.2fms)" % (avg_FPS, 1 / (end - start), (end_postprocessing - start_postprocessing)*1000.0, (end_drawing - start_drawing) * 1000.0)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 127), 1)
    cv2.imshow("detections", frame)
print(avg_FPS)
