## CMPE249 PROJECT: BUILDING PERCEPTION STACK FOR AUTONOMOUS DRIVING
### 2D & 3D OBJECT DETECTION

A simple ROS project that perform 2D & 3D object detection on Camera & Lidar sensor data. This package uses Ultralytics' YOLOv5 to perform 2D object detection on images and PointPillar to perform 3D object detection on point cloud data.

---
*Environment*:
- Ubuntu 20.04
- ROS Noetic on Miniconda (Robostack)
- Pytorch 1.10
- CUDA 11.3

**Setup**
- YOLOv5: download pretrained weight via the link below and place it at *av_perception/src/yolov5/weights* directory. This model was trained on Argoverse Dataset for 25 epochs.
  https://drive.google.com/file/d/1X-v6NvGIoHopULI3BPBFJMzV6K978ik5/view?usp=sharing.
- PointPillar: download pretrained weight via the link below and place it at *av_perception/src/pointpillar/pcdet/cfg* directory. You may also need to change the CONFIG PATH in pointpillar.yaml file to your local path.
https://drive.google.com/file/d/1cjFWjIhif4-EsiBtvS_WWUV7e7ycTEm7/view?usp=sharing
- Rosbag File: for testing, you can download the sample KITTI rosbag file below. Alternatively, you can also create your own KITTI rosbag file using kitti2bag python package. https://drive.google.com/file/d/1r-UndRpfRTltfJ-jJxKOLWUBy_5hmCo4/view?usp=sharing.
- Environment: make sure you have installed ROS & Pytorch with GPU. I'm currently using ROS Noetic on Miniconda via Robostack (https://robostack.github.io/). I'm considering switching to docker instead due to some issues with the PCL library.


**How to run**
```
After building this ROS package with catkin_make or catkin build

(terminal 1) $ roscore
(terminal 2) $ roslaunch av_perception demo.launch
(terminal 3) $ rosbag play <path-to-rosbag-file> -r 0.25

```
This package was tested on Geforce GTX 1060 4GB.

### Demo Result

![image](/fig/demo.png)
