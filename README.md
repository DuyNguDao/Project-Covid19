# FACE MASK RECOGNITION, DISTANCE AND CROWD DENSITY
### DEV
- Dao Duy Ngu
- Le Van Thien
### Install conda environments
- conda create --name detect_faces python=3.8
### Install package
- pip install -r requirements.txt
## DETECT FACE MASK
### Face detection and classification with deep learning
### Detect face with Retinaface
A simple package of face detection
This package is built on top of the [Retinaface](https://github.com/biubug6/Pytorch_Retinaface)

- Thank you authors of two backbone at link github:
    - [Retinaface](https:/ttps://github.com/hphuongdhsp/retinaface)
### Download model retinaface
- [Mobilenet](https://drive.google.com/drive/folders/1nvKaj3pZJNJmxEWWYSCu-Xe1B3iBQKVr?usp=sharing)
- [Resnet](https://drive.google.com/drive/folders/1nvKaj3pZJNJmxEWWYSCu-Xe1B3iBQKVr?usp=sharing)
- Note: Download file about folder (./Project_5K/retinaface/weights)
### Face Mask Classification with MobilenetV2
#### Test video
    - python detect_dir_image.py or detect_video_retinaface.py
### Face mask detection with yoloV5
#### Test video
    - python detect_video_yolov5.py
## Distance and Crowd density
- Detect distance with bird-eyes method 
    - python distance_bird_eyes_video.py
    - click four point with top left, top right, bottom right, bottom left
- Detect distance with distance camera 3D
    - python detect_video_yolov5.py
    - click seven point:
      - click first four point: left, top right, bottom right, bottom left
      - click again three point then compute pixel width, height: top left, top right, bottom left
## Demo
- video demo
-![](https://github.com/DuyNguDao/Project-Covid19/blob/master/video_demo.mp4)
- Bird eyes map
- ![](https://github.com/DuyNguDao/Project-Covid19/blob/master/bird_eyes_map.gif)
