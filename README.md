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
    - python detect_video_retinaface.py
### Face mask detection with yoloV5
#### Test video
    - python detect_video_yolov5.py
## Distance and Crowd density














