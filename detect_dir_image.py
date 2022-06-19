"""
Project: FACE MASK RECOGNITION, DISTANCE AND CROWD DENSITY
Member: DA0 DUY NGU, LE VAN THIEN
Instructor: PhD. TRAN THI MINH HANH
*************** CONTACT INFORMATION ***********************************
THE UNIVERSITY OF DA NANG
THE UNIVERSITY OF SCIENCE AND TECHNOLOGY
THE FACULTY OF ELECTRONIC AND TELECOMMUNICATION
Major: Computer engineering
Address: 54 Nguyen Luong Bang Street, Lien Chieu District, Da Nang city
***********************************************************************
"""

from glob import glob
import cv2
from retinaface.detector import FaceDetection
import argparse
import os
from classification.utils.load_model import Model
from pathlib import Path


# ******************************** ROOT PATH *****************************
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT / 'weight'


def draw_bbox(image, bboxs):
    for label, bbox in bboxs:
        x1, y1, x2, y2, score = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(image, str(round(score, 2)), (x1, y1+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, label, (x1, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2, cv2.LINE_AA)
    return image


def detect_image(path_image):
    """
    function: detect face mask of folder image
    :param path_image: path of folder contain image
    :return: None
    """
    if not os.path.exists(path_image):
        raise ValueError("Input folder (", path_image, ") does not exist.")
    list_images = glob(path_image + '/*.jpg') + glob(path_image + '/*.jpeg') + glob(path_image + '/*.png')
    list_images.sort()
    print("Number image: ", len(list_images))
    # load model face mask classification
    model = Model(WEIGHTS / 'result_mobilenetv2')
    # Load model face detect
    detector = FaceDetection(net='mobilenet').detect_faces
    # Set thresh
    thresh = 0
    for path in list_images:
        image = cv2.imread(path)
        h, w, _ = image.shape
        bboxs, landmark = detector(image)
        list_predict = []
        for bbox in bboxs:
            x1, y1, x2, y2 = [round(i) for i in bbox[0:4]]
            x1, y1, x2, y2 = max(x1 - thresh, 0), max(y1 - thresh, 0), min(x2 + thresh, w - 1), min(y2 + thresh, h - 1)
            pred = model.predict(image[y1:y2, x1:x2])
            list_predict.append((pred[0] + ': %d%%' % (round(max(pred[1]))), (x1, y1, x2, y2, bbox[4])))
        # save data file yolo
        image = draw_bbox(image, list_predict)
        cv2.imshow('image', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Face Image')
    parser.add_argument("-f", "--file_folder", help="folder file image", default='', type=str)
    args = parser.parse_args()
    path_image = args.file_folder
    detect_image(path_image)




