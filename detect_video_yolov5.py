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

import argparse
import time
from yolov5.detect import *
import numpy as np
from functions_processing import compute_distance, check_point_in_polygon
from pathlib import Path


# ******************************** ROOT PATH *****************************
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT / 'weight'

# ************** SETUP CAMERA ******************
# Focal length (pixel)
# F = D*P/H
# D: distance from person to camera (cm)
# P: pixel height of person in image (pixel)
# Height average of person (cm)
F = 615
H = 165
# **********************************************
list_point_area = []


def get_pixel(event, x, y, flags, param):
    global list_point_area
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(image_set, (x, y), 5, (0, 0, 255), 10)
        if (len(list_point_area) >= 1) and (len(list_point_area) <= 3):
            cv2.line(image_set, (x, y), list_point_area[len(list_point_area)-1], (70, 70, 70), 2)
            if len(list_point_area) == 3:
                cv2.line(image_set, (x, y), list_point_area[0], (70, 70, 70, 2))
        if 'list_point_area' not in globals():
            list_point_area = []
        list_point_area.append((x, y))


def detect_5k(url_video, path_model, flag_save=False, fps=None, name_video='video.avi'):
    """
    function: detect 5k, distance, face mask, total person
    :param url_video: url of video
    :param path_model: path model detect yolov5
    :param flag_save: flag save video True or False
    :param fps: value fps
    :param name_video: name of video
    :return: None
    """

    # load model detect yolov5
    y5_model = Y5Detect(weights=path_model)
    if url_video == '':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(url_video)

    # get size
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if frame_height > 1080 and frame_width > 1920:
        frame_width = 1920
        frame_height = 1080
    # get fps of camera
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    # save video
    if flag_save is True:
        video_writer = cv2.VideoWriter(name_video,
                                       cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))
    global image_set
    count = 0
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) == ord('q'):
            break

        h, w, _ = frame.shape

        if h > 1080 and w > 1920:
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            h, w, _ = frame.shape

        if count == 0:
            while True:
                image_set = frame
                cv2.imshow('image', image_set)
                cv2.waitKey(1)
                if len(list_point_area) == 4:
                    cv2.destroyWindow('image')
                    break
        count += 1
        # initial coordinate spatial x, y, z (cm) and x, y (pixel)
        list_spatial = dict()
        list_bbox_body = dict()

        # detect body of person
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox, label, score = y5_model.predict(image)
        bboxs, labels, scores = np.array(bbox), np.array(label), np.array(score)

        # *********************** TRACKING PERSON  ********************
        if bbox is not None:
            id_person = (labels == 'person')
            bbox_person = bboxs[id_person]
            score_person = scores[id_person]
            if len(bbox_person) > 0:
                # initial idx for dict
                idx = 0
                for box in bbox_person:
                    if check_point_in_polygon(box[:4], list_point_area[:4]) != 360:
                        continue
                    list_bbox_body[idx] = box[:4]
                    # convert center coordinate of bounding boxes (pixel) about spatial coordinate (cm)
                    result_cm = compute_distance(box[:4], H, F)
                    list_spatial[idx] = result_cm
                    idx += 1
        # draw polygon
        for idx, point in enumerate(list_point_area):
            cv2.circle(frame, point, 5, (0, 0, 255), 10)
            if idx < len(list_point_area) - 1:
                cv2.line(frame, list_point_area[idx], list_point_area[idx + 1], (255, 0, 0), 2)
            else:
                cv2.line(frame, list_point_area[idx], list_point_area[0], (255, 0, 0), 2)

        # count without mask
        count_without_mask = 0
        id_mask = (labels == ('with_mask' or 'without_mask'))
        bbox_mask = bboxs[id_mask]
        score_mask = scores[id_mask]
        label_mask = labels[id_mask]
        for idx, bbox in enumerate(bbox_mask):
            count_without_mask += 1
            # draw bounding box of with mask and without mask
            frame, _ = draw_boxes(frame, bbox, label=label_mask[idx], scores=score_mask[idx])

        # compute distance between every person detect in a frame
        # initial set contain the index of the person that violates the distance
        violates_person = set()
        for i in list_spatial.keys():
            for j in list_spatial.keys():
                if i < j:
                    # compute distance between two people
                    distance = np.sqrt(np.sum((np.array(list_spatial[i])-np.array(list_spatial[j]))**2))

                    # check distance (2m == 200cm)
                    if distance < 200:
                        # append set index
                        violates_person.add(i)
                        violates_person.add(j)

        # draw bounding of person
        for i in list_spatial.keys():
            if i in violates_person:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            x1, y1, x2, y2 = list_bbox_body[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        fps = int(1/(time.time()-start))
        # draw total without mask, distance violates, person, fps
        cv2.putText(frame, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    2, cv2.LINE_AA)
        cv2.putText(frame, 'without mask: ' + str(count_without_mask), (0, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Distance < 2m: ' + str(len(violates_person)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Total person: ' + str(len(list_bbox_body)), (0, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('video', frame)
        cv2.waitKey(1)

        if flag_save is True:
            video_writer.write(frame)

    cap.release()
    if flag_save is True:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Face On Video')
    parser.add_argument("-fn", "--file_name", help="video file name or rtsp", default='', type=str)
    parser.add_argument("-ds", "--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("-op", "--option", help="if save video then choice option = 1", default=False, type=bool)
    parser.add_argument("-o", "--output", help="path to output video file", default='face_recording.avi', type=str)
    parser.add_argument("-f", "--fps", default=20, help="FPS of output video", type=int)
    args = parser.parse_args()

    # path model
    path_models = WEIGHTS / 'result_yolov5/best.pt'

    # path video tesst
    url = '/home/duyngu/Downloads/Do_An/video_test/TownCentre.mp4'
    source = args.file_name
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_pixel)
    # if run  as terminal, replace url = source
    detect_5k(url_video=url, path_model=path_models,
              flag_save=args.option, fps=args.fps, name_video=args.output)

