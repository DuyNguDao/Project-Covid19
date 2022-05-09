"""
Project: FACE MASK RECOGNITION, DISTANCE AND CROWD DENSITY
Member: DA0 DUY NGU, LE VAN THIEN
Instructor: PhD. TRAN THI MINH HANH
*************** CONTACT INFORMATION ***********************************
THE UNIVERSITY OF DA NANG
THE UNIVERSITY OF SCIENCE AND TECHNOLOGY
THE FACULTY ELECTRONIC AND TELECOMMUNICATION
Major: Computer engineering
Address: 54 Nguyen Luong Bang Street, Lien Chieu District, Da Nang city
***********************************************************************
"""

import argparse
import time
from yolov5.detect import *
import numpy as np

# ************** SETUP CAMERA ******************
# Focal length (pixel)
# F = D*P/H
# D: distance from person to camera (cm)
# P: pixel height of person in image (pixel)
# Height average of person (cm)
F = 615
H = 165
# **********************************************


def compute_distance(bboxes):
    """
    function: compute distance from person to camera
    :param bboxes: bounding box of person in image
    :return: x, y, z (cm)
    """
    x1, y1, x2, y2 = bboxes
    # center coordinate of bounding box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    # height if person (pixel)
    height = y2 - y1
    # distance from camera base on triangle similarity technique (cm)
    distance = (H*F)/height

    # center coordinate of bounding boxes base on triangle similarity technique (cm)
    x_center_cm = (x_center * distance)/F
    y_center_cm = (y_center * distance)/F
    return x_center_cm, y_center_cm, distance


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

        # initial coordinate spatial x, y, z (cm) and x, y (pixel)
        list_spatial = dict()
        list_bbox_body = dict()

        # detect body of person
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox, label, score = y5_model.predict(image)

        # initial idx for dict
        idx = 0
        # count without mask
        count_without_mask = 0
        for i in range(len(label)):
            # check bbox of person
            if label[i] == 'person':
                list_bbox_body[idx] = bbox[i]
                # convert center coordinate of bounding boxes (pixel) about spatial coordinate (cm)
                result_cm = compute_distance(bbox[i])
                list_spatial[idx] = result_cm
                idx += 1
                continue
            if label[i] == 'withoutmask':
                count_without_mask += 1

            # draw bounding box of with mask and without mask
            frame, _ = draw_boxes(frame, bbox[i], label=label[i], scores=score[i])

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
    parser.add_argument("-op", "--option", help="if save video then choice option = 1", default=False, type=bool)
    parser.add_argument("-o", "--output", help="path to output video file", default='face_recording.avi', type=str)
    parser.add_argument("-f", "--fps", default=20, help="FPS of output video", type=int)
    args = parser.parse_args()

    path_models = '/home/duyngu/Desktop/Project_5K/model_training/Yolov5/best.pt'
    url = '/home/duyngu/Downloads/Do_An/video_cctv.mp4'
    source = args.file_name
    # if run  as terminal, replace url = source
    detect_5k(url_video=url, path_model=path_models, flag_save=args.option, fps=args.fps, name_video=args.output)

