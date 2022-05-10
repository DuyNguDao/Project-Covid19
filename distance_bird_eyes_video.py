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
import cv2
from yolov5.detect import *
import numpy as np
from functions_processing import get_transform, point_distance, check_point_in_polygon

# ************** SETUP CAMERA ******************
# Setup width, height of object (cm)
w_cm = 200
h_cm = 200
# set size view, map
width_map = 480
height_map = 640
# **********************************************
list_point_area = []


def get_pixel(event, x, y, flags, param):
    global list_point_area
    if event == cv2.EVENT_LBUTTONUP:
        if len(list_point_area) <= 3:
            cv2.circle(image_set, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image_set, (x, y), 5, (255, 0, 0), 10)
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
    count = 0
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
        video_writer_map = cv2.VideoWriter('bird_eyes_map.avi',
                                       cv2.VideoWriter_fourcc(*'MJPG'), fps, (width_map, height_map))

    global image_set
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
                if len(list_point_area) == 7:
                    cv2.destroyWindow('image')
                    break
        # convert bird-eye-view
        src = np.float32(np.array(list_point_area[:4]))
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        transform = cv2.getPerspectiveTransform(src, dst)

        # compute number pixel on distance(cm) at bird-eye map
        pts = np.float32([np.array(list_point_area[4:])])
        tran_pts = cv2.perspectiveTransform(pts, transform)[0]

        distance_w = np.sqrt(np.sum((tran_pts[0]-tran_pts[1])**2))
        distance_h = np.sqrt(np.sum((tran_pts[0]-tran_pts[2])**2))

        count += 1
        # initial coordinate spatial x, y, z (cm) and x, y (pixel)
        list_transform = dict()
        list_bbox_body = dict()

        # detect body of person
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox, label, score = y5_model.predict(image)

        # draw polygon
        for idx, point in enumerate(list_point_area[:4]):
            cv2.circle(frame, point, 5, (0, 0, 255), 10)
            if idx < len(list_point_area[:4]) - 1:
                cv2.line(frame, list_point_area[idx], list_point_area[idx + 1], (255, 0, 0), 2)
            else:
                cv2.line(frame, list_point_area[idx], list_point_area[0], (255, 0, 0), 2)

        # initial idx for dict
        idx = 0
        # count without mask
        count_without_mask = 0
        for i in range(len(label)):
            # check bbox of person
            if label[i] == 'person':
                if check_point_in_polygon(bbox[i], list_point_area[:4]) != 360:
                    continue
                list_bbox_body[idx] = bbox[i]
                # transform point about bird-eyes-view
                list_transform[idx] = get_transform(bbox[i], transform)
                idx += 1
                continue
            if label[i] == 'withoutmask':
                count_without_mask += 1

            # draw bounding box of with mask and without mask
            frame, _ = draw_boxes(frame, bbox[i], label=label[i], scores=score[i])

        # compute distance between every person detect in a frame
        # initial set contain the index of the person that violates the distance
        violates_person = set()
        for i in list_transform.keys():
            for j in list_transform.keys():
                if i < j:
                    # compute distance between two people
                    distance = point_distance(list_transform[i], list_transform[j], distance_w, distance_h, w_cm, h_cm)
                    # check distance (2m == 200cm)
                    if distance < 200:
                        # append set index
                        violates_person.add(i)
                        violates_person.add(j)

        # draw bounding of person
        view_map = np.zeros((height_map, width_map, 3), dtype='uint8')
        cv2.putText(view_map, 'High risk: ' + str(len(violates_person)), (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(view_map, 'Total person: ' + str(len(list_bbox_body)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2, cv2.LINE_AA)

        for i in list_transform.keys():
            if i in violates_person:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            # convert pixel w, h about size view map
            x1, y1, x2, y2 = list_bbox_body[i]
            p_x, p_y = list_transform[i]
            p_x, p_y = int(p_x*width_map/w), int(p_y*height_map/h)
            # draw
            cv2.circle(view_map, (p_x, p_y), 5, color, 10)
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
        cv2.imshow('bird_eyes', view_map)
        cv2.waitKey(1)
        if flag_save is True:
            video_writer.write(frame)
            video_writer_map.write(view_map)

    cap.release()
    if flag_save is True:
        video_writer.release()
        video_writer_map.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Face On Video')
    parser.add_argument("-fn", "--file_name", help="video file name or rtsp", default='', type=str)
    parser.add_argument("-op", "--option", help="if save video then choice option = 1", default=False, type=bool)
    parser.add_argument("-o", "--output", help="path to output video file", default='face_recording.avi', type=str)
    parser.add_argument("-f", "--fps", default=20, help="FPS of output video", type=int)
    args = parser.parse_args()

    path_models = '/home/duyngu/Downloads/Do_An/model_training/Yolov5/best.pt'
    url = '/home/duyngu/Downloads/Do_An/video_cctv.mp4'
    source = args.file_name
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_pixel)
    # if run  as terminal, replace url = source
    detect_5k(url_video=url, path_model=path_models, flag_save=args.option, fps=args.fps, name_video=args.output)

