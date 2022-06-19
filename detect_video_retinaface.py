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
from retinaface import FaceDetection
import time
from classification.utils.load_model import Model
from yolov5.detect import *
from pathlib import Path


# ******************************** ROOT PATH *****************************
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT / 'weight'


def draw_bbox(image, bboxs):
    """
    function: Draw bounding box of face
    :param image: image or frame
    :param bboxs: list bounding box and class
    :return: image
    """
    for label, bbox in bboxs:
        x1, y1, x2, y2, score = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(image, str(round(score, 2)), (x1, y1+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, label, (x1, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2, cv2.LINE_AA)
    return image


def detect_facemask(url_video, path_model, flag_save=False, fps=None, name_video='video.avi'):
    """
    function detect face mask
    :param url_video: url of video
    :param path_model: path model detect yolov5
    :param flag_save: flag save video
    :param fps: fps of video
    :param name_video: name of video
    :return: None
    """
    # load model face mask classification
    model = Model(path_model)

    # load model face detect
    detector = FaceDetection(net='mobilenet').detect_faces

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
    # set thresh bounding box
    thresh = 0
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
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if h > 1080 and w > 1920:
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            h, w, _ = frame.shape
        # detect face and classification face mask
        bboxs, landmark = detector(frame)
        list_predict = []
        count_without_mask = 0
        for index, bbox in enumerate(bboxs):
            x1, y1, x2, y2 = [round(a) for a in bbox[0:4]]
            x1, y1, x2, y2 = max(x1 - thresh // 2, 0), max(y1 + thresh, 0), min(x2 + thresh // 2, w - 1), min(
                y2 - thresh, h - 1)
            pred = model.predict(frame[y1:y2, x1:x2])
            if pred[0] == 'without_mask':
                count_without_mask += 1
            list_predict.append((pred[0] + ': %d%%' % (round(max(pred[1]))), (x1, y1, x2, y2, bbox[4])))

        # draw classification face mask and fps
        frame = draw_bbox(frame, list_predict)
        fps = int(1 / (time.time() - start))
        cv2.putText(frame, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    2, cv2.LINE_AA)
        cv2.putText(frame, 'Without mask: ' + str(count_without_mask), (0, 40), cv2.FONT_HERSHEY_SIMPLEX,
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
    parser.add_argument("-op", "--option", help="if save video then choice option = True", default=False, type=bool)
    parser.add_argument("-o", "--output", help="path to output video file", default='face_recording.avi', type=str)
    parser.add_argument("-f", "--fps", default=20, help="FPS of output video", type=int)

    args = parser.parse_args()
    path_models = WEIGHTS / 'result_mobilenetv2'
    source = '/home/duyngu/Downloads/Do_An/video_test/TownCentre.mp4' # args.file_name
    detect_facemask(url_video=source, path_model=path_models, flag_save=args.option, fps=args.fps, name_video=args.output)

