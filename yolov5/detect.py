
import torch
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.datasets import letterbox
import cv2


class Y5Detect:
    def __init__(self, weights):
        """
        :param weights: 'yolov5s.pt'
        """
        self.weights = weights
        self.model_image_size = 640
        self.conf_threshold = 0.4
        self.iou_threshold = 0.45
        self.model, self.device = self.load_model(use_cuda=False)

        stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(self.model_image_size, s=stride)
        self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def load_model(self, use_cuda=False):
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if use_cuda else 'cpu')

        model = attempt_load(self.weights, map_location=device)
        print('yolov5 running with {}'.format(device))
        return model, device

    def preprocess_image(self, image_rgb):
        # Padded resize
        img = letterbox(image_rgb.copy(), new_shape=self.image_size)[0]

        # Convert
        img = img.transpose(2, 0, 1)  # to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # 1 3 480 640
        return img

    def predict(self, image_rgb, show=False):
        image_rgb_shape = image_rgb.shape
        img = self.preprocess_image(image_rgb)
        pred = self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred,
                                   self.conf_threshold,
                                   self.iou_threshold,)
        bboxes = []
        labels = []
        scores = []
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_rgb_shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    with torch.no_grad():
                        x1 = xyxy[0].cpu().data.numpy()
                        y1 = xyxy[1].cpu().data.numpy()
                        x2 = xyxy[2].cpu().data.numpy()
                        y2 = xyxy[3].cpu().data.numpy()
                        #                        print('[INFO] bbox: ', x1, y1, x2, y2)
                        bboxes.append(list(map(int, [x1, y1, x2, y2])))
                        label = self.class_names[int(cls)]
                        #                        print('[INFO] label: ', label)
                        labels.append(label)
                        score = conf.cpu().data.numpy()
                        #                        print('[INFO] score: ', score)
                        scores.append(float(score))

        return bboxes, labels, scores


def draw_boxes(image, boxes, label, scores):

    if label == 'person' or label == 'with_mask':
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    xmin, ymin, xmax, ymax = boxes
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    # cv2.putText(image, label+"-{:.2f}".format(scores), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image, boxes


if __name__ == '__main__':
    y5_model = Y5Detect(weights="./weight/best_2.pt")
    img_path = "/home/thien/Downloads/4060.jpg"
    image_bgr = cv2.imread(img_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    bbox, label, score = y5_model.predict(image)
    for i in range(len(label)):
        image, _ = draw_boxes(image_bgr, bbox[i], label[i], scores=score[i])
    cv2.imshow('person', image_bgr)
    cv2.waitKey(0)
