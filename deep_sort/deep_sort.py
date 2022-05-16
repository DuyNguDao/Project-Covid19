import numpy as np

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, use_cuda=True):
        self.min_confidence = 0.3
        self.nms_max_overlap = 1.0

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def update(self, bbox_xyxy, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xyxy, ori_img)
        bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences)
                      if conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _xyxy_to_tlwh(bbox_xyxy):
        bbox_xyxy[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_xyxy[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_xyxy

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x-w/2), 0)
        x2 = min(int(x+w/2), self.width-1)
        y1 = max(int(y-h/2), 0)
        y2 = min(int(y+h/2), self.height-1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width-1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height-1)
        return x1, y1, x2, y2
    
    def _get_features(self, bbox_xyxy, ori_img):
        im_crops = []
        for box in bbox_xyxy:
            x1, y1, x2, y2 = box
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
