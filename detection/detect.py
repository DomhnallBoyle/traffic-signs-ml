import argparse
import base64
import json

import config
import cv2
import numpy as np
import redis
from keras.models import load_model


REQUIRED_WIDTH, REQUIRED_HEIGHT = (416, 416)
LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench", "bird", "cat", "dog",  "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee",  "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
    "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]
ANCHORS = [
    [116, 90, 156, 198, 373, 326],
    [30, 61, 62, 45, 59, 119],
    [10, 13, 16, 30, 33, 23]
]

model = None


def read_image_from_file(image_path):
    image = cv2.imread(image_path, 1)
    height, width, channels = image.shape

    return image, (height, width, channels)


def show_image(image, detections=None):
    for bb in detections:
        y1, x1, y2, x2 = bb.ymin, bb.xmin, bb.ymax, bb.xmax
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('Image', image)
    cv2.waitKey(0)


def pre_process_image(image):
    image = cv2.resize(image, dsize=(REQUIRED_HEIGHT, REQUIRED_WIDTH),
                       interpolation=cv2.INTER_CUBIC)
    image = image.astype('float32')
    image /= 255.0
    image = np.expand_dims(image, 0)

    return image


class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = None
        self.score = None

    def scale(self, scale_width, scale_height):
        x_offset, x_scale = 0.0, 1.0
        y_offset, y_scale = 0.0, 1.0
        self.xmin = int((self.xmin - x_offset) / x_scale * scale_width)
        self.xmax = int((self.xmax - x_offset) / x_scale * scale_width)
        self.ymin = int((self.ymin - y_offset) / y_scale * scale_height)
        self.ymax = int((self.ymax - y_offset) / y_scale * scale_height)

        return self

    def to_json(self):
        return {
            'label': self.label,
            'x1': self.xmin,
            'y1': self.ymin,
            'x2': self.xmax,
            'y2': self.ymax
        }


def decode_predictions(predictions, obj_thresh=0.6):
    sigmoid = lambda x: 1. / (1. + np.exp(-x))

    boxes = []
    for p, anchor in zip(predictions, ANCHORS):
        p = p[0]

        grid_h, grid_w = p.shape[:2]
        nb_box = 3
        netout = p.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5

        netout[..., :2] = sigmoid(netout[..., :2])
        netout[..., 4:] = sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for i in range(grid_h * grid_w):
            row = i / grid_w
            col = i % grid_w
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[int(row)][int(col)][b][4]
                if objectness.all() <= obj_thresh:
                    continue

                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]
                x = (col + x) / grid_w  # center position, unit: image width
                y = (row + y) / grid_h  # center position, unit: image height
                w = anchor[2 * b + 0] * np.exp(w) / REQUIRED_WIDTH  # unit: image width
                h = anchor[2 * b + 1] * np.exp(h) / REQUIRED_HEIGHT  # unit: image height

                # last elements are class probabilities
                classes = netout[int(row)][col][b][5:]

                box = BoundingBox(xmin=x - w / 2,
                                  ymin=y - h / 2,
                                  xmax=x + w / 2,
                                  ymax=y + h / 2,
                                  objness=objectness,
                                  classes=classes)
                boxes.append(box)

    return boxes


def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bb_iou(box1, box2):
    intersect_w = interval_overlap([box1.xmin, box1.xmax],
                                   [box2.xmin, box2.xmax])
    intersect_h = interval_overlap([box1.ymin, box1.ymax],
                                   [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def nms(boxes, nms_thresh=0.5):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0:
                continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bb_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

    return boxes


def whittle_boxes(boxes, object_type=None, threshold=0.6):
    leftover_boxes = []

    for box in boxes:
        for i, label in enumerate(LABELS):
            if box.classes[i] > threshold:
                if object_type and object_type != label:
                    continue

                box.label = label
                box.score = box.classes[i] * 100
                leftover_boxes.append(box)

    return leftover_boxes


def detect(image, object_type=None):
    global model

    if object_type and object_type not in LABELS:
        object_type = None

    height, width, channels = image.shape

    pre_processed_image = pre_process_image(image)
    predictions = model.predict(pre_processed_image)
    bbs = decode_predictions(predictions)
    bbs = [bb.scale(width, height) for bb in bbs]
    bbs = nms(bbs)
    bbs = whittle_boxes(bbs, object_type=object_type)

    return bbs


def api_receiver():
    cache = redis.Redis(host=config.REDIS_HOST)

    while True:
        item = cache.lpop(config.REDIS_DETECT_QUEUE)
        if not item:
            continue

        item = json.loads(item)
        _id = item['id']
        image = item['image']
        height, width = item['height'], item['width']
        object_type = item['object_type']

        # decode back to image
        image = np.frombuffer(
            base64.decodebytes(bytes(image, encoding='utf-8')), dtype=np.uint8)
        image = np.reshape(image, (height, width, 3))

        bbs = detect(image, object_type)

        cache.set(_id, json.dumps([bb.to_json() for bb in bbs]))


def _load_model(model_path):
    global model
    model = load_model(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('from_api')
    parser_1.add_argument('model_path', type=str)

    parser_2 = sub_parsers.add_parser('from_file')
    parser_2.add_argument('model_path', type=str)
    parser_2.add_argument('image_path', type=str)
    parser_2.add_argument('--object_type', type=str, default=None)

    args = parser.parse_args()
    if args.run_type == 'from_api':
        _load_model(args.model_path)
        api_receiver()
    elif args.run_type == 'from_file':
        _load_model(args.model_path)
        image = read_image_from_file(args.image_path)[0]
        detections = detect(image, args.object_type)
        show_image(image, detections)
    else:
        parser.print_help()
