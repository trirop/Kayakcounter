
"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import collections
import common
import cv2
import numpy as np
import os
from PIL import Image
import re
import csv
import time
import datetime
import tflite_runtime.interpreter as tflite
from gooey import Gooey

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

def main():
    default_model_dir = '../all_models'
    default_model = 'detect.tflite'
    default_labels = 'labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    CSV_FILENAME = 'result_video.csv'
    with open(CSV_FILENAME, 'a+', newline='') as f:
      fnames = ['Time', 'Class', 'Score']
      writer = csv.DictWriter(f, fieldnames=fnames)
      writer.writeheader()

      interpreter = common.make_interpreter(args.model)
      interpreter.allocate_tensors()
      labels = load_labels(args.labels)

      ######################################################
      #cap = cv2.VideoCapture(args.camera_idx)
      #cap = cv2.VideoCapture('../../workspace/training_demo/test_videos/20190808_Bootszählung_Isar.AVI')
      #cap = cv2.VideoCapture('../../workspace/training_demo/test_videos/20190808_Bootszählung_Isar.AVI')
      cap = cv2.VideoCapture('../../workspace/training_demo/test_videos/20190803_Bootszählung_Isar.AVI')
      ######################################################

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        common.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
        for obj in objs:
          now = datetime.datetime.today()
          writer.writerow({'Time': now.strftime('%Y-%m-%d_%H:%M:%S'), 'Class': labels.get(obj.id, obj.id), 'Score': obj.score})
          cv2_im = append_objs_to_img(cv2_im, objs, labels)
          cv2.imwrite(now.strftime('%Y-%m-%d_%H:%M:%S')+'.jpg', cv2_im)
        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    f.close()

def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()
