# autopep8 -i test_model_video_local.py
# idle
import numpy as np
import cv2
import os
import tensorflow as tf
import sys
import matplotlib
import time
import datetime
import csv

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
LOG_PATH = 'video_results/'

MODEL_NAME = 'trained_inference_graph'
CWD_PATH = "../../../workspace/training_demo/"
NUM_CLASSES = 3
LIMIT = 0.85

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'label_map.pbtxt')

# Path to image
VIDEO_NAME = '20190803_Bootszählung_Isar.AVI'
PATH_TO_VIDEO = os.path.join(
    '../../../workspace/training_demo/test_videos/', VIDEO_NAME)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

CSV_FILENAME = 'video_results/result_video.csv'
with open(CSV_FILENAME, 'a+', newline='') as f:
    fnames = ['Time', 'Class', 'Score']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

    cap = cv2.VideoCapture(PATH_TO_VIDEO)

    while(cap.isOpened()):
        ret, image = cap.read()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        final_score = np.squeeze(scores)
        final_classes = np.squeeze(classes)
        count = 0

        for i in range(100):
            if scores is None or final_score[i] > LIMIT:
                print("Number = ", i)
                print("Class = ", final_classes[i])
                print("Score = ", final_score[i])
                count = count + 1

                if count > 0:
                    # Bild abspeichern
                    now = datetime.datetime.today()
                    filename = LOG_PATH + \
                        now.strftime('%Y-%m-%d_%H:%M:%S') + "_orig_image.jpg"
                    print("Current time=", now.strftime('%Y-%m-%d_%H:%M;%S'))
                    print("Origiginal Filename=", filename)

                    j = 0
                    while j < count:
                        rowcontent = [result, category_index.get(
                            final_classes[j]).get('name'), final_score[j]]

                        print("Rowcontent = ", rowcontent)

                        writer.writerow({'Time': result, 'Class': category_index.get(
                            final_classes[j]).get('name'), 'Score': final_score[j]})

                        j = j+1
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=LIMIT)
                cv2.imwrite(filename, image)

            cv2.imshow('frame', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
f.close()
