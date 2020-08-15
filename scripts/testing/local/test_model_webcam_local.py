# autopep8 -i test_model_webcam_local.py
# idle
import tensorflow as tf
import cv2
import time
import numpy as np
import os
import sys
import matplotlib
import csv

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'trained_inference_graph'
CWD_PATH = "/home/tristan/Kayakcounter/workspace/training_demo/"
NUM_CLASSES = 3
LIMIT = 0.5

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'label_map.pbtxt')

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


capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)
img_counter = 0
frame_set = []
start_time = time.time()

with open('result_webcam.csv', 'w', newline='') as f:
    fnames = ['Time', 'Class', 'Score']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

    while True:
        ret, image = capture.read()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start_time >= 5:  # <---- Check if 5 sec passed
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})

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
                        localtime = time.localtime()
                        result = time.strftime("%I:%M:%S %p", localtime)
                        filename = result + "orig_image.jpg"
                        print("Akuelle Zeit=", result)
                        print("Orig Filename=", filename)
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

            #img_name = "opencv_frame_{}.png".format(img_counter)
            #cv2.imwrite(img_name, image)
            #print("{} written!".format(img_counter))
            start_time = time.time()
            img_counter += 1
f.close()
