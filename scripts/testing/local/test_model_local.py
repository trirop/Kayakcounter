######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
#import tkinter
import matplotlib
import time
#matplotlib.use('Qt5Agg')
#matplotlib.use('TkAgg')
#matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
#matplotlib.use('qt5agg')
import csv

#matplotlib.use('TKAgg',warn=False, force=True)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'trained_inference_graph'
IMAGE = 'image13'
IMAGE_NAME = IMAGE + '.jpg'
IMAGE_RESULT_NAME = IMAGE + '_result.jpg'

# Grab path to current working directory
#CWD_PATH = os.getcwd()
CWD_PATH = "/home/tristan/Kayakcounter/workspace/training_demo/"

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','label_map.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join('/home/tristan/Kayakcounter/workspace/training_demo/test_images',IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 3
LIMIT = 0.9

f = open('result_isar.csv', 'w')

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
print("Categories[0]=",categories[0])

category_index = label_map_util.create_category_index(categories)
print("category_index=", category_index)

print("1.Class = ",category_index.get(1).get('name'))
print("2.Class = ",category_index.get(2).get('name'))
print("3.Class = ",category_index.get(3).get('name'))

#print [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#print("Number of detections =", num_detections)

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

#print("Number of boxes = ", np.squeeze(boxes))
#print("Number of scores = ", np.squeeze(scores))
#print("Number of classes = ", np.squeeze(classes))
#print("Number of detections = ", num)

final_score = np.squeeze(scores)    
final_classes = np.squeeze(classes)
count = 0
for i in range(100):
	if scores is None or final_score[i] > LIMIT:
		print("I = ", i)
		print("Class = ",final_classes[i])
		print("Score = ",final_score[i])
		count = count + 1

print("Found objects = ", count)
if count > 0:
	localtime = time.localtime()
	result = time.strftime("%I:%M:%S %p", localtime)
	with f:
		fnames = ['Time', 'Class', 'Score']
		writer = csv.DictWriter(f, fieldnames=fnames)
		writer.writeheader()
		j=0
		while j < count:
			writer.writerow({'Time' : result, 
					'Class': category_index.get(final_classes[j]).get('name'), 
					'Score' : final_score[j] })
			j=j+1

# Draw the results of the detection (aka 'visulaize the results')

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=LIMIT)

# All the results have been drawn on image. Now display the image.
cv2.imwrite('image_result.jpg', image)
time.sleep(3)
cv2.imshow('Object detector', image)
time.sleep(3)

#matplotlib.use('TkAgg')
#plt.figure()
#plt.imshow(image)
#plt.show()


# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
