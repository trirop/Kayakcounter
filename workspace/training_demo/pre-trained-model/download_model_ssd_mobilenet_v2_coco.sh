# Download SSD MobileNet V2 model
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
curl -O http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz -C ../../../workspace/training_demo/pre-trained-model/
rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz
mv ssd_mobilenet_v2_coco_2018_03_29 ssd_mobilenet_v2_coco
