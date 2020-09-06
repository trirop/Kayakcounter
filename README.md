# Kayakcounter
[comment]: https://towardsdatascience.com/how-to-easily-download-googles-open-images-dataset-for-your-ai-apps-db552a82fc6
[comment]: https://medium.com/@intprogrammer/how-to-scrape-google-for-images-to-train-your-machine-learning-classifiers-on-565076972ce
[comment]: https://medium.com/analytics-vidhya/how-to-set-up-tensorflow-gpu-on-ubuntu-18-04-lts-7a09ffd5f30f

## Preliminar words
In the past sportif paddlers and whitewater kayakers were threaten by legal limitations of the offical authorithies by imposing driving bans on certain river sections in Bavaria, Germany, due to nature conservation grounds.  
The sports organization [BKV](https://www.kanu-bayern.de/) (Bayerischer Kanu Verband) and [CMK](https://www.cmk-muenchen.de) (Club Münchener Kayakfahrer e.V.) tried to argue with these detailed figures to the authorities in Bavaria, that the organized sportif paddlers and whitewater kayakers are not the root cause of the nature conservation problems.
This was the reason to the develop this software "Kayacounter" to check the different types of boats on a river section and count the results in a CSV file. 

The software itself was developed with the help of Python 3.7, Tensorflow 1.15, Opencv 3.5 and Matplotlib. 

The neuronal net of the Kayakcounter can distiguish three types of boates: 
1. Kayak
2. Rubber boat
3. Standup paddler

## Setting-up a working Python environment
Download and install Anaconda (Python 3.7 for Linux Ubuntu 18.04 in my case) from:
[Anaconda](https://www.anaconda.com/distribution/)

After this setup do this:  
Create a virtual environment for a special user, for instance user Kayakcounter
```
conda create -n Kayakcounter python=3.7
```
Then activate this new conda environment with:
```
conda activate Kayakcounter
```
Install the necessary Python software libraries

```
conda install tensorflow==1.15
conda install Pillow
conda install numpy
conda install mathplotlib
conda install Cython
conda install libxml
conda install pycocotools
conda install Pillow
conda install contextlib
conda install jupyter
```
```
pip3 install -r requirements.txt
```

### Setup and installation of the needed software requirements and prerequisites 
For installing the needed software libraries please check
[Medium](https://choosealicense.com/licenses/mit/) 
```
git clone https://github.com/tensorflow/models.git
```

```
protoc object_detection/protos/*.proto --python_out=.
os.environ['PYTHONPATH'] += ':/content/gdrive/My Drive/Kayakcounter/models/research/:/content/gdrive/My Drive/Kayakcounter/models/research/slim'
```

```
!python model_builder_test.py
```

```
python object_detection/builders/model_builder_test.py 
```

```
Directory Structure for Training input data
Kayakcounter
│   README.md
│      
│
└───workspace
│   └───training_demo
│        │
│        └───annotations
│        │      │   train.record
│        │      │   test.record
│        │      
│        └───images
│        │        └─all
                  └─train
                  └─test
│        └───models
│		 │	      └─(all Tensorflow models 
│        
│        └───pre-trained-model
│		 │		│   ssd_mobilenet_v2_quantized_300x300
│        └───test_images
│		 │		│   (some test images)
│        └───test_videos
│		 │		│   (some test videos)
│        └───tflite
│		 │		│   (generated Tensorflow model in TFLite and EdgeTPU format)
│		 └───trained_inference_graph
│		 │		│   (generated frozen graph model)
│        └───training
│
│   
└───scripts
    │   README.md
	└───dataset_creating
			│   file111.txt
	└───exporting_model
			│   file111.txt
	└───preprocessing
			│   file111.txt

		


```


Output of the Kayacounter trained inference graph (frozen.pb):
<p align="center">
  <img src="/docs/image_result.jpg" width="350" alt="accessibility text">
</p>

<p align="center">
  <img src="/docs/Rubberboat.jpg" width="350" alt="accessibility text">
</p>

<p align="center">
  <img src="/docs/Standuppaddler.jpg" width="350" alt="accessibility text">
</p>

<p align="center">
  <img src="/docs/Kayak.png" width="350" alt="accessibility text">
</p>

Output of the Kayaccounter tflite model:
<p align="center">
  <img src="/docs/Kayakcounter_tflite.png" width="350" alt="accessibility text">
</p>

## Preparing the dataset for the Kayakcounter

1. Download all images into the folder workspace/training_demo/images/all 
2. Resize all images to the same size of width and height (image-resize.sh) 
3. Change the name of all images to enumerated numbers
4. Annotate all images in the all folder
5. Partition the images in train und test data (partition_datateser.sh)
6. Create the csv_files (xml_to_csv_test.sh und xml_to_csv_train.sh).
7. Create the tfrecords files with create_test_tfrecord.sh and create_train_tfrecord.sh  

## Downloading a adaquate Tensorflow model for retraining the model
I have this Tensorflow model and downloaded it from the GitHub Tensorflow model zoo. Extract this zipped Tar file under workspace/training_demo/pre-trained-model folder.

```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
```



## Training the model

In order to train the Tensorflow model, run these scripts inside the training folder of the scripts folder.

1. train_model_ssd_mobilenet_v2_quantized_300x300.s

## Exporting the model to TFLite format

<p align="center">
  <img src="/docs/compile-workflow.png" width="350" alt="accessibility text">
</p>

1. create_frozen_graph_model_ssd_mobilenet_v2_quantized_300x300.sh
2. show_output_tensors.sh
3. export_tflite_model_ssd_mobilenet_v2_quantized_300x300.sh
4. convert_tflite_model_ssd_mobilenet_v2_quantized_300x300.sh
5. generate_edgetpu_tflite_model_ssd_mobilenet_v2_quantized_300x300



## Testing
There are several tesings scripts for checking the Tensorflow model in action.
Testing with the frozen infernce graph of Tensorflow:
1. test_model_image_local.sh
2. test_model_dir_local.sh
3. test_model_video_local.sh
4.test_model_webcam.sh

Testing with the TFLite format of Tensorflow:
1. test_model_tflite_image.sh

The result of the test procedures can you see in the diffrenet test folders.
1. image_results
2. dir_results
3. video_results
4. webcam_results

The result is wrtten down in a CSV file and contains such information:


| Time                | Class      | Score     |
|---------------------|------------|-----------|
|  2020-08-15_15:03:53| Kayak      |0.94796604 |
|  2020-08-15_17:09:35| Rubberboat |0.94523983 |
|  2020-08-15_17:09:37| Rubberboat |0.94400776 |



[Scripts for testing](/scripts/testing/Readme.md)

## Usage

## Contributing und Downloading links
[Images](https://drive.google.com/file/d/1zuxL50qTbJ0nLWU1Cfw0xsFmCli18ZQV/view?usp=sharing)  
[Inference Graph](https://drive.google.com/file/d/1Ck2XczZKoKdqOwSpJ0Le021tEmZanATq/view?usp=sharing)  
[Tflite sources](https://drive.google.com/file/d/1CmEBkx7-_xB0xn-ogYyWlho1nd735wmI/view?usp=sharing)  
 [TFLite file](https://drive.google.com/file/d/1SlUIGP3VpfPqffZj16xFp8Geq51kigWL/view?usp=sharing)  
[TFLite EdgeTPU file](https://drive.google.com/file/d/1-OyrvgU1-wiB69nfdOTbTzkhxGragJZO/view?usp=sharing)  
[Introduction to TFLite on Raspberry Pi and Android](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md)
## License
[MIT](https://choosealicense.com/licenses/mit/)
[BKV](https://www.kanu-bayern.de/)
[Isar Videos](https://www.dropbox.com/sh/p8ex39se92zxzic/AACZPNh_7BOBeSNWbmUtFpzga?dl=0)


