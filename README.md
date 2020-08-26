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
conda install jupyter
conda install Pillow
conda install contextlib
conda install jupyter
```
pip3 install -r requirements.txt 

### Setup and installation of the needed software requirements and prerequisites 
For installing the needed software libraries please check
[Medium](https://choosealicense.com/licenses/mit/)

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

## Preparing the dataset for the Kayakcounter

## Training of the Kayakcounter
<p align="center">
  <img src="/docs/compile-workflow.png" width="350" alt="accessibility text">
</p>

## Testing
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


