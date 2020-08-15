# Kayakcounter
[comment]: https://towardsdatascience.com/how-to-easily-download-googles-open-images-dataset-for-your-ai-apps-db552a82fc6
[comment]: https://medium.com/@intprogrammer/how-to-scrape-google-for-images-to-train-your-machine-learning-classifiers-on-565076972ce
[comment]: https://medium.com/analytics-vidhya/how-to-set-up-tensorflow-gpu-on-ubuntu-18-04-lts-7a09ffd5f30f

## Preliminar words
In the past sportif paddlers and whitewater kayakers were threaten by legal limitations of the offical authorithies by imposing driving bans on certain river sections in Bavaria, Germany due to nature conservation grounds.
The sports organization BKV (Bayerischer Kanuverband) [BKV](https://www.kanu-bayern.de/) tried to argue with this detailed figures the authorities in Bavara, that the organized sportif paddlers and whitewater kayakers are not the root cause of the nature conservation problems.
This was the reason to the devlop this software "Kayacounter" to check the different types of boats on a river section and count them in a CSV file. The software itself was developed with the help of Python 3.7, Tensorflow 1.15, Opencv 3.5 and Matplotlib.

## Setting-up a working Python environment
Download and install Anaconda (Python 3.7 for Linux Ubuntu 18.04 in my case) from:
[Anaconda](https://www.anaconda.com/distribution/)

After this setup do this:  
Create a virtual environment for a special user, for instance user Kayakcounter
```
conda create -n Kayakcounter python=3.7
```
Then install the necessary Python software libraries

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
```
pip3 install -r requirements.txt 

### Installion of the needed software requirements and prerequisites 
For installing the needed software libraries please check
[Medium](https://choosealicense.com/licenses/mit/)

```
conda activate Kayakcounter
```
<p align="center">
  <img src="/docs/image_result.jpg" width="350" alt="accessibility text">
</p>

## Testing
[Scripts for testing](/scripts/testing/Readme.md)
## Usage

## Contributing

## License
[MIT](https://choosealicense.com/licenses/mit/)
[BKV](https://www.kanu-bayern.de/)


