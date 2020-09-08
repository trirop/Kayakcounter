import cv2 as o
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import PIL
import Cython
from platform import python_version
import lxml
from pycocotools import mask as mask

import sys
print(sys.executable)
print(sys.version)
print(sys.version_info)

print("Version of OpenCV =",o.__version__)
print("Version of Tensorflow =",tf.__version__)
print("Version of Numpy =",np.__version__)
print("Version of Matplotlib =",mpl.__version__)
print("Version of Python =",python_version())
print("Version of PIL =",PIL.__version__)
print("Version of Cython =",Cython.__version__)
print("Version of XML =", lxml.__version__)
print("Name of the author of pycocotools=",mask.__author__)

