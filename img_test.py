#pylab inline
import os
import pylab
from pylab import imread
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
from PIL import Image

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)


# Directory to the .png data sets
root_dir_data = os.path.abspath('../Dehazing_Datasets')
data_dir_in = os.path.join(root_dir_data,'deepCorals', 'demosaiced')
data_dir_out = os.path.join(root_dir_data, 'deepCorals', 'proc')

# Directory to list of image names
list_dir = os.path.abspath('../neural_net/list')
train = pd.read_csv(os.path.join(list_dir, 'train', 'deepCorals.csv'))
test = pd.read_csv(os.path.join(list_dir, 'test', 'deepCorals.csv'))
train.head()

# check for existence of firectories
os.path.exists(root_dir_data)
os.path.exists(list_dir)


img_name = rng.choice(test.name)
filepath = os.path.join(data_dir_in, img_name)


# Reshape images into these desired dimensions.
height =1024
width =1360
depth = 3

img = imread(filepath)

print img.shape

img_reshaped = np.reshape(img, 1024*1360*3)
new = img.reshape(-1, 1024*1360*3)
reverse = new.reshape(1024,1360,3)

print train.name[0]
