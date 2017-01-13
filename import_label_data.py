import numpy as np
import os
import pandas as pd
from PIL import Image

##The following functions allow importing:
    ## all datasets with: import_all(scaleIn, scaleOut)
    ## single dataset with: import_single(scaleIn, scaleOut, dataName)
    ## multiple datasets with: import_multiple(scaleIn, scaleOut, dataNameList)

global root_dir_data, list_dir, DataNameList, NameListLength
# Directory to the .png data sets
root_dir_data = os.path.abspath('../Dehazing_Datasets')

# Directory to list of image names
list_dir = os.path.abspath('../neural_net/list')

# Check for existence of firectories
os.path.exists(root_dir_data)
os.path.exists(list_dir)

DataNameList = os.listdir(root_dir_data)
NameListLength = np.size(DataNameList)

#Assign labels to each dataset
def assign_label(a):
    if a=="Dataset Kelp":
        label = 0
    elif a=="Dataset Rocks":
        label = 1
    elif a=="Dataset Rocks-Sand":
        label = 2
    elif a=="shallowCorals":
        label = 3
    elif a=="mediumCorals":
        label = 4
    elif a=="deepCorals":
        label = 5
    return label

# Class for easy return of image matrices
class ReturnValue(object):
  def __init__(self, train_x, train_y, test_x, test_y, testImgs_x, testImgs_y):
     self.train_x = train_x
     self.train_y = train_y
     self.test_x = test_x
     self.test_y = test_y
     self.testImgs_x = testImgs_x
     self.testImgs_y = testImgs_y
print "Please wait. Importing image data..."

#Populating the vectors with all the data sets of images
def import_all(scaleIn, scaleOut):
    global root_dir_data, list_dir, DataNameList, NameListLength
    i = 0
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    testImgs_x = []
    testImgs_y = []
    while i < NameListLength:
        data_dir_in = os.path.join(root_dir_data, DataNameList[i], 'proc')
        train = pd.read_csv(os.path.join(list_dir, 'train', DataNameList[i] + "." + "csv"))
        test = pd.read_csv(os.path.join(list_dir, 'test', DataNameList[i] + "." + "csv"))

        for img_name in train.name:
            image_path = os.path.join(data_dir_in, img_name)
            img = Image.open(image_path)   # Open image as PIL image object
            rsize = img.resize((img.size[0]/scaleIn, img.size[1]/scaleIn)) # Use PIL to resize
            rsizeArr = np.asarray(rsize)  # Get array back
            train_x.append(rsizeArr)

        for img_name in train.name:
            label = assign_label(DataNameList[i])
            train_y.append(label)

        for img_name in test.name:
            image_path = os.path.join(data_dir_in, img_name)
	    testImgs_x.append(image_path) 	
            img = Image.open(image_path)   # Open image as PIL image object
            rsize = img.resize((img.size[0]/scaleIn,img.size[1]/scaleIn)) # Use PIL to resize
            rsizeArr = np.asarray(rsize)  # Get array back
            test_x.append(rsizeArr)

        for img_name in test.name:
            label = assign_label(DataNameList[i])
            testImgs_y.append(image_path) 
            test_y.append(label)

        print "{0} successfully imported!".format(DataNameList[i])
        i+=1
    train_x = np.stack(train_x)
    train_y = np.stack(train_y)
    test_x = np.stack(test_x)
    test_y = np.stack(test_y)
    testImgs_x = np.stack(testImgs_x)
    testImgs_y = np.stack(testImgs_y)
    return ReturnValue(train_x, train_y, test_x, test_y, testImgs_x, testImgs_y)

#Populating the vectors with only one dataset
def import_single(scaleIn, scaleOut, DataName):
	global root_dir_data, list_dir
	i = 0
	train_x = []
	train_y = []
	test_x = []
	test_y = []
	testImgs_x = []
	testImgs_y = []
	data_dir_in = os.path.join(root_dir_data, DataNameList, 'proc')
	train = pd.read_csv(os.path.join(list_dir, 'train', DataNameList + "." + "csv"))
	test = pd.read_csv(os.path.join(list_dir, 'test', DataNameList + "." + "csv"))

	for img_name in train.name:
	    image_path = os.path.join(data_dir_in, img_name)
	    img = Image.open(image_path)   # Open image as PIL image object
	    rsize = img.resize((img.size[0]/scaleIn, img.size[1]/scaleIn)) # Use PIL to resize
	    rsizeArr = np.asarray(rsize)  # Get array back
	    train_x.append(rsizeArr)

	for img_name in train.name:
	    label = assign_label(DataNameList[i])
	    train_y.append(label)

	for img_name in test.name:
	    image_path = os.path.join(data_dir_in, img_name)
	    testImgs_x.append(image_path) 	
	    img = Image.open(image_path)   # Open image as PIL image object
	    rsize = img.resize((img.size[0]/scaleIn,img.size[1]/scaleIn)) # Use PIL to resize
	    rsizeArr = np.asarray(rsize)  # Get array back
	    test_x.append(rsizeArr)

	for img_name in test.name:
	    label = assign_label(DataNameList[i])
	    testImgs_y.append(image_path) 
	    test_y.append(label)

	print "{0} successfully imported!".format(DataNameList[i])
	i+=1
	train_x = np.stack(train_x)
	train_y = np.stack(train_y)
	test_x = np.stack(test_x)
	test_y = np.stack(test_y)
	testImgs_x = np.stack(testImgs_x)
	testImgs_y = np.stack(testImgs_y)
	return ReturnValue(train_x, train_y, test_x, test_y, testImgs_x, testImgs_y)

#Populating the vectors with multiple sets of data
def import_multiple(scaleIn, scaleOut, DataNameList):
    global root_dir_data, list_dir
    i = 0
    NameListLength = np.size(DataNameList)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    testImgs_x = []
    testImgs_y = []
    while i < NameListLength:
        data_dir_in = os.path.join(root_dir_data, DataNameList[i], 'proc')
        train = pd.read_csv(os.path.join(list_dir, 'train', DataNameList[i] + "." + "csv"))
        test = pd.read_csv(os.path.join(list_dir, 'test', DataNameList[i] + "." + "csv"))
        for img_name in train.name:
            image_path = os.path.join(data_dir_in, img_name)
            img = Image.open(image_path)   # Open image as PIL image object
            rsize = img.resize((img.size[0]/scaleIn, img.size[1]/scaleIn)) # Use PIL to resize
            rsizeArr = np.asarray(rsize)  # Get array back
            train_x.append(rsizeArr)

        for img_name in train.name:
            label = assign_label(DataNameList[i])
            train_y.append(label)

        for img_name in test.name:
            image_path = os.path.join(data_dir_in, img_name)
	    testImgs_x.append(image_path) 	
            img = Image.open(image_path)   # Open image as PIL image object
            rsize = img.resize((img.size[0]/scaleIn,img.size[1]/scaleIn)) # Use PIL to resize
            rsizeArr = np.asarray(rsize)  # Get array back
            test_x.append(rsizeArr)

        for img_name in test.name:
            label = assign_label(DataNameList[i])
            testImgs_y.append(image_path) 
            test_y.append(label)

        print "{0} successfully imported!".format(DataNameList[i])
        i+=1
    train_x = np.stack(train_x)
    train_y = np.stack(train_y)
    test_x = np.stack(test_x)
    test_y = np.stack(test_y)
    testImgs_x = np.stack(testImgs_x)
    testImgs_y = np.stack(testImgs_y)
    return ReturnValue(train_x, train_y, test_x, test_y, testImgs_x, testImgs_y)
