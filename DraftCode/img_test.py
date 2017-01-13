import import_label_data as imp
import numpy as np

"""
scaleIn = 16
scaleOut = 16

data = imp.import_all(scaleIn, scaleOut)
print data.train_x.shape
print data.train_y[0:100]




def dense_to_one_hot(a, num_classes=6):
    labels_dense = np.asarray(a)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
 
a = [4, 3, 5, 2, 5, 2 ,4, 4, 5]

b = dense_to_one_hot(a, num_classes=6)

"""


def read_label(element):
    size = element.size
    if size > 1:
    	a = np.argmax(element)
    else: 
	a = element	
    if a==0:
        label= "Dataset Kelp"
    elif a==1:
        label = "Dataset Rocks"
    elif a==2:
        label = "Dataset Rocks-Sand"
    elif a==3:
        label = "shallowCorals"
    elif a==4:
        label = "mediumCorals"
    elif a==5:
        label = "deepCorals"
    return label


c = [10,2,3,4,5,6]
d = [0]
c = np.asarray(c)
d = np.asarray(d)

lab1 = read_label(c)
lab2 = read_label(d)
print lab1,lab2
