#pylab inline
import os
import pylab
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import random

###########
#PARAMETERS
###########

#Import functions that allow the importing of multiple datasets
import import_label_data as imp

#Neural network architecture parameters
epochs = 50 #number of times we will use all the images for processing
batch_size = 20 #number of images in each batch
learning_rate = .0001 #can be initialized inside AdamOptimizer(learning_rate=learning_rate)

#Receptive field size: every neuron in the Conv Layer would now have a total of conv1_n*conv2_n*depth
#connections to the input volume, with depth for a normal rgb image being 3
conv1_n = 3
conv2_n = 3
conv3_n = 3

#Number of features extracted from each square aka number of neurons per square
#Note the last convolution layer will be the reduction layer so output is reduction to 3
conv1_N = 15
conv2_N = 30

#Scale decrease as result of pooling
#ex. if we pool twice with 2x2 pooling then poolingParam = 4
#ex. if we do not pool then the pooling parameter = 1
poolingParam = 8

#Image dimension modifications
#Note afte we scale images, we must get an integer value. Otherwise we have to crop
scaleIn  = 4
scaleOut = 4
scaleOut = scaleIn*poolingParam

#Resulting Image dimensions after saling
wIn = 1360/scaleIn
hIn = 1024/scaleIn #no cropping
d   = 3

#These dimensions are formed after concolutional laters and pooling, 
#and will determine the size of the fully connected layer
wOut = 1360/scaleOut
hOut = 1024/scaleOut #no cropping

############
#IMPORT DATA
############

#Fill arrays with data with choices being:
	##List = ["Dataset Kelp", "Dataset Rocks"]  
	## data = imp.import_all(scaleIn, scaleOut, List)
	## DataName = ["Dataset Rocks"]
	## data = imp.import_single(scaleIn, scaleOut, DataName)
        ## data = imp.import_all(scaleIn, scaleOut)

List       = ["Dataset Kelp", "Dataset Rocks"]
data 	   = imp.import_multiple(scaleIn, scaleOut, List)
train_x    = data.train_x
train_y    = data.train_y
test_x     = data.test_x
test_y     = data.test_y
testImgs_x = data.testImgs_x
testImgs_y = data.testImgs_y

############################
#NEURAL NETWORK ARCHITECTURE
############################

#Indices of images rerandomized, each batch takes a set of these indices with each iteration
l = train_x.shape[0]
total_batch = random.sample(xrange(0,l),l)
total_batch = np.asarray(total_batch)

def dense_to_one_hot(labels_dense, num_classes=6):
    labels_dense = np.asarray(labels_dense)
    """Convert class labels from integers to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot 	

def preproc(unclean_batch_x):
    #Convert values to range 0-1
    temp_batch = unclean_batch_x / 255.0
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name, i, num_batches):

    print "Batch number: {0}/{1}".format(i+1, num_batches)
    print "Processing up to image number: {0}".format(batch_size*i+batch_size)

    indexStart = batch_size*i
    indexFinal = indexStart+batch_size
    batch_mask = total_batch[indexStart:indexFinal]
    batch_mask = batch_mask.astype(int)
    #print "Batch mask indices {0}".format(batch_mask)

    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, wIn*hIn*d)
    batch_x = preproc(batch_x)
	
    if dataset_name == 'train':
        batch_y = eval(dataset_name + '_y')[[batch_mask]]
	batch_y = dense_to_one_hot(batch_y)

    return batch_x, batch_y

print "Initializing neural network architecture..."
# Number of neurons in each layer
input_num_units  = wIn*hIn*d
output_scaled    = wOut*hOut*d
output_num_units = 6

# Define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

def conv2d(x, f):
    return tf.nn.conv2d(x, f, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    #initial = tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial)

x_image = tf.reshape(x, [-1, hIn, wIn, d])

W_conv1 = weight_variable([conv1_n, conv1_n, 3, conv1_N])
b_conv1 = bias_variable([conv1_N])

W_conv2 = weight_variable([conv2_n, conv2_n, conv1_N, conv2_N])
b_conv2 = bias_variable([conv2_N])

W_conv3 = weight_variable([conv3_n, conv3_n, conv2_N, 3])
b_conv3 = bias_variable([3])

W_fc1 = weight_variable([32*43*d, 32*43*d])
b_fc1 = bias_variable([32*43*d])

W_fc2 = weight_variable([32*43*d, output_num_units])
b_fc2 = bias_variable([output_num_units])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool0 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool0, W_conv2) + b_conv2)
h_pool1 = max_pool_2x2(h_conv2)

h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
h_pool2 = max_pool_2x2(h_conv3)

w_pool = h_pool2.get_shape()[1]
print w_pool
h_pool = h_pool2.get_shape()[2]
print h_pool

h_pool2_flat = tf.reshape(h_pool2, [-1, 32*43*d])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
output_layer = tf.matmul(h_fc1, W_fc2) + b_fc2
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
init = tf.initialize_all_variables()

saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
print "Neural netwrok successfully initialized!\n"

############
#RUN SESSION
############

print "\n***************************************************************************"
print "                         Recognizing Deep Water Images"
print "                     Neural Network Developed by: natacks"
print "***************************************************************************\n"
print "***************************************************************************"
print "{0} epochs with {1} images per batch".format(epochs, batch_size)
print "{0} train images with size {1}x{2}x{3} ".format( train_x.shape[0], train_x.shape[1], train_x.shape[2], train_x.shape[3])
print "{0} test image(s) with size {1}x{2}x{3}".format( test_x.shape[0], test_x.shape[1], test_x.shape[2], test_x.shape[3])
print "***************************************************************************\n"

with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    pre_cost = 0
    accuracy_vec = np.array([0])
    for epoch in range(epochs):
        avg_cost = 0
        num_batches = int(train_x.shape[0]/batch_size)
        print "========== Epoch {0} ==========".format(epoch+1)
        for i in range(num_batches):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train', i, num_batches)
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
	    avg_cost += c / num_batches
	pred_temp = tf.equal(tf.argmax(output_layer,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
	accuracy_disp = accuracy.eval({x: test_x.reshape(-1, wIn*hIn*d)/255.0, y: dense_to_one_hot(test_y)})
	accuracy_vec = np.vstack((accuracy_vec, accuracy_disp))
        print "===== Epoch Cost: {:.2f} ======".format(avg_cost)
        print "===== Delta Cost: {:.2f} ======".format(pre_cost - avg_cost)
	print "== Validation Accuracy: {:.2f} == \n".format(accuracy_disp)
	np.savetxt('Accuracy/recognize/accuracy_vec.csv',accuracy_vec, delimiter='\n',fmt='%1.3f')
    	saver.save(sess, 'Models/recognize/recognizeModel_1')
    print "\nTraining complete!"
    predict = output_layer
    pred = predict.eval({x: test_x.reshape(-1, wIn*hIn*d)/255.0})

################
#DISPLAY RESULTS
################

#Decode labels to each dataset
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

#Use this function if you want to visualize expected outputs
def output_vis(pred,test_y,image_path, scaleOut):
	wrong_set = []	
	wrong_set = np.asarray(wrong_set)
	num_wrong = 0	
	test_num = pred.shape[0]
	for i in range(test_num):
	    output_label = read_label(pred[i])
	    true_label = read_label(test_y[i])
	    if true_label != output_label:
	    	wrong_set = np.hstack((wrong_set, true_label))
		num_wrong = num_wrong + 1	  		    
	    print "Predicted label is: {0}, True label is: {1}".format(output_label, true_label)	
	print "Wrong labels: {0}".format(wrong_set) 
	print "Number wrong: {0}/{1}".format(num_wrong, i)   

output_vis(pred=pred,test_y=test_y,image_path=testImgs_y,scaleOut=scaleOut)
#pylab.show()



