#pylab inline
import os
import pylab
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

#Import functions that allow the importing of multiple datasets
import import_data as imp

#Neural network architecture parameters
epochs = 1000 #number of times we will use all the images for processing
batch_size = 30 #number of images in each batch
learning_rate = .0001 #can be initialized inside AdamOptimizer(learning_rate=learning_rate)

#Scale decrease as result of pooling
#ex. if we pool twice with 2x2 pooling then poolingParam = 4
#ex. if we do not pool then the pooling parameter = 1
poolingParam = 1

#Receptive field size: every neuron in the Conv Layer would now have a total of conv1_n*conv2_n*depth 
#connections to the input volume, with depth for a normal rgb image being 3
conv1_n = 3
conv2_n = 3
conv3_n = 3
conv4_n = 3
conv5_n = 3
conv6_n = 3

#Number of features extracted from each square aka number of neurons per square
#Note the last convolution layer will be the reduction layer so output is reduction to 3
conv1_N = 15
conv2_N = 20
conv3_N = 25
conv4_N = 20
conv5_N = 15

#Image dimension modifications
#Note afte we scale images, we must get an integer value. Otherwise we have to crop
scaleIn = 4
scaleOut = 4
scaleOut = scaleIn*poolingParam

wIn = 1360/scaleIn
hIn = 1024/scaleIn #no cropping
d = 3

wOut = 1360/scaleOut
hOut = 1024/scaleOut #no cropping

#Fill arrays with data
List = ["Dataset Rocks", "Dataset Kelp"]
data = imp.import_multiple(scaleIn, scaleOut, List)
train_x = data.train_x
train_y = data.train_y
test_x = data.test_x
test_y = data.test_y
testImgs_x = data.testImgs_x
testImgs_y = data.testImgs_y

def preproc(unclean_batch_x):
    #Convert values to range 0-1
    temp_batch = unclean_batch_x / 255.0
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name, i, total_batch):

    print "Batch number: {0}/{1}".format(i+1, total_batch)
    print "Processing up to image number: {0}".format(batch_size*i+batch_size)

    batch_mask = np.zeros(shape=(1,batch_size))
    index = batch_size*i
    indexFinal = batch_size*i+batch_size

    loc = 0;
    while index < indexFinal:
        batch_mask[0,loc] = index
        index = index+1
        loc=loc+1
    batch_mask = batch_mask.astype(int)

    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, wIn*hIn*d)
    batch_x = preproc(batch_x)

    if dataset_name == 'train':
        batch_y = eval(dataset_name + '_y')[[batch_mask]].reshape(-1, wOut*hOut*d)
        batch_y = preproc(batch_y)

    return batch_x, batch_y

print "Initializing neural network architecture..."
# Number of neurons in each layer
input_num_units = wIn*hIn*d
output_num_units = wOut*hOut*d

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
    return tf.Variable(initial)

x_image = tf.reshape(x, [-1, hIn, wIn, d])

W_conv1 = weight_variable([conv1_n, conv1_n, 3, conv1_N])
b_conv1 = bias_variable([conv1_N])

W_conv2 = weight_variable([conv2_n, conv2_n, conv1_N, conv2_N])
b_conv2 = bias_variable([conv2_N])

W_conv3 = weight_variable([conv3_n, conv3_n, conv2_N, conv3_N])
b_conv3 = bias_variable([conv3_N])

W_conv4 = weight_variable([conv4_n, conv4_n, conv3_N, conv4_N])
b_conv4 = bias_variable([conv4_N])

W_conv5 = weight_variable([conv5_n, conv5_n, conv4_N, conv5_N])
b_conv5 = bias_variable([conv5_N])

W_conv6 = weight_variable([conv6_n, conv6_n, conv5_N, 3])
b_conv6 = bias_variable([3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
h_conv6 = (conv2d(h_conv5, W_conv6) + b_conv6)

output_layer = tf.reshape(h_conv6, [-1,wOut*hOut*d])
cost = tf.nn.l2_loss(tf.sub(output_layer,y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
init = tf.initialize_all_variables()
saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
print "Neural netwrok successfully initialized!\n"

print "\n***************************************************************************"
print "                         Dehazing Neural Network"
print "                          Developed by: natacks"
print "***************************************************************************\n"
print "***************************************************************************"
print "{0} epochs with {1} images per batch".format(epochs, batch_size)
print "{0} train images with size {1}x{2}x{3} ".format( train_x.shape[0], train_x.shape[1], train_x.shape[2], train_x.shape[3])
print "{0} test image(s) with size {1}x{2}x{3}".format( test_x.shape[0], test_x.shape[1], test_x.shape[2], test_x.shape[3])
print "***************************************************************************\n"

#visualize the image after every single epoch
def showPred(pred,h,w,test_index):
	out_arr = pred[test_index]
        img_dehazed = out_arr.reshape(hIn,wIn,3)
	pylab.axis('off')
	pylab.imshow(img_dehazed)
	pylab.savefig("img_out_2.png")

with tf.Session() as sess:
    # create initialized variables
    sess.run(init)

    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    pre_cost = 0
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train_x.shape[0]/batch_size)
        print "========== Epoch {0} ==========".format(epoch+1)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train', i, total_batch)
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
	    avg_cost += c / total_batch
	
        print "===== Epoch Cost: {:.2f} ======".format(avg_cost)
        print "===== Delta Cost: {:.2f} ======\n".format(pre_cost - avg_cost)
        predict = output_layer
        pred = predict.eval({x: test_x.reshape(-1, wIn*hIn*d)/255.0})
        showPred(pred,hOut,wOut,test_index=0)
	pre_cost = avg_cost
    print "\nTraining complete!"
    predict = output_layer
    saver.save(sess, '/Models/dehaze/dehaze_model')	

global figure_num
figure_num = 1

#Use this function if you want to visualize expected outputs
def output_vis(title, image_path, testIndex, scaleOut):
	global figure_num
	for i in range(testIndex):
		fig = pylab.figure(figureNum)
		fig.suptitle(title)
		img = Image.open(image_path)   # Open image as PIL image object
		rsize = img.resize((img.size[0]/scaleOut,img.size[1]/scaleOut)) # Use PIL to resize
		rsizeArr = np.asarray(rsize)  # Get array back
		pylab.imshow(rsizeArr)
		pylab.axis('off')
		figureNum=figureNum+1

#Use this function to visualize predicted output
def predict_vis(figureNum, pred, h, w, d):
	global figure_num
	test_num = pred.shape[0]
	for i in range(test_num):		
		fig = pylab.figure(figureNum)
		fig.suptitle("Neural Network Predicted Output")	
		out_arr = pred[i]
		img_dehazed = out_arr.reshape(h,w,d)
		pylab.imshow(img_dehazed)
		pylab.axis('off')
		figureNum=figureNum+1

predict_vis(figureNum=1, pred=pred, h=hOut, w=wOut, d=d)
output_vis("Underwater Image", testImgs_x[0], testIndex=0, scaleOut=scaleOut)
output_vis("Expected Output", testImgs_y[0], testIndex=0, scaleOut=scaleOut)

pylab.show()

