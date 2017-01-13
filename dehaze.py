#pylab inline
import os
import pylab
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)


root_dir = os.path.abspath('../neural_net')
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)



train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'Test', 'test.csv'))
train.head()



#sample_submission = pd.read_csv(os.path.join(data_dir, 'Sample_Submission.csv'))
'''
#pick random image from the input file
img_name = rng.choice(train.file_in)
#find image path
filepath = os.path.join(data_dir, 'Train', img_name)

#display the image
img = imread(filepath, flatten=True)
pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()
'''

temp = []
for img_name in train.file_in:
    image_path = os.path.join(data_dir, 'Train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

train_x = np.stack(temp)

temp = []
for img_name in train.file_out:
    image_path = os.path.join(data_dir, 'Train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

train_y = np.stack(temp)

temp = []
for img_name in test.file_in:
    image_path = os.path.join(data_dir, 'Test', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

test_x = np.stack(temp)

temp = []
for img_name in test.file_out:
    image_path = os.path.join(data_dir, 'Test', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

test_y = np.stack(temp)


split_size = int(train_x.shape[0]*0.7)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    print ("Creating a batch")
    batch_mask = rng.choice(dataset_length, batch_size)

    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, 1392640)
    batch_x = preproc(batch_x)

    if dataset_name == 'train':
        batch_y = eval(dataset_name + '_y')[[batch_mask]].reshape(-1, 1392640)
        batch_y = preproc(batch_y)

    return batch_x, batch_y

#set all variables

# number of neurons in each layer
input_num_units = 1360*1024
hidden_num_units = 50
output_num_units = 1360*1024


# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 3
batch_size = 1
learning_rate = 0.01

#define weights and biases of the neural network (refer this article if you don't understand the terminologies)

weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}



hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()


with tf.Session() as sess:
    # create initialized variables
    sess.run(init)

    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize

    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})

            avg_cost += c / total_batch

        print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)

    print "\nTraining complete!"


    # find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    pred_temp = output_layer
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print "Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, 1392640), y: val_y.reshape(-1, 1392640)})

    predict = tf.argmax(output_layer, 1)
    pred = predict.eval({x: test_x.reshape(-1, 1392640)})


img_name = rng.choice(test.file_in)
filepath = os.path.join(data_dir, 'Test', img_name)

img = imread(filepath, flatten=True)
#test_index = int(img_name.split('.')[0]) - 49000
test_index = 0
print "Prediction is: ", pred[test_index]

#pylab.imshow(img, cmap='gray')
pylab.imshow(img)
pylab.axis('off')
pylab.show()
