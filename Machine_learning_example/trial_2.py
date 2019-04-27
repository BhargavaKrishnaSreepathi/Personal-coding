

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import imageio


class_labels = ['text', 'floorplan', 'map', 'face', 'collage', 'property', 'siteplan']

train = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')    # reading the csv file
train.head()      # printing first five rows of the file

train['class_text'] = np.zeros(len(train))
train['class_floorplan'] = np.zeros(len(train))
train['class_map'] = np.zeros(len(train))
train['class_face'] = np.zeros(len(train))
train['class_collage'] = np.zeros(len(train))
train['class_property'] = np.zeros(len(train))
train['class_siteplan'] = np.zeros(len(train))

train_data_original = []
train_labels_original = []
IMG_SIZE = 299

for i in range(len(train)):
    if 'text' in train.loc[i]['labels']:
        train.loc[i, 'class_text'] = 1

    if 'floorplan' in train.loc[i]['labels']:
        train.loc[i, 'class_floorplan'] = 1

    if 'map' in train.loc[i]['labels']:
        train.loc[i, 'class_map'] = 1

    if 'face' in train.loc[i]['labels']:
        train.loc[i, 'class_face'] = 1

    if 'collage' in train.loc[i]['labels']:
        train.loc[i, 'class_collage'] = 1

    if 'property' in train.loc[i]['labels']:
        train.loc[i, 'class_property'] = 1

    if 'siteplan' in train.loc[i]['labels']:
        train.loc[i, 'class_siteplan'] = 1

    # img = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))
    # img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    img = imageio.imread(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))
    arr = np.array(img)

    if len(arr.shape) > 2:

        train_data_original.append(arr)
        train_labels_original.append([train.loc[i, 'class_text'], train.loc[i, 'class_floorplan'],
                                      train.loc[i, 'class_map'], train.loc[i, 'class_face'], train.loc[i, 'class_collage'],
                                      train.loc[i, 'class_property'], train.loc[i, 'class_face']])


train_images_T = np.array(train_data_original)
train_labels_T = np.array(train_labels_original)
print ('Data Processed')

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = IMG_size * IMG_size * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 6, so -> 7)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    ### END CODE HERE ###

    return X, Y


# GRADED FUNCTION: forward_propagation

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv_net(x, weights, biases):

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

training_iters = 200
learning_rate = 0.001
batch_size = 128

# MNIST data input (img shape: 28*28)
n_input = 299

# MNIST total classes (0-9 digits)
n_classes = 7
#both placeholders are of type float
x = tf.placeholder("float", [None, IMG_SIZE,IMG_SIZE,3])
y = tf.placeholder("float", [None, n_classes])
weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,3,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W3', shape=(11552*16,128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}

pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
train_X = train_images_T
train_y = train_labels_T

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: train_X,y : train_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()

# ## START CODE HERE ## (PUT YOUR IMAGE NAME)
# ## END CODE HERE ##
#
# img = imageio.imread(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + '3853906.jpg')
# arr = np.array(img)
# test_data_original = []
#
# if len(arr.shape) > 2:
#
#     test_data_original.append(img.flatten())
#
#
# test_images_T = np.array(test_data_original).T

# # We preprocess your image to fit your algorithm.
#
#
# my_image_prediction = predict(test_images_T, parameters)
#
# print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))