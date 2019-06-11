
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import numpy as np
from PIL import Image
import tensorflow as tf
import math
import numpy as np
import h5py
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

train_data = []
train_labels = []
IMG_SIZE = 299

for i in range(len(train)):
    if 'text' in train.loc[i]['labels']:
        train.loc[i, 'class_text'] = 1

    if 'floorplan' in train.loc[i]['labels']:
        train.loc[i, 'class_floorplan'] = 2

    if 'map' in train.loc[i]['labels']:
        train.loc[i, 'class_map'] = 3

    if 'face' in train.loc[i]['labels']:
        train.loc[i, 'class_face'] = 4

    if 'collage' in train.loc[i]['labels']:
        train.loc[i, 'class_collage'] = 5

    if 'property' in train.loc[i]['labels']:
        train.loc[i, 'class_property'] = 6

    if 'siteplan' in train.loc[i]['labels']:
        train.loc[i, 'class_siteplan'] = 7

    # img = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))
    # img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    img = imageio.imread(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))

    # arr = np.array(img).reshape(-1,1)
    train_data.append(img.flatten())
    train_labels.append([train.loc[i, 'class_text'], train.loc[i, 'class_floorplan'],
                       train.loc[i, 'class_map'], train.loc[i, 'class_face'], train.loc[i, 'class_collage'],
                       train.loc[i, 'class_property'], train.loc[i, 'class_face']])


x_data = np.array([train_data])
pixels = x_data.flatten().reshape(10000, IMG_SIZE*IMG_SIZE*3)

x_data = np.array([np.array(Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id'])) for i in range(len(train)))] )

pixels = x_data.flatten().reshape(10000, IMG_SIZE*IMG_SIZE*3)

trainImages = np.array([i[0] for i in train_data]).reshape(10000, IMG_SIZE*IMG_SIZE*3)
trainLabels = np.array([i[0] for i in train_labels])




model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(train_images_arr, train_labels_arr, batch_size = 50, epochs = 5, verbose = 1)

model.evaluate(trainImages, trainImages, verbose = 0)


# # GRADED FUNCTION: create_placeholders
#
# def create_placeholders(n_x, n_y):
#     """
#     Creates the placeholders for the tensorflow session.
#
#     Arguments:
#     n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
#     n_y -- scalar, number of classes (from 0 to 5, so -> 6)
#
#     Returns:
#     X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
#     Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
#
#     Tips:
#     - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
#       In fact, the number of examples during test/train is different.
#     """
#
#     ### START CODE HERE ### (approx. 2 lines)
#     X = tf.placeholder(tf.float32, [n_x, None], name="X")
#     Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
#     ### END CODE HERE ###
#
#     return X, Y
#
#
# def initialize_parameters():
#     """
#     Initializes parameters to build a neural network with tensorflow. The shapes are:
#                         W1 : [25, 12288]
#                         b1 : [25, 1]
#                         W2 : [12, 25]
#                         b2 : [12, 1]
#                         W3 : [6, 12]
#                         b3 : [6, 1]
#
#     Returns:
#     parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
#     """
#
#     tf.set_random_seed(1)  # so that your "random" numbers match ours
#
#     ### START CODE HERE ### (approx. 6 lines of code)
#     W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
#     b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
#     W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
#     b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
#     W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
#     b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
#     ### END CODE HERE ###
#
#     parameters = {"W1": W1,
#                   "b1": b1,
#                   "W2": W2,
#                   "b2": b2,
#                   "W3": W3,
#                   "b3": b3}
#
#     return parameters
# ### END CODE HERE ###
#
# def forward_propagation(X, parameters):
#     """
#     Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
#
#     Arguments:
#     X -- input dataset placeholder, of shape (input size, number of examples)
#     parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
#                   the shapes are given in initialize_parameters
#
#     Returns:
#     Z3 -- the output of the last LINEAR unit
#     """
#
#     # Retrieve the parameters from the dictionary "parameters"
#     W1 = parameters['W1']
#     b1 = parameters['b1']
#     W2 = parameters['W2']
#     b2 = parameters['b2']
#     W3 = parameters['W3']
#     b3 = parameters['b3']
#
#     ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
#     Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
#     A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
#     Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
#     A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
#     Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
#     ### END CODE HERE ###
#
#     return Z3
#
#
# # GRADED FUNCTION: compute_cost
#
# def compute_cost(Z3, Y):
#     """
#     Computes the cost
#
#     Arguments:
#     Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
#     Y -- "true" labels vector placeholder, same shape as Z3
#
#     Returns:
#     cost - Tensor of the cost function
#     """
#
#     # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
#     logits = tf.transpose(Z3)
#     labels = tf.transpose(Y)
#
#     ### START CODE HERE ### (1 line of code)
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
#     ### END CODE HERE ###
#
#     return cost
#
#
# def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
#     """
#     Creates a list of random minibatches from (X, Y)
#
#     Arguments:
#     X -- input data, of shape (input size, number of examples)
#     Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
#     mini_batch_size - size of the mini-batches, integer
#     seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
#
#     Returns:
#     mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
#     """
#
#     m = X.shape[1]  # number of training examples
#     mini_batches = []
#     np.random.seed(seed)
#
#     # Step 1: Shuffle (X, Y)
#     permutation = list(np.random.permutation(m))
#     shuffled_X = X[:, permutation]
#     shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))
#
#     # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
#     num_complete_minibatches = math.floor(
#         m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
#     for k in range(0, num_complete_minibatches):
#         mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
#         mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
#         mini_batch = (mini_batch_X, mini_batch_Y)
#         mini_batches.append(mini_batch)
#
#     # Handling the end case (last mini-batch < mini_batch_size)
#     if m % mini_batch_size != 0:
#         mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
#         mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
#         mini_batch = (mini_batch_X, mini_batch_Y)
#         mini_batches.append(mini_batch)
#
#     return mini_batches
#
# def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
#           num_epochs=1500, minibatch_size=32, print_cost=True):
#     """
#     Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
#
#     Arguments:
#     X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
#     Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
#     X_test -- training set, of shape (input size = 12288, number of training examples = 120)
#     Y_test -- test set, of shape (output size = 6, number of test examples = 120)
#     learning_rate -- learning rate of the optimization
#     num_epochs -- number of epochs of the optimization loop
#     minibatch_size -- size of a minibatch
#     print_cost -- True to print the cost every 100 epochs
#
#     Returns:
#     parameters -- parameters learnt by the model. They can then be used to predict.
#     """
#
#     tf.set_random_seed(1)  # to keep consistent results
#     seed = 3  # to keep consistent results
#     (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
#     n_y = Y_train.shape[0]  # n_y : output size
#     costs = []  # To keep track of the cost
#
#     # Create Placeholders of shape (n_x, n_y)
#     ### START CODE HERE ### (1 line)
#     X, Y = create_placeholders(n_x, n_y)
#     ### END CODE HERE ###
#
#     # Initialize parameters
#     ### START CODE HERE ### (1 line)
#     parameters = initialize_parameters()
#     ### END CODE HERE ###
#
#     # Forward propagation: Build the forward propagation in the tensorflow graph
#     ### START CODE HERE ### (1 line)
#     Z3 = forward_propagation(X, parameters)
#     ### END CODE HERE ###
#
#     # Cost function: Add cost function to tensorflow graph
#     ### START CODE HERE ### (1 line)
#     cost = compute_cost(Z3, Y)
#     ### END CODE HERE ###
#
#     # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
#     ### START CODE HERE ### (1 line)
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#     ### END CODE HERE ###
#
#     # Initialize all the variables
#     init = tf.global_variables_initializer()
#
#     # Start the session to compute the tensorflow graph
#     with tf.Session() as sess:
#
#         # Run the initialization
#         sess.run(init)
#
#         # Do the training loop
#         for epoch in range(num_epochs):
#
#             epoch_cost = 0.  # Defines a cost related to an epoch
#             num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
#             seed = seed + 1
#             minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
#
#             for minibatch in minibatches:
#                 # Select a minibatch
#                 (minibatch_X, minibatch_Y) = minibatch
#
#                 # IMPORTANT: The line that runs the graph on a minibatch.
#                 # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
#                 ### START CODE HERE ### (1 line)
#                 _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
#                 ### END CODE HERE ###
#
#                 epoch_cost += minibatch_cost / num_minibatches
#
#             # Print the cost every epoch
#             if print_cost == True and epoch % 100 == 0:
#                 print("Cost after epoch %i: %f" % (epoch, epoch_cost))
#             if print_cost == True and epoch % 5 == 0:
#                 costs.append(epoch_cost)
#
#         # plot the cost
#         plt.plot(np.squeeze(costs))
#         plt.ylabel('cost')
#         plt.xlabel('iterations (per tens)')
#         plt.title("Learning rate =" + str(learning_rate))
#         plt.show()
#
#         # lets save the parameters in a variable
#         parameters = sess.run(parameters)
#         print("Parameters have been trained!")
#
#         # Calculate the correct predictions
#         correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
#
#         # Calculate accuracy on the test set
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
#         print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
#         print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
#
#         return parameters
#
# parameters = model(train_data, train_labels, train_data, train_labels)
