TRAIN_IMAGES_PATH = '../train-images-idx3-ubyte'
TRAIN_LABELS_PATH = '../train-labels-idx1-ubyte'

# NOTE: this code is heavily sourced from the TensorFlow MNIST tutorial found here: https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets('..', one_hot=True)
NUM_IMAGES = 60000
PIXELS = 784
DIGITS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 500
LEARNING_RATE = 1e-4

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1], padding='SAME')

#img as 1 dimensional array
x = tf.placeholder(tf.float32, [None, PIXELS])

#first convolution layer
Wc1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
bc1 = tf.Variable(tf.constant(.1, shape=[32]))
#reshape image to 28X28
img = tf.reshape(x, [-1, 28, 28, 1])
hc1 = tf.nn.relu(conv2d(img, Wc1) + bc1)
hp1 = max_pool_2x2(hc1)

#second convolution layer
Wc2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
bc2 = tf.Variable(tf.constant(.1, shape=[64]))

hc2 = tf.nn.relu(conv2d(hp1, Wc2) + bc2)
hp2 = max_pool_2x2(hc2)

#fully connected layer
Wfc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
bfc1 = tf.Variable(tf.constant(.1, shape=[1024]))

hp2_flat = tf.reshape(hp2, [-1, 7*7*64])
hfc1 = tf.nn.relu(tf.matmul(hp2_flat, Wfc1) + bfc1)

#dropping based on inputed probability
keep_prob = tf.placeholder(tf.float32)
hfc1_drop = tf.nn.dropout(hfc1, keep_prob)

Wfc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
bfc2 = tf.Variable(tf.constant(.1, shape=[10]))

p = tf.nn.softmax(tf.matmul(hfc1_drop, Wfc2) + bfc2)

lab = tf.placeholder(tf.float32, [None, DIGITS])

cross_ent = tf.reduce_mean(-tf.reduce_sum(lab * tf.log(p), reduction_indices = [1]))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_ent)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(2000):
	if i % 10 == 0:
		print ('Batch: {}'.format(i))	
	batch_imgs, batch_labs = mnist.train.next_batch(BATCH_SIZE)
	sess.run(train_step, feed_dict={x: batch_imgs, lab: batch_labs, keep_prob: .5})

correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(lab, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print (sess.run(accuracy, feed_dict={x: mnist.test.images, lab: mnist.test.labels, keep_prob: 1.0}))

