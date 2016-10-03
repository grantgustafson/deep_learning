TRAIN_IMAGES_PATH = '../train-images-idx3-ubyte'
TRAIN_LABELS_PATH = '../train-labels-idx1-ubyte'
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets('..', one_hot=True)
NUM_IMAGES = 60000
PIXELS = 784
DIGITS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 100
LEARNING_RATE = .5 

img = tf.placeholder(tf.float32, [None, PIXELS])
w = tf.Variable(tf.zeros([PIXELS, DIGITS]))
b = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(img, w) + b)
lab = tf.placeholder(tf.float32, [None, DIGITS])

cross_ent = tf.reduce_mean(-tf.reduce_sum(lab * tf.log(p), reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_ent)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):	
	batch_imgs, batch_labs = mnist.train.next_batch(BATCH_SIZE)
	sess.run(train_step, feed_dict={img: batch_imgs, lab: batch_labs})

correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(lab, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print (sess.run(accuracy, feed_dict={img: mnist.test.images, lab: mnist.test.labels}))

