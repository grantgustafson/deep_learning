TRAIN_WORDS_PATH = '../train.txt'
TEST_WORDS_PATH = '../test.txt'

import math
import tensorflow as tf

BATCH_SIZE = 20
LEARNING_RATE = 1e-4
EMBDSZ = 30
HIDDEN_SIZE = 100

word_map = {}
vocab_size = 0
with open(TRAIN_WORDS_PATH) as f:
	for word in f.read().split():
		if word not in word_map:
			word_map[word] = vocab_size
			vocab_size += 1
	print("vocab size: {}".format(vocab_size))
	# create map from string to ints
	# given a position i in a word


inpt = tf.placeholder(tf.int32, [None])
output = tf.placeholder(tf.int32, [None])
E = tf.Variable(tf.random_uniform([vocab_size, EMBDSZ], 0.0, 0.2))
# #take input and get embedding rep: 1 X EMBDSZ
embd = tf.nn.embedding_lookup(E, inpt)
w1 = tf.Variable(tf.truncated_normal([EMBDSZ, HIDDEN_SIZE], stddev=0.1))
# hidden size 100
b1 = tf.Variable(tf.zeros(HIDDEN_SIZE))
h1 = tf.nn.relu(tf.matmul(embd, w1) + b1)
b2 = tf.Variable(tf.zeros(vocab_size))
w2 = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, vocab_size], stddev=0.1))
logits = tf.matmul(h1, w2) + b2
error = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, output)
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(error)
sum_err = tf.reduce_mean(error)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

with open(TRAIN_WORDS_PATH) as f:
	words = f.read().split()
	i = 0
	batches = 0
	tot_err = 0.0
	while i < len(words) - BATCH_SIZE - 1:
		batch_input_words = words[i : i + BATCH_SIZE]
		batch_input = list(map(lambda w: word_map.get(w), batch_input_words))
		batch_output_words = words[i + 1 : i + BATCH_SIZE + 1]
		batch_output = list(map(lambda w: word_map.get(w), batch_output_words))
		_, err = sess.run([train_step, error], feed_dict={inpt: batch_input, output: batch_output})
		tot_err += sum(err)
		i += BATCH_SIZE
		if batches % 1000 == 0:
			print ('batch {}'.format(batches))
			print ('perplexity: {}'.format(math.exp(tot_err/i)))
		batches += 1
	print('Final training perplexity: {}'.format(math.exp((tot_err/i))))

with open(TEST_WORDS_PATH) as f:
	words = f.read().split()
	i = 0
	tot_err = 0.0
	while i < len(words) - BATCH_SIZE - 1:
		batch_input_words = words[i : i + BATCH_SIZE]
		batch_input = list(map(lambda w: word_map.get(w), batch_input_words))
		batch_output_words = words[i + 1 : i + BATCH_SIZE + 1]
		batch_output = list(map(lambda w: word_map.get(w), batch_output_words))
		err = sess.run(error, feed_dict={inpt: batch_input, output: batch_output})
		tot_err += sum(err)
		i += BATCH_SIZE
	print ('Test perplexity: {}'.format(math.exp(tot_err / (i + BATCH_SIZE))))