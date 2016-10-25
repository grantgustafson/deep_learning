import tensorflow as tf
from collections import Counter
import re
import math
import numpy as np

sess = tf.Session()

# constants
TEXT_PATH = "../great_expectations.txt"
EMBEDSIZE = 50
BATCHSIZE = 50
TRAININGRATE = 1e-4
NUMSTEPS = 20
LSTMSIZE = 256
EPOCHS = 4
VOCABSIZE = 8000

# tokenizer (referecned from Piazza post)
def basic_tokenizer(sentence, word_split=re.compile(b"([.,!?\"':;)(])")):
    """
    Very basic tokenizer: split the sentence into a list of tokens, lowercase.
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(word_split, space_separated_fragment))
    return [w.lower() for w in words if w]

# open and tokenize source text
f = open(TEXT_PATH, "rb")
words = basic_tokenizer(f.read())
# create word counts
word_counts = Counter(words)
vocab_size = 0

# threshold is the minimum times we need to see a word to put it in our dictionary (not as UNK)
threshold = 1
counts_list = []
for word, count in word_counts.items():
	counts_list.append(count)

# set threshold to the VOCABSIZE'th frequecy
counts_list = sorted(counts_list)
if len(counts_list) > VOCABSIZE + 1:
	threshold = counts_list[-VOCABSIZE]

print('frequency threshold: {}'.format(threshold))

#create our word -> int mat. UNK is VOCABSIZE
word_map = {}
for word in words:
	# check if word isn't in our dictionary and meets threshold requirements 
	if word not in word_map and word_counts[word] >= threshold and vocab_size < VOCABSIZE - 1:
		word_map[word] = vocab_size
		vocab_size += 1
	# UNK the word
	elif word not in word_map:
		word_map[word] = VOCABSIZE - 1
print("vocab size: {}".format(vocab_size + 1))


train_words = []
test_words = []

# build training and testing data lists
i = 0
for word in words:
	i += 1
	if i < math.floor(len(words)*.9):
		train_words.append(word_map[word])
	else:
		test_words.append(word_map[word])
print('training size: {} words'.format(len(train_words)))
print('testing size: {} words'.format(len(train_words)))
trains = np.array(train_words)
tests = np.array(test_words)



### THE RNN ###

# Create the model
# placeholders for x, y, and keep_prob
x = tf.placeholder(tf.int32, [BATCHSIZE, NUMSTEPS])
y = tf.placeholder(tf.int32, [BATCHSIZE, NUMSTEPS])
keep_prob = tf.placeholder(tf.float32)

# Embedding matrix
E = tf.Variable(tf.random_uniform([VOCABSIZE, EMBEDSIZE], minval=-1, maxval=1, dtype=tf.float32, seed=0))
embd = tf.nn.embedding_lookup(E, x)

# lstm
basicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell(LSTMSIZE, state_is_tuple=True)
initialState = basicLSTMCell.zero_state(BATCHSIZE, tf.float32)

# weight and bias vars
w = tf.Variable(tf.truncated_normal([LSTMSIZE, VOCABSIZE], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[VOCABSIZE]))

# dropout step 
embd_drop = tf.nn.dropout(embd, keep_prob)

# run embeddings through lstm
rnn, outst = dyrnn = tf.nn.dynamic_rnn(basicLSTMCell, embd_drop, initial_state = initialState)

# reshape to 2d
rnn2 = tf.reshape(rnn, [BATCHSIZE * NUMSTEPS, LSTMSIZE])

# get logits
logits = tf.matmul(rnn2, w) + b

# new w's
w2 = tf.Variable(tf.constant(1.0, shape=[BATCHSIZE*NUMSTEPS]))

# reshape y to 1d
y1d = tf.reshape(y, [BATCHSIZE*NUMSTEPS])

# loss calculation
loss1 = tf.nn.seq2seq.sequence_loss_by_example([logits], [y1d], [w2])
loss = tf.reduce_sum(loss1)

trainStep = tf.train.AdamOptimizer(TRAININGRATE).minimize(loss)

sess.run(tf.initialize_all_variables())

print("training for {} epochs".format(EPOCHS))
# train
lsum = 0
batches = 0
for e in range(EPOCHS):
	print("epoch: {}".format(e + 1))
	i = 0
	state = (np.zeros([BATCHSIZE, LSTMSIZE]), np.zeros([BATCHSIZE, LSTMSIZE]))
	while i + BATCHSIZE*NUMSTEPS + 1 < len(trains):
		batches += 1
		nextstate, _, l = sess.run([outst, trainStep, loss], 
			feed_dict = {x: np.reshape(trains[i : i + BATCHSIZE*NUMSTEPS], (BATCHSIZE, NUMSTEPS)),
						 y: np.reshape(trains[i+1 : i+1+BATCHSIZE*NUMSTEPS], (BATCHSIZE, NUMSTEPS)),  
						 keep_prob: .5,
						 initialState: state})
		#print (np.reshape(trains[i : i + BATCHSIZE*NUMSTEPS], (BATCHSIZE, NUMSTEPS)))
		#print (np.reshape(trains[i+1 : i+1+BATCHSIZE*NUMSTEPS], (BATCHSIZE, NUMSTEPS)))
		#exit()
		i += BATCHSIZE*NUMSTEPS
		lsum += (l / (BATCHSIZE*NUMSTEPS))
		state = nextstate
print("training perplexity: ")
print(math.exp(lsum/batches))


print("testing")
# test
batches = 0
lsum = 0
i = 0
state = (np.zeros([BATCHSIZE, LSTMSIZE]), np.zeros([BATCHSIZE, LSTMSIZE]))
while i + BATCHSIZE*NUMSTEPS + 1 < len(tests):
	batches += 1
	nextstate, _, l = sess.run([outst, trainStep, loss], 
		feed_dict = {x: np.reshape(tests[i : i + BATCHSIZE*NUMSTEPS], (BATCHSIZE, NUMSTEPS)),
					 y: np.reshape(tests[i+1 : i+1+BATCHSIZE*NUMSTEPS], (BATCHSIZE, NUMSTEPS)),
					 keep_prob: 1.0, 
					 initialState: state})
	i += BATCHSIZE*NUMSTEPS
	state = nextstate
	lsum += (l / (BATCHSIZE*NUMSTEPS))
print("testing perplexity: ")
print(math.exp(lsum/batches))
