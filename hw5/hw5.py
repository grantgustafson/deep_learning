import tensorflow as tf
from collections import Counter
import re
import math
import numpy as np

sess = tf.Session()


TEXT_PATH = "../great_expectations.txt"
EMBDSZ = 50
BATCHSZ = 50
LEARNING_RATE = 1e-4
NUMSTEPS = 20
LSTMSIZE = 256
EPOCHS = 15
VOCABSZ = 8000

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

# set threshold to the VOCABSZ'th frequecy
counts_list = sorted(counts_list)
if len(counts_list) > VOCABSZ + 1:
	threshold = counts_list[-VOCABSZ]

print('frequency threshold: {}'.format(threshold))

#create our word -> int mat. UNK is VOCABSZ
word_map = {}
for word in words:
	# check if word isn't in our dictionary and meets threshold requirements 
	if word not in word_map and word_counts[word] >= threshold and vocab_size < VOCABSZ - 1:
		word_map[word] = vocab_size
		vocab_size += 1
	# UNK the word
	elif word not in word_map:
		word_map[word] = VOCABSZ - 1
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
print('testing size: {} words'.format(len(test_words)))
trains = np.array(train_words)
tests = np.array(test_words)



### THE RNN ###

# Create the model
# placeholders for x, y, and keep_prob
x = tf.placeholder(tf.int32, [BATCHSZ, NUMSTEPS])
y = tf.placeholder(tf.int32, [BATCHSZ, NUMSTEPS])
keep_prob = tf.placeholder(tf.float32)

# Embedding matrix
E = tf.Variable(tf.random_uniform([VOCABSZ, EMBDSZ], minval=-1, maxval=1, dtype=tf.float32, seed=0))
embd = tf.nn.embedding_lookup(E, x)

# lstm
lstm = tf.nn.rnn_cell.BasicLSTMCell(LSTMSIZE, state_is_tuple=True)
initial_state = lstm.zero_state(BATCHSZ, tf.float32)

# weight and bias vars
w = tf.Variable(tf.truncated_normal([LSTMSIZE, VOCABSZ], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[VOCABSZ]))

# dropout step 
embd_drop = tf.nn.dropout(embd, keep_prob)

# run embeddings through lstm
rnn, out_state = dyrnn = tf.nn.dynamic_rnn(lstm, embd_drop, initial_state = initial_state)

# reshape to 2d
rnn2 = tf.reshape(rnn, [BATCHSZ * NUMSTEPS, LSTMSIZE])

# get logits
logits = tf.matmul(rnn2, w) + b

# new w's
w2 = tf.Variable(tf.constant(1.0, shape=[BATCHSZ*NUMSTEPS]))

# reshape y to 1d
y1d = tf.reshape(y, [BATCHSZ*NUMSTEPS])

# loss calculation
loss1 = tf.nn.seq2seq.sequence_loss_by_example([logits], [y1d], [w2])
loss = tf.reduce_sum(loss1)

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

sess.run(tf.initialize_all_variables())

print("training for {} epochs".format(EPOCHS))
# train
lsum = 0
batches = 0
for e in range(EPOCHS):
	print("epoch: {}".format(e + 1))
	i = 0
	state = (np.zeros([BATCHSZ, LSTMSIZE]), np.zeros([BATCHSZ, LSTMSIZE]))
	while i + BATCHSZ*NUMSTEPS + 1 < len(trains):
		batches += 1
		nextstate, _, l = sess.run([out_state, train_step, loss], 
			feed_dict = {x: np.reshape(trains[i : i + BATCHSZ*NUMSTEPS], (BATCHSZ, NUMSTEPS)),
						 y: np.reshape(trains[i+1 : i+1+BATCHSZ*NUMSTEPS], (BATCHSZ, NUMSTEPS)),  
						 keep_prob: .5,
						 initial_state: state})
		#print (np.reshape(trains[i : i + BATCHSZ*NUMSTEPS], (BATCHSZ, NUMSTEPS)))
		#print (np.reshape(trains[i+1 : i+1+BATCHSZ*NUMSTEPS], (BATCHSZ, NUMSTEPS)))
		#exit()
		i += BATCHSZ*NUMSTEPS
		lsum += (l / (BATCHSZ*NUMSTEPS))
		state = nextstate
print("training perplexity: ")
print(math.exp(lsum/batches))


print("testing")
# test
batches = 0
lsum = 0
i = 0
state = (np.zeros([BATCHSZ, LSTMSIZE]), np.zeros([BATCHSZ, LSTMSIZE]))
while i + BATCHSZ*NUMSTEPS + 1 < len(tests):
	batches += 1
	nextstate, _, l = sess.run([outst, trainStep, loss], 
		feed_dict = {x: np.reshape(tests[i : i + BATCHSZ*NUMSTEPS], (BATCHSZ, NUMSTEPS)),
					 y: np.reshape(tests[i+1 : i+1+BATCHSZ*NUMSTEPS], (BATCHSZ, NUMSTEPS)),
					 keep_prob: 1.0, 
					 initial_state: state})
	i += BATCHSZ*NUMSTEPS
	state = nextstate
	lsum += (l / (BATCHSZ*NUMSTEPS))
print("testing perplexity: ")
print(math.exp(lsum/batches))
