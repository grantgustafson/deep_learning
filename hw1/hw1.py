import sys, struct, math
import numpy as np
from collections import defaultdict

TRAIN_IMAGES_PATH = '../train-images-idx3-ubyte'
TRAIN_LABELS_PATH = '../train-labels-idx1-ubyte'
NUM_IMAGES = 60000
PIXELS = 784
DIGITS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 1
LEARNING_RATE = .5 

def index(x, y):
	return 28*x + y

digit_count = defaultdict(int)

weights = np.zeros((PIXELS, DIGITS))
biases = np.zeros(DIGITS)

def softmax(h, idx):
	h_sum = 0.0
	for i in range(DIGITS):
		h_sum += math.exp(h[i])
	return math.exp(h[idx]) / h_sum

def dE_dH(p, i, d):
	if i == d:
		return -(1-p)
	else:
		return -p


with open(TRAIN_LABELS_PATH, 'rb') as f:
	bytes = f.read(8)
	labels = []
	for ct in range(NUM_IMAGES):
		bytes = f.read(1)
		label = struct.unpack('1b', bytes)
		label = int(label[0])
		digit_count[label] += 1
		labels.append(label)

images = []

with open(TRAIN_IMAGES_PATH, 'rb') as f:
	bytes = f.read(16)

	for ct in range(NUM_IMAGES):
		bytes = f.read(PIXELS)
		img_raw = struct.unpack('784B', bytes)
		img = np.array(img_raw, dtype=float)
		images.append(img)

for img_idx in range(10000):
	img = images[img_idx]
	h = np.zeros(DIGITS)
	for i in range(PIXELS):
		for j in range(DIGITS):
			h[j] += weights[i][j] * img[i]
	for i in range(DIGITS):
		h[i] += biases[i]
	predicted = 0, 0
	for i in range(DIGITS):
		if softmax(h, i) > predicted[1]:
			predicted = i, softmax(h, i)
	for i in range(PIXELS):
		for j in range(DIGITS):
			weights[i][j] =+ -LEARNING_RATE * dE_dH(predicted[1], j, labels[img_idx])
	biases += np.dot(dE_dH(predicted[1], predicted[0], labels[img_idx]), -.5)


