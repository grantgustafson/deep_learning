import sys, struct
import numpy as np
from collections import defaultdict

TRAIN_IMAGES_PATH = '../train-images-idx3-ubyte'
TRAIN_LABELS_PATH = '../train-labels-idx1-ubyte'

def index(x, y):
	return 28*x + y

digit_count = defaultdict(int)

with open(TRAIN_LABELS_PATH, 'rb') as f:
	bytes = f.read(8)
	labels = []
	for ct in range(60000):
		bytes = f.read(1)
		label = struct.unpack('1b', bytes)
		label = int(label[0])
		digit_count[label] += 1
		labels.append(label)

avgs66 = defaultdict(float)
avgs1313 = defaultdict(float)

with open(TRAIN_IMAGES_PATH, 'rb') as f:
	bytes = f.read(16)

	for ct in range(60000):
		bytes = f.read(784)
		img_raw = struct.unpack('784B', bytes)
		img = np.array(img_raw, dtype=float)
		avgs66[labels[ct]] += img[index(6, 6)]
		avgs1313[labels[ct]] += img[index(13, 13)]

for i in range(10):
	print "Avg at 6, 6 for digit {} is {}".format(i, avgs66[i] / digit_count[i])

for i in range(10):
	print "Avg at 13, 13 for digit {} is {}".format(i, avgs1313[i] / digit_count[i])