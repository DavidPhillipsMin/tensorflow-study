#!/usr/bin/python2

import sys
import os
import math
import time
import random
import numpy as np

# defines
CHART_PATH = "charts/"
CHART_SIZE = ((7*12)-(12*2))*361
LABEL_PATH = "labels/"

def labels():
    result = np.array([]).reshape(-1, 64)

    for root, dirs, files in os.walk(LABEL_PATH):
        for file in files:
            labels = np.load(LABEL_PATH + file)
            result = np.append(result, labels, axis=0)

    return result

def next_batch(batchSize, repeat=False):
    if batchSize > num_set: batchSize = num_set

    batch_labels = np.array([]).reshape(-1, 64)

    if next_batch.counter + batchSize <= num_set:        
        batch_labels = next_batch.labels[next_batch.counter:next_batch.counter+batchSize]
        next_batch.counter = (next_batch.counter + batchSize) % num_set
    else:
        batch_labels = np.append(batch_labels, next_batch.labels[next_batch.counter:], axis=0)
        if repeat == False: next_batch.counter = 0
        else:
            next_batch.counter = (next_batch.counter + batchSize) % num_set
            batch_labels = np.append(batch_labels, next_batch.labels[0:next_batch.counter], axis=0)

    batch_charts = np.array([], dtype=int).reshape(-1, CHART_SIZE, 7) 

    for label in batch_labels[:,0:2]:
        startIndex = int(label[1]) * 361
        endIndex = startIndex + CHART_SIZE
        batch_charts = np.append(batch_charts, [np.load(CHART_PATH + label[0])[startIndex:endIndex,:]], axis=0)
    
    return batch_charts[:,:,2:].reshape(-1, CHART_SIZE, 1, 5), batch_labels[:,3:].astype(int)

def next_test_batch(batchSize, repeat=False):
    if batchSize > num_test_set: batchSize = num_test_set

    batch_labels = np.array([]).reshape(-1, 64)

    if next_test_batch.counter + batchSize <= num_test_set:
        batch_labels = next_test_batch.labels[next_test_batch.counter:next_test_batch.counter+batchSize]
        next_test_batch.counter = (next_test_batch.counter + batchSize) % num_test_set
    else:
        batch_labels = np.append(batch_labels, next_test_batch.labels[next_test_batch.counter:], axis=0)
        if repeat == False: next_test_batch.counter = 0
        else:
            next_test_batch.counter = (next_test_batch.counter + batchSize) % num_test_set
            batch_labels = np.append(batch_labels, next_test_batch.labels[0:next_test_batch.counter], axis=0)

    batch_charts = np.array([], dtype=int).reshape(-1, CHART_SIZE, 7) 

    for label in batch_labels[:,0:2]:
        startIndex = int(label[1]) * 361
        endIndex = startIndex + CHART_SIZE
        batch_charts = np.append(batch_charts, [np.load(CHART_PATH + label[0])[startIndex:endIndex,:]], axis=0)
    
    return batch_charts[:,:,2:].reshape(-1, CHART_SIZE, 1, 5), batch_labels[:,3:].astype(int)


def shuffle_batch_set():
    np.random.shuffle(next_batch.labels)


next_batch.counter = 0
next_test_batch.counter = 0

all_labels = labels()
num_set = int(len(all_labels) * 0.8)
num_test_set = len(all_labels) - num_set

next_batch.labels = all_labels[:num_set]
next_test_batch.labels = all_labels[num_set:]

num_set = 200

#np.set_printoptions(threshold=np.nan, linewidth=200)
if __name__ == "__main__":
    startTime = time.time()

    print("training set size: {:d}".format(num_set))
    print("test set size:     {:d}".format(num_test_set))

    shuffle_batch_set()

    total_batch_set = int(math.ceil(num_set/50.0))
    for i in range(total_batch_set):
        print(next_batch(50))

    total_test_batch_set = int(math.ceil(num_test_set/50.0))
    for i in range(total_test_batch_set):
        print(next_test_batch(50))

    print("Elapsed time: {:.6f}".format(time.time() - startTime))
