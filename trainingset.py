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
    if batchSize > len(next_batch.labels): return None

    batch_labels = np.array([]).reshape(-1, 64)

    if next_batch.counter + batchSize <= len(next_batch.labels):        
        batch_labels = next_batch.labels[next_batch.counter:next_batch.counter+batchSize]
        next_batch.counter = (next_batch.counter + batchSize) % len(next_batch.labels)
    else:
        batch_labels = np.append(batch_labels, next_batch.labels[next_batch.counter:], axis=0)
        if repeat == False: next_batch.counter = 0
        else:
            next_batch.counter = (next_batch.counter + batchSize) % len(next_batch.labels)
            batch_labels = np.append(batch_labels, next_batch.labels[0:next_batch.counter], axis=0)

    batch_charts = np.array([], dtype=int).reshape(-1, CHART_SIZE, 7) 

    for label in batch_labels[:,0:2]:
        startIndex = int(label[1]) * 361
        endIndex = startIndex + CHART_SIZE
        batch_charts = np.append(batch_charts, [np.load(CHART_PATH + label[0])[startIndex:endIndex,:]], axis=0)
    
    return batch_charts.reshape(-1, CHART_SIZE, 7, 1), batch_labels[:,3:].astype(int)



def shuffle_batch_set():
    np.random.shuffle(next_batch.labels)

next_batch.counter = 0
next_batch.labels = labels()
num_set = len(next_batch.labels)

#np.set_printoptions(threshold=np.nan, linewidth=200)
if __name__ == "__main__":
    startTime = time.time()

    print("training set size: {:d}".format(len(next_batch.labels)))
    shuffle_batch_set()

    total_batch_set = int(math.ceil(len(next_batch.labels)/100.0))
    for i in range(total_batch_set):
        print(next_batch(100))

    print("Elapsed time: {:.6f}".format(time.time() - startTime))
