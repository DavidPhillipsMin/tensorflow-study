#!/usr/bin/python2

import os
import sys
import datetime
import time
import numpy as np
from multiprocessing import Process, JoinableQueue

# defines
CHART_PATH = "charts/"
LABEL_PATH = "labels/"

dataQueue = JoinableQueue()

np.set_printoptions(threshold=100, linewidth=200)

def makeLabel(filename):
    chart = np.load(CHART_PATH + filename)
    #print(chart)

    template = [0] * 64
    result = np.array([], dtype=object).reshape(-1, 64)

    i = 0
    for repeat in range((7*12)-(12*2)): #exclude weekend!
        if chart[i+360][0] == chart[i][0] and chart[i+360][1] == 0 and chart[i][1] == 360:
            startPrice = float(chart[i+360][2])
            endPrice = float(chart[i][2])

            if startPrice > 0:
                ratio = (endPrice - startPrice) / startPrice * 100
            else: 
                ratio = 0

            if ratio > 30: ratio = 30
            if ratio < -30: ratio = -30
            
            template[0] = filename
            template[1] = repeat+1
            template[2] = ratio
            template[int(ratio)+33] = 1
            result = np.append(result, [template], axis=0)
            template[int(ratio)+33] = 0 # restore
        else:
            break
        i = i + 361

    return result


# Process task
def singleProc(queue):
    sum = [0]*61

    while True:
        filename = queue.get()
        
        labels = makeLabel(filename)
        print(filename + " -> Done.")
        print(labels)
        #print(labels[:,[0,15,16]])
        #print
        for row in labels[:,3:]:
            sum = sum + row.astype(int)
        #print(sum)

        np.save(LABEL_PATH + filename[0:6] + "_label", labels)
        
        queue.task_done()


if __name__ == "__main__":
    startTime = time.time()

    # make process pool
    for i in range(1):
        p = Process(target=singleProc, args=(dataQueue,))
        p.daemon = True
        p.start()

    for root, dirs, files in os.walk(CHART_PATH):
        for file in files:
            #print(file + " -> Pass task to process pool.")
            dataQueue.put(file)
            #dataQueue.put("005720.npy")

    dataQueue.join()
    #np.savez("finalCharts", **finalCharts)

    print("Elapsed time: {:.6f}".format(time.time() - startTime))
