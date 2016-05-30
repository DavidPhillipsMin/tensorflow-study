#!/usr/bin/python2

import os
import sys
import math
import datetime
import time
import numpy as np
from multiprocessing import Process, JoinableQueue

# defines
CHART_PATH = "charts/"

dataQueue = JoinableQueue()

emptySet = [[0, 0, 0, 0, 0, 0] for row in reversed(range(361))]
secondList = list(reversed(range(361)))
emptySet = np.insert(emptySet, 1, secondList, axis=1)

#np.set_printoptions(threshold=np.nan, linewidth=200)

def getDate(date_str):
    return datetime.datetime.strptime(date_str[0:8], "%Y%m%d")

def getMinute(date_str):
    date = datetime.datetime.strptime(date_str, "%Y%m%d%H%M%S")
    return (date.hour-9)*60 + date.minute 

def fillEmptySpace(chart, wannaDays):
    #print(chart)
    date = chart[:,2].astype(str)
    price = np.abs(chart[:,[0,1,3,4,5]].astype(int))

    result = np.reshape(np.array([], dtype=int), [-1, 7])

    i = 0
    Date = getDate(date[i])
    diffDate = Date - getDate(date[i])
    
    for repeatDate in range(wannaDays):
        if diffDate.days > 0 or i == len(chart):
            #if Date.weekday() >= 5: result = np.append(result, [[0, 0, 0, 0, 0, 0, 0]], axis=0) # weekend
            #else: result = np.append(result, emptySet, axis=0)
            if Date.weekday() < 5: result = np.append(result, emptySet, axis=0) # workingday
        else:
            day = int(Date.strftime("%Y%m%d"))
            for Minute in reversed(range(361)):
                if i < len(chart):
                    while getMinute(date[i]) > 360 or getMinute(date[i]) < 0:
                        i = i + 1 # skip
                        if i == len(chart):
                            result = np.append(result, [[0, 0, 0, 0, 0, 0, 0]], axis=0)
                            break;
                        else:
                            diffDate = Date - getDate(date[i])
                    if i == len(chart): continue
                    
                    if diffDate.days == 0 and getMinute(date[i]) == Minute: 
                        result = np.append(result, [[day, Minute, price[i][0], price[i][1], price[i][2], price[i][3], price[i][4]]], axis=0)
                        i = i + 1
                    else:
                        result = np.append(result, [[day, Minute, price[i][0], 0, 0, 0, 0]], axis=0)

                else:
                    result = np.append(result, [[0, 0, 0, 0, 0, 0, 0]], axis=0)

        Date = Date - datetime.timedelta(hours=24)
        if i < len(chart): diffDate = Date - getDate(date[i])
        
    print(len(chart), i, len(result))
    return result

# Process task
def singleProc(queue):
    while True:
        
        filename = queue.get()        
        result = fillEmptySpace(np.load(CHART_PATH+filename), 7*26) # for 26 weeks(182days)
        print(filename + " -> Done.")
        
        #print(result)
        
        np.save("tmp/"+filename, result)
        
        queue.task_done()


def main():
    # make process pool
    for i in range(2):
        p = Process(target=singleProc, args=(dataQueue,))
        p.daemon = True
        p.start()

    for root, dirs, files in os.walk(CHART_PATH):
        for file in files:
            print(file + " -> Pass task to process pool.")
            dataQueue.put(file)

    dataQueue.join()
    #np.savez("finalCharts", **finalCharts)

startTime = time.time()
main()
print("Elapsed time: {:.6f}".format(time.time() - startTime))
