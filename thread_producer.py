#!/usr/bin/python2

import os
import sys
import math
import Queue
import threading
import datetime
import time
import numpy as np

# defines
dataQueue = Queue.Queue()
lock = threading.Lock()
CHART_PATH = "./charts/"

#np.set_printoptions(threshold=np.nan, linewidth=200)

class produceTrainingSet(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            filename = self.queue.get()
            
            lock.acquire()
            print(filename)
            lock.release()
            
            self.fillEmptySpace(np.load(CHART_PATH+filename), 180)

            self.queue.task_done()

    def getDate(self, date_str):
        ret = datetime.datetime.strptime(date_str[0:8], "%Y%m%d")
        return ret

    def getMinute(self, date_str):
        date = datetime.datetime.strptime(date_str, "%Y%m%d%H%M%S")
        ret = (date.hour-9)*60 + date.minute 
        return ret

    def fillEmptySpace(self, chart, wannaDays):
        emptySet = [[0, 0, 0, 0, 0, 0] for row in range(361)]

        date = chart[:,2].astype(str)
        price = np.abs(chart[:,[0,1,3,4,5]].astype(int))

        result = np.reshape(np.array([], dtype=int), [-1, 6])

        i = 0
        Date = self.getDate(date[i])
        diffDate = Date - self.getDate(date[i])
    
        for repeatDate in range(wannaDays):
            if diffDate.days > 0 or i == len(chart):
                if Date.weekday() >= 5: result = np.append(result, [[0, 0, 0, 0, 0, 0]], axis=0) # weekend
                else: result = np.append(result, emptySet, axis=0) 
            else:
                for Minute in reversed(range(361)):
                    if i < len(chart):
                        while self.getMinute(date[i]) > 360 or self.getMinute(date[i]) < 0:
                            i = i + 1 # skip
                            if i == len(chart):
                                result = np.append(result, [[0, 0, 0, 0, 0, 0]], axis=0)
                                break;
                            else:
                                diffDate = Date - self.getDate(date[i])
                        if i == len(chart): continue
                    
                        if diffDate.days == 0 and self.getMinute(date[i]) == Minute: 
                            result = np.append(result, [[Minute, price[i][0], price[i][1], price[i][2], price[i][3], price[i][4]]], axis=0)
                            i = i + 1
                        else:
                            result = np.append(result, [[Minute, price[i][0], 0, 0, 0, 0]], axis=0)

                    else:
                        result = np.append(result, [[0, 0, 0, 0, 0, 0]], axis=0)

            Date = Date - datetime.timedelta(hours=24)
            if i < len(chart): diffDate = Date - self.getDate(date[i])
        
        lock.acquire()
        print(len(result))
        lock.release()
        return result

def main():
    datetime.datetime.strptime("2016-05-26", "%Y-%m-%d") # dummy calling for thread safe!
    for i in range(2):
        thread = produceTrainingSet(dataQueue)
        thread.setDaemon(True)
        thread.start()

    for root, dirs, files in os.walk(CHART_PATH):
        for file in files:
            dataQueue.put(file)

    dataQueue.join()

startTime = time.time()
main()
print("Elapsed time: {:.6f}".format(time.time() - startTime))
