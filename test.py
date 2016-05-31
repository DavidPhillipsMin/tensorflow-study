#!/usr/bin/python2

import sys
import tensorflow as tf
import numpy as np
import datetime

def fillEmptySpace(chart, wannaSize):
    result = np.reshape(np.array([]), [-1, 6])

    time = 15
    i = 0
    for repeat in range(wannaSize):
        if time < 0: time = 15

        if i < len(chart):
            while chart[i][0] > 15:
                i = i + 1 # skip

            if chart[i][0] == time:
                result = np.append(result, np.reshape(chart[i], [1, 6]), axis=0)
                i = i + 1
            else:
                result = np.append(result, [[time, chart[i][1], chart[i][2], chart[i][3], chart[i][4], chart[i][5]]], axis=0)

        else:
            result = np.append(result, [[time, 0, 0, 0, 0, 0]], axis=0)
            
        time = time - 1

    return result

tmp =   [
            [15, 5, 0, 0, 0, 0],
            [14, 6, 0, 0, 0, 0],
            [11, 7, 0, 0, 0, 0],
            [9, 8, 0, 0, 0, 0],
            [8, 1, 0, 0, 0, 0],
            [7, 2, 0, 0, 0, 0],
            [2, 4, 0, 0, 0, 0],
            [1, 12, 0, 0, 0, 0],
            [16, 1, 0, 0, 0, 0],
            [13, 11, 0, 0, 0, 0],
            [10, 9, 0, 0, 0, 0],
            [9, 0, 0, 0, 0, 0],
            [8, 7, 0, 0, 0, 0],
            [7, 0, 0, 0, 0, 0],
            [4, 0, 0, 2, 0, 0]
        ]

tmp2 =   [
            [15, 5, 0, 0, 0, 0],
            [14, 6, 0, 0, 0, 0],
            [11, 7, 0, 0, 0, 0],
            [9, 8, 0, 0, 0, 0],
            [8, 1, 0, 0, 0, 0],
            [7, 2, 0, 0, 0, 0],
            [2, 4, 0, 0, 0, 0],
            [1, 12, 0, 0, 0, 0],
            [16, 1, 0, 0, 0, 0],
            [13, 11, 0, 0, 0, 0],
            [10, 9, 0, 0, 0, 0],
            [9, 0, 0, 0, 0, 0],
            [8, 7, 0, 0, 0, 0],
            [7, 0, 0, 0, 0, 0],
            [4, 0, 0, 2, 0, 0]
        ]

a = np.array(tmp)
b = np.array(tmp2)

#print(np.reshape(a, [-1, 1, 7]))
#sys.exit(-1)


c = fillEmptySpace(a, 30)
d = fillEmptySpace(b, 30)

m = np.reshape(np.array([]), [-1, 30, 6])
m = np.append(m, np.reshape(c, [1, 30, 6]), axis=0)
m = np.append(m, np.reshape(d, [1, 30, 6]), axis=0)
print(m)
sys.exit(-1)

str = "201603240900"
str2 = "201603230900"
date = datetime.datetime.strptime(str, "%Y%m%d%H%M")
date2 = datetime.datetime.strptime(str2, "%Y%m%d%H%M")
now = datetime.datetime.now()
nowstr = now.strftime("%Y%m%d%H%M")
print(date)
print(now)
diff = date-date2
print(diff)
print(diff.days)
print(diff.seconds)
print(nowstr)
print(now - datetime.timedelta(minutes=15))

tmp = np.darray(["1", 2, 3])
print(tmp)
