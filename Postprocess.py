#%% import package
import numpy as np
from openpiv import tools, scaling, pyprocess, validation, process, filters
import matplotlib.pyplot as plt
import os
import sys

import cv2
from PIL import Image
import pandas as pd

#%% read data from file
inputFilePath = '/Volumes/LiuHongData/StreamingData/Test/testResult8'
os.chdir(inputFilePath)
data = np.loadtxt('test0.txt')
m, n = data.shape
Result = data
Result[:,2] = 0.
Result[:,3] = 0.
Result[:,4] = 0.
for i in range(992):
    txtName = 'test' + str(i) + '.txt'
    data = np.loadtxt(txtName)
    m, n = data.shape
    for j in range(m):
        if(data[j][4] == 0):
            Result[j][4] += 1.
            Result[j][2] += data[j][2]
            Result[j][3] += data[j][3]
for i in range(m):
    if (Result[i][4] != 0.):
        Result[i][2] /= Result[i][4]
        Result[i][3] /= Result[i][4]

#%% save the Result data as txt for tecplot
CalibrateData = 1.624 #micro meter per pixel
Result[:,0:5] *= CalibrateData
np.savetxt('result8.txt', Result,delimiter=',')


#%% Check data size
print(data.shape)
