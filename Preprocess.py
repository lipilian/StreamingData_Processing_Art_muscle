#%% import package
import numpy as np
from openpiv import tools, scaling, pyprocess, validation, process, filters
import matplotlib.pyplot as plt
import os
import sys
import cv2
from PIL import Image

#%% important file information
fileNumber = 3000 #each image we have 3000 for 25 fps
fps = 25
# Image crop information got from imageJ
x = 0 # Cut Start point x position
y = 0 # Cut Start point y position
height = 1296 # Cut off muscle region
width = 2048 # keep the origin size of the image width
bitSize = 16 # the number of bit for tif image
maxIntensity = 1500 # highest intensity value from the image
minIntensity = 700 # lowest intensity value from the image

#%% output the current working directory, will not influence the following steps
print(os.getcwd())
inputFilePath = '/Volumes/LiuHongData/StreamingData/Test/test8'
os.chdir(inputFilePath)
print(os.getcwd())

#%% sort the file to get file list
fileNameList = os.listdir(os.getcwd())
len(fileNameList)
def getint(name): # Function to sort the fileName
    value = name.split('.')
    value = value[0].split('_')
    if len(value) == 2:
        return int(value[1])
    else:
        return sys.maxsize
fileNameList = sorted(fileNameList, key = getint) #Sort the fileNameList
fileNameList = fileNameList[:-1]


#%% loop to generate velocity data to txt file
i = 0
winsize = 100 # pixels
searchsize = 100 #pixels
overlap = 50 # piexels
dt = 7*1./25 # piexels
while (i+7) <=2999:
    frame_a = tools.imread(fileNameList[i])
    frame_b = tools.imread(fileNameList[i+7])
    u0, v0, sig2noise = process.extended_search_area_piv(frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )
    x, y = process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )
    u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.3 )
    u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=5, kernel_size=5)
    tools.save(x,y,u2,v2,mask,'../testResult8/test' + str(i) + '.txt')
    i += 1

#%% read the image and preplot insure it is correct
frame_a = tools.imread(fileNameList[0])
frame_b = tools.imread(fileNameList[7])
fig,ax = plt.subplots(1, 2, figsize = (50, 100))
ax[0].imshow(frame_a, cmap = plt.cm.gray)
ax[1].imshow(frame_b, cmap = plt.cm.gray)

#%% crop the image based on cropping information to analysis
frame_aCrop = frame_a[0:1296-1, 0:2048-1]
frame_bCrop = frame_b[0:1296-1, 0:2048-1]
fig,ax = plt.subplots(1, 2, figsize = (50, 100))
ax[0].imshow(frame_aCrop, cmap = plt.cm.gray)
ax[1].imshow(frame_bCrop, cmap = plt.cm.gray)

#%% processing parameter
winsize = 100 # pixels
searchsize = 100 #pixels
overlap = 50 # piexels
dt = 7*1./25 # piexels
u0, v0, sig2noise = process.extended_search_area_piv(frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )
x, y = process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )
u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.3 )
u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=5, kernel_size=5)
#x, y, u, v = scaling.uniform(x, y, u2, v2, scaling_factor = 96.52 )
tools.save(x, y, u2, v2, mask, '../testResult8/test1.txt' )
tools.display_vector_field('../testResult8/test1.txt', scale=1000, width=0.0025)

#%% intensity and background information cancelling
# 1. Plot histogram for image a and image b
'''
plt.subplot(1, 2, 1)
plt.hist(frame_aCrop.ravel(),(maxIntensity - minIntensity),[minIntensity,maxIntensity])
plt.subplot(1, 2, 2)
plt.hist(frame_bCrop.ravel(),(maxIntensity - minIntensity),[minIntensity,maxIntensity])
plt.show
'''
#%% detect the background color from the raw image
