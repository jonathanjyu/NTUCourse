#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np
import cv2
import math
import copy
import random
lena = mpimg.imread('lena.bmp') # 讀取和程式碼處於同一目錄下的 lena.bmp
# 此時 lena 就已經是一個 np.array 了，可以對它進行任意處理
print(lena.shape) #(512, 512)
plt.imshow(lena) # 顯示圖片
#plt.show()


# In[2]:


def padding(img):
    img = np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=0)
    row, col = img.shape
    for j in range(1, col-1):
        img[0, j] = img[1, j]
        img[row-1, j] = img[row-2, j]
    for i in range(1, row-1):
        img[i, 0] = img[i, 1]
        img[i, col-1] = img[i, col-2]
    img[0, 0] = img[1, 1]
    img[0, col-1] = img[1, col-2]
    img[row-1, 0] = img[row-2 ,1]
    img[row-1, col-1] = img[row-2, col-2]
    return img


# ## part(a) Laplace Mask1: 15

# In[3]:


def Laplacian_Mask_1(img, threshold):
    laplacian_img = np.zeros(img.shape, np.int)
    img = padding(img)
    row, col = img.shape
    for i in range(1, row-1):
        for j in range(1, col-1):
            gradient = int(img[i-1, j]) + int(img[i, j-1]) + int(img[i, j+1]) + int(img[i+1, j]) - 4*int(img[i, j])
            if gradient >= threshold:
                laplacian_img[i-1, j-1] = 1
            elif gradient <= -threshold:
                laplacian_img[i-1, j-1] = -1
            else:
                laplacian_img[i-1, j-1] = 0
    zero_crossing_img = np.zeros(laplacian_img.shape, np.int)
    laplacian_img = padding(laplacian_img)
    for i in range(1, row-1):
            for j in range(1, col-1):
                zero_crossing_img[i-1, j-1] = 255
                if laplacian_img[i, j] == 1:
                    if laplacian_img[i-1, j-1] == -1 or laplacian_img[i-1, j] == -1 or laplacian_img[i-1, j+1] == -1 or laplacian_img[i, j-1] == -1 and laplacian_img[i, j+1] == -1 or laplacian_img[i+1, j-1] == -1 or laplacian_img[i+1, j] == -1 or laplacian_img[i+1, j+1] == -1:
                        zero_crossing_img[i-1, j-1] = 0
                else:
                    zero_crossing_img[i-1, j-1] = 255
    return zero_crossing_img


# In[4]:


part_a = Laplacian_Mask_1(lena,15)


# In[5]:


cv2.imwrite("part(a).bmp", part_a)
plt.imshow(part_a)


# ## part(b) Laplace Mask2: 15

# In[6]:


def Laplacian_Mask_2(img, threshold):
    laplacian_img = np.zeros(img.shape, np.int)
    img = padding(img)
    row, col = img.shape
    for i in range(1, row-1):
        for j in range(1, col-1):
            gradient = (int(img[i-1, j-1]) + int(img[i-1, j]) + int(img[i-1, j+1]) + int(img[i, j-1]) + int(img[i, j+1]) + int(img[i+1, j-1]) + int(img[i+1, j]) + int(img[i+1, j+1]) - 8*int(img[i, j]))/3
            if gradient >= threshold:
                laplacian_img[i-1, j-1] = 1
            elif gradient <= -threshold:
                laplacian_img[i-1, j-1] = -1
            else:
                laplacian_img[i-1, j-1] = 0
    zero_crossing_img = np.zeros(laplacian_img.shape, np.int)
    laplacian_img = padding(laplacian_img)
    for i in range(1, row-1):
            for j in range(1,col-1):
                zero_crossing_img[i-1, j-1] = 255
                if laplacian_img[i, j] == 1:
                    if laplacian_img[i-1, j-1] == -1 or laplacian_img[i-1, j] == -1 or laplacian_img[i-1, j+1] == -1 or laplacian_img[i, j-1] == -1 and laplacian_img[i, j+1] == -1 or laplacian_img[i+1, j-1] == -1 or laplacian_img[i+1, j] == -1 or laplacian_img[i+1, j+1] == -1:
                        zero_crossing_img[i-1, j-1] = 0
                else:
                    zero_crossing_img[i-1, j-1] = 255
    return zero_crossing_img


# In[7]:


part_b = Laplacian_Mask_2(lena,15)


# In[8]:


cv2.imwrite("part(b).bmp", part_b)
plt.imshow(part_b)


# ## part(c) Minimum-variance Laplacian: 20

# In[9]:


def minimum_variance_Laplacian(img, threshold):
    laplacian_img = np.zeros(img.shape, np.int)
    img = padding(img)
    row, col = img.shape
    for i in range(1, row-1):
        for j in range(1, col-1):
            gradient = (2*int(img[i-1, j-1]) - int(img[i-1, j]) + 2*int(img[i-1, j+1]) - int(img[i, j-1]) - int(img[i, j+1]) + 2*int(img[i+1, j-1]) - int(img[i+1, j]) + 2*int(img[i+1, j+1]) - 4*int(img[i, j]))/3
            if gradient >= threshold:
                laplacian_img[i-1, j-1] = 1
            elif gradient <= -threshold:
                laplacian_img[i-1, j-1] = -1
            else:
                laplacian_img[i-1, j-1] = 0
    zero_crossing_img = np.zeros(laplacian_img.shape, np.int)
    laplacian_img = padding(laplacian_img)
    for i in range(1, row-1):
            for j in range(1, col-1):
                zero_crossing_img[i-1, j-1] = 255
                if laplacian_img[i, j] == 1:
                    if laplacian_img[i-1, j-1] == -1 or laplacian_img[i-1, j] == -1 or laplacian_img[i-1, j+1] == -1 or laplacian_img[i, j-1] == -1 or laplacian_img[i, j+1] == -1 or laplacian_img[i+1, j-1] == -1 or laplacian_img[i+1, j] == -1 or laplacian_img[i+1, j+1] == -1:
                        zero_crossing_img[i-1, j-1] = 0
                else:
                    zero_crossing_img[i-1, j-1] = 255
    return zero_crossing_img


# In[10]:


part_c = minimum_variance_Laplacian(lena,20)


# In[11]:


cv2.imwrite("part(c).bmp", part_c)
plt.imshow(part_c)


# ## part(d) Laplace of Gaussian: 3000

# In[12]:


def Laplacian_of_Gaussian(img, threshold):
    mask = [[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]
           ]
    laplacian_img = np.zeros(img.shape, np.int)
    img = padding(img)
    img = padding(img)
    img = padding(img)
    img = padding(img)
    img = padding(img)
    row, col = img.shape
    for i in range(5, row-5):
        for j in range(5, col-5):
            gradient = 0
            for m in range(-5, 6):
                for n in range(-5, 6):
                    gradient += int(img[i+m, j+n]) * mask[m+5][n+5]
            if gradient >= threshold:
                laplacian_img[i-5, j-5] = 1
            elif gradient <= -threshold:
                laplacian_img[i-5, j-5] = -1
            else:
                laplacian_img[i-5, j-5] = 0
    zero_crossing_img = np.zeros(laplacian_img.shape, np.int)
    laplacian_img = padding(laplacian_img)
    row, col = laplacian_img.shape
    for i in range(1, row-1):
            for j in range(1, col-1):
                zero_crossing_img[i-1, j-1] = 255
                if laplacian_img[i, j] == 1:
                    if laplacian_img[i-1, j-1] == -1 or laplacian_img[i-1, j] == -1 or laplacian_img[i-1, j+1] == -1 or laplacian_img[i, j-1] == -1 or laplacian_img[i, j+1] == -1 or laplacian_img[i+1, j-1] == -1 or laplacian_img[i+1, j] == -1 or laplacian_img[i+1, j+1] == -1:
                        zero_crossing_img[i-1, j-1] = 0
                else:
                    zero_crossing_img[i-1, j-1] = 255
    return zero_crossing_img


# In[13]:


part_d = Laplacian_of_Gaussian(lena,3000)


# In[14]:


cv2.imwrite("part(d).bmp", part_d)
plt.imshow(part_d)


# ## part(e) Difference of Gaussian: 1

# In[15]:


def Difference_of_Gaussian(img, threshold):
    mask = [[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]
           ]
    laplacian_img = np.zeros(img.shape, np.int)
    img = padding(img)
    img = padding(img)
    img = padding(img)
    img = padding(img)
    img = padding(img)
    row, col = img.shape
    for i in range(5, row-5):
        for j in range(5, col-5):
            gradient = 0
            for m in range(-5, 6):
                for n in range(-5, 6):
                    gradient += int(img[i+m, j+n]) * mask[m+5][n+5]
            if gradient >= threshold:
                laplacian_img[i-5, j-5] = 1
            elif gradient <= -threshold:
                laplacian_img[i-5, j-5] = -1
            else:
                laplacian_img[i-5, j-5] = 0
    zero_crossing_img = np.zeros(laplacian_img.shape, np.int)
    laplacian_img = padding(laplacian_img)
    row, col = laplacian_img.shape
    for i in range(1, row-1):
            for j in range(1, col-1):
                zero_crossing_img[i-1, j-1] = 255
                if laplacian_img[i, j] == 1:
                    if laplacian_img[i-1, j-1] == -1 or laplacian_img[i-1, j] == -1 or laplacian_img[i-1, j+1] == -1 or laplacian_img[i, j-1] == -1 or laplacian_img[i, j+1] == -1 or laplacian_img[i+1, j-1] == -1 or laplacian_img[i+1, j] == -1 or laplacian_img[i+1, j+1] == -1:
                        zero_crossing_img[i-1, j-1] = 0
                else:
                    zero_crossing_img[i-1, j-1] = 255
    
    return zero_crossing_img


# In[16]:


part_e = Difference_of_Gaussian(lena,1)


# In[17]:


cv2.imwrite("part(e).bmp", part_e)
plt.imshow(part_e)


# In[ ]:




