#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np
import cv2
import math
import copy
lena = mpimg.imread('lena.bmp') # 讀取和程式碼處於同一目錄下的 lena.bmp
# 此時 lena 就已經是一個 np.array 了，可以對它進行任意處理
print(lena.shape) #(512, 512)
plt.imshow(lena) # 顯示圖片
plt.show()


# In[2]:


#binarize
def binarize(lena,ans):
    for i in range(lena.shape[0]):
        for j in range(lena.shape[1]):
            if lena[i][j]>=128:
                ans[i][j] = 1
            else:
                ans[i][j] = 0
    return ans


# In[3]:


bina = [[0] * lena.shape[1]] * lena.shape[0]
bina = np.array(bina)
bina = binarize(lena,bina)


# In[4]:


#downsampling
def downsampling(img):
    row, col = int(img.shape[0]/8), int(img.shape[1]/8)
    downsampling_img = np.zeros([row,col], np.int)
    for i in range(row):
        for j in range (col):
            downsampling_img[i,j] = img[i*8,j*8]
    return downsampling_img


# In[5]:


downsampling_img = downsampling(bina)


# In[6]:


plt.imshow(downsampling_img)


# In[7]:


#padding
downsampling_img = np.pad(downsampling_img, ((1, 1), (1, 1)), 'constant', constant_values=0)


# In[8]:


# initial
shrink_img = copy.deepcopy(downsampling_img)


# In[9]:


def yokoi_h(b, c, d, e):
    if (b == c) and (d != b or e != b):
        return 'q'
    elif (b == c) and (d == b and e == b):
        return 'r'
    else:
        return 's'


# In[11]:


def shrink_h(b, c, d, e):
    if b == c and (b != d or b != e):
        return 1
    else:
        return 0


# In[10]:


def yokoi_f(a1, a2, a3, a4):
    if (a1 == 'r' and a2 == 'r' and a3 == 'r' and a4 == 'r'):
        return 5
    else:
        n=0
        if a1 == 'q':
            n += 1
        if a2 == 'q':
            n += 1
        if a3 == 'q':
            n += 1
        if a4 == 'q':
            n +=1
        return n


# In[12]:


def shrink_f(a1, a2, a3, a4):
    cnt = 0
    if a1 == 1:
        cnt += 1
    if a2 == 1:
        cnt += 1
    if a3 == 1:
        cnt += 1
    if a4 == 1:
        cnt += 1
    if cnt == 1:
        return 1
    else:
        return 0


# In[13]:


def yokoi(img):
    row, col = img.shape[0]-2, img.shape[1]-2
    yokoi_matrix = np.zeros([row, col], np.int)
    for i in range(1, row+1):
        for j in range(1, col+1):
            if img[i, j] == 1:
                a1 = yokoi_h(img[i, j], img[i, j+1], img[i-1, j+1], img[i-1, j])
                a2 = yokoi_h(img[i, j], img[i-1, j], img[i-1, j-1], img[i, j-1])
                a3 = yokoi_h(img[i, j], img[i, j-1], img[i+1, j-1], img[i+1, j])
                a4 = yokoi_h(img[i, j], img[i+1, j], img[i+1, j+1], img[i, j+1])
                yokoi_matrix[i-1, j-1] = yokoi_f(a1, a2, a3, a4)
    return yokoi_matrix


# In[14]:


def pair_relationship(yokoi_matrix):
    row, col = yokoi_matrix.shape[0]-2, yokoi_matrix.shape[1]-2
    pair_relationship_matirx = np.zeros([row, col], np.str)
    for i in range(1, row+1):
        for j in range(1, col+1):
            if (yokoi_matrix[i, j] != 1) or (yokoi_matrix[i-1, j] != 1 and yokoi_matrix[i, j-1] != 1 and yokoi_matrix[i+1, j] != 1 and yokoi_matrix[i, j+1] != 1):
                pair_relationship_matirx[i-1, j-1] = 'q'
            elif (yokoi_matrix[i, j] == 1) and (yokoi_matrix[i-1, j] == 1 or yokoi_matrix[i, j-1] == 1 or yokoi_matrix[i+1, j] == 1 or yokoi_matrix[i, j+1] != 1):
                pair_relationship_matirx[i-1, j-1] = 'p'
    return pair_relationship_matirx


# In[15]:


def shrink(pair_relationship_matrix, img):
    row, col = img.shape[0]-2, img.shape[1]-2
    for i in range(1, row+1):
        for j in range(1, col+1):
            if img[i, j] == 1:
                a1 = shrink_h(img[i, j], img[i, j+1], img[i-1, j+1], img[i-1, j])
                a2 = shrink_h(img[i, j], img[i-1, j], img[i-1, j-1], img[i, j-1])
                a3 = shrink_h(img[i, j], img[i, j-1], img[i+1, j-1], img[i+1, j])
                a4 = shrink_h(img[i, j], img[i+1, j], img[i+1, j+1], img[i, j+1])
                if pair_relationship_matrix[i-1, j-1] == 'p':
                    if shrink_f(a1, a2, a3, a4):
                        img[i, j] = 0
    return img


# In[22]:


while(1):
    # copy
    tmp_img = copy.deepcopy(shrink_img)
    # yokoi number
    yokoi_matrix = yokoi(tmp_img)
    # pair relationship
    pair_relationship_matrix = pair_relationship(np.pad(yokoi_matrix, ((1, 1), (1, 1)), 'constant', constant_values=0))
    # shrink operation
    shrink_img = shrink(pair_relationship_matrix, shrink_img)
    if np.array_equal(shrink_img, tmp_img):
        break


# In[23]:


plt.imshow(shrink_img)


# In[24]:


cv2.imwrite("thinning.bmp", shrink_img * 255)


# In[ ]:




