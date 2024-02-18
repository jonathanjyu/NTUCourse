#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np
import cv2
import math
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


#plt.imshow(bina)


# In[5]:


#downsampling
def downsampling(img):
    row, col = int(img.shape[0]/8), int(img.shape[1]/8)
    downsampling_img = np.zeros([row,col], np.int)
    for i in range(row):
        for j in range (col):
            downsampling_img[i,j] = img[i*8,j*8]
    return downsampling_img


# In[6]:


downsampling_img = downsampling(bina)


# In[7]:


plt.imshow(downsampling_img)


# In[8]:


def h(b, c, d, e):
    if (b == c) and (d != b or e != b):
        return 'q'
    elif (b == c) and (d == b and e == b):
        return 'r'
    else:
        return 's'


# In[9]:


def f(a1, a2, a3, a4):
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


# In[10]:


def yokoi(img):
    yokoi_matrix = np.zeros(img.shape, np.int)
    row, col = img.shape 
    for i in range(row):
        for j in range(col):
            if img[i, j] == 1:
                neighborhood = np.zeros(9,np.int)
                if i-1 >= 0:
                    neighborhood[2] = img[i-1, j]
                    if j-1 >= 0:
                        neighborhood[7] = img[i-1, j-1]
                    if j+1 < col:
                        neighborhood[6] = img[i-1, j+1]
                neighborhood[0] = img[i, j]
                if j-1 >= 0:
                    neighborhood[3] = img[i, j-1]
                if j+1 < col:
                    neighborhood[1] = img[i, j+1]
                if i+1 < row:
                    neighborhood[4] = img[i+1, j]
                    if j-1 >= 0:
                        neighborhood[8] = img[i+1, j-1]
                    if j+1 < col:
                        neighborhood[5] = img[i+1, j+1]
                a1 = h(neighborhood[0], neighborhood[1], neighborhood[6], neighborhood[2])
                a2 = h(neighborhood[0], neighborhood[2], neighborhood[7], neighborhood[3])
                a3 = h(neighborhood[0], neighborhood[3], neighborhood[8], neighborhood[4])
                a4 = h(neighborhood[0], neighborhood[4], neighborhood[5], neighborhood[1])
                yokoi_matrix[i, j] = f(a1, a2, a3, a4)
    return yokoi_matrix


# In[11]:


# yokoi number
yokoi_matrix = yokoi(downsampling_img)
f = open("yokoi.txt", "w")
#f.write("hi")
for i in range(64):
    for j in range(64):
        if yokoi_matrix[i][j] != 0:
            f.write(str(yokoi_matrix[i][j]))
        else:
            f.write(" ")
    f.write("\n")
f.close()

