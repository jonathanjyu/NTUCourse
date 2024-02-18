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


# In[11]:


#set kernel
kernel = np.array([0, 1, 1, 1, 0,
                   1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   0, 1, 1, 1, 0,]).reshape((5, 5))


# ## part(a) Dilation

# In[12]:


def dilation(img,kernel):
    dilation_img = np.zeros(img.shape, np.int)
    row, col = img.shape
    kernel_row, kernel_col = kernel.shape
    for i in range(row):
        for j in range(col):
            if img[i, j] > 0:
                maximum = 0
                for m in range(-round(kernel_row/2), -round(kernel_row/2)+kernel_row):
                    for n in range(-round(kernel_col/2), -round(kernel_col/2)+kernel_col):
                        if kernel[m+round(kernel_row/2), n+round(kernel_col/2)] > 0:
                            if (i+m) < row and (j+n) < col and (i+m) >= 0 and (j+n) >= 0:
                                maximum = max(maximum, img[i+m, j+n])
                for m in range(-round(kernel_row/2), -round(kernel_row/2)+kernel_row):
                    for n in range(-round(kernel_col/2), -round(kernel_col/2)+kernel_col):
                        if kernel[m+round(kernel_row/2), n+round(kernel_col/2)] > 0:
                            if (i+m) < row and (j+n) < col and (i+m) >= 0 and (j+n) >= 0:
                                dilation_img[i+m, j+n] = maximum
    return dilation_img


# In[13]:


ans_dil = dilation(lena,kernel)


# In[16]:


#顯示圖片
plt.imshow(ans_dil)
#存圖
cv2.imwrite("part_a.bmp",ans_dil)


# ## part(b) Erosion

# In[17]:


def erosion(img,kernel):
    erosion_img = np.zeros(img.shape, np.int)
    row, col = img.shape
    kernel_row, kernel_col = kernel.shape
    for i in range(row):
        for j in range(col):
            flag = True
            minimun = 255
            for m in range(-round(kernel_row/2), -round(kernel_row/2)+kernel_row):
                for n in range(-round(kernel_col/2), -round(kernel_col/2)+kernel_col):
                    if kernel[m+round(kernel_row/2), n+round(kernel_col/2)] > 0:
                        if (i+m) > (row-1) or (j+n) > (col-1) or (i+m) < 0 or (j+n) < 0 or img[i+m, j+n] == 0:
                            flag = False
                            break
                        else:
                            minimun = min(minimun, img[i+m, j+n])
            if flag:
                erosion_img[i, j] = minimun
    return erosion_img


# In[18]:


ans_ero = erosion(lena,kernel)


# In[19]:


#顯示圖片
plt.imshow(ans_ero)
#存圖
cv2.imwrite("part_b.bmp",ans_ero)


# ## part(c) Opening

# In[20]:


ans_open = dilation(erosion(lena,kernel),kernel)


# In[21]:


#顯示圖片
plt.imshow(ans_open)
#存圖
cv2.imwrite("part_c.bmp",ans_open)


# ## part(d) Closing

# In[22]:


ans_clo = erosion(dilation(lena,kernel),kernel)


# In[23]:


#顯示圖片
plt.imshow(ans_clo)
#存圖
cv2.imwrite("part_d.bmp",ans_clo)

