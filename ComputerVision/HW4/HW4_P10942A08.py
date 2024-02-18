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


#set kernal
kernel = np.array([0, 1, 1, 1, 0,
                   1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   0, 1, 1, 1, 0,]).reshape((5, 5))

kernel_j = np.array([0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0,
                     1, 1, 0, 0, 0,
                     0, 1, 0, 0, 0,
                     0, 0, 0, 0, 0,]).reshape((5, 5))
                     
kernel_k = np.array([0, 0, 0, 0, 0,
                     0, 1, 1, 0, 0,
                     0, 0, 1, 0, 0,
                     0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0,]).reshape((5, 5))


# ## part(a) Dilation

# In[5]:


def dilation(img,kernel):
    dilation_img = np.zeros(img.shape, np.int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 1:
                for m in range(kernel.shape[0]):
                    for n in range(kernel.shape[1]):
                        if kernel[m, n] == 1:
                            if (i+m) < (img.shape[0]) and (j+n) < (img.shape[1]) and (i+m) >= 0 and (j+n) >= 0:
                                dilation_img[(i+m), (j+n)] = 1
    return dilation_img


# In[6]:


ans_dil = dilation(bina,kernel)


# In[7]:


#顯示圖片
plt.imshow(ans_dil)
#存圖
cv2.imwrite("part_a.bmp",ans_dil*255)


# ## part(b) Erosion

# In[8]:


def erosion(img,kernel):
    erosion_img = np.zeros(img.shape, np.int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            flag = True
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    if kernel[m, n] == 1:
                        if (i+m) > (img.shape[0]-1) or (j+n) > (img.shape[1]-1) or (i+m) < 0 or (j+n) < 0 or img[i+m, j+n] == 0:
                            flag = False
                            break
            if flag:
                erosion_img[i, j] = 1
    return erosion_img


# In[9]:


ans_ero = erosion(bina,kernel)


# In[10]:


#顯示圖片
plt.imshow(ans_ero)
#存圖
cv2.imwrite("part_b.bmp",ans_ero*255)


# ## part(c) Opening

# In[11]:


ans_open = dilation(erosion(bina,kernel),kernel)


# In[12]:


#顯示圖片
plt.imshow(ans_open)
#存圖
cv2.imwrite("part_c.bmp",ans_open*255)


# ## part(d) Closing

# In[13]:


ans_clo = erosion(dilation(bina,kernel),kernel)


# In[14]:


#顯示圖片
plt.imshow(ans_clo)
#存圖
cv2.imwrite("part_d.bmp",ans_clo*255)


# ## part(e) Hit-and-miss transform

# In[24]:


def hit_and_miss(img,kernel_j,kernel_k):
    hit_and_miss_img = np.zeros(img.shape, np.int)
    A_erosion_j = erosion(img, kernel_j) 
    print(A_erosion_j)
    Ac_erosion_k = erosion(~img+2, kernel_k) 
    print(Ac_erosion_k)
    hit_and_miss_img = A_erosion_j * Ac_erosion_k
    return hit_and_miss_img


# In[23]:


#顯示圖片
plt.imshow(ans_hit_miss)
#存圖
cv2.imwrite("part_e.bmp",ans_hit_miss*255)


# In[ ]:




