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


# ## part(a) original image and its histogram

# In[2]:


#original image save
cv2.imwrite("part_a_ori.bmp",lena)


# In[3]:


#hist
def histpic(img):
    img = img.reshape(img.shape[0]*img.shape[1],)
    print(img)
    
    hist = np.zeros(256,dtype = int)
    for i in img:
        for j in range(256):
            if i == j:
                hist[j] = hist[j] + 1
    return hist


# In[4]:


hist = histpic(lena)


# In[5]:


#顯示圖片
x=np.arange(0,255)
y=hist[x]
plt.xlabel("Values")
plt.ylabel("Count")
plt.title("lena pixel histogram")
plt.bar(x,y)

#存圖
plt.savefig('part_a_ori_hist.png')


# ## part(b) image with intensity divided by 3 and its histogram

# In[6]:


#divided by 3
def div3(img):
    img = img // 3
    return img


# In[7]:


div_3 = div3(lena)


# In[8]:


#顯示圖片
plt.imshow(div_3) 
#存圖
cv2.imwrite("part_b_div3.bmp",div_3)


# In[9]:


#hist
hist_div3 = histpic(div_3)


# In[10]:


div_3


# In[11]:


hist_div3


# In[12]:


#顯示圖片
x=np.arange(0,255)
y=hist_div3[x]
plt.xlabel("Values")
plt.ylabel("Count")
plt.title("lena divided by 3 pixel histogram")
plt.bar(x,y)

#存圖
plt.savefig('part_b_div3_hist.png')


# ## part(c) image after applying histogram equalization to (b) and its histogram

# In[13]:


def histeq(img_in, hist):
    row, col  = img_in.shape
    ans = np.array([[0] * img_in.shape[1]] * img_in.shape[0])
    cdf_list = [0 for i in range(256)]
    cdf = 0.0
    max_value = 0
    min_value = 1 << 31
    for i in range(0, len(hist)):
        if hist[i]:
            max_value = max(max_value, i)
            min_value = min(min_value, i)
            cdf += hist[i]
            cdf_list[i] = cdf
    for i in range(0, row):
        for j in range(0, col):
            ans[i, j] =  int((cdf_list[img_in[i, j]] - cdf_list[min_value])                             /(row * col - cdf_list[min_value])                             * (0xff - 1)) # 256 for grayscale
            
    return ans


# In[14]:


hist_eq = histeq(div_3, hist_div3)


# In[15]:


#顯示圖片
plt.imshow(hist_eq) 
#存圖
cv2.imwrite("part_c_hist_eq.bmp",div_3)


# In[16]:


hist_eq_hist = histpic(hist_eq)


# In[20]:


#顯示圖片
x=np.arange(0,255)
y=hist_eq_hist[x]
plt.xlabel("Values")
plt.ylabel("Count")
plt.title("lena applying histogram equalization histogram")
plt.bar(x,y)

#存圖
plt.savefig('part_c_hist_eq_hist.png')

