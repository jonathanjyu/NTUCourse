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


# ## part(a) Robert's Operator: 30

# In[2]:


def Roberts_Operator(img,threshold):
    answer = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r1 = 0
            r2 = 0
            if i+1 < img.shape[0] and j+1 < img.shape[1]:
                r1 = int(img[i+1,j+1]) - int(img[i,j])
                r2 = int(img[i+1,j]) - int(img[i,j+1])
            if math.sqrt(r1*r1+r2*r2) >= threshold:
                answer[i,j] = 0
            else:
                answer[i,j] = 255
    return answer


# In[3]:


part_a = Roberts_Operator(lena,30)


# In[4]:


cv2.imwrite("part(a).bmp", part_a)
plt.imshow(part_a)


# ## part(b) Prewitt's Edge Detector: 90

# In[5]:


def padding(img,size):
    answer = cv2.copyMakeBorder(img, size//2, size//2, size//2, size//2, cv2.BORDER_REFLECT)
    return answer


# In[6]:


def Prewitts_Edge_Detector(img,threshold):
    answer = img.copy()
    pad = padding(img,3)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p1 = 0
            p2 = 0
            up = 0
            down = 0
            left = 0
            right = 0
            up = int(pad[i,j])+int(pad[i,j+1])+int(pad[i,j+2])
            down = int(pad[i+2][j])+int(pad[i+2][j+1])+int(pad[i+2][j+2])
            left = int(pad[i,j])+int(pad[i+1,j])+int(pad[i+2,j])
            right = int(pad[i,j+2])+int(pad[i+1,j+2])+int(pad[i+2,j+2])
            p1 = down - up
            p2 = right - left
            if math.sqrt(p1*p1+p2*p2) >= threshold:
                answer[i,j] = 0
            else:
                answer[i,j] = 255
    return answer


# In[7]:


#part_b = Prewitts_Edge_Detector(lena,24)
part_b = Prewitts_Edge_Detector(lena,90)


# In[8]:


cv2.imwrite("part(b).bmp", part_b)
plt.imshow(part_b)


# ## part(c) Sobel's Edge Detector: 120

# In[10]:


def Sobels_Edge_Detector(img,threshold):
    answer = img.copy()
    pad = padding(img,3)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p1 = 0
            p2 = 0
            up = 0
            down = 0
            left = 0
            right = 0
            up = int(pad[i,j])+2*int(pad[i,j+1])+int(pad[i,j+2])
            down = int(pad[i+2][j])+2*int(pad[i+2][j+1])+int(pad[i+2][j+2])
            left = int(pad[i,j])+2*int(pad[i+1,j])+int(pad[i+2,j])
            right = int(pad[i,j+2])+2*int(pad[i+1,j+2])+int(pad[i+2,j+2])
            p1 = down - up
            p2 = right - left
            if math.sqrt(p1*p1+p2*p2) >= threshold:
                answer[i,j] = 0
            else:
                answer[i,j] = 255
    return answer


# In[11]:


#part_c = Sobels_Edge_Detector(lena,38)
part_c = Sobels_Edge_Detector(lena,120)


# In[12]:


cv2.imwrite("part(c).bmp", part_c)
plt.imshow(part_c)


# ## part(d) Frei and Chen's Gradient Operator: 103

# In[13]:


def Frei_and_Chens_Gradient_Operator(img,threshold):
    answer = img.copy()
    pad = padding(img,3)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p1 = 0
            p2 = 0
            up = 0
            down = 0
            left = 0
            right = 0
            up = int(pad[i,j])+math.sqrt(2)*int(pad[i,j+1])+int(pad[i,j+2])
            down = int(pad[i+2][j])+math.sqrt(2)*int(pad[i+2][j+1])+int(pad[i+2][j+2])
            left = int(pad[i,j])+math.sqrt(2)*int(pad[i+1,j])+int(pad[i+2,j])
            right = int(pad[i,j+2])+math.sqrt(2)*int(pad[i+1,j+2])+int(pad[i+2,j+2])
            p1 = down - up
            p2 = right - left
            if math.sqrt(p1*p1+p2*p2) >= threshold:
                answer[i,j] = 0
            else:
                answer[i,j] = 255
    return answer


# In[14]:


#part_d = Frei_and_Chens_Gradient_Operator(lena,30)
part_d = Frei_and_Chens_Gradient_Operator(lena,103)


# In[15]:


cv2.imwrite("part(d).bmp", part_d)
plt.imshow(part_d)


# ## part(e) Kirsch's Compass Operator: 500

# In[16]:


def Kirschs_Compass_Operator(img,threshold):
    answer = img.copy()
    pad = padding(img,3)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k = []
            for l in range(8):
                #print(l)
                if l == 0:
                    k.append(5*(int(pad[i+2,j+2])+int(pad[i+1,j+2])+int(pad[i,j+2]))
                             -3*(int(pad[i,j+1])+int(pad[i,j])+int(pad[i+1,j])+int(pad[i+2,j])+int(pad[i+2,j+1])))
                elif l == 1:
                    k.append(5*(int(pad[i+1,j+2])+int(pad[i,j+2])+int(pad[i,j+1]))
                             -3*(int(pad[i,j])+int(pad[i+1,j])+int(pad[i+2,j])+int(pad[i+2,j+1])+int(pad[i+2,j+2])))
                elif l == 2:
                    k.append(5*(int(pad[i,j+2])+int(pad[i,j+1])+int(pad[i,j]))
                             -3*(int(pad[i+1,j])+int(pad[i+2,j])+int(pad[i+2,j+1])+int(pad[i+2,j+2])+int(pad[i+1,j+2])))
                elif l == 3:
                    k.append(5*(int(pad[i,j+1])+int(pad[i,j])+int(pad[i+1,j]))
                             -3*(int(pad[i+2,j])+int(pad[i+2,j+1])+int(pad[i+2,j+2])+int(pad[i+1,j+2])+int(pad[i,j+2])))
                elif l == 4:
                    k.append(5*(int(pad[i,j])+int(pad[i+1,j])+int(pad[i+2,j]))
                             -3*(int(pad[i+2,j+1])+int(pad[i+2,j+2])+int(pad[i+1,j+2])+int(pad[i,j+2])+int(pad[i,j+1])))
                elif l == 5:
                    k.append(5*(int(pad[i+1,j])+int(pad[i+2,j])+int(pad[i+2,j+1]))
                             -3*(int(pad[i+2,j+2])+int(pad[i+1,j+2])+int(pad[i,j+2])+int(pad[i,j+1])+int(pad[i,j])))
                elif l == 6:
                    k.append(5*(int(pad[i+2,j])+int(pad[i+2,j+1])+int(pad[i+2,j+2]))
                             -3*(int(pad[i+1,j+2])+int(pad[i,j+2])+int(pad[i,j+1])+int(pad[i,j])+int(pad[i+1,j])))
                elif l == 7:
                    k.append(5*(int(pad[i+2,j+1])+int(pad[i+2,j+2])+int(pad[i+1,j+2]))
                             -3*(int(pad[i,j+2])+int(pad[i,j+1])+int(pad[i,j])+int(pad[i+1,j])+int(pad[i+2,j])))
            k = np.array(k)
            if max(k) >= threshold:
                answer[i,j] = 0
            else:
                answer[i,j] = 255            
    return answer


# In[17]:


#part_e = Kirschs_Compass_Operator(lena,135)
part_e = Kirschs_Compass_Operator(lena,500)


# In[18]:


cv2.imwrite("part(e).bmp", part_e)
plt.imshow(part_e)


# ## part(f) Robinson's Compass Operator: 123

# In[19]:


def Robinsons_Compass_Operator(img,threshold):
    answer = img.copy()
    pad = padding(img,3)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = []
            for l in range(8):
                if l == 0:
                    r.append(int(pad[i+2,j+2])+2*int(pad[i+1,j+2])+int(pad[i,j+2])
                             -int(pad[i,j])-2*int(pad[i+1,j])-int(pad[i+2,j]))
                elif l == 1:
                    r.append(int(pad[i+1,j+2])+2*int(pad[i,j+2])+int(pad[i,j+1])
                             -int(pad[i+1,j])-2*int(pad[i+2,j])-int(pad[i+2,j+1]))
                elif l == 2:
                    r.append(int(pad[i,j])+2*int(pad[i,j+1])+int(pad[i,j+2])
                             -int(pad[i+2,j])-2*int(pad[i+2,j+1])-int(pad[i+2,j+2]))
                elif l == 3:
                    r.append(int(pad[i,j+1])+2*int(pad[i,j])+int(pad[i+1,j])
                             -int(pad[i+2,j+1])-2*int(pad[i+2,j+2])-int(pad[i+1,j+2]))
                elif l == 4:
                    r.append(int(pad[i,j])+2*int(pad[i+1,j])+int(pad[i+2,j])
                             -int(pad[i,j+2])-2*int(pad[i+1,j+2])-int(pad[i+2,j+2]))
                elif l == 5:
                    r.append(int(pad[i+1,j])+2*int(pad[i+2,j])+int(pad[i+2,j+1])
                             -int(pad[i,j+1])-2*int(pad[i,j+2])-int(pad[i+1,j+2]))
                elif l == 6:
                    r.append(int(pad[i+2,j])+2*int(pad[i+2,j+1])+int(pad[i+2,j+2])
                             -int(pad[i,j])-2*int(pad[i,j+1])-int(pad[i,j+2]))
                elif l == 7:
                    r.append(int(pad[i+2,j+1])+2*int(pad[i+2,j+2])+int(pad[i+1,j+2])
                             -int(pad[i,j+1])-2*int(pad[i,j])-int(pad[i+1,j]))
            r = np.array(r)
            if max(r) >= threshold:
                answer[i,j] = 0
            else:
                answer[i,j] = 255            
    return answer


# In[20]:


#part_f = Robinsons_Compass_Operator(lena,43)
part_f = Robinsons_Compass_Operator(lena,123)


# In[21]:


cv2.imwrite("part(f).bmp", part_f)
plt.imshow(part_f)


# ## part(g) Nevatia-Babu 5x5 Operator: 32100

# In[22]:


def Nevatia_Babu5X5_Operator(img,threshold):
    answer = img.copy()
    pad = padding(img,5)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            N = []
            for l in range(6):
                if l == 0:
                    N.append(100*(int(pad[i,j])+int(pad[i,j+1])+int(pad[i,j+2])+int(pad[i,j+3])+int(pad[i,j+4])
                                 +int(pad[i+1,j])+int(pad[i+1,j+1])+int(pad[i+1,j+2])+int(pad[i+1,j+3])+int(pad[i+1,j+4]))
                             -100*(int(pad[i+3,j])+int(pad[i+3,j+1])+int(pad[i+3,j+2])+int(pad[i+3,j+3])+int(pad[i+3,j+4])
                                 +int(pad[i+4,j])+int(pad[i+4,j+1])+int(pad[i+4,j+2])+int(pad[i+4,j+3])+int(pad[i+4,j+4]))
                    )
                elif l == 1:
                    N.append(100*(int(pad[i,j])+int(pad[i,j+1])+int(pad[i,j+2])+int(pad[i,j+3])+int(pad[i,j+4])
                                  +int(pad[i+1,j])+int(pad[i+1,j+1])+int(pad[i+1,j+2])+int(pad[i+2,j]))
                             +92*(int(pad[i+2,j+1]))-92*(int(pad[i+2,j+3]))
                             +78*(int(pad[i+1,j+3]))-78*(int(pad[i+3,j+1]))
                             +32*(int(pad[i+3,j]))-32*(int(pad[i+1,j+4]))
                             -100*(int(pad[i+2,j+4])+int(pad[i+3,j+2])+int(pad[i+3,j+3])+int(pad[i+3,j+4])
                                  +int(pad[i+4,j])+int(pad[i+4,j+1])+int(pad[i+4,j+2])+int(pad[i+4,j+3])+int(pad[i+4,j+4]))
                    )
                elif l == 2:
                    N.append(100*(int(pad[i,j])+int(pad[i,j+1])+int(pad[i,j+2])+int(pad[i+1,j])+int(pad[i+1,j+1])
                                  +int(pad[i+2,j])+int(pad[i+2,j+1])+int(pad[i+3,j])+int(pad[i+4,j]))
                             +92*(int(pad[i+1,j+2]))-92*(int(pad[i+3,j+2]))
                             +78*(int(pad[i+3,j+1]))-78*(int(pad[i+1,j+3]))
                             +32*(int(pad[i,j+3]))-32*(int(pad[i+1,j+4]))
                             -100*(int(pad[i,j+4])+int(pad[i+1,j+4])+int(pad[i+2,j+3])+int(pad[i+2,j+4])+int(pad[i+3,j+3])
                                  +int(pad[i+3,j+4])+int(pad[i+4,j+2])+int(pad[i+4,j+3])+int(pad[i+4,j+4]))
                    )
                elif l == 3:
                    N.append(100*(int(pad[i,j+3])+int(pad[i+1,j+3])+int(pad[i+2,j+3])+int(pad[i+3,j+3])+int(pad[i+4,j+3])
                                 +int(pad[i,j+4])+int(pad[i+1,j+4])+int(pad[i+2,j+4])+int(pad[i+3,j+4])+int(pad[i+4,j+4]))
                             -100*(int(pad[i,j])+int(pad[i+1,j])+int(pad[i+2,j])+int(pad[i+3,j])+int(pad[i+4,j])
                                   +int(pad[i,j+1])+int(pad[i+1,j+1])+int(pad[i+2,j+1])+int(pad[i+3,j+1])+int(pad[i+4,j+1]))
                    )
                elif l == 4:
                    N.append(100*(int(pad[i,j+2])+int(pad[i,j+3])+int(pad[i,j+4])+int(pad[i+1,j+3])+int(pad[i+1,j+4])
                                 +int(pad[i+2,j+3])+int(pad[i+2,j+4])+int(pad[i+3,j+4])+int(pad[i+4,j+4]))
                             +92*(int(pad[i+1,j+2]))-92*(int(pad[i+3,j+2]))
                             +78*(int(pad[i+3,j+3]))-78*(int(pad[i+1,j+1]))
                             +32*(int(pad[i,j+1]))-32*(int(pad[i+4,j+3]))
                             -100*(int(pad[i,j])+int(pad[i+1,j])+int(pad[i+2,j])+int(pad[i+3,j])+int(pad[i+4,j])
                                  +int(pad[i+2,j+1])+int(pad[i+3,j+1])+int(pad[i+4,j+1])+int(pad[i+4,j+2]))
                    )
                elif l == 5:
                    N.append(100*(int(pad[i,j])+int(pad[i,j+1])+int(pad[i,j+2])+int(pad[i,j+3])+int(pad[i,j+4])
                                 +int(pad[i+1,j+2])+int(pad[i+1,j+3])+int(pad[i+1,j+4])+int(pad[i+1,j+4]))
                             +92*(int(pad[i+2,j+3]))-92*(int(pad[i+2,j+1]))
                             +78*(int(pad[i+1,j+1]))-78*(int(pad[i+3,j+3]))
                             +32*(int(pad[i+3,j+4]))-32*(int(pad[i+1,j]))
                             -100*(int(pad[i+2,j])+int(pad[i+3,j])+int(pad[i+3,j+1])+int(pad[i+3,j+2])+int(pad[i+4,j])
                                  +int(pad[i+4,j+1])+int(pad[i+4,j+2])+int(pad[i+4,j+3])+int(pad[i+4,j+4]))
                    )
            N = np.array(N)
            if max(N) >= threshold:
                answer[i,j] = 0
            else:
                answer[i,j] = 255            
    return answer


# In[23]:


#part_g = Nevatia_Babu5X5_Operator(lena,12500)
part_g = Nevatia_Babu5X5_Operator(lena,32100)


# In[24]:


cv2.imwrite("part(g).bmp", part_g)
plt.imshow(part_g)


# In[ ]:




