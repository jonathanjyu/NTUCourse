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


# ## part_a

# In[2]:


def part_a(lena,ans):
    for i in range(lena.shape[0]):
        for j in range(lena.shape[1]):
            if lena[i][j]>=128:
                ans[i][j] = 255
            else:
                ans[i][j] = 0
    return ans


# In[3]:


#part_a
#宣告答案array
ans = [[0] * lena.shape[1]] * lena.shape[0]
ans = np.array(ans)

#傳入function
ans = part_a(lena,ans)
#顯示圖片
plt.imshow(ans) 
#存圖
cv2.imwrite("part_a.bmp",ans)


# ## part_b

# In[4]:


lena_b = lena.reshape(lena.shape[0]*lena.shape[1],)
print(lena_b)


# In[5]:


hist = np.zeros(256,dtype = int)
for i in lena_b:
    for j in range(256):
        if i == j:
            hist[j] = hist[j] + 1


# In[6]:


#顯示圖片
x=np.arange(0,255)
y=hist[x]
plt.xlabel("Values")
plt.ylabel("Count")
plt.title("lena pixel histogram")
plt.bar(x,y)

#存圖
plt.savefig('part_b.png')


# ## part_c

# In[4]:


lena=cv2.imread("lena.bmp")
height = lena.shape[0]
width = lena.shape[1]
tmp = 0
label = np.zeros((width, height),np.int)
CHANGE = True
# initial
for i in range(width):
        for j in range(height):
                if ans[i,j] == 255:
                        tmp += 1
                        label[i][j] = tmp
#print(label)
# Iterative Algorithm
while CHANGE == True:
        CHANGE = False
        for i in range(width):
                for j in range(height):
                        if label[i][j] != 0:
                                if i > 0:
                                        if label[i-1][j] != 0 and label[i-1][j] < label[i][j]:
                                                label[i][j] = label[i-1][j]
                                                CHANGE = True
                                if j > 0:
                                        if label[i][j-1] != 0 and label[i][j-1] < label[i][j]:
                                                label[i][j] = label[i][j-1]
                                                CHANGE = True
                                if i < width-1:
                                        if label[i+1][j] != 0 and label[i+1][j] < label[i][j]:
                                                label[i][j] = label[i+1][j]
                                                CHANGE = True
                                if j < height-1:
                                        if label[i][j+1] != 0 and label[i][j+1] < label[i][j]:
                                                label[i][j] = label[i][j+1]
                                                CHANGE = True
        for i in range(width-1,-1,-1):
                for j in range(height-1,-1,-1):
                        if label[i][j] != 0:
                                if i > 0:
                                        if label[i-1][j] != 0 and label[i-1][j] < label[i][j]:
                                                label[i][j] = label[i-1][j]
                                                CHANGE = True
                                if j > 0:
                                        if label[i][j-1] != 0 and label[i][j-1] < label[i][j]:
                                                label[i][j] = label[i][j-1]
                                                CHANGE = True
                                if i < width-1:
                                        if label[i+1][j] != 0 and label[i+1][j] < label[i][j]:
                                                label[i][j] = label[i+1][j]
                                                CHANGE = True
                                if j < height-1:
                                        if label[i][j+1] != 0 and label[i][j+1] < label[i][j]:
                                                label[i][j] = label[i][j+1]
                                                CHANGE = True
# declare zero matrix for connected component
cc = np.zeros(np.max(label)+1, np.int)
# count number of same label
for i in range(width):
        for j in range(height):
                if label[i][j] > 0:
                        cc[label[i][j]] += 1
# Omit regions that have pixel count less than 500
for i in range(width):
        for j in range(height):
                if cc[label[i][j]] < 500:
                        label[i][j] = 0


# In[14]:


# bounding box
connected_component = cv2.imread("part_a.bmp")
for i in range(1, np.max(label)+1):
        if cc[i] >= 500:
                element_set = np.array(np.where(label==i)).T
                a = element_set.shape[0]
                r=0
                c=0
                for i in range(element_set.shape[0]):
                        r = r + element_set[i][1]
                        c = c + element_set[i][0]
                r=int(r/a)
                c=int(c/a)
                # region with "+" at centroid
                cv2.rectangle(connected_component, (r-7, c), (r+7, c), (0, 0, 255), -1)
                cv2.rectangle(connected_component, (r, c-7), (r, c+7), (0, 0, 255), -1)
                start = np.array([np.min(element_set[:,0]),np.min(element_set[:,1])])
                end = np.array([np.max(element_set[:,0]),np.max(element_set[:,1])])
                # draw a bounding box
                cv2.rectangle(connected_component, (start[1],start[0]), (end[1],end[0]), (255,0,0), 2)
                

cv2.imwrite('part_c.png', connected_component)


# In[ ]:




