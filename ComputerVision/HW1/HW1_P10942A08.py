#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np
import cv2
lena = mpimg.imread('lena.bmp') # 讀取和程式碼處於同一目錄下的 lena.bmp
# 此時 lena 就已經是一個 np.array 了，可以對它進行任意處理
print(lena.shape) #(512, 512)
plt.imshow(lena) # 顯示圖片
plt.show()


# In[2]:


print(lena)
print(type(lena))


# ## test

# In[3]:


a = np.array([[1,2,3],[4,5,6]])
print(a)
print(a.shape)


# ## part1_a

# In[4]:


def part1_a(lena,ans):
    for i in range(lena.shape[0]):
        for j in range(lena.shape[1]):
            ans[i][j] = lena[lena.shape[0]-1-i][j]
    return ans


# In[5]:


#part1_a
#宣告答案array
ans = [[0] * lena.shape[1]] * lena.shape[0]
ans = np.array(ans)

#傳入function
ans = part1_a(lena,ans)

#顯示圖片
plt.imshow(ans)
#存圖
cv2.imwrite("part1_a.bmp",ans)


# ## part1_b

# In[6]:


def part1_b(lena,ans):
    for i in range(lena.shape[0]):
        for j in range(lena.shape[1]):
            ans[i][j] = lena[i][len(lena[0])-1-j]
    return ans


# In[7]:


#part1_b
#宣告答案array
ans = [[0] * lena.shape[1]] * lena.shape[0]
ans = np.array(ans)

#傳入function
ans = part1_b(lena,ans)

#顯示圖片
plt.imshow(ans) 
#存圖
cv2.imwrite("part1_b.bmp",ans)


# ## part1_c

# In[8]:


def part1_c(lena,ans):
    for i in range(lena.shape[0]):
        for j in range(lena.shape[1]):
            ans[i][j] = lena[j][i]
    return ans


# In[9]:


#part1_c
#宣告答案array
ans = [[0] * lena.shape[1]] * lena.shape[0]
ans = np.array(ans)

#傳入function
ans = part1_c(lena,ans)

#顯示圖片
plt.imshow(ans) 
#存圖
cv2.imwrite("part1_c.bmp",ans)


# ## part2_d

# In[10]:


def part2_d(lena):
    (h,w) = lena.shape # 讀取圖片大小
    center = (w // 2, h // 2) # 找到圖片中心
    
    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, -45,1)
    
    # 第三個參數變化後的圖片大小
    ans = cv2.warpAffine(lena, M, (w, h))
    return ans


# In[11]:


#傳入function
ans = part2_d(lena)

#顯示圖片
plt.imshow(ans) 
#存圖
cv2.imwrite("part2_d.bmp",ans)


# ## part2_e

# In[12]:


def part2_e(lena):
    
    scale_percent = 50 # 要放大縮小幾%
    width = int(lena.shape[1] * scale_percent / 100) # 縮放後圖片寬度
    height = int(lena.shape[0] * scale_percent / 100) # 縮放後圖片高度
    dim = (width, height) # 圖片形狀 
    ans = cv2.resize(lena, dim, interpolation = cv2.INTER_AREA)  
    
    return ans


# In[13]:


#傳入function
ans = part2_e(lena)

#顯示圖片
plt.imshow(ans) 
#存圖
cv2.imwrite("part2_e.bmp",ans)


# ## part2_f

# In[17]:


def part2_f(lena,ans):
    for i in range(lena.shape[0]):
        for j in range(lena.shape[1]):
            if lena[i][j]>=128:
                ans[i][j] = 255
            else:
                ans[i][j] = 0
    return ans


# In[18]:


#part2_f
#宣告答案array
ans = [[0] * lena.shape[1]] * lena.shape[0]
ans = np.array(ans)

#傳入function
ans = part2_f(lena,ans)
#顯示圖片
plt.imshow(ans) 
plt.show()
#存圖
cv2.imwrite("part2_f.bmp",ans)


# ## check ans

# In[19]:


#重讀圖檔check
test = mpimg.imread('part2_f.bmp')
plt.imshow(test) # 顯示圖片

test.shape


# In[ ]:




