#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[2]:


#load gene.txt
gene = np.loadtxt("hw3_Data1/gene.txt", dtype=str).T
print(gene)


# In[3]:


gene = gene.astype(np.float)


# In[4]:


#gene.shape


# In[5]:


#load label
label = np.loadtxt("hw3_Data1/label.txt", dtype=str).T


# In[6]:


label = label.astype(np.float)


# In[7]:


#label.shape


# In[8]:


#binarize label
for i in range(len(label)):
    if label[i] > 0:
        label[i] = 1
    else:
        label[i] = -1


# In[9]:


#label


# In[10]:


# define feature selection
fs = SelectKBest(score_func=f_classif, k=10)


# In[11]:


KBest = fs.fit_transform(gene,label)


# In[12]:


#找index position
position = []
for i in range(len(KBest[0])):
    #print(np.where(gene[0] == test_selected[0][i])[0][0])
    position.append(np.where(gene[0] == KBest[0][i])[0][0])
position = np.array(position)
position


# In[13]:


#find what features I pick


# In[14]:


text_file = open("hw3_Data1/index.txt", "r")
lines = text_file.readlines()
#print (lines)
#print (len(lines))
text_file.close()


# In[15]:


lines = np.array(lines)


# In[16]:


print(lines[position])


# In[17]:


f = open('hw3_p1_feature.txt','w')
f.writelines(lines[position])
f.close()

