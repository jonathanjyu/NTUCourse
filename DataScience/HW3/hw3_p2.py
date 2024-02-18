#!/usr/bin/env python
# coding: utf-8

# In[1]:


from genetic_selection import GeneticSelectionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
import pandas as pd
import numpy as np


# In[2]:


#load gene.txt
gene = np.loadtxt("hw3_Data1/gene.txt", dtype=str).T
print(gene)


# In[3]:


gene = gene.astype(np.float)


# In[4]:


#load label
label = np.loadtxt("hw3_Data1/label.txt", dtype=str).T


# In[5]:


label = label.astype(np.float)


# In[6]:


#binarize label
for i in range(len(label)):
    if label[i] > 0:
        label[i] = 1
    else:
        label[i] = -1


# In[7]:


gene = pd.DataFrame(gene)
label = pd.DataFrame(label)


# In[8]:


#objective function
estimator = DecisionTreeClassifier()
#estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")


# In[9]:


model = GeneticSelectionCV(estimator, #A supervised learning estimator with a `fit` method.
                           cv = 5, #Determines the cross-validation splitting strategy.
                           verbose = 0, #Controls verbosity of output.
                           scoring = "accuracy", #a scorer callable object / function with signature
                           max_features = 10, #The maximum number of features selected.
                           n_population = 100, #Number of population for the genetic algorithm.
                           crossover_proba = 0.5, #Probability of crossover for the genetic algorithm.
                           mutation_proba = 0.2, #Probability of mutation for the genetic algorithm.
                           n_generations = 40, #Number of generations for the genetic algorithm.
                           crossover_independent_proba = 0.1, #Independent probability for each attribute to be exchanged, for the genetic algorithm.
                           mutation_independent_proba = 0.05, #Independent probability for each attribute to be mutated, for the genetic algorithm.
                           tournament_size = 3, #Tournament size for the genetic algorithm.
                           n_gen_no_change = 10, #terminate optimization when best individual is not changing in all of the previous ``n_gen_no_change`` number of generations.
                           caching = True, #If True, scores of the genetic algorithm are cached.
                           n_jobs = -1 #Number of cores to run in parallel.
                          )


# In[10]:


fitted = model.fit(gene,label)


# In[11]:


print('Features:',gene.columns[fitted.support_])


# In[12]:


fitted.n_features_,fitted.generation_scores_,fitted.support_,fitted.estimator_


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


# In[18]:


position = np.array(gene.columns[fitted.support_])


# In[19]:


print(lines[position])


# In[20]:


f = open('hw3_p2_feature.txt','w')
f.writelines(lines[position])
f.close()


# In[ ]:




