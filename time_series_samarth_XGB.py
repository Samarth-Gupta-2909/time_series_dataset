#!/usr/bin/env python
# coding: utf-8

# # Samarth
# #102083050
# #4CO27

# # Importing Libraries

# In[197]:


import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
import xgboost as xg


# # Importing Dataset

# In[198]:


df = pd.read_excel('DATASET.xlsx')
df.info()


# # Data Preprocessing

# In[199]:


df = df.drop([910, 911, 912, 913, 914, 915, 916, 917, 918])
df


# In[200]:


#Replacong empty cells with zero
df.fillna(0, inplace=True)

col = 0
for row in range(100):
  df.iloc[col:col + 10, 0] = df.iloc[col, 0]
  col += 10
    


# In[201]:


test_data = df.loc[df['year'] == 10]
test_data


# In[ ]:


df.columns = df.columns.str.replace('Unnamed: 0', 'Group')
test_data.columns = test_data.columns.str.replace('Unnamed: 0', 'Group')
df


# # Defining Model

# In[217]:


model = xg.XGBRegressor(n_estimators=500, max_depth=5, eta=0.1, seed=100)


# In[218]:


predicted = {'Para-9': [], 'Para-10': [], 'Para-11': [], 'Para-12': [], 'Para-13': []}
headers = ['Para-9', 'Para-10', 'Para-11', 'Para-12', 'Para-13']


# # Training And Predicting Model

# In[219]:


rmse = []
n = 0
idx = 0
for col in [10, 11, 12, 13, 14]:
  res = 0
  count = 0
  for row in range(0, 1000, 10):
    xtrain = df.iloc[row:row+9, 1:10]
    ytrain = df.iloc[row:row+9, col]
    xtest = df.iloc[row+9, 1:10].to_numpy()
    ytest = df.iloc[row+9, col]
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest.reshape(1,-1))[0]
    predicted[headers[idx]].append(ypred)
    res += (ytest - ypred)**2
    count += 1
    n += 1
    print(res)
  rmse.append(res)
  idx += 1


# # RMSE Calculations
# 

# In[221]:


df_pred = pd.DataFrame.from_dict(predicted)
df_pred
print(rmse)


# In[222]:


for i in rmse:
    val = (i/count)**0.5
    print(val)


# In[223]:


print(rmse)


# In[224]:


print((sum(rmse)/500)**0.5)


# # Plots

# In[225]:


import random
generator = []
for i in range(25):
  generator.append(random.randint(0,100))


# In[226]:


for i in range(5):
  plt.figure(figsize=(5, 5))
  true_value = df_pred.iloc[generator, i]
  predicted_value = test_data.iloc[generator, i + 10]
  plt.scatter(true_value, predicted_value, c='blue')

  p1 = max(max(predicted_value), max(true_value))
  p2 = min(min(predicted_value), min(true_value))
  plt.plot([p1, p2], [p1, p2], 'b-')
  plt.title(headers[i])
  plt.xlabel('True', fontsize=18)
  plt.ylabel('Predicted', fontsize=18)
  plt.axis('equal')
  plt.show()


# In[ ]:





# In[ ]:




