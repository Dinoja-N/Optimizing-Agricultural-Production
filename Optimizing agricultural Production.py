#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd
 


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns
from ipywidgets import interact


# In[10]:


data = pd.read_csv('data.csv')


# In[11]:


print("Shape of the dataset :", data.shape)


# In[12]:


data.head()


# In[1]:


data.isnull().sum()


# In[5]:


import pandas as pd


# In[6]:


data = pd.read_csv('data.csv')


# In[7]:


data.isnull().sum()


# In[1]:


data['label'].value_counts()


# In[2]:


import pandas as pd


# In[3]:


data = pd.read_csv('data.csv')


# In[4]:


data['label'].value_counts()


# In[5]:


print("Average ratio of Nitrogen in the soil : {0:.2f}".format(data['N'].mean()))
print("Average ratio of Phosphorous in the soil : {0:.2f}".format(data['P'].mean()))
print("Average ratio of Pottassium in the soil : {0:.2f}".format(data['K'].mean()))
print("Average ratio of Celsius in the soil : {0:.2f}".format(data['temperature'].mean()))
print("Average ratio of Humidity in the soil : {0:.2f}".format(data['humidity'].mean()))
print("Average ratio of PH value in the soil : {0:.2f}".format(data['ph'].mean()))
print("Average ratio of Rainfall in mm  : {0:.2f}".format(data['rainfall'].mean()))


# In[6]:


@interact
def summary(crops = list(data['label'].value_counts().index)):
	x = data[data['label'] == crops]
	print("                                          ")
	print("------------------------------------------")
	print("Statistics for Nitrogen")
	print("Minimum Nitrogen required : {0:.2f}".format(X['N'].min()))
	print("Average Nitrogen required : {0:.2f}".format(X['N'].mean()))
	print("Maximum Nitrogen required : {0:.2f}".format(X['N'].max()))
	print("------------------------------------------")
	print("Statistics for Phosphorous")
	print("Minimum Phosphorous required : {0:.2f}".format(X['P'].min()))
	print("Average Phosphorous required : {0:.2f}".format(X['P'].mean()))
	print("Maximum Phosphorous required : {0:.2f}".format(X['P'].max()))
	print("------------------------------------------")
	print("Statistics for Pottassium")
	print("Minimum Pottassium required : {0:.2f}".format(X['K'].min()))
	print("Average Pottassium required : {0:.2f}".format(X['K'].mean()))
	print("Maximum Pottassium required :{0:.2f}".format(X['K'].max()))
	print("------------------------------------------")
	print("Statistics for Humidity")
	print("Minimum Humidity required : {0:.2f}".format(X['humidity'].min()))
	print("Average Humidity required : {0:.2f}".format(X['humidity'].mean()))
	print("Maximum Humidity required : {0:.2f}".format(X['humidity'].max()))
	print("------------------------------------------")
	print("Statistics for PH")
	print("Minimum PH required : {0:.2f}".format(X['ph'].min()))
	print("Average PH required : {0:.2f}".format(X['ph'].mean()))
	print("Maximum PH required : {0:.2f}".format(X['ph'].max()))
	print("------------------------------------------")
	print("Statistics for Rainfall")
	print("Minimum Rainfall required : {0:.2f}".format(X['rainfall'].min()))
	print("Average Rainfall required : {0:.2f}".format(X['rainfall'].mean()))
	print("Maximum Rainfall required : {0:.2f}".format(X['rainfall'].max()))
	print("------------------------------------------")


# In[7]:


import seaborn as sns
from ipywidgets import interact


# In[9]:


@interact
def summary(crops = list(data['label'].value_counts().index)):
	X = data[data['label'] == crops]
	print("                                          ")
	print("------------------------------------------")
	print("Statistics for Nitrogen")
	print("Minimum Nitrogen required : {0:.2f}".format(X['N'].min()))
	print("Average Nitrogen required : {0:.2f}".format(X['N'].mean()))
	print("Maximum Nitrogen required : {0:.2f}".format(X['N'].max()))
	print("------------------------------------------")
	print("Statistics for Phosphorous")
	print("Minimum Phosphorous required : {0:.2f}".format(X['P'].min()))
	print("Average Phosphorous required : {0:.2f}".format(X['P'].mean()))
	print("Maximum Phosphorous required : {0:.2f}".format(X['P'].max()))
	print("------------------------------------------")
	print("Statistics for Pottassium")
	print("Minimum Pottassium required : {0:.2f}".format(X['K'].min()))
	print("Average Pottassium required : {0:.2f}".format(X['K'].mean()))
	print("Maximum Pottassium required :{0:.2f}".format(X['K'].max()))
	print("------------------------------------------")
	print("Statistics for Humidity")
	print("Minimum Humidity required : {0:.2f}".format(X['humidity'].min()))
	print("Average Humidity required : {0:.2f}".format(X['humidity'].mean()))
	print("Maximum Humidity required : {0:.2f}".format(X['humidity'].max()))
	print("------------------------------------------")
	print("Statistics for PH")
	print("Minimum PH required : {0:.2f}".format(X['ph'].min()))
	print("Average PH required : {0:.2f}".format(X['ph'].mean()))
	print("Maximum PH required : {0:.2f}".format(X['ph'].max()))
	print("------------------------------------------")
	print("Statistics for Rainfall")
	print("Minimum Rainfall required : {0:.2f}".format(X['rainfall'].min()))
	print("Average Rainfall required : {0:.2f}".format(X['rainfall'].mean()))
	print("Maximum Rainfall required : {0:.2f}".format(X['rainfall'].max()))
	print("------------------------------------------")


# In[12]:


@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
	print("Crops which require greater than average", conditions,'\n')
	print(data[data[conditions] > data[conditions].mean()]['label'].unique())
	print("----------------------------------------------")
	print("Crops which require less than average", conditions,'\n')
	print(data[data[conditions] <= data[conditions].mean()]['label'].unique())


# # Distribution

# In[13]:


plt.rcParams['figure.figsize'] = (15,7)

plt.subplot(2,4,1)
sns.distplot(data['N'],color = 'lightgrey')
plt.xlabel('Ratio of Nitrogen',fontsize = 12)
plt.grid()

plt.subplot(2,4,2)
sns.distplot(data['P'],color = 'skyblue')
plt.xlabel('Ratio of Phosphorous',fontsize = 12)
plt.grid()

plt.subplot(2,4,3)
sns.distplot(data['K'],color = 'darkblue')
plt.xlabel('Ratio of Pottassium',fontsize = 12)
plt.grid()

plt.subplot(2,4,2)
sns.distplot(data['P'],color = 'skyblue')
plt.xlabel('Ratio of Phosphorous',fontsize = 12)
plt.grid()


# In[16]:


import matplotlib.pyplot as plt


# In[17]:


import seaborn as sns
from ipywidgets import interact


# In[31]:


plt.rcParams['figure.figsize'] = (15,7)

plt.subplot(2,4,1)
sns.distplot(data['N'],color = 'lightgrey')
plt.xlabel('Ratio of Nitrogen',fontsize = 12)
plt.grid()

plt.subplot(2,4,2)
sns.distplot(data['P'],color = 'skyblue')
plt.xlabel('Ratio of Phosphorous',fontsize = 12)
plt.grid()

plt.subplot(2,4,3)
sns.distplot(data['K'],color = 'darkblue')
plt.xlabel('Ratio of Pottassium',fontsize = 12)
plt.grid()

plt.subplot(2,4,4)
sns.distplot(data['humidity'],color = 'pink')
plt.xlabel('Humidity',fontsize = 12)
plt.grid()

plt.subplot(2,4,5)
sns.distplot(data['ph'],color = 'green')
plt.xlabel('PH',fontsize = 12)
plt.grid()

plt.subplot(2,4,6)
sns.distplot(data['rainfall'],color = 'orange')
plt.xlabel('Rainfall',fontsize = 12)
plt.grid()

plt.subplot(2,4,7)
sns.distplot(data['temperature'],color = 'black')
plt.xlabel('Temperature',fontsize = 12)
plt.grid()


# In[32]:


print("Some interesting patterns")
print("---------------------------------------")
print("Crops which requires very high Ratio of Nitrogen Content in Soil :", data[data['N'] > 120]['label'].unique())
print("Crops which requires very high Ratio of Phophorous Content in Soil :", data[data['P'] > 100]['label'].unique())
print("Crops which requires very high Ratio of Potassium Content in Soil :", data[data['K'] > 200]['label'].unique())
print("Crops which requires very high Rainfall :", data[data['rainfall'] > 200]['label'].unique())
print("Crops which requires very Low Temperature :", data[data['temperature'] < 10]['label'].unique())
print("Crops which requires very High Temperature :", data[data['temperature'] > 40]['label'].unique())
print("Crops which requires very Low Humidity :", data[data['humidity'] < 20]['label'].unique())
print("Crops which requires very Low pH :", data[data['ph'] < 4]['label'].unique())
print("Crops which requires very high pH :", data[data['ph'] > 9]['label'].unique())


# In[36]:


print("Summer Crops")
print(data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())
print("-------------------------------------------------------------------------------")
print("Winter Crops")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique())
print("-------------------------------------------------------------------------------")
print("Rainy Crops")
print(data[(data['rainfall'] > 200) & (data['humidity'] > 50)]['label'].unique())
print("-------------------------------------------------------------------------------")


# In[42]:


from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

x = data.loc[:, ['N','P','K','temperature','ph','humidity','rainfall']].values

print(x.shape)

x_data = pd.DataFrame(x)
x_data.head()


# In[45]:


plt.rcParams['figure.figsize'] = (10,4)

wcss = []
for i in range(1,11):
	km = KMeans(n_clusters = i, init = 'k-means++' , max_iter = 300, n_init = 10, random_state = 0)
	km.fit(x)
	wcss.append(km.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow method', fontsize = 30)
plt.xlabel('No of Clusters')
plt.ylabel('wcss')
plt.show()



# In[48]:


km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename(columns = {0: 'cluster'})

print("Lets check the results after applying the K means Clustering analysis \n")
print("Crops in First Cluster :", z[z['cluster'] == 0]['label'].unique())
print("---------------------------------------------------------------")
print("Crops in Second Cluster :", z[z['cluster'] == 1]['label'].unique())
print("---------------------------------------------------------------")
print("Crops in Third Cluster :", z[z['cluster'] == 2]['label'].unique())
print("---------------------------------------------------------------")
print("Crops in Fourth Cluster :", z[z['cluster'] == 3]['label'].unique())


# In[49]:


y = data['label']
x = data.drop(['label'], axis = 1)

print("Shape of x:",x.shape)
print("Shape of y:",y.shape)


# In[51]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

print("The Shape of x train :", x_train.shape)
print("The Shape of x test :",x_test.shape)
print("The Shape of y train :", y_train.shape)
print("The Shape of y test :",y_test.shape)




# In[52]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

print("The Shape of x train :", x_train.shape)
print("The Shape of x test :",x_test.shape)
print("The Shape of y train :", y_train.shape)
print("The Shape of y test :",y_test.shape)


# In[53]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[54]:


from sklearn.metrics import classification_report

cr =  classification_report(y_test, y_pred)
print(cr)


# In[55]:


data.head()


# In[56]:


prediction = model.predict((np.array([[90,40,40,20,80,7,200]])))

print("The suggested crop for the given Climatic condition is :", prediction)


# In[57]:


import numpy as np


# In[58]:


prediction = model.predict((np.array([[90,40,40,20,80,7,200]])))

print("The suggested crop for the given Climatic condition is :", prediction)


# In[59]:


data[data['label'] == 'oranges'].head()


# In[60]:


data[data['label'] == 'orange'].head()


# In[62]:


prediction = model.predict((np.array([[20,
                                       30,
                                       10,
                                       15,
                                       90,
                                       7.5,
                                       100]])))

print("The suggested crop for the given Climatic condition is :", prediction)


# In[ ]:




