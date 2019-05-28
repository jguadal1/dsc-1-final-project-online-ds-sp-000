#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats


df = pd.read_csv('kc_house_data.csv')


# My goal going into this project is to determine a few things:
# 1. How close can we get to determining the house price? 
# 2. Is there any data that violates the integrity of our model and can we remove it without negatively affecting the other categories?
# 3. What is the clearest way of displaying the results?

# With already having a set of data to work with, my first step according to the OSEMN model of explatory data analysis (which I will be using) is to investigate the data I've been given and to scrub away any faulty information (null values, major outliers, etc...)

# In[3]:


df.info()


# In[148]:


df.head()


# When I originially loaded the data I noticed that sqft_basement was classified as an object. I assumed that the presence of a basement affects the price of a house so I decided it would be helpful to convert the column to numeric.

# In[2]:


df.sqft_basement.replace("?",'0',inplace=True)


# To convert the data from strings to numeric I had to first take care of the '?'s. With the unknown information making up a small portion of the data and the houses with 0 sqft_basement making up much of this data, I decided to replace all of the unknown data with a 0.

# In[3]:


df.sqft_basement = df.sqft_basement.astype('float64')


# The column that is next classified as an object is date. According to the data source, this column describes the date a house was sold. Because I believe that, along with 'id', this information will have no bearing on the price of a house I will go ahead and drop both columns.

# In[4]:


df.drop('date',axis=1,inplace=True)


# In[5]:


df.drop('id',axis=1,inplace=True)


# In[15]:


df.corr()


# One of the first things I am curious about is the following: Without scrubbing any data, what is the correlation between price and the other independent variables? I am curious also with seeing any major correlation between any two variables. Some notable independent variables are: sqft_living, grade, sqft_above, sqft_living15, and bathrooms. They have a decent correlation with price. It's possible that with some additional scrubbing we can the correlation value up. It should also be noted that we are missing a couple of variables in the above table. We are missing sqft_basement and date.

# According to the above info, sqft_basement and date are both dates which lets us know why we may not be seeing them displayed on the correlation table.

# In[16]:


total = df.isnull().sum().sort_values(ascending=False)
total


# According to the above data, there exists 3 variables with null values. I will start by scrubbing each of the variables.

# In[17]:


df.waterfront.unique()


# In[18]:


df.waterfront.value_counts()


# In[6]:


df.waterfront.fillna(0,inplace=True)


# The waterfront column contained 2,376 missing values. This equates to 11 percent of the possible data being a null value. Given that the data doesn't correlate too strong with any other variable, but still wanting to keep the information the rows offer, I will simply replace each null value with a 0.

# In[110]:


df.view.value_counts()


# In[111]:


df.view.unique()


# The 'views' column has a total of 63 null values.The overwhelming amount of houses in this column contain 0 views. Because there is such a small number of null values, the model will not be too affected by the presence of this data, however, I will still convert the null values to 0 since more data is better than less. 

# In[7]:


df.view.fillna(0,inplace=True)


# Finally for the last column with null values which also contains the most.

# In[33]:


df.yr_renovated.value_counts()


# In[81]:


df.yr_renovated.unique()


# In[84]:


df.yr_renovated.describe()


# The 'yr_renovated' column has 3,842 null values. 17,011 values in this column are 0. I don't particularly want to delete 3,842 more rows and the majority of the houses in this set of data haven't been renovated. I also see that this column has very little correlation with all other variables. With these observations being duly noted I will go ahead and replace all the null values in this column with a 0. Replacing it with the mean would make no sense since the mean of this data set is 83.

# In[8]:


df.yr_renovated.fillna(0,inplace=True)


# In[114]:


df.info()


# The data now has no null values to worry about. My next point of interest is seeing how the remaining data relates to our dependent variable, 'price'. To do that I will be using a heatmap courtesy of seaborn.

# In[24]:


correlation = df.corr()
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")


# In[39]:


formula = 'price ~ waterfront + yr_renovated + view + bathrooms + sqft_living + sqft_living15 + grade + bedrooms + floors + condition + lat + zipcode + long + yr_built + sqft_lot + sqft_lot15'
model = ols(formula= formula, data=df).fit()


# In[40]:


model.summary()


# A couple of things to note from the above model summary is the following:
# 1. The r-squared value of our model is .699 which means that the model can explain 69.9 percent of the variation of data. I have yet to normalize the data so I am curious to see what effect that will have.
# 2. Just about all our of variables have a p-value of 0 which means they are significant and relevant for our model.

# In[49]:


df.hist(figsize=[12,12]);


# Much of the data does not follow a normal distriubtion so I will attempt to make the data appear more normal.

# In[9]:


df2 = pd.DataFrame([])


# In[10]:


df2['grade'] = df.grade
df2['baths'] = df.bathrooms
df2['beds'] = df.bedrooms
df2['cond'] = df.condition
df2['floor'] = df.floors
df2['lat'] = df.lat
df2['long'] = df.long
df2['price'] = np.log(df.price)
df2['sqft_above'] = np.log(df.sqft_above)
df2['sqft_basement'] = np.log(df.sqft_basement)
df2['sqft_living'] = np.log(df.sqft_living)
df2['sqft_living15'] = np.log(df.sqft_living15)
df2['sqft_lot'] = np.log(df.sqft_lot)
df2['sqft_lot15'] = np.log(df.sqft_lot15)


# In[18]:


scaled_price = (df2.price - min(df2.price))/(max(df2.price) - min(df2.price))
scaled_sqft_above = (df2.sqft_above - min(df2.sqft_above))/(max(df2.sqft_above) - min(df2.sqft_above))
scaled_sqft_basement = (df.sqft_basement - min(df.sqft_basement))/(max(df.sqft_basement) - min(df.sqft_basement))
scaled_sqft_living = (df2.sqft_living - min(df2.sqft_living))/(max(df2.sqft_living) - min(df2.sqft_living))
scaled_sqft_living15 = (df2.sqft_living15 - min(df2.sqft_living15))/(max(df2.sqft_living15) - min(df2.sqft_living15))
scaled_sqft_lot = (df2.sqft_lot - min(df2.sqft_lot))/(max(df2.sqft_lot) - min(df2.sqft_lot))
scaled_sqft_lot15 = (df2.sqft_lot15 - min(df2.sqft_lot15))/(max(df2.sqft_lot15) - min(df2.sqft_lot15))


# In[12]:


dfscaled = pd.DataFrame([])


# In[19]:


dfscaled['waterfront'] = df.waterfront
dfscaled['view'] = df.view
dfscaled['yr_renovated'] = df.yr_renovated
dfscaled['yr_built'] = df.yr_built
dfscaled['zipcode'] = df.zipcode
dfscaled['grade'] = df.grade
dfscaled['baths'] = df.bathrooms
dfscaled['beds'] = df.bedrooms
dfscaled['cond'] = df.condition
dfscaled['floor'] = df.floors
dfscaled['lat'] = df.lat
dfscaled['long'] = df.long
dfscaled['price'] = scaled_price
dfscaled['sqft_above'] = scaled_sqft_above
dfscaled['sqft_basement'] = scaled_sqft_basement
dfscaled['sqft_living'] = scaled_sqft_living
dfscaled['sqft_living15'] = scaled_sqft_living15
dfscaled['sqft_lot'] = scaled_sqft_lot
dfscaled['sqft_lot15'] = scaled_sqft_lot15


# In[20]:


dfscaled.hist(figsize=[12,12]);


# In[21]:


formula2 = 'price ~ waterfront + view + yr_renovated + yr_built + zipcode + grade + baths + beds + cond + floor + lat + long + sqft_above + sqft_basement + sqft_living + sqft_living15 + sqft_lot + sqft_lot15'


# In[22]:


model2 = ols(formula= formula2, data=dfscaled).fit()


# In[23]:


model2.summary()


# In[24]:


fig = sm.graphics.qqplot(model2.resid, dist=stats.norm, line='45', fit=True)


# In[27]:


plt.scatter(model2.predict(dfscaled), model2.resid)
plt.plot(model2.predict(dfscaled), [0 for i in range(len(df))])


# In[ ]:




