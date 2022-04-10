#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None) 


# In[4]:


df = pd.read_csv('casestudy.csv')
df.sample(5)


# In[60]:


df.describe()


# We can see that, there are 685927 rows in the dataset which are distributed in year 2015, 2016 and 2017. In order to make the answer logically readable, they are not presented in the order the same as questions.

# ## Total revenue

# In[67]:


total_rev = df.groupby('year').sum().drop(['Unnamed: 0'], axis=1)
total_rev.columns = ['total_revenue']
total_rev


# ## Total Customers Current Year

# In[68]:


total_cus = df[['year', 'Unnamed: 0']].drop_duplicates(subset=['Unnamed: 0']).groupby('year').count()
total_cus.columns = ['total_customers']
total_cus


# In[27]:


# df[['year', 'Unnamed: 0']].groupby('year').count()
# There are no duplicates


# ## Total Customers Previous Year

# In[69]:


pre_cus = df[['year', 'Unnamed: 0']].drop_duplicates(subset=['Unnamed: 0']).groupby('year').count().shift(1)
pre_cus.columns = ['previous_customers']
pre_cus


# ## New Customers & Revenue
# Here, we define a new customer as a customer that not appears in the last year only.

# In[53]:


new_cus = pd.DataFrame(data = None,columns = ['new_customers', 'new_revenue'])
for i in df['year'].drop_duplicates():
    if i == 2015:
        new_num = len(df[df.year == i])
        new_rev = df[df.year == 2015]['net_revenue'].sum()
        new_cus.loc[i, 'new_customers'] = new_num
        new_cus.loc[i, 'new_revenue'] = new_rev
    else:
        previous = df.loc[df.year == i-1]
        current = df.loc[df.year == i]
        new_num = len(current.loc[current.customer_email.isin(previous.customer_email) == False])
        new_rev = current.loc[current.customer_email.isin(previous.customer_email) == False]['net_revenue'].sum()
        new_cus.loc[i, 'new_customers'] = new_num
        new_cus.loc[i, 'new_revenue'] = new_rev
        
new_cus.index.name='year'
new_cus


# ## Lost Customers & Revenue Lost from Attrition
# Here, we define a lost customer as a customer that appears in the last year but not in this year.

# In[51]:


lost_cus = pd.DataFrame(data = None,columns = ['lost_customers', 'lost_revenue'])
for i in df['year'].drop_duplicates():
    if i == 2015:
        lost_cus.loc[i, 'lost_customers'] = None
        lost_cus.loc[i, 'lost_revenue'] = None
    else:
        previous = df.loc[df.year == i-1]
        current = df.loc[df.year == i]
        lost_num = len(previous.loc[previous.customer_email.isin(current.customer_email) == False])
        lost_rev = previous.loc[previous.customer_email.isin(current.customer_email) == False]['net_revenue'].sum()
        lost_cus.loc[i, 'lost_customers'] = lost_num
        lost_cus.loc[i, 'lost_revenue'] = lost_rev
        
lost_cus.index.name = 'year'
lost_cus


# ## Existing Customer Revenue & Growth
# Here, we define an existing customer as a customer that appears both in the last year and in this year.

# In[117]:


exi_cus = pd.DataFrame(data = None,columns = ['existing_customers', 'existing_revenue', 'prior_revenue', 'growth'])
for i in df['year'].drop_duplicates():
    if i == 2015:
        exi_num = len(df[df.year == i])
        exi_rev = df[df.year == 2015]['net_revenue'].sum()
        exi_cus.loc[i, 'existing_customers'] = 0
        exi_cus.loc[i, 'existing_revenue'] = 0
        exi_cus.loc[i, 'prior_revenue'] = 0
        exi_cus.loc[i, 'growth'] = 0
    else:
        previous = df.loc[df.year == i-1]
        current = df.loc[df.year == i]
        exi_cur = current.loc[current.customer_email.isin(previous.customer_email) == True]
        exi_pri = previous.loc[previous.customer_email.isin(current.customer_email) == True]
        exi_num = len(exi_cur)
        exi_rev = exi_cur['net_revenue'].sum()
        pri_rev = exi_pri['net_revenue'].sum()
        exi_cus.loc[i, 'existing_customers'] = exi_num
        exi_cus.loc[i, 'existing_revenue'] = exi_rev
        exi_cus.loc[i, 'prior_revenue'] = pri_rev
        exi_cus.loc[i, 'growth'] = exi_rev - pri_rev
        
exi_cus.index.name='year'
exi_cus


# Now we join up all results into one data frame.

# In[118]:


res = pd.concat([total_rev, total_cus, pre_cus, new_cus, lost_cus, exi_cus], axis = 1)
res


# In[119]:


fig=plt.figure('Revenue')
labels = ['2015', '2016', '2017']
width=1
fig, ax1=plt.subplots()
ax1.bar(labels, res.new_revenue, width, color='red', edgecolor='white', label='new')
ax1.bar(labels, res.existing_revenue, width, bottom=res.new_revenue, color='blue', edgecolor='white', label='existing')
plt.title('Revenue in each year')
plt.xlabel('year')
plt.ylabel('revenue')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
# plt.show()


# From the plot, we can see that total revenue in 2017 is the highest and that in 2016 is the lowest. Most of the revenue comes from new customers.

# In[135]:


fig=plt.figure('Revenue')
labels = ['2015', '2016', '2017']
fig, ax1=plt.subplots()
ax1.plot(labels, res.total_revenue/res.total_customers, marker='o', label='total')
ax1.plot(labels, res.new_revenue/res.new_customers, marker='o', label='new')
ax1.plot(labels, res.lost_revenue/res.lost_customers, marker='o', label='lost')
plt.title('Average revenue')
plt.ylim((125, 126))
plt.xlabel('year')
plt.ylabel('revenue')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
# plt.show()


# We can see that the average revenue remains stable regardless of year and customer type.

# In[ ]:




