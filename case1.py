#!/usr/bin/env python
# coding: utf-8

# In[226]:


import numpy as np
import pandas as pd
import altair as alt
pd.set_option('display.max_columns',None) 
alt.data_transformers.disable_max_rows()

from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[3]:


df = pd.read_csv('loans_full_schema.csv')
df.sample(5)


# # Description

# In[6]:


df.shape


# We can see that there are 10000 loans and 55 features in the dataset.

# In[7]:


df.info()


# Among those types of features, 17 are float, 25 are int and 13 are object. More statistical values are as follow.

# In[8]:


df.describe()


# Now we want to see the portion of empty values.

# In[11]:


100 * df.isnull( ).sum( ) / len(df)


# We can see that, `annual_income_joint`, `verification_income_joint`, and `debt_to_income_joint` have more than 85% null values, perhaps because most people do not own a joint account. Also, `months_since_90d_late` has 77% null values and `months_since_last_delinq` has more than 56% null values, since a large portion of people never experience a delinquency or 90 days late pay. At last, `months_since_last_credit_inquiry` with 12% null values, `emp_title` and `emp_length` with 8% null values, `num_accounts_120d_past_due` with 3%null values, and `debt_to_income` with 0.24 null values, these features may not be applicable for some people.

# Overall, the dataset is quite clean. We only need to delete the columns with null values or fill them with suitable values.

# In[141]:


# df['issue_month']=pd.to_datetime(df.issue_month.str.upper(), format='%b-%y', yearfirst=False)
df['earliest_credit_line'].unique()


# # Visualization (with interaction)

# ## Vis 1 & 2: Credit among Different Customers

# In[134]:


selection=alt.selection_interval()

scatter = alt.Chart(df).mark_point(filled=True).encode(
    alt.X('total_credit_limit:Q', title='total_credit_limit'),
    alt.Y("annual_income:Q"),
    alt.Color('interest_rate:Q', scale=alt.Scale(type='log',range=['green','red']))
).add_selection(
    selection
).properties(height=300, width=500,
    title='Distribution of Income, Credit, and Interest Rate'
)

bars = alt.Chart(df).transform_filter(
    selection
).transform_aggregate(
    groupby=['grade', 'homeownership'],
    count='count()'
).mark_bar().encode(
    alt.Y('grade:N'),
    alt.X('count:Q'),
    alt.Color('homeownership:N', sort=['RENT', 'OWN', 'MORTGAGE']),
    alt.Order('homeownership:N', sort='ascending'),
    tooltip='count:Q'
).interactive(
).properties(height=300, width=500,
    title='Distribution of Home Ownership among Grades'
)

scatter & bars


# * User guide <br>
# These two graphs support interaction, and they are linked together. In the first plot, if you select an area with mouse, then the dots in the rectangular space will become the data source of the second plot. You can drag the rectangle to select different space. You can also scroll to enlarge or reduce the space. Please single click to cancel selection.
# In the scond graph, you can scroll to zoom in and out, and drag to view different parts of the graph. Double click to return to the default view. Hover above the bar to view the exact count.

# * Findings <br>
# Overall, the majority of customers have annual income less tham 200000 dollars. Customers having higher income tend to have a higher credit limit. The majority of customers have annual income below 200000 dollars and credit limit below 800000 dollars. Interset rate does not have an obvious correlation with annual income or total credit limit.<br>
# Generally, about a half of customers have a mortgage of their home. Three eighths of the customers rent their home, and about one eighth own their home. B grade has the most people, then come to C grade and A grade.<br>
# There is one interesting finding. If I select the region with annual income $200000 or above, then I surprisingly find that A grade has the most customers, then B and C grade. The same happens when I select a range of high credit limit. Also, more than 2/3 of them has a mortgage or own their home. Less people rent their house. We can conclude that customers with higher annual income or total credit limit tend to have better credit grade and buy their home instead of renting it.

# ## Vis 3: Different subgrades

# In[149]:


base = alt.Chart(df).encode(
    alt.X("sub_grade:N")
)

line = base.mark_line(size=2.5, color='pink', point=True).transform_aggregate(
    groupby=['sub_grade'],
    avg_rate='average(interest_rate):Q'
).encode(
    alt.Y("avg_rate:Q", axis=alt.Axis(title='Average interest rate')),
    tooltip=['avg_rate:Q']
).interactive()

bar = base.mark_bar().transform_aggregate(
    groupby=['sub_grade'],
    avg_in='average(annual_income):Q',
    avg_dti='average(debt_to_income):Q'
).encode(
    alt.Y("avg_in:Q", axis=alt.Axis(title='Average annual income')),
    tooltip=['avg_in:Q', 'avg_dti:Q'],
    color = alt.Color('avg_dti:Q', 
        scale=alt.Scale(type='log',range=['green','red']),
        title='Average debt to income')
).properties(height=300, width=700,
    title='Stats about Grades'
).interactive()

alt.layer(bar, line).resolve_scale(
    y = 'independent'
)


# * User guide <br>
# This graph supports interaction. You can scroll to zoom in and out, and drag to view different parts of the graph. Double click to return to the default view. Hover above the bar to view the detailed annual income and debt to income. Hover above the line to view interest rate.

# * Findings <br>
# It is obvious that higher grade leads to lower interest rate. A1 grade has excessively high average income. Generally, for grade over D4, grade is directly proportional to income, but that is not the case for grades under D5. The debt to income ratio is negatively proportional to grades.F4 and F5 have the highest debt to income ratio.

# ## Vis 4: Purpose of loans

# In[148]:


alt.Chart(df).mark_rect().encode(
    alt.X('loan_amount:Q', bin=alt.Bin(maxbins=60)),
    alt.Y('loan_purpose:N'),
    alt.Color('count():Q', scale=alt.Scale(type='log', scheme='greenblue'))
).properties(height=300, width=700,
    title='Purpose of Loans'
)


# * Findings <br>
# The most popular purpose of loan is debit consolidation, followed by credit card, other, and home improvement. Some purpose like vacation and moving mainly focus on small amount, while others tend to have higher loan amount.

# ## Vis 5: Bankruptcy and Loan Amount

# In[165]:


alt.Chart(df).mark_boxplot(extent='min-max').encode(
    x='public_record_bankrupt:O',
    y='loan_amount:Q'
).properties(height=400, width=200,
    title='Bankruptcy and Loan Amount'
)


# * Findings <br>
# We can find that, as the bankrupt record increases, people tend to get less loan amount.

# # Interest Prediction

# Considering that columns with empty values is not so relative to interest rate, in order to reduce error, we drop columns with empty values. Also, we seperate the features to numerical and categorical.

# In[166]:


df1 = df.dropna(axis=1)


# In[169]:


interest = df1.pop('interest_rate')


# In[180]:


numerical = df1.select_dtypes(include=['int64','float64'])

categorical = df1.select_dtypes(include="object")
categorical_dum = pd.get_dummies(categorical)


# In[182]:


train = pd.concat([numerical, categorical_dum],axis=1)
train.shape


# In[195]:


X_num = numerical.values
X_cat = categorical_dum.values
Y = interest.values.astype('int')


# We use chi2 test to figure out which features are most correlated with interest rate. We will select 25 numerical and 25 categorical features.

# In[207]:


Selector1 = SelectKBest(chi2, k=25)
X_num_sel = Selector1.fit_transform(X_num, Y)
numerical_sel = numerical.columns.values[Selector1.get_support()]
numerical_sel


# In[208]:


Selector2 = SelectKBest(chi2, k=25)
X_cat_sel = Selector2.fit_transform(X_cat, Y)
categorical_sel = categorical_dum.columns.values[Selector2.get_support()]
categorical_sel


# Above are the feature set I generate. Next I will predict interset rate with linear regression, gradient boosting and random forest.

# In[209]:


X = np.concatenate([X_num_sel, X_cat_sel], axis = 1)
X.shape


# In[211]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# In[216]:


lr = LinearRegression().fit(X_train, y_train)
lr.score(X_test, y_test)


# In[228]:


params = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_split": 10,
    "learning_rate": 0.1
}

gb = GradientBoostingRegressor(**params).fit(X_train, y_train)
gb.score(X_test, y_test)


# In[223]:


rf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
rf.score(X_test, y_test)


# We can see that gradient boosting has the best accuracy, then followed by random forest, and linear regression has the lowest accuracy.

# In[237]:


prediction_lr = lr.predict(X_train)
prediction_gb = gb.predict(X_train)
prediction_rf = rf.predict(X_train)


# In[250]:


data_comp = {'Actual Values': y_train, 'lr': prediction_lr, 'gb': prediction_gb, 'rf': prediction_rf}
df_comp = pd.DataFrame(data=data_comp)


# In[256]:


df_comp


# In[264]:


alt.Chart(df_comp).transform_fold(
    ['Actual Values', 'lr', 'gb', 'rf'],
    as_=['Experiment', 'Measurement']
).mark_bar(
    opacity=0.3,
    binSpacing=0
).encode(
    alt.X('Measurement:Q', bin=alt.Bin(maxbins=50)),
    alt.Y('count()', stack=None),
    alt.Color('Experiment:N')
)


# From the plot above, we can see that linear regression has the lowest accuracy and random forest has the highest accuracy.

# In[ ]:




