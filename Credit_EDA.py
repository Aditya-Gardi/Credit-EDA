#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_columns", None)


# In[3]:


app_data = pd.read_csv("application_data.csv")
app_data.head()


# In[4]:


#Data Inspection on application dataset
#Get info & shape on the dataset
app_data.info()


# In[5]:


app_data.describe()


# In[6]:


#data quality check
#check for percentage null values in application dataset
pd.set_option("display.max_rows", 200)
app_data.isnull().mean()*100

# -Conclusion: Columns with null values more than 47% may give wrong insights, hence will drop them
# In[7]:


# Dropping columns with missing values greater than 47%
percentage = 47
threshold = int(((100-percentage)/100)*app_data.shape[0] + 1)
app_df = app_data.dropna(axis=1, thresh=threshold)
app_df.head()


# In[8]:


app_df.shape


# In[9]:


app_df.isnull().mean()*100


# ### Impute missing values
# #### check missing values in application  dataset before imputing

# In[10]:


app_df.info()


# #OCCUPATION_TYPE column has 31% missing values, since its a categorical column, imputing the missing values with unknown or others value

# In[11]:


app_df.OCCUPATION_TYPE.isnull().mean()*100


# In[12]:


app_df.OCCUPATION_TYPE.value_counts(normalize=True)*100


# In[13]:


app_df.OCCUPATION_TYPE.fillna("Others", inplace=True)


# In[14]:


app_df.OCCUPATION_TYPE.isnull().mean()*100


# In[15]:


app_df.OCCUPATION_TYPE.value_counts(normalize=True)*100


# ### EXT_Source_3 Column has 19% missing values

# In[16]:


app_df.EXT_SOURCE_3.isnull().mean()*100


# In[17]:


app_df.EXT_SOURCE_3.value_counts(normalize=True)*100


# In[18]:


app_df.EXT_SOURCE_3.describe()


# In[19]:


sns.boxplot(app_df.EXT_SOURCE_3)
plt.show()


# - Conclusion: Since its a numerical columns with no outliers and there is not much difference between Mean and median, Hence we can impute with mean or median

# In[20]:


app_df.EXT_SOURCE_3.fillna(app_df.EXT_SOURCE_3.median(), inplace=True)


# In[21]:


app_df.EXT_SOURCE_3.isnull().mean()*100


# In[22]:


app_df.EXT_SOURCE_3.value_counts(normalize=True)*100


# In[23]:


null_cols = list(app_df.columns[app_df.isna().any()])
len(null_cols)


# In[24]:


app_df.isnull().mean()*100


# ### Handling Missing values in Columns with 13% null values

# In[25]:


app_df.AMT_REQ_CREDIT_BUREAU_HOUR.value_counts(normalize=True)*100


# - Conclusion: We could see that 99% of values in the column AMT_REQ_CREDIT_BUREAU_HOUR, AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_WEEK, AMT_REQ_CREDIT_BUREAU_MON, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_YEAR is 0.0.
# Hence impute these columns with mode
# 

# In[26]:


Cols = ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON" , "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]


# In[27]:


for col in Cols:
    app_df[col].fillna(app_df[col].mode()[0], inplace=True)


# In[28]:


app_df.isnull().mean()*100


# ## Handling Missing values less than 1%

# In[29]:


null_cols = list(app_df.columns[app_df.isna().any()])
len(null_cols)


# In[30]:


app_df.NAME_TYPE_SUITE.value_counts(normalize=True)*100


# - Conclusion:
#     - For categorical columns impute the missing values with mode
#     - For numerical columns, impute missing values with median

# In[31]:


app_df.NAME_TYPE_SUITE.fillna(app_df.NAME_TYPE_SUITE.mode()[0], inplace=True)


# In[32]:


app_df.CNT_FAM_MEMBERS.fillna(app_df.CNT_FAM_MEMBERS.mode()[0], inplace=True)


# In[33]:


# imputing Numerical Columns
app_df.EXT_SOURCE_2.fillna(app_df.EXT_SOURCE_2.median(), inplace=True)
app_df.AMT_GOODS_PRICE.fillna(app_df.AMT_GOODS_PRICE.median(), inplace=True)
app_df.AMT_ANNUITY.fillna(app_df.AMT_ANNUITY.median(), inplace=True)
app_df.DEF_60_CNT_SOCIAL_CIRCLE.fillna(app_df.DEF_60_CNT_SOCIAL_CIRCLE.median(), inplace=True)
app_df.DEF_30_CNT_SOCIAL_CIRCLE.fillna(app_df.DEF_30_CNT_SOCIAL_CIRCLE.median(), inplace=True)
app_df.OBS_30_CNT_SOCIAL_CIRCLE.fillna(app_df.OBS_30_CNT_SOCIAL_CIRCLE.median(), inplace=True)
app_df.OBS_60_CNT_SOCIAL_CIRCLE.fillna(app_df.OBS_60_CNT_SOCIAL_CIRCLE.median(), inplace=True)
app_df.DAYS_LAST_PHONE_CHANGE.fillna(app_df.DAYS_LAST_PHONE_CHANGE.median(), inplace=True)


# In[34]:


null_cols = list(app_df.columns[app_df.isna().any()])
len(null_cols)


# In[35]:


app_df.isnull().mean()*100


# #### Convert negative values to positive values in days variable so that median is not affected

# In[36]:


app_df.DAYS_BIRTH = app_df.DAYS_BIRTH.apply(lambda x: abs(x))
app_df.DAYS_EMPLOYED= app_df.DAYS_EMPLOYED.apply(lambda x: abs(x))
app_df.DAYS_REGISTRATION = app_df.DAYS_REGISTRATION.apply(lambda x: abs(x))
app_df.DAYS_ID_PUBLISH = app_df.DAYS_ID_PUBLISH.apply(lambda x: abs(x))
app_df.DAYS_LAST_PHONE_CHANGE = app_df.DAYS_LAST_PHONE_CHANGE.apply(lambda x: abs(x))


# ### Binning of Continious Variables
# #### Standardizing Days columns in Years for easy binning

# In[37]:


app_df["YEARS_BIRTH"] = app_df.DAYS_BIRTH.apply(lambda x:int(x//365))
app_df["YEARS_EMPLOYED"] = app_df.DAYS_EMPLOYED.apply(lambda x:int(x//365))
app_df["YEARS_REGISTRATION"] = app_df.DAYS_REGISTRATION.apply(lambda x:int(x//365))
app_df["YEARS_ID_PUBLISH"] = app_df.DAYS_ID_PUBLISH.apply(lambda x:int(x//365))
app_df["YEARS_LAST_PHONE_CHANGE"] = app_df.DAYS_LAST_PHONE_CHANGE.apply(lambda x:int(x//365))


# ## Binning AMT_CREDIT Column

# In[38]:


app_df.AMT_CREDIT.value_counts(normalize=True)*100


# In[39]:


app_df["AMT_CREDIT_Category"] = pd.cut(app_df.AMT_CREDIT, [0, 200000, 400000, 600000, 800000, 1000000], labels=["Very low Credit", "Low Credit", "Medium Credit", "High Credit", "Very High Credit"])


# In[40]:


app_df.AMT_CREDIT_Category.value_counts(normalize=True)*100


# In[41]:


app_df["AMT_CREDIT_Category"].value_counts(normalize=True).plot.bar()
plt.show()


# - Conclusion: The credit amount of the loan for amount low (2L to 4L) or Very High (above 8L)

# ## Binning YEARS_BIRTH Column

# In[42]:


app_df["AGE_Category"] = pd.cut(app_df.YEARS_BIRTH, [0, 25, 45, 65, 85], labels = ["Below 25", "25-45", "45-65", "65-85"])


# In[43]:


app_df.AGE_Category.value_counts(normalize=True)*100


# In[44]:


app_df["AGE_Category"].value_counts(normalize=True).plot.pie(autopct = '%1.2f%%')
plt.show()


#  - Conclusion: Most of the applicants are between the age group 25-45

# ## Data Imbalance Check

# In[45]:


app_df.head()


# ## Diving Application dataset with Target Variable as 0 & 1

# In[46]:


tar_0 = app_df[app_df.TARGET == 0]
tar_1 = app_df[app_df.TARGET == 1]


# In[47]:


app_df.TARGET.value_counts(normalize=True)*100


# - Conclusion: 1 out of 9/10 applicants are defaulters

# ## Univariate Analysis

# In[48]:


cat_cols = list(app_df.columns[app_df.dtypes == object])
num_cols = list(app_df.columns[app_df.dtypes == np.int64])  +  list(app_df.columns[app_df.dtypes == np.float64])


# In[49]:


cat_cols


# In[50]:


num_cols


# In[51]:


for col in cat_cols:
    print(app_df[col].value_counts(normalize=True))
    plt.figure(figsize=[5,5])
    app_df[col].value_counts(normalize=True).plot.pie(labeldistance = None, autopct = '%1.2f%%')
    plt.legend()
    plt.show()


# - Conclusion >> Insights on below columns
# 
# 
#         1.NAME_CONTRACT_TYPE - More applicants have cash loans than revolving loans
#         2.CODE_GENDER - Number of female applicants is twice than that of male applicants
#         3.FLAG_OWN_CAR - Most(70%) applicants do not own cars
#         4.FLAG_OWN_REALTY - Most(70%) applicants do not own house
#         5.NAME_TYPE_SUITE - Most(81%) applicants are unaccompanied
#         6.NAME_INCOME_TYPE - Most(51%) applicants are earning their income from work
#         7.NAME_EDUCATION_TYPE - 71% applicants have completed secondary / secondary special education
#         8.NAME_FAMILY_STATUS - 63% applicants are married
#         9.NAME_HOUSING_TYPE - 88% of the housing type of applicants are house/appartment
#        10.OCCUPATION_TYPE - Most(31%) of the applicants have other occupation type
#        11.WEEKDAY_APPR_PROCESS_START - Most of the applicants have applied on Tuesday
#        12.ORGANIZATION_TYPE - Most of the Organization type of applicants are Business entity type-3

# ### Plot on Numerical Columns
# 
# #### Categorizing columns with and without flags

# In[52]:


num_cols_withoutflag = []
num_cols_withflag = []
for col in num_cols:
    if col.startswith("FLAG"):
        num_cols_withflag.append(col)
    else:
        num_cols_withoutflag.append(col)


# In[53]:


num_cols_withoutflag


# In[54]:


num_cols_withflag


# In[55]:


for col in num_cols_withoutflag:
    print(app_df[col].describe())
    plt.figure(figsize = [8,5])
    sns.boxplot(data=app_df, x=col)
    plt.show()
    print("--------------")


# ### Univariate Analysis on Columns with Target 0 & 1

# In[56]:


for col in cat_cols:
    print(f"plot on {col} for Target 0 and 1")
    plt.figure(figsize=[10,7])
    plt.subplot(1,2,1)
    tar_0[col].value_counts(normalize=True).plot.bar()
    plt.title("Target 0")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.subplot(1,2,2)
    tar_1[col].value_counts(normalize=True).plot.bar()
    plt.title("Target 1")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()
    print("\n\n ----------------------------------------------------------------------------------------------------------------- \n\n")


# #### Analysis on AMT_GOODS_PRICE on Target 0 and 1

# In[57]:


plt.figure(figsize=(10,6))
sns.distplot(tar_0["AMT_GOODS_PRICE"], label='tar_0', hist=False)
sns.distplot(tar_1["AMT_GOODS_PRICE"], label='tar_1', hist=False)
plt.legend()
plt.show()


# - Conclusion: The price of the goods for which loan is given has the same variation for Target 0 and 1

# # Bivariate and Multivariate Analysis
# 
# ### Bivariate analysis between WEEKDAY_APPR_PROCESS_START vs HOUR_APPR_PROCESS_START

# In[58]:


plt.figure(figsize=(15,10))

plt.subplot(1,2,1)
sns.boxplot(x='WEEKDAY_APPR_PROCESS_START', y='HOUR_APPR_PROCESS_START', data = tar_0)
plt.subplot(1,2,2)
sns.boxplot(x='WEEKDAY_APPR_PROCESS_START', y='HOUR_APPR_PROCESS_START', data = tar_1)
plt.show()


# - Conclusion >>
# 
# 1.The bank operates between 10am to 3pm except for Saturday and Sunday, its between 10am to 2pm
# 
# 2.We can observe that around 11:30am to 12pm around 50% of customers visit the branch for loan application on all the days except for sunday where the time is between 10am to 11am for both target 0 and 1
# 
# 3.The Loan Defaulters have applied for the loan between 9:30am to 10am and 2pm where as the applicants who repay the loan on time have applied for loan between 10am to 3pm    

# #### Bivariate Analysis between AGE_Category vs AMT_CREDIT

# In[59]:


plt.figure(figsize=(15,10))

plt.subplot(1,2,1)
sns.boxplot(x='AGE_Category', y='AMT_CREDIT', data = tar_0)
plt.subplot(1,2,2)
sns.boxplot(x='AGE_Category', y='AMT_CREDIT', data = tar_1)
plt.show()


# ### Pair plot of Amount Columns For Target 0

# In[60]:


sns.pairplot(tar_0[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]])
plt.show()


# ### Pair plot of Amount Columns For Target 1

# In[61]:


sns.pairplot(tar_1[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]])
plt.show()


# ### Co-relation between Numerical Columns

# In[62]:


corr_data = app_df[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "YEARS_BIRTH", "YEARS_EMPLOYED", "YEARS_REGISTRATION", "YEARS_ID_PUBLISH", "YEARS_LAST_PHONE_CHANGE"]]
corr_data.head()


# In[63]:


corr_data.corr()


# In[64]:


plt.figure(figsize = (10,10))
sns.heatmap(corr_data.corr(), annot=True, cmap="RdYlGn")
plt.show()


# ### Split the Numerical variables based on Target 0 and 1 to find the co-relation

# In[65]:


corr_data_0 = tar_0[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "YEARS_BIRTH", "YEARS_EMPLOYED", "YEARS_REGISTRATION", "YEARS_ID_PUBLISH", "YEARS_LAST_PHONE_CHANGE"]]
corr_data_0.head()


# In[66]:


corr_data_1 = tar_1[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "YEARS_BIRTH", "YEARS_EMPLOYED", "YEARS_REGISTRATION", "YEARS_ID_PUBLISH", "YEARS_LAST_PHONE_CHANGE"]]
corr_data_1.head()


# In[67]:


plt.figure(figsize = (10,10))
sns.heatmap(corr_data_0.corr(), annot=True, cmap="RdYlGn")
plt.show()


# In[68]:


plt.figure(figsize = (10,10))
sns.heatmap(corr_data_1.corr(), annot=True, cmap="RdYlGn")
plt.show()


# ## Read Previous Application CSV 

# In[69]:


papp_data = pd.read_csv("previous_application.csv")
papp_data.head()


# ### Data Inspection on previous_application dataset
# #### Get info & shape on the dataset

# In[70]:


papp_data.info()


# In[71]:


papp_data.shape


# ## Data Quality check
# 
# ### Check for percentage null values in application dataset

# In[72]:


papp_data.isnull().mean()*100


# In[73]:


percentage = 49
threshold_p = int(((100-percentage)/100)*app_data.shape[0] + 1)
papp_df = papp_data.dropna(axis=1, thresh=threshold_p)
papp_df.head()


# ### Impute missing values
# #### check missing values in previous_application  dataset before imputing

# In[74]:


for col in papp_df.columns:
    if papp_df[col].dtypes == np.int64 or papp_df[col].dtypes == np.float64:
        papp_df[col] = papp_df[col].apply(lambda x: abs(x))


# ### Validate if any null values present in dataset

# In[75]:


null_cols = list(papp_df.columns[papp_df.isna().any()])
len(null_cols)


# ### Binning of Continious Variables
# #### Binning AMT_CREDIT Column

# In[76]:


papp_df.AMT_CREDIT.describe()


# In[77]:


papp_df["AMT_CREDIT_Category"] = pd.cut(papp_df.AMT_CREDIT, [0, 200000, 400000, 600000, 800000, 1000000], labels=["Very low Credit", "Low Credit", "Medium Credit", "High Credit", "Very High Credit"])


# In[78]:


papp_df["AMT_CREDIT_Category"].value_counts(normalize=True).plot.bar()
plt.show()


# - Conclusion: The credit amount of the loan for most applicants is less than 2L to 4L

# In[79]:


# Determining quantiles and the creating categorial bins
quantiles = papp_df['AMT_GOODS_PRICE'].quantile([0, 0.25, 0.45, 0.65, 0.85, 1]).values
papp_df['AMT_GOODS_PRICE_CATEGORY'] = pd.cut(papp_df['AMT_GOODS_PRICE'], bins=quantiles,labels=["Very Low Price", "Low Price", "Medium Price", "High Price", "Very High Price"],include_lowest=True)


# In[80]:


papp_df["AMT_GOODS_PRICE_CATEGORY"].value_counts(normalize=True).plot.pie(autopct = '%1.2f%%')
plt.legend()
plt.show()


# ### Data Imbalance Check
# 
# #### Dividing previous_application Dataset with NAME_CONTRACT_STATUS

# In[81]:


approved = papp_df[papp_df.NAME_CONTRACT_STATUS == "Approved"]
canceled = papp_df[papp_df.NAME_CONTRACT_STATUS == "Canceled"]
refused = papp_df[papp_df.NAME_CONTRACT_STATUS == "Refused"]
unused = papp_df[papp_df.NAME_CONTRACT_STATUS == "Unused"]


# In[82]:


papp_df.NAME_CONTRACT_STATUS.value_counts(normalize = True)*100


# In[83]:


papp_df.NAME_CONTRACT_STATUS.value_counts(normalize=True).plot.pie(autopct = '%1.2f%%')
plt.legend()
plt.show()


# - Conclusion: Loan approved applicants = 62%, 19% = Cancelled, 17% = Rejected, 2% = Unused

# ## Univariate Analysis

# In[84]:


cat_cols = list(papp_df.columns[papp_df.dtypes == object])
num_cols = list(papp_df.columns[papp_df.dtypes == np.int64])  +  list(papp_df.columns[papp_df.dtypes == np.float64])


# In[85]:


cat_cols


# In[86]:


num_cols


# In[87]:


cat_cols = ["NAME_CONTRACT_TYPE","WEEKDAY_APPR_PROCESS_START","NAME_CONTRACT_STATUS","NAME_CLIENT_TYPE","NAME_PAYMENT_TYPE","NAME_SELLER_INDUSTRY","CHANNEL_TYPE","NAME_YIELD_GROUP","PRODUCT_COMBINATION"]


# In[88]:


num_cols = ["HOUR_APPR_PROCESS_START","DAYS_DECISION","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE","CNT_PAYMENT"]


# ### Plot on Categorical columns

# In[89]:


for col in cat_cols:
    print(papp_df[col].value_counts(normalize=True)*100)
    plt.figure(figsize=[5,5])
    papp_df[col].value_counts(normalize=True).plot.pie(labeldistance=None, autopct = '%1.2f%%')
    plt.legend()
    plt.show()
    print('---------------------------------')


# ## Plot on Numerical Columns

# In[90]:


for col in num_cols:
    print("99th percentile", np.percentile(papp_df[col],99))
    print(papp_df[col].describe())
    plt.figure(figsize=[10,6])
    sns.boxplot(data=papp_df, x=col)
    plt.show()
    print('--------------------')


# ## Bivariate and Multivariate Analysis
# 
# ### Bivariate analysis between WEEKDAY_APPR_PROCESS_START vs AMT_APPLICATION

# In[91]:


plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START', y='AMT_APPLICATION', data=approved)
plt.title("Plot for Approved")
plt.show()


# In[92]:


plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START', y='AMT_APPLICATION', data=canceled)
plt.title("Plot for canceled")
plt.show()


# In[93]:


plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START', y='AMT_APPLICATION', data=refused)
plt.title("Plot for refused")
plt.show()


# ### Bivariate analysis between AMT_ANNUITY vs AMT_GOODS_PRICE

# In[95]:


plt.figure(figsize=(15,10))
plt.subplot(1,3,1)
plt.title("Approved")
sns.scatterplot(x='AMT_ANNUITY', y="AMT_GOODS_PRICE", data=approved)
plt.subplot(1,3,2)
plt.title("Canceled")
sns.scatterplot(x='AMT_ANNUITY', y="AMT_GOODS_PRICE", data=canceled)
plt.subplot(1,3,3)
plt.title("Refused")
sns.scatterplot(x='AMT_ANNUITY', y="AMT_GOODS_PRICE", data=refused)


# ### Co relation between Numerical Columns

# In[97]:


corr_approved = approved[["DAYS_DECISION", "AMT_ANNUITY","AMT_APPLICATION","AMT_GOODS_PRICE","CNT_PAYMENT"]]
corr_canceled = canceled[["DAYS_DECISION", "AMT_ANNUITY","AMT_APPLICATION","AMT_GOODS_PRICE","CNT_PAYMENT"]]
corr_refused = refused[["DAYS_DECISION", "AMT_ANNUITY","AMT_APPLICATION","AMT_GOODS_PRICE","CNT_PAYMENT"]]


# ### Co Relation for Numerical columns for approved

# In[98]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_approved.corr(),annot=True,cmap="Blues")
plt.title("Heat map plot for Approved")
plt.show


# ### Co Relation for Numerical columns for refused

# In[99]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_refused.corr(),annot=True,cmap="Blues")
plt.title("Heat map plot for Refused")
plt.show()


# ### Co Relation for Numerical columns for canceled
# 

# In[100]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_canceled.corr(),annot=True,cmap="Blues")
plt.title("Heat map plot for Canceled")
plt.show()


# ## Merge the Application and previous_application Dataframes

# In[101]:


merge_df = app_df.merge(papp_df, on=["SK_ID_CURR"], how='left')
merge_df.head()


# In[102]:


merge_df.info()


# ### Filtering required columns for our anaysis

# In[104]:


for col in merge_df.columns:
    if col.startswith("FLAG"):
        merge_df.drop(columns=col, axis=1, inplace=True)


# In[106]:


merge_df.shape


# In[110]:


res1 = pd.pivot_table(data=merge_df, index=["NAME_INCOME_TYPE", "NAME_CLIENT_TYPE"], columns=["NAME_CONTRACT_STATUS"], values="TARGET", aggfunc="mean")
res1


# In[111]:


plt.figure(figsize=[10,10])
sns.heatmap(res1, annot=True, cmap='BuPu')
plt.show()


# In[113]:


res2 = pd.pivot_table(data=merge_df, index=["CODE_GENDER", "NAME_SELLER_INDUSTRY"], columns=["TARGET"], values="AMT_GOODS_PRICE_x", aggfunc="sum")
res2


# In[114]:


plt.figure(figsize=[10,10])
sns.heatmap(res2, annot=True, cmap='BuPu')
plt.show()


# In[ ]:




