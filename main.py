
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("./Upgrad/Assignment2_LendingCaseStudy/loan/loan.csv", encoding="ISO-8859-1")
df.head()


# In[3]:


df.shape


# `##Listing down business meaning / understainding   of Important fields to proceed with the analysis :
#     for Loan_status =Charged Off what are the factors driving it .
#     dti: ratio of individual’s monthly debt payment to his or her monthly gross income
#     earliest_cr_line:an amount of money a person or company is allowed to borrow during a particular period of time from one or more financial organizations
#     mths_since_last_delinq: month since last detauulting to pay loan amount
#     revol_bal: similar to credit card ince amount is paid the credit linit is increased
#     total_pymnt: total payment done
#     ID: unique id for each row in the LC 
#     Loan_amnt: Loam applied amount by borrower
#     Funded_amnt:amount committed to that loan at that point in time
#     Funded_amnt_inv: amount  committed by investors for  loan.
#     term:duration of loan in months
#     grade : Business grading of loan
#     sub_grade : business driven sub grading
#     emp_title: employee occupation designation or tile
#     emP_lenght: employee employment duration (total experience)
#     home_ownership : whether the borrower is staying in rented , own or mortaged house
#     annual_inc: earning of the employee per year
#     verification_status: whether income earned was verified
#     issue_d : month on which loan was issued
#     purpose of the loan as claimed by borrower 
#     title of the loan 
#     addr_state: name of state where loan is taken
#    

#     # basic cleaning starts from here

# In[4]:


pd.set_option('display.max_rows', 500)
missing_count=(df.isnull().sum()/len(df))*100 #check missing data 
missing_count=missing_count[missing_count>0]
missing_count.sort_values(ascending=False) #Data cleaning


# In[5]:


df.dropna(axis=1,how='all',inplace =True) # drop columns which has all the values as NAN (100 percent blank record)
df.shape


# In[6]:


missingPostdrop=(df.isna().sum()/len(df))*100
missingPostdrop=missingPostdrop[missingPostdrop>0]
missingPostdrop.sort_values(ascending=False) #later we will take care of filling missing values


# In[7]:


#dropping zip code as all the columns have xx character in it, address state field is enough for further analysis
df.drop('zip_code',axis =1, inplace =True)
df.shape


#     ##Type driven derived metrics creation int_rate_percent to enable ease of analysis 

# In[8]:


# int_rate and few columns have % symbol in the value , remove the symbol from data and adding percent name in column name 
df.rename(columns={'int_rate':'int_rate_percent','revol_util':'revol_util_percent'},inplace=True)
#remove the % value from the data. 
df['int_rate_percent']=df['int_rate_percent'].apply(lambda x: x.replace('%',''))
df['int_rate_percent']=df['int_rate_percent'].astype(float) # converting to float for numeric plotting
df.info()# 0.125891 percent record of revol_util_precent is blank ,hence removing that records and then deleting the %sign 


#     ##Type driven derived metrics creation revol_util_percent from revol_util to enable ease of analysis 

# In[10]:


df.dropna(axis=0,subset=['revol_util_percent'],inplace=True)#dropping na values as its .2 percent data without dropping we cant remove % symbol from data
df['revol_util_percent']=df['revol_util_percent'].apply(lambda x: x.replace('%',''))#removing %symbol from revol_util field
df['revol_util_percent']=df['revol_util_percent'].astype(float) # converting to float for numeric plotting
df.shape


# `## Derived metrics year and Month which help in further analysis

# In[11]:


import datetime as dt
## Derived metrics year and Month which help in further analysis
df.issue_d=pd.to_datetime(df.issue_d.str.upper(),format='%b-%y', yearfirst=False)
df['Year']=df['issue_d'].dt.year
df['Month']=df['issue_d'].dt.month


# In[12]:


df.isnull().sum()


# In[13]:


df.shape


# In[14]:


missingPostdrop=(df.isna().sum()/len(df))*100
missingPostdrop=missingPostdrop[missingPostdrop>0]
missingPostdrop.sort_values(ascending=False) #later we will take care of filling missing values


# `##Missing value impute 
#   How prevalent is the missing value  : its less than 20 percent in all the below mentioned columns
#   Is missing data Random or does it have a pattern : Its Random  

# In[15]:


#dropping columns as the total number of NAN values are almost 10 percent its better to drop it rather than replace as its random and without pattern
df.dropna(axis=0,subset=['last_credit_pull_d','title','tax_liens','collections_12_mths_ex_med','chargeoff_within_12_mths','last_pymnt_d','pub_rec_bankruptcies','emp_length','emp_title'],inplace=True)
df.shape


# In[16]:


#dropping colmns next_pymnt_d  and mths_since_last_record as more thatn 92 percent records are blank and droping it rather than replace as its random and without pattern
df.drop(['next_pymnt_d','mths_since_last_record'],axis=1,inplace=True)
df.shape


# In[17]:


#dont see significance of url and initial_list_status loan was initially marked for the whole loan program via the “initial_list_status” variable, which takes on values of “w” and “f.”
df.drop(['url','initial_list_status'],axis=1,inplace=True) 
df.shape


# In[18]:


df.info()


# In[19]:


pd.set_option('display.max_columns', 100)
df


# In[20]:


df.dtypes.value_counts() # unique data type counts in dataframe


# In[21]:


df.shape


# In[22]:


df.groupby(['member_id']).count().max() # a given member does not have more than one record in the table , hence member id column can be droped as it doesnt add any value


# In[23]:


df.drop(['member_id'],axis=1,inplace =True)


# In[24]:


#seperate out numeric and categorical columns which will help us in plotting all the columns together
#numeric_data = df_train.select_dtypes(include = [np.number])
numeric_data=df.select_dtypes(include=[np.number])
categorical_data=df.select_dtypes(exclude=[np.number])


# In[25]:


numeric_data.info() #31 numenric columns


#     #univariate analysis starts from here

# In[26]:


#Outlier detection  in data as first step in analyses to enable clean up as required ,here dealing with numeric column
for column in numeric_data: 
    plt.figure(figsize=(20,10))
    numeric_data.boxplot([column],fontsize='large')
    plt.show()


#  `##Analysis with Box plot 
# 1)Loan_amnt has outliner beyond 75th percentile,hence better to consider record between 25th and 75 percent for normal distri.
# 2)funded_amnt has outliner beyond 75th percentile,hence better to consider record between 25th and 75 percent for normal distri.
# 3)funded_amnt_in  has outliers beyond 25th  beyond 75th percentile,hence better to consider record between 25th and 75 percent for normal distri.
# 4)installment  has outliers beyond 75th percentile,hence better to consider record between 25th and 75 percent for normal distri.
# 5)annual_inc field data is not uniformly distributed needs further analysis 
# 6)dti field  has outliers beyond 25th and 75th percentile,hence better to consider record between 25th and 75 percent for normal distri.
# 7)delinq_2yrs data is not normally distributed , analyse further and check if this column is arequired for analysis
# 8) inq_last_6mnths has beyond 75th percentile,hence better to consider record less than 75 percent for normal distri.
# `9) mths_since_last_delinq  has outliers beyond 25th and 75th percentile,hence better to consider record between 25th and 75 percent for normal distri
# 10)open_acc  has outliers beyond 25th and 75th percentile,hence better to consider record between 25th and 75 percent for normal distri
# 11)pub_rec this column can be dropped as there are no significant data for analysis.
# 12)revol_bal there is huge outlier data , needs further analysis before deciding.
# 13)total_acc  has outliers beyond 25th and 75th percentile,hence better to consider record between 25th and 75 percent for normal distri
# 14)out_prncp there is no distributed data , needs more analysis before handing the data.
# 15)out_prncp_inv there is no distributed data , needs more analysis before handing the data.
# 16)total_pymnt has outliers beyond 75th percentile,hence better to consider record between 25th and 75 percent for normal distri.
# 17) total_pymnt_inv has outliers beyond 75th percentile,hence better to consider record between 25th and 75 percent for normal distri.
# `18)total_rec_prncp has outliers beyond 75th percentile,hence better to consider record between 25th and 75 percent for normal distri.
# 19)total_rec_int has huge amount of data in outliers , needs more analysis 
# 20)total_rec_late_fee there is no distributed data , needs more analysis
# 21)recoveries there is no distributed data , needs more analysis
# 22)collection_recovery_fee this column needs to be analysed further , seems this column dont have much signifcance .
# 23)last_pumnt_amnt has huge outliers need to decide on how to handle it .
# 24)collections_12_mnths_ex_med this column doesnt have much significant data , hence it can be deleted.
# 25)policy_code this column doesnt have much significant data , hence it can be deleted.
# 26)acc_now_delinq looks this column can be dropped , analyse it further 
# 27)chargoff_within_12_mnths , drop this column as data seems to be almost same acorss all rows.
# 28)delin1_amnt , drop this column as data seems to be almost same acorss all rows.
# 29)pub_rec_bankruptcies this column doesnt have any significant data and data seems to be same across all rows hence can be dropped.
# 30)tax_liens this column doesnt have any significant data and data seems to be same across all rows hence can be dropped.
# 

# `##Decission based on the analysis
# 
# ##Drop following columns from data set
# pub_rec,
# collections_12_mnths_ex_med
# policy_code
# chargoff_within_12_mnths
# delin1_amnt
# pub_rec_bankruptcies
# tax_liens
# 
# ##Needs further analysis
#     annual_inc
# delinq_2yrs
#     revol_bal
# out_prncp
# out_prncp_inv
#     total_rec_int
# total_rec_late_fee
# recoveries 
# collection_recovery_fee
#     last_pymnt_amnt
# acc_now_delinq
# 
# ##filter records or remove outliers
# Loan_amnt
# funded_amnt
# funded_amnt_in
# installment
# dti field 
# inq_last_6mnths
# mths_since_last_delinq
# open_acc
# total_acc
# total_pymnt
# total_pymnt_inv
# total_rec_prncp

#     ## Analysing columns marked for further analysis as above

# In[27]:


print(df['collection_recovery_fee'].describe())
(df['collection_recovery_fee']!=0).value_counts() #3332 records are having non zero values which means roughly 91 percent values are 0 . hence we can drop this column


# In[28]:


print(df['acc_now_delinq'].describe())
#print(df['acc_now_delinq']) 
(df['acc_now_delinq']!=0).value_counts()# all the values are zero in this column , hence marked for dropping 


# In[29]:


print(df['last_pymnt_amnt'].describe()) # can consider this column as whole for analysis as last payment amount differ in loan amount
print(((df['last_pymnt_amnt']>3457) | (df['last_pymnt_amnt']<226)).value_counts())#roughly 50 percent of records fall outside of  25th and 7th quarantile , hence considering this column as whole


# In[30]:


print(df['recoveries'].describe())
#print(df['recoveries']) 
(df['recoveries']!=0).value_counts() 
#32704 records thats apporx 90 percent having value as 0 hence droping it rather than replace as its random and without pattern


# In[31]:


print(df['total_rec_late_fee'].describe())
#print(df['total_rec_late_fee']) 
(df['total_rec_late_fee']!=0).value_counts()
#34616 rows thats approx 95 percent having 0 value hence drop it rather than replace as its random and without pattern


# In[32]:


print(df['total_rec_int'].describe()) # this column can be analysed as it is as 
print(((df['total_rec_int']>2891) | (df['total_rec_int']<678)).value_counts()) 
#roughly 50 percent of recors fall outside of  25th and 7th quarantile , hence considiring this column as whole 


# In[33]:


print(df['out_prncp_inv'].describe())# dropping this column as more than 90 percent is 0, drop it rather than replace as its random and without pattern
#print(df['out_prncp_inv']) 
#(df['out_prncp_inv']!=0).value_counts()


# In[34]:


print(df['out_prncp'].describe())# dropping this column as more than 90 percent is 0 ,droping it rather than replace as its random and without pattern
#print(df['out_prncp']) 
(df['out_prncp']!=0).value_counts()


# In[35]:


print(df['revol_bal'].describe())# approximately 50 percent columns are beyond 25th and 75th quartile , hence consider whole column or remove above 90 percent quantile. 
#print(df['revol_bal']) 
print(((df['revol_bal']>17231) | (df['revol_bal']<3832)).value_counts())


# In[36]:


print(df['annual_inc'].describe())# approximately 50 percent columns are beyond 25th and 75th quartile , hence consider whole column for analysis or remove only outliers above 90
print(((df['annual_inc']>8.300000e+04) | (df['annual_inc']<4.200000e+04)).value_counts())
#print(((df['annual_inc']>8.300000e+04)).value_counts())
df['annual_inc'].quantile(0.9)


# In[37]:


print(df['delinq_2yrs'].describe())# dropping this column as more than 90 percent is 0
#print(df['delinq_2yrs']) 
(df['delinq_2yrs']!=0).value_counts()


#     ##Action based on univariate analysis

#   `##following columns from needs further analysis would be deleted , based on the above analysis
#   droping  these columns  rather than replace as its random and without any pattern to replace with mean , meadian
# delinq_2yrs
# out_prncp
# out_prncp_inv
# total_rec_late_fee
# recoveries 
# collection_recovery_fee
# acc_now_delinq
# 
# #below columns are from already decided list above to be dropped 
# pub_rec,
# collections_12_mnths_ex_med
# policy_code
# chargoff_within_12_mnths
# delin1_amnt
# pub_rec_bankruptcies
# tax_liens

# In[38]:


print(df.shape)# shape before drop
numeric_data.info()


# In[39]:


#dropping above mentioned columns from analysis
df.drop(['delinq_2yrs','out_prncp','out_prncp_inv','total_rec_late_fee','recoveries','collection_recovery_fee','acc_now_delinq','pub_rec','collections_12_mths_ex_med','policy_code','chargeoff_within_12_mths','delinq_amnt','pub_rec_bankruptcies','tax_liens'],axis=1,inplace=True)
df.shape# total columns left in main data frame after dropping few numeric columns 


# In[40]:


np.set_printoptions(suppress=True)
print(df['annual_inc'].describe()) # there is huge out lier in annual income , hence reducing the 3 plus or minus sd value percent beyond outliers using SD
df=df[np.abs(df.annual_inc-df.annual_inc.mean())<=(3*df.annual_inc.std())]
print(df.shape)


# In[41]:


print(df['revol_bal'].describe())#revol_bal revolving balance min is 0 and max is 149588 where as mean is 13312 retaining  3+_sd record
df=df[np.abs(df.revol_bal-df.revol_bal.mean())<=(3*df.revol_bal.std())]
print(df.shape)


# In[42]:


df.last_pymnt_amnt.describe()#ther is huge difference in last payment amount in min max and quantile hence to maintain normalisation remvoing 3rd sd + and negative
df=df[np.abs(df.last_pymnt_amnt-df.last_pymnt_amnt.mean())<=(3*df.last_pymnt_amnt.std())]
print(df.shape)


# In[43]:


plt.figure(figsize=(15,5))
sns.distplot(df['loan_amnt']) #more amount of loan is distributed between 4500 to 6000
#plt.xticks(np.arange(1000,35000,10000))
plt.show()


# In[44]:


plt.title('loan amount frequency')
sns.distplot(df['int_rate_percent']) #interest rate  between 11 and 12 is highest
plt.show()


# In[45]:


plt.figure(figsize=(15,5))
sns.distplot(df['annual_inc']) #majority of anual income is in the range of 60000
plt.xticks(np.arange(0,260000,20000))
plt.show()  


# In[46]:


#analysing categorical data 
categorical_data.info()


# In[47]:


df['home_ownership'].value_counts().plot(kind='bar') # The home ownership status provided by the borrower during registration indicates more home are rented followed by mortagage


# In[48]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
df['addr_state'].value_counts().plot(kind='bar')# majority of the loans is in Calafornia , followed by New york , Florida and Texas


# In[49]:


sns.set(rc={'figure.figsize':(5,5)}) # majority of the loans are of 36 month duration
df['term'].value_counts().plot(kind='bar')


# In[50]:


df['loan_status'].value_counts().plot(kind='bar')


# In[51]:


df['purpose'].value_counts().plot(kind='bar')


# In[52]:


df['application_type'].value_counts().plot(kind='bar') # since all the application are of type distribution ..this column can be dropped


# In[53]:


df['grade'].value_counts().plot(kind='bar') # majority of loan grade are B ,A and followed by C


# In[54]:


plt.figure(figsize=(15,5)) 
df['sub_grade'].value_counts().plot(kind='bar')


# In[55]:


df['emp_length'].value_counts().plot(kind='bar') # majority of the loan takers are having experince of 10+ years


# In[56]:


df['verification_status'].value_counts().plot(kind='bar') # majority of the loan takers income was not verified by LC


# In[57]:


plt.figure(figsize=(15,5))  
df['issue_d'].value_counts().plot(kind='bar') # more number of loan taken in month of 11


# In[58]:


df['pymnt_plan'].value_counts().plot(kind='bar')  # this column can be dropped as it contains same value 


# In[59]:


# as per the anaysis done above droping columns application_type and pymnt_plan
df.drop(['application_type','pymnt_plan'],axis=1,inplace=True)
df.shape


#     ## multivariate analysis starts from here

# In[60]:


sns.boxplot(x=df['term'],y=df['loan_amnt']) 
# as per univariate analysis 36 month loan amount was more than double that of 60 month , in this plot its clear that 60 month loan amount is more than double of 36 month loan amount


# In[61]:


sns.boxplot(x=df['loan_status'],y=df['loan_amnt']) 
#loan amounts in the range of 6000 to 16000 are in charged off status 


# In[62]:


df_charged_off=df[df['loan_status']=='Charged Off']#creating a df with just charged off status 
df_good=df[df['loan_status']=='Fully Paid']#creating a df with fully paid 


# In[63]:


plt.figure(figsize=(15,5))# for charged off loan loan amount increase as annual income increase
sns.scatterplot(x=df_charged_off['annual_inc'],y=df_charged_off['loan_amnt'])


# In[64]:


# below two plots shows that charged of loan has higher interest as compared with paid and current status loan


# In[65]:


plt.title('loan amount frequency of charged off loan')
sns.distplot(df_charged_off['int_rate_percent']) #interest rate  between 13 and 15 is highest
plt.show()


# In[66]:


plt.title('loan amount frequency of good Fully paid  loan')
sns.distplot(df_good['int_rate_percent']) #interest rate  between 11 and 12 is highest
plt.show()


# In[67]:


plt.figure(figsize=(15,5))# analysing how annual income and loan status is related . for lower income group charged off status is more
sns.boxplot(x=df.loan_status,y=df.annual_inc)


# In[68]:


plt.figure(figsize=(15,5)) #plotting the relation beween dti and Loan status
sns.boxplot(x=df.loan_status,y=df.dti)


# In[69]:


plt.figure(figsize=(15,5))# plotting the relation ship between loan status and interest rate
sns.boxplot(x=df.loan_status,y=df.int_rate_percent)


# In[70]:


plt.figure(figsize=(15,5)) # plotting the relation ship between loan status and installment amount
sns.boxplot(x=df.loan_status,y=df.installment)


# In[71]:


#interest rate of charged off loan is high and also tenure is high for 60 and 36 month loan
plt.figure(figsize=(10,5))
sns.boxplot(x=df.loan_status,y=df.int_rate_percent,hue=df.term)


# In[72]:


plt.figure(figsize=(10,5))# income of charged of loan is low for both term loans
sns.barplot(x=df.loan_status,y=df.annual_inc,hue=df.term)


# In[73]:


plt.figure(figsize=(10,5))  # dti of charged off loan is slightly higher compared to paid loan for both term loan
sns.barplot(x=df.loan_status,y=df.dti,hue=df.term)


# In[74]:


pd.crosstab(df['purpose'], df['loan_status']).plot(kind='bar', stacked=True) # purpose of the loan doesnt have much significance 


# In[75]:


#
# Uni-Variate Analysis : Regions of states where chrged off values are higher
#

#Preparing dataframe after mapping addr_state with it's Region, with the help of below source :
# Ref: https://www.infoplease.com/us/postal-information/state-abbreviations-and-state-postal-codes 

States = ['ME',  'NH',  'VT',  'MA',  'RI',  'CT',  'NY',  'PA',  'NJ',  'IN',  'OH',  'ND',  'SD',  'NE',  'KA',  'MN',  'LO',  'MO',  'DE',  'MD',  'DC',  'VA',  'WV',  'NC',  'SC',  'GA',  'FL',  'KY',  'TN',  'MS',  'AL',  'OK',  'TX',  'AR',  'LA',  'ID',  'MT',  'WY',  'NV',  'UT',  'CO',  'AZ',  'NM',  'AK',  'WA',  'OR',  'CA',  'HI']
Regions = ['Northeast',  'Northeast',  'Northeast',  'Northeast',  'Northeast',  'Northeast',  'Northeast',  'Northeast',  'Northeast',  'Midwest',  'Midwest',  'Midwest',  'Midwest',  'Midwest',  'Midwest',  'Midwest',  'Midwest',  'Midwest',  'South',  'South',  'South',  'South',  'South',  'South',  'South',  'South',  'South',  'South',  'South',  'South',  'South',  'South',  'South',  'South',  'South',  'West',  'West',  'West',  'West',  'West',  'West',  'West',  'West',  'West',  'West',  'West',  'West',  'West']
df_state_region = pd.DataFrame(list(zip(States, Regions)), columns=["addr_state","Region"])

#Prepareing 
ds_grpby_addr_state_loan_status = df.groupby(['addr_state', 'loan_status']).size()
ds_grpby_addr_state_loan_status.drop(columns=['index'],inplace=True)
df_grpby_addr_state_loan_status = ds_grpby_addr_state_loan_status.to_frame()
df_grpby_addr_state_loan_status.reset_index(inplace=True)
df_grpby_addr_state_loan_status.rename(columns={0:'Value'},inplace=True)

df_grpby_addr_state_loan_status = df_grpby_addr_state_loan_status.pivot(index='addr_state', columns='loan_status', values='Value')
df_grpby_addr_state_loan_status.reset_index(inplace=True)
df_grpby_addr_state_loan_status.drop('Current',inplace=True, axis=1)

#
# Now, I'll merge prviously created df_state_region with df_grpby_addr_state_loan_status to map state's region 
# with their total count of loan_status ('Charged Off' and 'Fully Paid').
#

df_grpby_addr_state_loan_status_merged_state_region = df_grpby_addr_state_loan_status.merge(df_state_region,how='inner')
df_grpby_addr_state_loan_status_merged_state_region.drop('addr_state',inplace=True, axis=1)

#
# Aggregating sum of their counts.
#
df_grpby_addr_state_loan_status_merged_state_region = df_grpby_addr_state_loan_status_merged_state_region.groupby('Region').agg({'Fully Paid': 'sum', 'Charged Off': 'sum'})
df_grpby_addr_state_loan_status_merged_state_region.sort_values(by='Charged Off', ascending=False, inplace=True)
df_grpby_addr_state_loan_status_merged_state_region = df_grpby_addr_state_loan_status_merged_state_region.astype({'Fully Paid': int, 'Charged Off': int})

#
# Plotting
#
ax = df_grpby_addr_state_loan_status_merged_state_region.plot(kind='bar',alpha=1,figsize=(12,8))

#
# Putting values for bars
#
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
ax.set_ylabel("Total Count")


#   #### Conclusion : granting loan for parties in lower income group ,with higher interest rate , higher installment and higher dti results in defaulting of the loan .
