#!/usr/bin/env python
# coding: utf-8

# In[1]:


<import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

def encode_categorical(churn, cat_cols):
    for col in cat_cols:
        churn[col] = churn[col].astype(str)  # Converting all values to strings

encoder = OneHotEncoder(drop='first', sparse_output=False)  # Dropping first category to avoid multicollinearity
encoded_data = encoder.fit_transform(churn[cat_cols])
    
# Convert to DataFrame with proper column names
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols), index=churn.index)
    
# Drop original categorical columns and merge encoded ones
churn = churn.drop(columns=cat_cols).reset_index(drop=True)
churn = pd.concat([churn, encoded_df], axis=1)
    
return churn, encoder

cat_cols = ['Gender', 'Marital_Status', 'account_segment', 'rev_per_month', 'Payment']
churn, encoder = encode_categorical(churn, cat_cols)

print(churn.head())


# In[2]:


df= pd.read_excel("D:\Mak\Jain study\innercity.xlsx")
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.isnull()


# In[7]:


obj = (df.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))
 
int_ = (df.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))
 
fl = (df.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))


# In[8]:


df.isna().sum()


# In[9]:


#Converting $ to null values
df['total_area'] = df['total_area'].astype(str).str.replace('$', '', regex=False)
df['total_area'] = df['total_area'].replace('', np.nan)
df['total_area'] = pd.to_numeric(df['total_area'], errors='coerce')

df['ceil'] = df['ceil'].astype(str).str.replace('$', '', regex=False)
df['ceil'] = df['ceil'].replace('', np.nan)
df['ceil'] = pd.to_numeric(df['ceil'], errors='coerce')

df['coast'] = df['coast'].astype(str).str.replace('$', '', regex=False)
df['coast'] = df['coast'].replace('', np.nan)
df['coast'] = pd.to_numeric(df['coast'], errors='coerce')

df['condition'] = df['condition'].astype(str).str.replace('$', '', regex=False)
df['condition'] = df['condition'].replace('', np.nan)
df['condition'] = pd.to_numeric(df['condition'], errors='coerce')

df['yr_built'] = df['yr_built'].astype(str).str.replace('$', '', regex=False)
df['yr_built'] = df['yr_built'].replace('', np.nan)
df['yr_built'] = pd.to_numeric(df['yr_built'], errors='coerce')

df['long'] = df['long'].astype(str).str.replace('$', '', regex=False)
df['long'] = df['long'].replace('', np.nan)
df['long'] = pd.to_numeric(df['long'], errors='coerce')


# In[10]:


df.isna().sum()


# In[11]:


#median imputation
from sklearn.impute import SimpleImputer
median_imputer = SimpleImputer(strategy='median')

df['room_bed'] = median_imputer.fit_transform(df[['room_bed']]) 
df['living_measure'] = median_imputer.fit_transform(df[['living_measure']]) 
df['lot_measure'] = median_imputer.fit_transform(df[['lot_measure']]) 
df['coast'] = median_imputer.fit_transform(df[['coast']]) 
df['sight'] = median_imputer.fit_transform(df[['sight']]) 
df['condition'] = median_imputer.fit_transform(df[['condition']]) 
df['quality'] = median_imputer.fit_transform(df[['quality']]) 
df['ceil_measure'] = median_imputer.fit_transform(df[['ceil_measure']])
df['basement'] = median_imputer.fit_transform(df[['basement']]) 
df['yr_built'] = median_imputer.fit_transform(df[['yr_built']]) 
df['long'] = median_imputer.fit_transform(df[['long']])
df['living_measure15'] = median_imputer.fit_transform(df[['living_measure15']]) 
df['lot_measure15'] = median_imputer.fit_transform(df[['lot_measure15']]) 
df['furnished'] = median_imputer.fit_transform(df[['furnished']]) 
df['total_area'] = median_imputer.fit_transform(df[['total_area']])


# In[12]:


# Convert to categorical using type
df['room_bath'] = df['room_bath'].astype('category') 
df['ceil'] = df['ceil'].astype('category')

# Assign numerical codes
df['room_bath'] = df['room_bath'].cat.codes + 1
df['ceil'] = df['ceil'].cat.codes + 1


# In[13]:


#mode imputation
from sklearn.impute import SimpleImputer
mode_imputer = SimpleImputer(strategy='most_frequent')

df['room_bed'] = mode_imputer.fit_transform(df[['room_bed']]) 
df['ceil'] = mode_imputer.fit_transform(df[['ceil']]) 


# In[14]:


df.isna().sum()


# In[15]:


from datetime import datetime


# In[16]:


# Get the current year
current_year = datetime.now().year

# Calculate the 'age of the house'
df['age'] = current_year - df['yr_built']
df.loc[df['yr_renovated'] > 0, 'age'] = current_year - df['yr_renovated']

# Convert 'dayhours' column to just date format (yyyy/mm/dd)
df['dayhours'] = pd.to_datetime(df['dayhours'].str[:8], format='%Y%m%d')

# Display the first few rows of the dataframe to check the changes
df[['yr_built', 'yr_renovated', 'age', 'dayhours']].head()


# In[17]:


df.describe()


# In[18]:


#univariate (histogram)
plt.hist(df['price'],bins=20)
plt.title('Statistical summary of Price',fontsize=10)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[19]:


#univariate (density plot)
sns.kdeplot(df['price'], fill=True)
plt.title('Density Plot of Price', fontsize=10)
plt.xlabel('Price')
plt.ylabel('Density')
plt.show()


# In[20]:


#bivariate for price vs living space
plt.scatter(x=df['price'],y=df['living_measure'],color='blue')
plt.title('Price Vs Living space',fontsize=10)
plt.xlabel('Price')
plt.ylabel('Living measure')
plt.show()


# In[21]:


#multivariate for price, living measure, ceil measure and basement
correlation_matrix=df[['price','living_measure','ceil_measure','basement']].corr()
sns.heatmap(correlation_matrix,annot=True)
plt.title('correlation between dependent variables',fontsize=20)
plt.show()


# In[22]:


# Columns to be processed
columns = ['cid', 'price', 'room_bed', 'room_bath', 'living_measure', 'lot_measure', 'ceil',
           'coast', 'sight', 'condition', 'quality', 'ceil_measure', 'basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 
           'long', 'living_measure15', 'lot_measure15', 'furnished', 'total_area', 'age']

# Plotting boxplots before capping
plt.figure(figsize=(20, 14))

for i, column in enumerate(columns, 1):
  plt.subplot(5, 5, i)  # Removed extra space
  sns.boxplot(df[f'{column}'])
  plt.title(f'{column}')

plt.tight_layout()
plt.show()


# In[23]:


plt.figure(figsize=(12,8))

plt.subplot(4,5,1)
sns.boxplot(df['cid'])

plt.subplot(4,5,2)
sns.boxplot(df['price'])

plt.subplot(4,5,3)
sns.boxplot(df['room_bed'])

plt.subplot(4,5,4)
sns.boxplot(df['room_bath'])

plt.subplot(4,5,5)
sns.boxplot(df['living_measure'])

plt.subplot(4,5,6)
sns.boxplot(df['lot_measure'])

plt.subplot(4,5,7)
sns.boxplot(df['ceil'])

plt.subplot(4,5,8)
sns.boxplot(df['coast'])

plt.subplot(4,5,9)
sns.boxplot(df['sight'])

plt.subplot(4,5,10)
sns.boxplot(df['condition'])

plt.subplot(4,5,11)
sns.boxplot(df['yr_built'])

plt.subplot(4,5,12)
sns.boxplot(df['yr_renovated'])

plt.subplot(4,5,13)
sns.boxplot(df['zipcode'])

plt.subplot(4,5,14)
sns.boxplot(df['lat'])

plt.subplot(4,5,15)
sns.boxplot(df['long'])

plt.subplot(4,5,16)
sns.boxplot(df['living_measure15'])

plt.subplot(4,5,17)
sns.boxplot(df['lot_measure15']) 

plt.subplot(4,5,18)
sns.boxplot(df['furnished'])

plt.subplot(4,5,19)
sns.boxplot(df['total_area'])

plt.subplot(4,5,20)
sns.boxplot(df['age'])

plt.show()


# In[24]:


plt.figure(figsize=(12,8))

plt.subplot(1,3,1)
sns.boxplot(df['quality'])

plt.subplot(1,3,2)
sns.boxplot(df['basement'])

plt.subplot(1,3,3)
sns.boxplot(df['ceil_measure'])

plt.show()


# In[25]:


# Function to detect outliers using IQR
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    outliers = (data < lower_limit) | (data > upper_limit)
    return outliers

# Function to cap outliers
def cap_outliers(data, lower_percentile=0.05, upper_percentile=0.95):
    lower_cap = data.quantile(lower_percentile)
    upper_cap = data.quantile(upper_percentile)
    data = data.clip(lower=lower_cap, upper=upper_cap)
    return data

# Apply the IQR method to detect and cap outliers for the specified columns
for column in columns:
    df[f'{column}_capped'] = cap_outliers(df[column])


# In[26]:


# Plotting boxplots after capping
plt.figure(figsize=(20, 20))

for i, column in enumerate(columns, 1):
    plt.subplot(5, 5, i)
    sns.boxplot(df[f'{column}_capped'])
    plt.title(f'{column} (capped)')

plt.tight_layout()
plt.show()


# In[27]:


#the exact percentile values for sepcific outlier columns
lot_measure_25th = 5.043000e+03
lot_measure_75th = 1.066000e+04
lot_measure15_25th = 5100.000000
lot_measure15_75th = 10080.000000
total_area_25th = 7.040000e+03
total_area_75th = 1.297000e+04

# Calculate IQR and bounds
lot_measure_iqr = lot_measure_75th - lot_measure_25th
lot_measure_lower_limit = lot_measure_25th - 1.5 * lot_measure_iqr
lot_measure_upper_limit = lot_measure_75th + 1.5 * lot_measure_iqr

lot_measure15_iqr = lot_measure15_75th - lot_measure15_25th
lot_measure15_lower_limit = lot_measure15_25th - 1.5 * lot_measure15_iqr
lot_measure15_upper_limit = lot_measure15_75th + 1.5 * lot_measure15_iqr

total_area_iqr = total_area_75th - total_area_25th
total_area_lower_limit = total_area_25th - 1.5 * total_area_iqr
total_area_upper_limit = total_area_75th + 1.5 * total_area_iqr

# Apply capping based on the calculated bounds
df['lot_measure_custom_capped'] = df['lot_measure'].clip(lower=lot_measure_lower_limit, upper=lot_measure_upper_limit)
df['lot_measure15_custom_capped'] = df['lot_measure15'].clip(lower=lot_measure15_lower_limit, upper=lot_measure15_upper_limit)
df['total_area_custom_capped'] = df['total_area'].clip(lower=total_area_lower_limit, upper=total_area_upper_limit)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(df['lot_measure_custom_capped'])
plt.title('lot_measure (custom capped)')

plt.subplot(1, 3, 2)
sns.boxplot(df['lot_measure15_custom_capped'])
plt.title('lot_measure15 (custom capped)')

plt.subplot(1, 3, 3)
sns.boxplot(df['total_area_custom_capped'])
plt.title('total_area (custom capped)')

plt.tight_layout()
plt.show()


# In[28]:


#the exact percentile values for sepcific outlier columns
price_25th = 3.219500e+05
price_75th = 6.450000e+05
quality_25th = 7.000000
quality_75th = 8.000000


# Calculate IQR and bounds
price_iqr = price_75th - price_25th
price_lower_limit = price_25th - 1.5 * price_iqr
price_upper_limit = price_75th + 1.5 * price_iqr

quality_iqr = quality_75th - quality_25th
quality_lower_limit = quality_25th - 1.5 * quality_iqr
quality_upper_limit = quality_75th + 1.5 * quality_iqr


# Apply capping based on the calculated bounds
df['price_custom_capped'] = df['price'].clip(lower=price_lower_limit, upper=price_upper_limit)
df['quality_custom_capped'] = df['quality'].clip(lower=quality_lower_limit, upper=quality_upper_limit)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.boxplot(df['price_custom_capped'])
plt.title('price (custom capped)')

plt.subplot(1, 2, 2)
sns.boxplot(df['quality_custom_capped'])
plt.title('quality (custom capped)')

plt.tight_layout()
plt.show()


# In[29]:


plt.figure(figsize=(20,15))

plt.subplot(5,5,1)
sns.boxplot(df['cid_capped'])
plt.title('cid (capped)')

plt.subplot(5,5,2)
sns.boxplot(df['price_custom_capped'])
plt.title('price (custom capped)')

plt.subplot(5,5,3)
sns.boxplot(df['room_bed_capped'])
plt.title('room_bed (capped)')

plt.subplot(5,5,4)
sns.boxplot(df['room_bath_capped'])
plt.title('room_bath (capped)')

plt.subplot(5,5,5)
sns.boxplot(df['living_measure_capped'])
plt.title('living_measure (capped)')

plt.subplot(5,5,6)
sns.boxplot(df['lot_measure_custom_capped'])
plt.title('lot_measure (custom capped)')

plt.subplot(5,5,7)
sns.boxplot(df['ceil_capped'])
plt.title('ceil (capped)')

plt.subplot(5,5,8)
sns.boxplot(df['coast_capped'])
plt.title('coast (capped)')

plt.subplot(5,5,9)
sns.boxplot(df['sight_capped'])
plt.title('sight (capped)')

plt.subplot(5,5,10)
sns.boxplot(df['condition_capped'])
plt.title('condition (capped)')

plt.subplot(5,5,11)
sns.boxplot(df['quality_custom_capped'])
plt.title('quality (custom capped)')

plt.subplot(5,5,12)
sns.boxplot(df['ceil_measure_capped'])
plt.title('ceil_measure (capped)')

plt.subplot(5,5,13)
sns.boxplot(df['basement_capped'])
plt.title('basement (capped)')

plt.subplot(5,5,14)
sns.boxplot(df['yr_built_capped'])
plt.title('yr_built (capped)')

plt.subplot(5,5,15)
sns.boxplot(df['yr_renovated_capped'])
plt.title('yr_renovated (capped)')

plt.subplot(5,5,16)
sns.boxplot(df['zipcode_capped'])
plt.title('zipcode (capped)')

plt.subplot(5,5,17)
sns.boxplot(df['lat_capped'])
plt.title('lat (capped)')

plt.subplot(5,5,18)
sns.boxplot(df['long_capped'])
plt.title('long (capped)')

plt.subplot(5,5,19)
sns.boxplot(df['living_measure15_capped'])
plt.title('living_measure15 (capped)')

plt.subplot(5,5,20)
sns.boxplot(df['lot_measure15_custom_capped']) 
plt.title('lot_measure15 (custom capped)')

plt.subplot(5,5,21)
sns.boxplot(df['furnished_capped'])
plt.title('furnished (capped)')

plt.subplot(5,5,22)
sns.boxplot(df['total_area_custom_capped'])
plt.title('total_area (custom capped)')

plt.subplot(5,5,23)
sns.boxplot(df['age_capped'])
plt.title('age (capped)')

plt.show()


# In[30]:


from sklearn.preprocessing import MinMaxScaler


# In[31]:


scaler = MinMaxScaler()


# In[32]:


numeric_columns = [
    'price_custom_capped', 'room_bed_capped', 'room_bath_capped', 'living_measure_capped', 'lot_measure_custom_capped',
    'sight_capped', 'quality_custom_capped', 'basement_capped', 'ceil_measure_capped', 'furnished_capped', 
    'living_measure15_capped', 'lot_measure15_custom_capped'
]

df_normalized = df.copy()
df_normalized[numeric_columns] = scaler.fit_transform(df[numeric_columns])

df_normalized.head()



# In[33]:


from sklearn.preprocessing import LabelEncoder

# One-Hot Encoding for 'ceil', 'zipcode' and 'coast'
df_encoded = pd.get_dummies(df, columns=['ceil_capped', 'coast_capped', 'zipcode_capped'], drop_first=True)

#Convert'yr_built', 'long', 'total_area' to numeric
df['yr_built_capped'] = pd.to_numeric(df['yr_built_capped'], errors = 'coerce')
df['long_capped'] = pd.to_numeric(df['long_capped'], errors = 'coerce')
df['total_area_custom_capped'] = pd.to_numeric(df['total_area_custom_capped'], errors = 'coerce')

# Label Encoding for 'condition'
label_encoder = LabelEncoder()
df_encoded['condition_capped'] = label_encoder.fit_transform(df['condition_capped'])

# Display the first few rows of the encoded DataFrame
df_encoded.head()


# In[34]:


df_encoded.ceil_measure_capped


# In[35]:


df['total_rooms'] = df['room_bed_capped'] + df['room_bath_capped']
df['total_area'] = df['living_measure_capped'] + df['lot_measure_custom_capped']

#Verify
df[['room_bed_capped', 'room_bath_capped', 'total_rooms', 
    'living_measure_capped', 'lot_measure_custom_capped', 'total_area']].head()


# In[36]:


df[['total_rooms', 'total_area']].isnull().sum()


# In[37]:


columns = [
    'cid_capped', 'room_bed_capped', 'room_bath_capped',
    'living_measure_capped', 'lot_measure_custom_capped', 'ceil_capped',
    'coast_capped', 'sight_capped', 'condition_capped', 'quality_custom_capped',
    'ceil_measure_capped', 'basement_capped', 'yr_built_capped', 'yr_renovated_capped',
    'zipcode_capped', 'lat_capped', 'long_capped', 'living_measure15_capped',
    'lot_measure15_custom_capped', 'furnished_capped', 'total_area_custom_capped',
    'age_capped', 'total_rooms', 'total_area'
]

X = df[columns]


# In[38]:


X


# In[39]:


X.isna().sum()


# In[40]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# K-Means with n=15 clusters
kmeans_15 = KMeans(n_clusters=15, random_state=42)
kmeans_15.fit(X)
df['Cluster_15'] = kmeans_15.labels_

# Calculating the Silhouette Scores
sil_score_15 = silhouette_score(X, df['Cluster_15'])

print(f"Silhouette Score for 15 clusters: {sil_score_15}")


# In[41]:


X = df[columns] #feature_matrix
y = df['price_custom_capped'] #target_variable

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[42]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Creating a linear regression model
lr = LinearRegression()

# Performing RFE to select the top features
rfe = RFE(estimator=lr, n_features_to_select=10)
rfe = rfe.fit(X_train, y_train)

selected_features = rfe.support_
ranking = rfe.ranking_

# checking the selected features gs
print("Selected Features: ", selected_features)


# In[43]:


# Get the column names that were selected by RFE
selected_columns = X_train.columns[selected_features]

# Filter the features in X_train and X_test using the selected columns
X_train_selected = X_train[selected_columns]
X_test_selected = X_test[selected_columns]


# In[44]:


import statsmodels.api as sm

X_train_selected_sm = sm.add_constant(X_train_selected)

# Fit the model using statsmodels with the selected features
model = sm.OLS(y_train, X_train_selected_sm).fit()

# Print the p-value summary
print(model.summary())

# Identifying the column names (features) with p-values > 0.05
significant_features = model.pvalues[model.pvalues <= 0.05].index
significant_features = significant_features.drop('const')

# Filter the dataset to keep only the significant features
X_train_significant = X_train_selected[significant_features]
X_test_significant = X_test_selected[significant_features]


# In[45]:


print(X_train_significant.columns)
print(X_test_significant.columns)


# In[46]:


#Linear Regression Model

from sklearn.linear_model import LinearRegression
from sklearn import metrics

model_lr = LinearRegression()

model_lr.fit(X_train_significant,y_train)
y_pred = model_lr.predict(X_test_significant)

sns.distplot((y_test-y_pred), bins=50)
plt.title('Dist plot of Linear Regression of significant features')
plt.show()


# In[47]:


# Calculating metrics
import math

mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r_squared = metrics.r2_score(y_test, y_pred)
adjusted_r_squared = 1 - (1-r_squared) * (len(y_test)-1)/(len(y_test)-X_test_significant.shape[1]-1)

print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")


# In[48]:


# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual plot of Linear Regression")
plt.show()


# In[49]:


#Lasso Regression Model

from sklearn.linear_model import Lasso
from sklearn import metrics

model_lm = Lasso(alpha=1)

model_lm.fit(X_train_significant,y_train)
y_pred = model_lm.predict(X_test_significant.astype(int))

sns.distplot((y_test-y_pred), bins=50)
plt.title('Dist plot of Lasso Regression')
plt.show()


# In[50]:


# Calculating metrics
import math

mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r_squared = metrics.r2_score(y_test, y_pred)
adjusted_r_squared = 1 - (1-r_squared) * (len(y_test)-1)/(len(y_test)-X_test_significant.shape[1]-1)

print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")


# In[51]:


# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual plot of Lasso Regression")
plt.show()


# In[52]:


#Ridge Regression Model

from sklearn.linear_model import Ridge
from sklearn import metrics

model_rm = Ridge()

model_rm.fit(X_train_significant,y_train)
y_pred = model_rm.predict(X_test_significant.astype(int))

sns.distplot((y_test-y_pred), bins=50)
plt.title('Dist plot of Ridge Regression')
plt.show()


# In[53]:


# Calculating metrics
import math

mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r_squared = metrics.r2_score(y_test, y_pred)
adjusted_r_squared = 1 - (1-r_squared) * (len(y_test)-1)/(len(y_test)-X_test_significant.shape[1]-1)

print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")


# In[54]:


# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual plot of Ridge Regression")
plt.show()


# In[55]:


#Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
 
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train_significant, y_train)
y_pred = model_RFR.predict(X_test_significant)

sns.distplot((y_test-y_pred), bins=50)
plt.title('Dist plot of RF Regression')
plt.show()


# In[56]:


# Calculating metrics
import math

mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r_squared = metrics.r2_score(y_test, y_pred)
adjusted_r_squared = 1 - (1-r_squared) * (len(y_test)-1)/(len(y_test)-X_test_significant.shape[1]-1)

print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")


# In[57]:


# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual plot of RF Regression")
plt.show()


# In[58]:


#Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
 
model_DTR = DecisionTreeRegressor()
model_DTR.fit(X_train_significant, y_train)
y_pred = model_DTR.predict(X_test_significant)

sns.distplot((y_test-y_pred), bins=50)
plt.title('Dist plot of DT Regression')
plt.show()


# In[59]:


# Calculating metrics
import math

mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r_squared = metrics.r2_score(y_test, y_pred)
adjusted_r_squared = 1 - (1-r_squared) * (len(y_test)-1)/(len(y_test)-X_test_significant.shape[1]-1)

print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")


# In[60]:


# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual plot of DT Regression")
plt.show()


# In[62]:


#XG Boost Regressor

from xgboost import XGBRegressor
 
model_XGB = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_XGB.fit(X_train_significant, y_train)

y_pred = model_XGB.predict(X_test_significant)

sns.distplot((y_test-y_pred), bins=50)
plt.title('Dist plot of XGB Regression')
plt.show()


# In[63]:


# Calculating metrics
import math

mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r_squared = metrics.r2_score(y_test, y_pred)
adjusted_r_squared = 1 - (1-r_squared) * (len(y_test)-1)/(len(y_test)-X_test_significant.shape[1]-1)

print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")


# In[64]:


# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual plot of XGB Regression")
plt.show()


# In[65]:


from sklearn.model_selection import KFold, cross_val_score

# Initialize the models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regressiom': Lasso(alpha=1),
    'Ridge Regression': Ridge(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=10),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'XGBoost Regressor': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

# Set up KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate each model
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train_significant, y_train, cv=kf, scoring='r2')
    print(f"{model_name} - Mean R-squared: {np.mean(cv_scores):.4f}, Std: {np.std(cv_scores):.4f}")

# For additional metrics such as RMSE
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train_significant, y_train, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    print(f"{model_name} - Mean RMSE: {np.mean(rmse_scores):.4f}, Std: {np.std(rmse_scores):.4f}")


# In[66]:


# XGBoost Regressor achieves the highest mean R-squared (0.8482) and the lowest mean RMSE (97632.7747). 
# Indicating it is the best model among the evaluated ones.

# Random Forest Regressor has an R-squared of 0.8339, which is close to XGBoost but slightly lower.
# Random Forest Regressor has the second-lowest RMSE (102284.0492).


# In[67]:


#Randomized Search CV for XGB Regression

from sklearn.model_selection import RandomizedSearchCV

#defining the hyperparameter grid

param_distributions_xgb = {
    'n_estimators': [100,200,300,400,500],
    'max_depth': [3,4,5,6,7],
    'learning_rate': [0.01,0.05,0.1,0.15,0.2],
    'subsample': [0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree': [0.6,0.7,0.8,0.9,1.0]
}
    
xgb_model = XGBRegressor(random_state=42)

xgb_random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_distributions_xgb,
                                   n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)


xgb_random_search.fit(X_train_significant,y_train)

#Get the best parameters and score
best_params = xgb_random_search.best_params_
best_score = xgb_random_search.best_score_

print("Best Parameters:", best_params)
print("Best Score(RMSE):", np.sqrt(-best_score))


# In[68]:


#XG Boost Regressor with best parameters

from xgboost import XGBRegressor

#using the optimized parameters
model_XGB_optimised = XGBRegressor(
     n_estimators=300,
     max_depth=7,
     learning_rate=0.05,
     subsample=0.6,
     colsample_bytree=0.7
)
 
model_XGB_optimised.fit(X_train_significant, y_train)

y_pred_optimised = model_XGB_optimised.predict(X_test_significant)

sns.distplot((y_test-y_pred_optimised), bins=50)
plt.title('Dist plot of Optimised XGB Regression')
plt.show()


# In[69]:


# Calculating metrics for optimised models
import math

mape = metrics.mean_absolute_percentage_error(y_test, y_pred_optimised)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_optimised))
r_squared = metrics.r2_score(y_test, y_pred_optimised)
adjusted_r_squared = 1 - (1-r_squared) * (len(y_test)-1)/(len(y_test)-X_test_significant.shape[1]-1)

print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")


# In[70]:


# Residual plot
residuals = y_test - y_pred_optimised
plt.scatter(y_pred_optimised, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual plot of optimised XGB Regression")
plt.show()

