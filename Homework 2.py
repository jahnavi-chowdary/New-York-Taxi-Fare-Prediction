
# coding: utf-8

# In[77]:


# LOADING MODULES

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')

from pydoc import help
from scipy.stats.stats import pearsonr

from sklearn import datasets, linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

from datetime import datetime
from datetime import timedelta
import datetime as dt
import calendar

import math


# In[2]:


# !conda install -c conda-forge haversine
# y


# In[3]:


# !conda install -c conda-forge/label/gcc7 haversine


# In[4]:


# READING TRAINING DATA

train = pd.read_csv('./train.csv', nrows = 1000000)
train.columns


# In[5]:


# CHECK FOR NULL ENTRIES
train[pd.isnull(train)].sum()


# In[6]:


# EXTRACT DATA FROM PICKUP_DATETIME FEATURE IN THE DATASET 

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format = '%Y-%m-%d %H:%M:%S UTC')

#train['pickup_date']= train['pickup_datetime'].dt.date

train['pickup_hour']=train['pickup_datetime'].dt.hour
train['pickup_day']=train['pickup_datetime'].dt.day
train['pickup_month']=train['pickup_datetime'].dt.month
train['pickup_year']=train['pickup_datetime'].dt.year
train['pickup_day_of_week']=train['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])


# In[7]:


print (train.shape)
print (train.columns)


# In[8]:


# REMOVING ENTRIES WITH -VE FARE AMOUNT

train=train.loc[train['fare_amount']>=0]

print (train.shape)


# ##### TRUNCATING THE LONGITUDE AND LATTITUDE COORDINATES BASED ON THAT OF NEW YORK
# 
# Used the below link to obtain the Boundary Co-ordinates of New York
# https://www.mapdevelopers.com/geocode_bounding_box.php
# 
# The boundaries of New York are:  
# North Latitude: 40.917577   
# South Latitude: 40.477399   
# East Longitude: -73.700272   
# West Longitude: -74.259090  
# 
# Used the above to build a boundary by for the allowed Drop_off and Pick_up Longitude, Latittudes.
# 
# This method of data cleaning did not produce substantial results. On analysing the possible reasons for the bad behaviour of the model, I felt that the narrow possible values for the Longitude and Lattitude could be a reason.
# To verify the same, I checked the boundaries of the Longitudes and Lattitudes on the Test Data and found that a lot of samples had coordinates outside the above used boundary. 
# 
# Inorder to cater to this issue, I used the Test Data to obtain the boundaries on the Lattitude and Longitude.
# 
# Which came out to be  
# North Latitude: 41.709555   
# South Latitude: 40.573143   
# East Longitude: -72.986532   
# West Longitude: -74.263242  
# 
# Created the Boundary using the below values as Test Data contained co-ordinates a bit outside the boundaries specified by the link above.

# In[9]:


#Before we  ahead and identify outlier location, let us read the test data and see what the boundaries are.
test =  pd.read_csv('./test.csv')

print("Longitude Boundary in test data")
print (min(test.pickup_longitude.min(), test.dropoff_longitude.min()),max(test.pickup_longitude.max(), test.dropoff_longitude.max()))

print("Latitude Boundary in test data")
print (min(test.pickup_latitude.min(), test.pickup_latitude.min()),max(test.pickup_latitude.max(), test.pickup_latitude.max()))


# In[10]:


# boundary={'min_lng':-74.263242,
#           'min_lat':40.573143,
#           'max_lng':-72.986532, 
#           'max_lat':41.709555}

boundary={'north_lat':41.709555,
          'south_lat':40.573143,
          'east_long':-72.986532, 
          'west_long':-74.263242}


# In[11]:


# train[(train.pickup_latitude==0) | (train.pickup_longitude)==0 | (train.dropoff_latitude==0)|(train.dropoff_longitude==0)].shape


# In[12]:


train.loc[~((train.pickup_longitude >= boundary['west_long'] ) & (train.pickup_longitude <= boundary['east_long']) &
            (train.pickup_latitude >= boundary['south_lat']) & (train.pickup_latitude <= boundary['north_lat']) &
            (train.dropoff_longitude >= boundary['west_long']) & (train.dropoff_longitude <= boundary['east_long']) &
            (train.dropoff_latitude >=boundary['south_lat']) & (train.dropoff_latitude <= boundary['north_lat'])),'is_outlier_loc']=1
train.loc[((train.pickup_longitude >= boundary['west_long'] ) & (train.pickup_longitude <= boundary['east_long']) &
            (train.pickup_latitude >= boundary['south_lat']) & (train.pickup_latitude <= boundary['north_lat']) &
            (train.dropoff_longitude >= boundary['west_long']) & (train.dropoff_longitude <= boundary['east_long']) &
            (train.dropoff_latitude >=boundary['south_lat']) & (train.dropoff_latitude <= boundary['north_lat'])),'is_outlier_loc']=0

# print("Outlier vs Non Outlier Counts")
# print(train['is_outlier_loc'].value_counts())

train=train.loc[train['is_outlier_loc']==0]
train.drop(['is_outlier_loc'],axis=1,inplace=True)
print (train.shape)


# In[13]:


# CALCULATING HAVERSIAN DISTANCE

def haversian_distance(lat1, lat2, lon1,lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))

train['hav_distance'] = train.apply(lambda row:haversian_distance(row['pickup_latitude'],row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)


# In[14]:


# CALCULATING EUCLEDIAN DISTANCE

train['euc_distance'] = 69 * np.sqrt((np.array(train.dropoff_longitude) - np.array(train.pickup_longitude))**2 + (np.array(train.pickup_latitude) - np.array(train.dropoff_latitude))**2)


# In[15]:


print (train.columns)


# #### WEATHER DATA
# 
# Used the below link for reference to obtain a part of New York Weather Data for the period that The Taxi Fare Dataset spans over.
# 
# https://sdaulton.github.io/TaxiPrediction/
# 
# The dataset set used is https://raw.githubusercontent.com/sdaulton/TaxiPrediction/master/data/nyc-weather-data.csv
# 
# This dataset covers Daily Summary of Weather of New York From Jan 1st 2009 to Nov 11th 2015.
# 

# In[16]:


# EXTRACTING FEATURES FROM THE WEATHER DATASET

weather=pd.read_csv("./Weather_Data.csv")

# Replacing Values with -9999 with 0 as they indicate Missing Data
weather.loc[weather.SNWD <= -9999, 'SNWD'] = 0
weather.loc[weather.SNOW <= -9999, 'SNOW'] = 0
weather.loc[weather.AWND <= -9999, 'AWND'] = 0

#Extracting the Year, Month, Day with the same Column Name as that of the Existing Data for Merging
weather['pickup_year'] = (weather['DATE']/10000).apply(math.floor)
weather['pickup_month'] = ((weather['DATE'].mod(10000))/100).apply(math.floor)
weather['pickup_day'] = weather['DATE'].mod(100)
weather = weather[['pickup_year','pickup_month','pickup_day','PRCP','SNWD','SNOW','TMAX','TMIN','AWND']]

# weather['PRCP'] = weather['PRCP'] / 10.
# weather['TMAX'] = weather['TMAX'] / 10.
# weather['TMIN'] = weather['TMIN'] / 10.
# weather['AWND'] = weather['AWND'] / 10. * 3.6
weather.columns = ['pickup_year','pickup_month','pickup_day','precipitation','snow_depth','snowfall','max_temp','min_temp','avg_wind']


# In[17]:


# MERGING EXISTING DATA WITH WEATHER DATA TO GENERATE NEW FEATURES

train_new = pd.merge(train, weather, how='left', on=['pickup_year','pickup_month','pickup_day'])

print (train_new.columns)


# In[18]:


# CHECKING FOR NULL VALUES

train_new[pd.isnull(train_new)].sum()
print (train_new.dtypes)


# In[19]:


# FINAL TRAIN DATA SET THAT WILL BE USED FOR TRAINING AND VALIDATION

train = train_new
print (train.columns)
print (train.shape)
print (train.dtypes)


# #### PEARSON CORRELATION

# In[20]:


print ("Corr between Eucledian Distance and Fare Amount")
corr = pearsonr(train.fare_amount, train.euc_distance)
print (corr)


# In[120]:


# Scatter Plot Between Eucledian Distance and Fare Amount

sns.lmplot(x='euc_distance', y='fare_amount', data=train)

plt.title('Distance of Ride Vs Taxi Fare')
# Set x-axis label
plt.xlabel('Distance of Ride')
# Set y-axis label
plt.ylabel('Taxi Fare')


# In[122]:


# Scatter Plot Between Eucledian Distance <=50 and Fare Amount

lm = sns.lmplot(x='euc_distance', y='fare_amount', data=train)

lm.set(xlim=(0, 50))
plt.title('Distance of Ride <=50 Vs Taxi Fare')
# Set x-axis label
plt.xlabel('Distance of Ride <= 50')
# Set y-axis label
plt.ylabel('Taxi Fare')


# In[22]:


# pickup_hour is distributed from 0-23 for each hour
print ("Corr between Time of the Day and Eucledian Distance")
corr = pearsonr(train.euc_distance, train.pickup_hour)
print (corr)


# In[105]:


# Scatter Plot Between Eucledian Distance and Time of the Day

sns.lmplot(x='pickup_hour', y='euc_distance', data=train)

plt.title('Time of Day Vs Distance of Ride')
# Set x-axis label
plt.xlabel('Time of Day')
# Set y-axis label
plt.ylabel('Distance of Ride')


# In[24]:


print ("Corr between Time of the Day and Fare Amount")
corr = pearsonr(train.fare_amount, train.pickup_hour)
print (corr)


# In[106]:


# Scatter Plot Between Fare Amount and Time of the Day

sns.lmplot(x='pickup_hour', y='fare_amount', data=train)

plt.title('Time of Day Vs Taxi Fare')
# Set x-axis label
plt.xlabel('Time of Day')
# Set y-axis label
plt.ylabel('Taxi Fare')


# #### OTHER INTERESTING PLOTS AND CORRELATIONS BETWEEN CERTAIN FEATURES

# In[26]:


# Checking the Correlation between Fare Amount and all other Features

print ("Corr between Fare Amount and all other parameters")
print (train.corr('pearson')["fare_amount"])


# In[27]:


# And, checking the Correlation between Haversian Distance and all other Features

print ("Corr between Haversian Distance and all other parameters")
print (train.corr('pearson')["hav_distance"])


# In[32]:


# Histogram of No.of.Trips taken per Hour in a Day

plt.hist(train.pickup_hour, bins=24)
plt.xticks(np.arange(0,24,step=1))
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Trips')
plt.title('Histogram of No.of.Trips taken per Hour in a Day')
plt.show()


# In[54]:


# Histogram of No.of.Trips taken per Day in a Week

def encodeDays(day_of_week):
    day_dict={'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6}
    return day_dict[day_of_week]
train['pickup_day_of_week']=train['pickup_day_of_week'].apply(lambda x:encodeDays(x))

plt.hist(train.pickup_day_of_week, bins=7)
plt.xticks(np.arange(0,7,step=1))
plt.xlabel('Day of the Week')
plt.ylabel('Number of Trips')
plt.title('Histogram of No.of.Trips taken per Day in a Week')
plt.show()


# In[35]:


# Histogram of No.of.Trips taken per Month in a Year

plt.hist(train.pickup_month, bins=12)
plt.xticks(np.arange(0,13,step=1))
plt.xlabel('Month of the Year')
plt.ylabel('Number of Trips')
plt.title('Histogram of No.of.Trips taken per Month in a Year')
plt.show()


# In[36]:


# Histogram of No.of.Trips taken per Year

plt.hist(train.pickup_year, bins=8)
plt.xticks(np.arange(2009,2016,step=1))
plt.xlabel('Year')
plt.ylabel('Number of Trips')
plt.title('Histogram of No.of.Trips taken per Year')
plt.show()


# #### PLOTS BETWEEN FARE AMOUNT AND WEATHER CONDITIONS

# In[123]:


sns.lmplot(x='precipitation', y='fare_amount', data=train)

plt.title('Precipitation Vs Taxi Fare')
# Set x-axis label
plt.xlabel('Precipitation')
# Set y-axis label
plt.ylabel('Taxi Fare')


# In[108]:


sns.lmplot(x='snow_depth', y='fare_amount', data=train)

plt.title('Snow Depth Vs Taxi Fare')
# Set x-axis label
plt.xlabel('Snow Depth')
# Set y-axis label
plt.ylabel('Taxi Fare')


# In[109]:


sns.lmplot(x='snowfall', y='fare_amount',data=train)

plt.title('Snowfall Vs Taxi Fare')
# Set x-axis label
plt.xlabel('Snowfall')
# Set y-axis label
plt.ylabel('Taxi Fare')


# In[110]:


sns.lmplot(x='max_temp', y='fare_amount',data=train)

plt.title('Max Temp Vs Taxi Fare')
# Set x-axis label
plt.xlabel('Max Temp')
# Set y-axis label
plt.ylabel('Taxi Fare')


# In[111]:


sns.lmplot(x='min_temp', y='fare_amount',data=train)

plt.title('Min Temp Vs Taxi Fare')
# Set x-axis label
plt.xlabel('Min Temp')
# Set y-axis label
plt.ylabel('Taxi Fare')


# In[112]:


sns.lmplot(x='avg_wind', y='fare_amount', data=train)

plt.title('Avg Wind Vs Taxi Fare')
# Set x-axis label
plt.xlabel('Avg Wind')
# Set y-axis label
plt.ylabel('Taxi Fare')


# #### PLOTS BETWEEN TRIP DISTANCE AND WEATHER CONDITIONS

# In[113]:


sns.lmplot(x='precipitation', y='hav_distance', data=train)

plt.title('Precipitation Vs Distance of Ride')
# Set x-axis label
plt.xlabel('Precipitation')
# Set y-axis label
plt.ylabel('Distance of Ride')


# In[114]:


sns.lmplot(x='snow_depth', y='hav_distance', data=train)

plt.title('Snow Depth Vs Distance of Ride')
# Set x-axis label
plt.xlabel('Snow Depth')
# Set y-axis label
plt.ylabel('Distance of Ride')


# In[115]:


sns.lmplot(x='snowfall', y='hav_distance',data=train)

plt.title('Snowfall Vs Distance of Ride')
# Set x-axis label
plt.xlabel('Snowfall')
# Set y-axis label
plt.ylabel('Distance of Ride')


# In[119]:


sns.lmplot(x='max_temp', y='hav_distance',data=train)

plt.title('Max Temp Vs Distance of Ride')
# Set x-axis label
plt.xlabel('Max Temp')
# Set y-axis label
plt.ylabel('Distance of Ride')


# In[117]:


sns.lmplot(x='min_temp', y='hav_distance',data=train)

plt.title('Min Temp Vs Distance of Ride')
# Set x-axis label
plt.xlabel('Min Temp')
# Set y-axis label
plt.ylabel('Distance of Ride')


# In[118]:


sns.lmplot(x='avg_wind', y='hav_distance', data=train)

plt.title('Avg Wind Vs Distance of Ride')
# Set x-axis label
plt.xlabel('Avg Wind')
# Set y-axis label
plt.ylabel('Distance of Ride')


# #### PROCESSING TEST DATA 

# In[66]:


# Pre-Processing Test Data

test =  pd.read_csv('./test.csv')
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')

test['pickup_date']= test['pickup_datetime'].dt.date
test['pickup_day']=test['pickup_datetime'].dt.day
test['pickup_hour']=test['pickup_datetime'].dt.hour
test['pickup_day_of_week']=test['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
test['pickup_month']=test['pickup_datetime'].dt.month
test['pickup_year']=test['pickup_datetime'].dt.year


test['hav_distance']=test.apply(lambda row:haversian_distance(row['pickup_latitude'],row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)
test['euc_distance'] = 69 * np.sqrt((np.array(test.dropoff_longitude) - np.array(test.pickup_longitude))**2 + (np.array(test.pickup_latitude) - np.array(test.dropoff_latitude))**2)


# In[67]:


# Merging with Weather Data to obtain extra Features

test_new = pd.merge(test, weather, how='left', on=['pickup_year','pickup_month','pickup_day'])
print (test_new.columns)


# In[68]:


# Checking for NULL Values

test_new[pd.isnull(test_new)].sum()
print (test_new.dtypes)


# In[69]:


# Final Test Set that will be Used For Testing

test = test_new
print (test.columns)
print (test.shape)
print (test.dtypes)


# #### PREPARING TRAIN AND VALIDATION SET FROM TRAINING DATA 

# In[124]:


# Considering all the Features except 'Key', 'pickup_date', 'pickup_day_of_week' and 'euc_distance' (as haversian distance is already being considered, and both together could lead to redundant features.)
traindata = train[train.columns[3:]]
traindata = traindata[traindata.columns.difference(['pickup_day_of_week'])]
traindata = traindata[traindata.columns.difference(['pickup_date'])]
traindata = traindata[traindata.columns.difference(['euc_distance'])]

# Trainoutput is the Fare Amount column
trainoutput = train[train.columns[1:2]]

# Considering 2/3rd of the data for Training and 1/3rd for Validation
x, y = traindata.shape
x1 = 2 * int(x / 3)

traindata_X = traindata[0:x1]
testdata_X = traindata[x1:x]

traindata_Y = trainoutput[0:x1]
testdata_Y = trainoutput[x1:x]

x, y
print (traindata.columns)


# #### PREPARING TEST DATA

# In[125]:


# Considering all Features except 'Key', 'pickup_date', 'pickup_day_of_week'
testdata = test[test.columns[2:]]
testdata = testdata[testdata.columns.difference(['pickup_day_of_week'])]
testdata = testdata[testdata.columns.difference(['pickup_date'])]
testdata = testdata[testdata.columns.difference(['euc_distance'])]

print (testdata.columns)


# ## PREDICTION MODELS

# #### LINEAR REGRESSION MODEL

# In[126]:


#TRAINING USING LINEAR REGRESSION

regr = linear_model.LinearRegression()
regr.fit(traindata_X, traindata_Y)
testdata_Y_pred_LR = regr.predict(testdata_X)

# The coefficients
print('Coefficients: \n', regr.coef_)

# ERROR METRICS

# Root Mean Square Error
rmse = np.sqrt(mean_squared_error(testdata_Y, testdata_Y_pred_LR))
print("RMSE: %f" % (rmse))

# The Mean Squared Error
print("Mean squared error: %.2f"
      % mean_squared_error(testdata_Y, testdata_Y_pred_LR))

# R2 Score
print('Variance score: %.2f' % r2_score(testdata_Y, testdata_Y_pred_LR))


# In[127]:


# TESTING ON TEST USED USING THE ABOVE MODEL

actual_testdata_pred_lr = regr.predict(testdata)

#EXPORTING PREDICTIONS TO CSV

key = pd.DataFrame(test[test.columns[0:1]])

key['fare_amount'] = actual_testdata_pred_lr
key.to_csv('test_predictions_lr.csv')


# ##### USING K-FOLD CROSS VALIDATION
# The whole traindata and trainoutput are used here without splitting.  
# K-Fold does the splitting based on K (Here K=6)

# In[78]:


# USING K-FOLD CROSS VALIDATION

scores = cross_val_score(regr, traindata, trainoutput, cv=6)
print ("Cross validated scores:", scores)
    
predictions = cross_val_predict(regr, traindata, trainoutput, cv=6)
plt.scatter(trainoutput, predictions)

accuracy = metrics.r2_score(trainoutput, predictions)
print ("Cross-Predicted Accuracy:", accuracy)


# In[79]:


actual_testdata_pred_lr = regr.predict(testdata)

#EXPORTING PREDICTIONS TO CSV

key = pd.DataFrame(test[test.columns[0:1]])

key['fare_amount'] = actual_testdata_pred_lr
key.to_csv('test_predictions_lr_cv.csv')


# #### DECISION TREE REGRESSOR MODEL

# In[137]:


# TRAINING USING DECISION TREE REGRESSOR MODEL

dtr = DecisionTreeRegressor().fit(traindata_X, traindata_Y)
testdata_Y_pred_DTR = dtr.predict(testdata_X)

# Error Metrics

# Root Mean Square Error
rmse = np.sqrt(mean_squared_error(testdata_Y, testdata_Y_pred_DTR))
print("RMSE: %f" % (rmse))

# Mean Squared Error
print("Mean squared error: %.2f"
      % mean_squared_error(testdata_Y, testdata_Y_pred_DTR))

# R2 Score
print('Variance score: %.2f' % r2_score(testdata_Y, testdata_Y_pred_DTR))


# In[138]:


actual_testdata_pred_dtr = dtr.predict(testdata)

#EXPORTING PREDICTIONS TO CSV

key = pd.DataFrame(test[test.columns[0:1]])

key['fare_amount'] = actual_testdata_pred_dtr
key.to_csv('test_predictions_dtr.csv')


# #### RANDOM FOREST REGRESSOR MODEL

# In[128]:


#TRAINING USING RANDOM FOREST REGRESSOR

rf = RandomForestRegressor()
rf.fit(traindata_X, traindata_Y)
testdata_Y_pred_RFR = rf.predict(testdata_X)

# Error Metrics

# Root Mean Square Error
rmse = np.sqrt(mean_squared_error(testdata_Y, testdata_Y_pred_RFR))
print("RMSE: %f" % (rmse))

# Mean Squared Error
print("Mean squared error: %.2f"
      % mean_squared_error(testdata_Y, testdata_Y_pred_RFR))

# R2 Score
print('Variance score: %.2f' % r2_score(testdata_Y, testdata_Y_pred_RFR))


# In[129]:


actual_testdata_pred_rfr = rf.predict(testdata)

#EXPORTING PREDICTIONS TO CSV

key = pd.DataFrame(test[test.columns[0:1]])

key['fare_amount'] = actual_testdata_pred_rfr
key.to_csv('test_predictions_rfr.csv')


# #### XGB REGRESSOR

# In[130]:


#TRAINING USING XGB REGRESSOR

xg_reg = xgb.XGBRegressor()
xg_reg.fit(traindata_X,traindata_Y)
testdata_Y_pred_XGB = xg_reg.predict(testdata_X)

# Error Metrics

# Root Mean Square Error
rmse = np.sqrt(mean_squared_error(testdata_Y, testdata_Y_pred_XGB))
print("RMSE: %f" % (rmse))

# Mean Square Error
print("Mean squared error: %.2f"
      % mean_squared_error(testdata_Y, testdata_Y_pred_XGB))

# R2 Score
print('Variance score: %.2f' % r2_score(testdata_Y, testdata_Y_pred_XGB))


# In[131]:


actual_testdata_pred_xgb = xg_reg.predict(testdata)

#EXPORTING PREDICTIONS TO CSV

key = pd.DataFrame(test[test.columns[0:1]])

key['fare_amount'] = actual_testdata_pred_xgb
key.to_csv('test_predictions_xgb.csv')


# #### XGB WITH HYPERPARAMETERS

# In[132]:


dtrain = xgb.DMatrix(traindata_X, label=traindata_Y)
dtest = xgb.DMatrix(testdata_X)

#set parameters for xgboost
params = {'max_depth':7,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.1
         }
num_rounds = 50

xb = xgb.train(params, dtrain, num_rounds)

y_pred_xgb = xb.predict(dtest)

rmse = np.sqrt(mean_squared_error(testdata_Y, y_pred_xgb))
print("RMSE: %f" % (rmse))

print("Mean squared error: %.2f"
      % mean_squared_error(testdata_Y, y_pred_xgb))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(testdata_Y, y_pred_xgb))


# Tried changing the Parameters to check for better performance

# In[133]:


dtrain = xgb.DMatrix(traindata_X, label=traindata_Y)
dtest = xgb.DMatrix(testdata_X)

#Set parameters for xgboost
params = {'max_depth':9,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.1
         }
num_rounds = 100

xb = xgb.train(params, dtrain, num_rounds)

y_pred_xgb = xb.predict(dtest)

# Error Metrics

# Root Mean Square Error
rmse = np.sqrt(mean_squared_error(testdata_Y, y_pred_xgb))
print("RMSE: %f" % (rmse))

# Mean Squared Error
print("Mean squared error: %.2f"
      % mean_squared_error(testdata_Y, y_pred_xgb))

# R2 Score
print('Variance score: %.2f' % r2_score(testdata_Y, y_pred_xgb))


# Tried changing the Parameters again.
# ##### This gave the best performance among all the previous models.

# In[134]:


dtrain = xgb.DMatrix(traindata_X, label=traindata_Y)
dtest = xgb.DMatrix(testdata_X)

#set parameters for xgboost
params = {'max_depth':10,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.15
         }
num_rounds = 100

xb = xgb.train(params, dtrain, num_rounds)

y_pred_xgb = xb.predict(dtest)

# Error Metrics

# Root Mean Square Error
rmse = np.sqrt(mean_squared_error(testdata_Y, y_pred_xgb))
print("RMSE: %f" % (rmse))

# Mean Squared Error
print("Mean squared error: %.2f"
      % mean_squared_error(testdata_Y, y_pred_xgb))

# R2 Score
print('Variance score: %.2f' % r2_score(testdata_Y, y_pred_xgb))


# In[135]:


dtest_actual = xgb.DMatrix(testdata)
actual_testdata_pred_xgb_params = xb.predict(dtest_actual)

#EXPORTING PREDICTIONS TO CSV

key = pd.DataFrame(test[test.columns[0:1]])

key['fare_amount'] = actual_testdata_pred_xgb_params
key.to_csv('test_predictions_xgb_params.csv')


# Below Plot shows the importance of each feature towards building the Prediction Model.

# In[136]:


xgb.plot_importance(xb)
plt.rcParams['figure.figsize'] = [7, 7]
plt.show()

