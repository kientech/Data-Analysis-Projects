<h1 style="font-weight: 900; color: #000; text-align:center;">Ploice Data Analyst - Data Analyst with Python</h1>

<style>
    h2{
        color:bisque;
        text-decoration:underline;
    }
    li{
        font-size: 16px;
        margin: 10px 0;
    }
    span{
        font-weight: 700;
    }
</style>
<h2 style="font-weight: bold; font-size: 20px;">💻 Problem Solving: </h2>
<ul>
    <li>
        <span>Question 01:</span> Remove the column that only contains missing values.
    </li>
    <li>
        <span>Question 02:</span> For Speeding , were Men or Women stopped more often ?
    </li>
    <li>
        <span>Question 03:</span> Does gender affect who gets searched during a stop ?
    </li>
    <li>
        <span>Question 04:</span> What is the mean stop_duration ?
    </li>
    <li>
        <span>Question 05:</span> Compare the age distributions for each violation.
    </li>
</ul>

<h1>----------</h1>

<h2 style="font-weight: 900; font-size: 18px;">Exploratory Data Analyst</h2>


```python
# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import datetime as dt
import tensorflow as tf
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[1], line 7
          5 import seaborn as sns
          6 import datetime as dt
    ----> 7 import tensorflow as tf


    ModuleNotFoundError: No module named 'tensorflow'



```python
# read csv file
df = pd.read_csv('./data.csv')
```


```python
# show data in dataframe
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stop_date</th>
      <th>stop_time</th>
      <th>country_name</th>
      <th>driver_gender</th>
      <th>driver_age_raw</th>
      <th>driver_age</th>
      <th>driver_race</th>
      <th>violation_raw</th>
      <th>violation</th>
      <th>search_conducted</th>
      <th>search_type</th>
      <th>stop_outcome</th>
      <th>is_arrested</th>
      <th>stop_duration</th>
      <th>drugs_related_stop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/2/2005</td>
      <td>1:55</td>
      <td>NaN</td>
      <td>M</td>
      <td>1985.0</td>
      <td>20.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/18/2005</td>
      <td>8:15</td>
      <td>NaN</td>
      <td>M</td>
      <td>1965.0</td>
      <td>40.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/23/2005</td>
      <td>23:15</td>
      <td>NaN</td>
      <td>M</td>
      <td>1972.0</td>
      <td>33.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/20/2005</td>
      <td>17:15</td>
      <td>NaN</td>
      <td>M</td>
      <td>1986.0</td>
      <td>19.0</td>
      <td>White</td>
      <td>Call for Service</td>
      <td>Other</td>
      <td>False</td>
      <td>NaN</td>
      <td>Arrest Driver</td>
      <td>True</td>
      <td>16-30 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3/14/2005</td>
      <td>10:00</td>
      <td>NaN</td>
      <td>F</td>
      <td>1984.0</td>
      <td>21.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check information of each columns
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 65535 entries, 0 to 65534
    Data columns (total 15 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   stop_date           65535 non-null  object 
     1   stop_time           65535 non-null  object 
     2   country_name        0 non-null      float64
     3   driver_gender       61474 non-null  object 
     4   driver_age_raw      61481 non-null  float64
     5   driver_age          61228 non-null  float64
     6   driver_race         61475 non-null  object 
     7   violation_raw       61475 non-null  object 
     8   violation           61475 non-null  object 
     9   search_conducted    65535 non-null  bool   
     10  search_type         2479 non-null   object 
     11  stop_outcome        61475 non-null  object 
     12  is_arrested         61475 non-null  object 
     13  stop_duration       61475 non-null  object 
     14  drugs_related_stop  65535 non-null  bool   
    dtypes: bool(2), float64(3), object(10)
    memory usage: 6.6+ MB



```python
# show shape of dataframe
df.shape
```




    (65535, 15)




```python
# display all columns within dataframe
df.columns
```




    Index(['stop_date', 'stop_time', 'country_name', 'driver_gender',
           'driver_age_raw', 'driver_age', 'driver_race', 'violation_raw',
           'violation', 'search_conducted', 'search_type', 'stop_outcome',
           'is_arrested', 'stop_duration', 'drugs_related_stop'],
          dtype='object')




```python
# show statistics
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_name</th>
      <th>driver_age_raw</th>
      <th>driver_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>0.0</td>
      <td>61481.000000</td>
      <td>61228.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>1967.791106</td>
      <td>34.148984</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>121.050106</td>
      <td>12.760710</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>1965.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>1978.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>1985.000000</td>
      <td>43.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>8801.000000</td>
      <td>88.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# show datatype 
df.dtypes
```




    stop_date              object
    stop_time              object
    country_name          float64
    driver_gender          object
    driver_age_raw        float64
    driver_age            float64
    driver_race            object
    violation_raw          object
    violation              object
    search_conducted         bool
    search_type            object
    stop_outcome           object
    is_arrested            object
    stop_duration          object
    drugs_related_stop       bool
    dtype: object




```python
# checking mssing values
df.isna().sum()
```




    stop_date                 0
    stop_time                 0
    country_name          65535
    driver_gender          4061
    driver_age_raw         4054
    driver_age             4307
    driver_race            4060
    violation_raw          4060
    violation              4060
    search_conducted          0
    search_type           63056
    stop_outcome           4060
    is_arrested            4060
    stop_duration          4060
    drugs_related_stop        0
    dtype: int64




```python
# convert datatype object to datetime for stop_date column
# df['stop_date'] = pd.to_datetime(df['stop_date'])
```


```python
# reshow dataframe
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stop_date</th>
      <th>stop_time</th>
      <th>country_name</th>
      <th>driver_gender</th>
      <th>driver_age_raw</th>
      <th>driver_age</th>
      <th>driver_race</th>
      <th>violation_raw</th>
      <th>violation</th>
      <th>search_conducted</th>
      <th>search_type</th>
      <th>stop_outcome</th>
      <th>is_arrested</th>
      <th>stop_duration</th>
      <th>drugs_related_stop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/2/2005</td>
      <td>1:55</td>
      <td>NaN</td>
      <td>M</td>
      <td>1985.0</td>
      <td>20.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/18/2005</td>
      <td>8:15</td>
      <td>NaN</td>
      <td>M</td>
      <td>1965.0</td>
      <td>40.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/23/2005</td>
      <td>23:15</td>
      <td>NaN</td>
      <td>M</td>
      <td>1972.0</td>
      <td>33.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/20/2005</td>
      <td>17:15</td>
      <td>NaN</td>
      <td>M</td>
      <td>1986.0</td>
      <td>19.0</td>
      <td>White</td>
      <td>Call for Service</td>
      <td>Other</td>
      <td>False</td>
      <td>NaN</td>
      <td>Arrest Driver</td>
      <td>True</td>
      <td>16-30 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3/14/2005</td>
      <td>10:00</td>
      <td>NaN</td>
      <td>F</td>
      <td>1984.0</td>
      <td>21.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



<h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 01: </span>Remove the column that only contains missing values.</h3>



```python
# reshow dataframe
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stop_date</th>
      <th>stop_time</th>
      <th>country_name</th>
      <th>driver_gender</th>
      <th>driver_age_raw</th>
      <th>driver_age</th>
      <th>driver_race</th>
      <th>violation_raw</th>
      <th>violation</th>
      <th>search_conducted</th>
      <th>search_type</th>
      <th>stop_outcome</th>
      <th>is_arrested</th>
      <th>stop_duration</th>
      <th>drugs_related_stop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/2/2005</td>
      <td>1:55</td>
      <td>NaN</td>
      <td>M</td>
      <td>1985.0</td>
      <td>20.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/18/2005</td>
      <td>8:15</td>
      <td>NaN</td>
      <td>M</td>
      <td>1965.0</td>
      <td>40.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# show all columns missing value
df.isna().sum()
```




    stop_date                 0
    stop_time                 0
    country_name          65535
    driver_gender          4061
    driver_age_raw         4054
    driver_age             4307
    driver_race            4060
    violation_raw          4060
    violation              4060
    search_conducted          0
    search_type           63056
    stop_outcome           4060
    is_arrested            4060
    stop_duration          4060
    drugs_related_stop        0
    dtype: int64




```python
# drop columns
df.drop(columns=('country_name'), inplace=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stop_date</th>
      <th>stop_time</th>
      <th>driver_gender</th>
      <th>driver_age_raw</th>
      <th>driver_age</th>
      <th>driver_race</th>
      <th>violation_raw</th>
      <th>violation</th>
      <th>search_conducted</th>
      <th>search_type</th>
      <th>stop_outcome</th>
      <th>is_arrested</th>
      <th>stop_duration</th>
      <th>drugs_related_stop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/2/2005</td>
      <td>1:55</td>
      <td>M</td>
      <td>1985.0</td>
      <td>20.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/18/2005</td>
      <td>8:15</td>
      <td>M</td>
      <td>1965.0</td>
      <td>40.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/23/2005</td>
      <td>23:15</td>
      <td>M</td>
      <td>1972.0</td>
      <td>33.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/20/2005</td>
      <td>17:15</td>
      <td>M</td>
      <td>1986.0</td>
      <td>19.0</td>
      <td>White</td>
      <td>Call for Service</td>
      <td>Other</td>
      <td>False</td>
      <td>NaN</td>
      <td>Arrest Driver</td>
      <td>True</td>
      <td>16-30 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3/14/2005</td>
      <td>10:00</td>
      <td>F</td>
      <td>1984.0</td>
      <td>21.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



<h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 02: </span>For Speeding, were Man or Women stopped often?</h3>



```python
# filter violation
speeding = df[df['violation'] == 'Speeding']
```


```python
# answer the question 02
ans_02 = speeding['driver_gender'].value_counts()
ans_02
```




    driver_gender
    M    25517
    F    11686
    Name: count, dtype: int64



<h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 03: </span>Does gender affect who gets searched during a stop ?</h3>



```python
# reshow dataframe
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stop_date</th>
      <th>stop_time</th>
      <th>driver_gender</th>
      <th>driver_age_raw</th>
      <th>driver_age</th>
      <th>driver_race</th>
      <th>violation_raw</th>
      <th>violation</th>
      <th>search_conducted</th>
      <th>search_type</th>
      <th>stop_outcome</th>
      <th>is_arrested</th>
      <th>stop_duration</th>
      <th>drugs_related_stop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/2/2005</td>
      <td>1:55</td>
      <td>M</td>
      <td>1985.0</td>
      <td>20.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/18/2005</td>
      <td>8:15</td>
      <td>M</td>
      <td>1965.0</td>
      <td>40.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/23/2005</td>
      <td>23:15</td>
      <td>M</td>
      <td>1972.0</td>
      <td>33.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/20/2005</td>
      <td>17:15</td>
      <td>M</td>
      <td>1986.0</td>
      <td>19.0</td>
      <td>White</td>
      <td>Call for Service</td>
      <td>Other</td>
      <td>False</td>
      <td>NaN</td>
      <td>Arrest Driver</td>
      <td>True</td>
      <td>16-30 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3/14/2005</td>
      <td>10:00</td>
      <td>F</td>
      <td>1984.0</td>
      <td>21.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
groupby_gender = df.groupby('driver_gender')['search_conducted'].sum()
```


```python
groupby_gender
```




    driver_gender
    F     366
    M    2113
    Name: search_conducted, dtype: int64



<h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 04: </span>What is the mean stop_duration ?</h3>



```python
# reshow dataframe
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stop_date</th>
      <th>stop_time</th>
      <th>driver_gender</th>
      <th>driver_age_raw</th>
      <th>driver_age</th>
      <th>driver_race</th>
      <th>violation_raw</th>
      <th>violation</th>
      <th>search_conducted</th>
      <th>search_type</th>
      <th>stop_outcome</th>
      <th>is_arrested</th>
      <th>stop_duration</th>
      <th>drugs_related_stop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/2/2005</td>
      <td>1:55</td>
      <td>M</td>
      <td>1985.0</td>
      <td>20.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/18/2005</td>
      <td>8:15</td>
      <td>M</td>
      <td>1965.0</td>
      <td>40.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/23/2005</td>
      <td>23:15</td>
      <td>M</td>
      <td>1972.0</td>
      <td>33.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/20/2005</td>
      <td>17:15</td>
      <td>M</td>
      <td>1986.0</td>
      <td>19.0</td>
      <td>White</td>
      <td>Call for Service</td>
      <td>Other</td>
      <td>False</td>
      <td>NaN</td>
      <td>Arrest Driver</td>
      <td>True</td>
      <td>16-30 Min</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3/14/2005</td>
      <td>10:00</td>
      <td>F</td>
      <td>1984.0</td>
      <td>21.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# count value in stop duration
df['stop_duration'].value_counts()
```




    stop_duration
    0-15 Min     47379
    16-30 Min    11448
    30+ Min       2647
    2                1
    Name: count, dtype: int64




```python
# category stop duration columns
df['category_stop_duration'] = df['stop_duration'].map({'0-15 Min': 7.5, '16-30 Min': 23, '30+ Min': 45})
```


```python
# answer the question 04
ans_04 = df['category_stop_duration'].mean()
```


```python
ans_04
```




    12.001195627419722



<h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 05: </span>Compare the age distributions for each violation.</h3>



```python
# reshow dataframe
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stop_date</th>
      <th>stop_time</th>
      <th>driver_gender</th>
      <th>driver_age_raw</th>
      <th>driver_age</th>
      <th>driver_race</th>
      <th>violation_raw</th>
      <th>violation</th>
      <th>search_conducted</th>
      <th>search_type</th>
      <th>stop_outcome</th>
      <th>is_arrested</th>
      <th>stop_duration</th>
      <th>drugs_related_stop</th>
      <th>category_stop_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/2/2005</td>
      <td>1:55</td>
      <td>M</td>
      <td>1985.0</td>
      <td>20.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/18/2005</td>
      <td>8:15</td>
      <td>M</td>
      <td>1965.0</td>
      <td>40.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/23/2005</td>
      <td>23:15</td>
      <td>M</td>
      <td>1972.0</td>
      <td>33.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/20/2005</td>
      <td>17:15</td>
      <td>M</td>
      <td>1986.0</td>
      <td>19.0</td>
      <td>White</td>
      <td>Call for Service</td>
      <td>Other</td>
      <td>False</td>
      <td>NaN</td>
      <td>Arrest Driver</td>
      <td>True</td>
      <td>16-30 Min</td>
      <td>False</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3/14/2005</td>
      <td>10:00</td>
      <td>F</td>
      <td>1984.0</td>
      <td>21.0</td>
      <td>White</td>
      <td>Speeding</td>
      <td>Speeding</td>
      <td>False</td>
      <td>NaN</td>
      <td>Citation</td>
      <td>False</td>
      <td>0-15 Min</td>
      <td>False</td>
      <td>7.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# answer the quetion 05
ans_05 = df.groupby('violation')['driver_age'].describe()
```


```python
ans_05
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>violation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Equipment</th>
      <td>6507.0</td>
      <td>31.682957</td>
      <td>11.380671</td>
      <td>16.0</td>
      <td>23.0</td>
      <td>28.0</td>
      <td>39.0</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>Moving violation</th>
      <td>11876.0</td>
      <td>36.736443</td>
      <td>13.258350</td>
      <td>15.0</td>
      <td>25.0</td>
      <td>35.0</td>
      <td>47.0</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>3477.0</td>
      <td>40.362381</td>
      <td>12.754423</td>
      <td>16.0</td>
      <td>30.0</td>
      <td>41.0</td>
      <td>50.0</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>Registration/plates</th>
      <td>2240.0</td>
      <td>32.656696</td>
      <td>11.150780</td>
      <td>16.0</td>
      <td>24.0</td>
      <td>30.0</td>
      <td>40.0</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>Seat belt</th>
      <td>3.0</td>
      <td>30.333333</td>
      <td>10.214369</td>
      <td>23.0</td>
      <td>24.5</td>
      <td>26.0</td>
      <td>34.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>Speeding</th>
      <td>37120.0</td>
      <td>33.262581</td>
      <td>12.615781</td>
      <td>15.0</td>
      <td>23.0</td>
      <td>30.0</td>
      <td>42.0</td>
      <td>88.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
