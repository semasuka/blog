---
title:  "Pandas Exercises Part 2"
image: /assets/post_images/pandas.jpg
excerpt_separator: <!-- more -->
tags:
- python
- exercises
- pandas
---


Welcome back, guys! We will continue with part 2 in this series of Pandas exercise. I am very excited about this post because we will introducing DataFrame, the most used Pandas data structure. I hope you guys will enjoy this post.<!-- more -->

With no further due, let's get started.

We will start by importing Pandas and NumPy


```python
import pandas as pd
import numpy as np
```

### Ex 26: How to get the mean of a series grouped by another series?

Q: Compute the mean of weights of each fruit.


```python
fruits = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
weights = pd.Series(np.linspace(1, 10, 10))
```

#### Desired output


```python
# Keep in mind that your values will be different from mine and you might only randomly select only 2 fruits instead of 3.
```

![Pandas_ex26](/blog/assets/post_cont_image/pandas_ex26.png)

#### Solution


```python
fruits_weights = pd.concat({"fruits":fruits,"weights":weights},axis=1)
```


```python
fruits_weights.groupby(by="fruits").mean()
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
      <th>weights</th>
    </tr>
    <tr>
      <th>fruits</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>apple</th>
      <td>5.4</td>
    </tr>
    <tr>
      <th>banana</th>
      <td>6.5</td>
    </tr>
    <tr>
      <th>carrot</th>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



We concatenate horizontally (by setting the axis = 1) the two series into a dataframe by using the concat function and use that dataframe to group the fruits by the name of the fruit. After the grouping the dataframe, we get the mean of each fruit using the mean function.

### Ex 27: How to compute the euclidean distance between two series?

Q: Compute the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between series (points) p and q, using a packaged formula and another solution without.

Euclidean distance formular:

![Pandas_ex27](/blog/assets/post_cont_image/pandas_ex27.png)


```python
p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
```

#### Desired output


```python
# 18.165
```

#### Solution

#### 1st Method using a built-in function


```python
np.linalg.norm(p-q)
```




    18.16590212458495



We can get the Euclidean distance by calling the NumPy function linalg.norm function and pass in the difference in the two series.

#### 2nd Method without using a built-in function


```python
sum((p - q)**2)**.5
```




    18.16590212458495



Using the Euclidean formula provided, we can use operators to find the Euclidean distance. We first subtract the corresponding elements in the two series and apply 2 as an exponent then sum it up and finally get the square root.  

### Ex 28: How to find all the local maxima (or peaks) in a numeric series?

Q: Get the positions of peaks (values surrounded by smaller values on both sides) in ser.


```python
ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])
```

#### Desired output


```python
# array([1, 5, 7])
```

#### Solution


```python
from scipy.signal import argrelextrema

argrelextrema(ser.values, np.greater)
```




    (array([1, 5, 7]),)



To calculate the relative extrema of the series, we use argrelextrema function from the scipy (Scientific Python) which is a Python library close to NumPy used for mathematics, science, and engineering. 

In that function, we pass in the series and the comparator. Since we are looking for the maxima, in this case, the comparator will be np.greater.

### Ex 29: How to replace missing spaces in a string with the least frequent character?

Q: Replace the spaces in my_str with the least frequent character.


```python
ser = pd.Series(list('dbc deb abed gagde'))
```

#### Desired output


```python
# least frequent element is c

# ['d',
#  'b',
#  'c',
#  'c',
#  'd',
#  'e',
#  'b',
#  'c',
#  'a',
#  'b',
#  'e',
#  'd',
#  'c',
#  'g',
#  'a',
#  'g',
#  'd',
#  'e']
```

#### Solution


```python
from collections import Counter

least_common_char = Counter(ser.replace(" ","")).most_common()[-1][0]
```


```python
Counter(ser.replace(" ","")).most_common()
```




    [('d', 4), ('b', 3), ('', 3), ('e', 3), ('a', 2), ('g', 2), ('c', 1)]




```python
least_common_char
```




    'c'




```python
ser.replace(" ",least_common_char)
```




    ['d',
     'b',
     'c',
     'c',
     'd',
     'e',
     'b',
     'c',
     'a',
     'b',
     'e',
     'd',
     'c',
     'g',
     'a',
     'g',
     'd',
     'e']



To replace the white space with the most common element in the series, we need first to find the most common character in the series. 

To find it, we use the counter function from the collection library. We pass in the series without the white space (by replacing " " by "") and apply to the counter function, the most_common function. We will get back a list of tuples will all characters and their counts in decreasing order. We use -1 to target the last tuple and 0 to get the character in that tuple.

Now that we have the least common character, we can replace all the instances of white space by the least common character.

### Ex 30: How to create a TimeSeries starting ‘2000-01-01’ and 10 weekends (Saturdays) and have random numbers as values?

Q: Create a TimeSeries starting ‘2000-01-01’ and 10 weekends (Saturdays) and have random numbers as values?

#### Desired output


```python
# values will be different due to randomness

# 2000-01-01    4
# 2000-01-08    1
# 2000-01-15    8
# 2000-01-22    4
# 2000-01-29    4
# 2000-02-05    2
# 2000-02-12    4
# 2000-02-19    9
# 2000-02-26    6
# 2000-03-04    6
```

#### Solution


```python
pd.Series(np.random.randint(1,high=10,size=10),pd.date_range(start="2000-01-01",periods=10,freq="W-SAT"))
```




    2000-01-01    1
    2000-01-08    9
    2000-01-15    1
    2000-01-22    3
    2000-01-29    8
    2000-02-05    7
    2000-02-12    3
    2000-02-19    8
    2000-02-26    4
    2000-03-04    7
    Freq: W-SAT, dtype: int64



We create as the values of the series, the ten random numbers from 1 to 10 and as indexes, we create a date_range function which returns a date rage starting from 2000-01-01 and set the number of periods to generate to 10 with the frequency set to Saturday weekly. To get the list of all the frequencies, visit [this link.](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases)

### 31: How to fill an intermittent time series so all missing dates show up with values of previous non-missing date?

Q: 
```ser``` has missing dates and values. Make all missing dates appear and fill up with value from previous date.


```python
import pandas as pd 
import numpy as np
```


```python
ser = pd.Series([1,10,3,np.nan], index=pd.to_datetime(['2000-01-01', '2000-01-03', '2000-01-06', '2000-01-08']))
```

#### Desired output


```python
# 2000-01-01     1.0
# 2000-01-02     1.0
# 2000-01-03    10.0
# 2000-01-04    10.0
# 2000-01-05    10.0
# 2000-01-06     3.0
# 2000-01-07     3.0
# 2000-01-08     NaN
```

#### Solution


```python
ser.resample("D").ffill()
```




    2000-01-01     1.0
    2000-01-02     1.0
    2000-01-03    10.0
    2000-01-04    10.0
    2000-01-05    10.0
    2000-01-06     3.0
    2000-01-07     3.0
    2000-01-08     NaN
    Freq: D, dtype: float64



We use the resample function to fill up all the days that are missing starting from ```2000-01-01``` to ```2000-01-08``` and uses the character ```D``` to specify that we want days as the interval. We then use ```ffill``` function to fill up the missing values from the previous row et voila!

### 32: How to compute the autocorrelations of a numeric series?

Q: Compute autocorrelations for the first 10 lags of ser. Find out which lag has the largest correlation.


```python
ser = pd.Series(np.arange(20) + np.random.normal(1, 10, 20))
```

#### Desired output


```python
# values will change due to randomness

# [-0.462232351922819,
#  0.24702149262453904,
#  -0.3667824631718427,
#  0.09378057953432406,
#  0.3382941938771548,
#  -0.04450324725676436,
#  0.16361925861505003,
#  -0.5351035019540977,
#  0.26359968436232056,
#  0.03944833988252732]
# the lag with the highest correlation is 8
```

#### Solution


```python
autocorr = [ser.autocorr(lag=i) for i in range(11)][1:]
```


```python
autocorr
```




    [-0.462232351922819,
     0.24702149262453904,
     -0.3667824631718427,
     0.09378057953432406,
     0.3382941938771548,
     -0.04450324725676436,
     0.16361925861505003,
     -0.5351035019540977,
     0.26359968436232056,
     0.03944833988252732]




```python
print("the lag with the highest correlation is {}".format(np.argmax(np.abs(autocorr))+1))
```

    the lag with the highest correlation is 8


We first have to calculate the correlation between each consecutive number and to do that we loop through all the elements in the series using range and list comprehension. We use indexing to ignore the first correlation since the correction with the same element is 1.

After finding all the correlation, it is time to find the position of the largest correlation. To do this, we use the NumPy function ```argmax``` to get back the position of the largest absolute (by changing negative correlation to positive) correlation number and add 1 since the count starts from 0.

### Ex 33: How to import only every nth row from a csv file to create a dataframe?

Q: Import every 50th row of [BostonHousing dataset]("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv") as a dataframe.

#### Desired output

![Pandas_ex33](/blog/assets/post_cont_image/pandas_ex33.png)

#### Solution


```python
boston_housing_dataset = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
```


```python
boston_housing_dataset.head()
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
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>b</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
boston_housing_dataset[::50]
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
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>b</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.08873</td>
      <td>21.0</td>
      <td>5.64</td>
      <td>0</td>
      <td>0.439</td>
      <td>5.963</td>
      <td>45.7</td>
      <td>6.8147</td>
      <td>4</td>
      <td>243</td>
      <td>16.8</td>
      <td>395.56</td>
      <td>13.45</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.14866</td>
      <td>0.0</td>
      <td>8.56</td>
      <td>0</td>
      <td>0.520</td>
      <td>6.727</td>
      <td>79.9</td>
      <td>2.7778</td>
      <td>5</td>
      <td>384</td>
      <td>20.9</td>
      <td>394.76</td>
      <td>9.42</td>
      <td>27.5</td>
    </tr>
    <tr>
      <th>150</th>
      <td>1.65660</td>
      <td>0.0</td>
      <td>19.58</td>
      <td>0</td>
      <td>0.871</td>
      <td>6.122</td>
      <td>97.3</td>
      <td>1.6180</td>
      <td>5</td>
      <td>403</td>
      <td>14.7</td>
      <td>372.80</td>
      <td>14.10</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0.01778</td>
      <td>95.0</td>
      <td>1.47</td>
      <td>0</td>
      <td>0.403</td>
      <td>7.135</td>
      <td>13.9</td>
      <td>7.6534</td>
      <td>3</td>
      <td>402</td>
      <td>17.0</td>
      <td>384.30</td>
      <td>4.45</td>
      <td>32.9</td>
    </tr>
    <tr>
      <th>250</th>
      <td>0.14030</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0</td>
      <td>0.431</td>
      <td>6.487</td>
      <td>13.0</td>
      <td>7.3967</td>
      <td>7</td>
      <td>330</td>
      <td>19.1</td>
      <td>396.28</td>
      <td>5.90</td>
      <td>24.4</td>
    </tr>
    <tr>
      <th>300</th>
      <td>0.04417</td>
      <td>70.0</td>
      <td>2.24</td>
      <td>0</td>
      <td>0.400</td>
      <td>6.871</td>
      <td>47.4</td>
      <td>7.8278</td>
      <td>5</td>
      <td>358</td>
      <td>14.8</td>
      <td>390.86</td>
      <td>6.07</td>
      <td>24.8</td>
    </tr>
    <tr>
      <th>350</th>
      <td>0.06211</td>
      <td>40.0</td>
      <td>1.25</td>
      <td>0</td>
      <td>0.429</td>
      <td>6.490</td>
      <td>44.4</td>
      <td>8.7921</td>
      <td>1</td>
      <td>335</td>
      <td>19.7</td>
      <td>396.90</td>
      <td>5.98</td>
      <td>22.9</td>
    </tr>
    <tr>
      <th>400</th>
      <td>25.04610</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0</td>
      <td>0.693</td>
      <td>5.987</td>
      <td>100.0</td>
      <td>1.5888</td>
      <td>24</td>
      <td>666</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>26.77</td>
      <td>5.6</td>
    </tr>
    <tr>
      <th>450</th>
      <td>6.71772</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0</td>
      <td>0.713</td>
      <td>6.749</td>
      <td>92.6</td>
      <td>2.3236</td>
      <td>24</td>
      <td>666</td>
      <td>20.2</td>
      <td>0.32</td>
      <td>17.44</td>
      <td>13.4</td>
    </tr>
    <tr>
      <th>500</th>
      <td>0.22438</td>
      <td>0.0</td>
      <td>9.69</td>
      <td>0</td>
      <td>0.585</td>
      <td>6.027</td>
      <td>79.7</td>
      <td>2.4982</td>
      <td>6</td>
      <td>391</td>
      <td>19.2</td>
      <td>396.90</td>
      <td>14.33</td>
      <td>16.8</td>
    </tr>
  </tbody>
</table>
</div>



To import a csv(comma-separated value) dataset, we use the ```read_csv``` function and pass in the link to the csv file. Now we have the dataset imported and stored in the ```boston_housing_dataset```. 

To get every 50th row in the dataset, we use indexing with a step of 50.

### Ex 34: How to change column values when importing csv to a dataframe?

Q: Import [the  BostonHousing dataset]("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv") dataset, but while importing change the 'medv' (median house value) column so that values < 25 becomes ‘Low’ and > 25 becomes ‘High’.

#### Desired output

![Pandas_ex34](/blog/assets/post_cont_image/pandas_ex34.png)

#### Solution


```python
boston_housing_dataset = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",converters={"medv": lambda x: "High" if float(x) > 25 else "Low"})
```


```python
boston_housing_dataset.head()
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
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>b</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>High</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>High</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>High</td>
    </tr>
  </tbody>
</table>
</div>



To change the value of a column while importing the dataset, we use the parameter converters from the read_csv function. We pass in a dictionary with a key corresponding to the name of the column we want to change and value to be a lambda expression where "High" is for the value greater than 25 and "Low" is for the value less than 25.

### Ex 35: How to create a dataframe with rows as strides from a given series?

Q: Create a dataframe with rows as strides from a given series with 2 as the stride length and 4 as length of each row.


```python
the_series = pd.Series(range(15))
```

#### Desired output


```python
# array([[ 0,  1,  2,  3],
#        [ 2,  3,  4,  5],
#        [ 4,  5,  6,  7],
#        [ 6,  7,  8,  9],
#        [ 8,  9, 10, 11],
#        [10, 11, 12, 13]])
```

#### Solution


```python
def gen_strides(the_series, stride_len=2, window_len=4):
    n_strides = ((the_series.size - window_len)//stride_len) + 1
    return np.array([the_series[i:(i+window_len)] for i in np.arange(0,the_series.size,stride_len)[:n_strides]])

gen_strides(the_series)

```

    [0 2 4]


Strides are used in CNN (convolutional neural network), which will be covered in a future post. To get the stride, we first create a function that takes in the stride length 2 (which mean that the two last elements in the row will be the same as the first two elements in the following row) and a window length of 4 (corresponding to the number of elements in the row or the number of columns).

We need first to get the numbers of rows by subtracting the size of the series by the number of element desired in a row and floor divide with the stride length and finally add 1.

We proceed by using the list comprehension by looping through an array that starts from 0, step by ```stride_len``` and stops at ```the_series.size```. We use indexing with ```n_stride``` to get the first six elements because we only have six rows. 

Now it is time to populate the rows, for each row we use the original series and start from index ```i``` to index ```i + window_len``` to get the strides.

### Ex 36: How to import only specified columns from a csv file?

Q: Import ```crim``` and ```medv``` columns of the [BostonHousing dataset](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv) as a dataframe.

#### Desired output

![Pandas_ex36](/blog/assets/post_cont_image/pandas_ex36.png)

#### Solution


```python
pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",usecols=["crim","medv"]).head()
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
      <th>crim</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



To import only the ```crim``` and the ```medv``` column, we pass a list of the names of those two columns the ```usecols``` parameter.

### Ex 37: How to get the nrows, ncolumns, datatype, summary stats of each column of a dataframe? Also, get the array and list equivalent.

Q: Get the number of rows, columns, datatype, columns for  each datatype and statistical summary of each column of the [Cars93]("https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv") dataset. Also, get the numpy array and list equivalent of the dataframe.

#### Desired output


```python
# Rows and columns

#(93, 27)

# Datatype

# Manufacturer           object
# Model                  object
# Type                   object
# Min.Price             float64
# Price                 float64
# Max.Price             float64
# MPG.city              float64
# MPG.highway           float64
# AirBags                object
# DriveTrain             object
# Cylinders              object
# EngineSize            float64
# Horsepower            float64
# RPM                   float64
# Rev.per.mile          float64
# Man.trans.avail        object
# Fuel.tank.capacity    float64
# Passengers            float64
# Length                float64
# Wheelbase             float64
# Width                 float64
# Turn.circle           float64
# Rear.seat.room        float64
# Luggage.room          float64
# Weight                float64
# Origin                 object
# Make                   object
# dtype: object

# Columns for each datatype

# float64    18
# object      9
# dtype: int64

# Statistical summary

# Min.Price	Price	Max.Price	MPG.city	MPG.highway	EngineSize	Horsepower	RPM	Rev.per.mile	Fuel.tank.capacity	Passengers	Length	Wheelbase	Width	Turn.circle	Rear.seat.room	Luggage.room	Weight
# count	86.000000	91.000000	88.000000	84.000000	91.000000	91.000000	86.000000	90.000000	87.000000	85.000000	91.000000	89.000000	92.000000	87.000000	88.000000	89.000000	74.000000	86.000000
# mean	17.118605	19.616484	21.459091	22.404762	29.065934	2.658242	144.000000	5276.666667	2355.000000	16.683529	5.076923	182.865169	103.956522	69.448276	38.954545	27.853933	13.986486	3104.593023
# std	8.828290	9.724280	10.696563	5.841520	5.370293	1.045845	53.455204	605.554811	486.916616	3.375748	1.045953	14.792651	6.856317	3.778023	3.304157	3.018129	3.120824	600.129993
# min	6.700000	7.400000	7.900000	15.000000	20.000000	1.000000	55.000000	3800.000000	1320.000000	9.200000	2.000000	141.000000	90.000000	60.000000	32.000000	19.000000	6.000000	1695.000000
# 25%	10.825000	12.350000	14.575000	18.000000	26.000000	1.800000	100.750000	4800.000000	2017.500000	14.500000	4.000000	174.000000	98.000000	67.000000	36.000000	26.000000	12.000000	2647.500000
# 50%	14.600000	17.700000	19.150000	21.000000	28.000000	2.300000	140.000000	5200.000000	2360.000000	16.500000	5.000000	181.000000	103.000000	69.000000	39.000000	27.500000	14.000000	3085.000000
# 75%	20.250000	23.500000	24.825000	25.000000	31.000000	3.250000	170.000000	5787.500000	2565.000000	19.000000	6.000000	192.000000	110.000000	72.000000	42.000000	30.000000	16.000000	3567.500000
# max	45.400000	61.900000	80.000000	46.000000	50.000000	5.700000	300.000000	6500.000000	3755.000000	27.000000	8.000000	219.000000	119.000000	78.000000	45.000000	36.000000	22.000000	4105.000000

# NumPy array

# array(['Acura', 'Integra', 'Small', 12.9, 15.9, 18.8, 25.0, 31.0, 'None',
#        'Front', '4', 1.8, 140.0, 6300.0, 2890.0, 'Yes', 13.2, 5.0, 177.0,
#        102.0, 68.0, 37.0, 26.5, nan, 2705.0, 'non-USA', 'Acura Integra'],
#       dtype=object)

# List

# ['Acura',
#   'Integra',
#   'Small',
#   12.9,
#   15.9,
#   18.8,
#   25.0,
#   31.0,
#   'None',
#   'Front',
#   '4',
#   1.8,
#   140.0,
#   6300.0,
#   2890.0,
#   'Yes',
#   13.2,
#   5.0,
#   177.0,
#   102.0,
#   68.0,
#   37.0,
#   26.5,
#   nan,
#   2705.0,
#   'non-USA',
#   'Acura Integra']

```

#### Solution


```python
cars_dataset = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv")
```

#### Shape


```python
cars_dataset.shape
```




    (93, 27)



We call the shape function on the dataset, we back a tuple with the first element as the number of rows and the second element is the number of columns in the dataframe.

#### Datatype


```python
cars_dataset.dtypes
```




    Manufacturer           object
    Model                  object
    Type                   object
    Min.Price             float64
    Price                 float64
    Max.Price             float64
    MPG.city              float64
    MPG.highway           float64
    AirBags                object
    DriveTrain             object
    Cylinders              object
    EngineSize            float64
    Horsepower            float64
    RPM                   float64
    Rev.per.mile          float64
    Man.trans.avail        object
    Fuel.tank.capacity    float64
    Passengers            float64
    Length                float64
    Wheelbase             float64
    Width                 float64
    Turn.circle           float64
    Rear.seat.room        float64
    Luggage.room          float64
    Weight                float64
    Origin                 object
    Make                   object
    dtype: object



To get the datatype for each column, we call the ```dtypes``` function on the dataset. 

#### Columns for each datatype


```python
cars_dataset.dtypes.value_counts()
```




    float64    18
    object      9
    dtype: int64



To get the columns count for each datatype, we call the ```value_counts``` on the ```dtype``` function.

#### Statistical summary


```python
cars_dataset.describe()
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
      <th>Min.Price</th>
      <th>Price</th>
      <th>Max.Price</th>
      <th>MPG.city</th>
      <th>MPG.highway</th>
      <th>EngineSize</th>
      <th>Horsepower</th>
      <th>RPM</th>
      <th>Rev.per.mile</th>
      <th>Fuel.tank.capacity</th>
      <th>Passengers</th>
      <th>Length</th>
      <th>Wheelbase</th>
      <th>Width</th>
      <th>Turn.circle</th>
      <th>Rear.seat.room</th>
      <th>Luggage.room</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>86.000000</td>
      <td>91.000000</td>
      <td>88.000000</td>
      <td>84.000000</td>
      <td>91.000000</td>
      <td>91.000000</td>
      <td>86.000000</td>
      <td>90.000000</td>
      <td>87.000000</td>
      <td>85.000000</td>
      <td>91.000000</td>
      <td>89.000000</td>
      <td>92.000000</td>
      <td>87.000000</td>
      <td>88.000000</td>
      <td>89.000000</td>
      <td>74.000000</td>
      <td>86.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>17.118605</td>
      <td>19.616484</td>
      <td>21.459091</td>
      <td>22.404762</td>
      <td>29.065934</td>
      <td>2.658242</td>
      <td>144.000000</td>
      <td>5276.666667</td>
      <td>2355.000000</td>
      <td>16.683529</td>
      <td>5.076923</td>
      <td>182.865169</td>
      <td>103.956522</td>
      <td>69.448276</td>
      <td>38.954545</td>
      <td>27.853933</td>
      <td>13.986486</td>
      <td>3104.593023</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.828290</td>
      <td>9.724280</td>
      <td>10.696563</td>
      <td>5.841520</td>
      <td>5.370293</td>
      <td>1.045845</td>
      <td>53.455204</td>
      <td>605.554811</td>
      <td>486.916616</td>
      <td>3.375748</td>
      <td>1.045953</td>
      <td>14.792651</td>
      <td>6.856317</td>
      <td>3.778023</td>
      <td>3.304157</td>
      <td>3.018129</td>
      <td>3.120824</td>
      <td>600.129993</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.700000</td>
      <td>7.400000</td>
      <td>7.900000</td>
      <td>15.000000</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>55.000000</td>
      <td>3800.000000</td>
      <td>1320.000000</td>
      <td>9.200000</td>
      <td>2.000000</td>
      <td>141.000000</td>
      <td>90.000000</td>
      <td>60.000000</td>
      <td>32.000000</td>
      <td>19.000000</td>
      <td>6.000000</td>
      <td>1695.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>10.825000</td>
      <td>12.350000</td>
      <td>14.575000</td>
      <td>18.000000</td>
      <td>26.000000</td>
      <td>1.800000</td>
      <td>100.750000</td>
      <td>4800.000000</td>
      <td>2017.500000</td>
      <td>14.500000</td>
      <td>4.000000</td>
      <td>174.000000</td>
      <td>98.000000</td>
      <td>67.000000</td>
      <td>36.000000</td>
      <td>26.000000</td>
      <td>12.000000</td>
      <td>2647.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>14.600000</td>
      <td>17.700000</td>
      <td>19.150000</td>
      <td>21.000000</td>
      <td>28.000000</td>
      <td>2.300000</td>
      <td>140.000000</td>
      <td>5200.000000</td>
      <td>2360.000000</td>
      <td>16.500000</td>
      <td>5.000000</td>
      <td>181.000000</td>
      <td>103.000000</td>
      <td>69.000000</td>
      <td>39.000000</td>
      <td>27.500000</td>
      <td>14.000000</td>
      <td>3085.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>20.250000</td>
      <td>23.500000</td>
      <td>24.825000</td>
      <td>25.000000</td>
      <td>31.000000</td>
      <td>3.250000</td>
      <td>170.000000</td>
      <td>5787.500000</td>
      <td>2565.000000</td>
      <td>19.000000</td>
      <td>6.000000</td>
      <td>192.000000</td>
      <td>110.000000</td>
      <td>72.000000</td>
      <td>42.000000</td>
      <td>30.000000</td>
      <td>16.000000</td>
      <td>3567.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>45.400000</td>
      <td>61.900000</td>
      <td>80.000000</td>
      <td>46.000000</td>
      <td>50.000000</td>
      <td>5.700000</td>
      <td>300.000000</td>
      <td>6500.000000</td>
      <td>3755.000000</td>
      <td>27.000000</td>
      <td>8.000000</td>
      <td>219.000000</td>
      <td>119.000000</td>
      <td>78.000000</td>
      <td>45.000000</td>
      <td>36.000000</td>
      <td>22.000000</td>
      <td>4105.000000</td>
    </tr>
  </tbody>
</table>
</div>



To get the statistical summary (mean, std, percentile, min, max and count), we call the ```describe``` function on the dataset.

#### Dataframe to NumPy


```python
cars_dataset.iloc[0].to_numpy()
```




    array(['Acura', 'Integra', 'Small', 12.9, 15.9, 18.8, 25.0, 31.0, 'None',
           'Front', '4', 1.8, 140.0, 6300.0, 2890.0, 'Yes', 13.2, 5.0, 177.0,
           102.0, 68.0, 37.0, 26.5, nan, 2705.0, 'non-USA', 'Acura Integra'],
          dtype=object)



We extract the first row and call the ```to_numpy``` function to cast the row to a NumPy array. It is also possible to cast the whole dataframe to an array. 

#### Dataframe to list


```python
cars_dataset.iloc[0].values.tolist()
```




    ['Acura',
     'Integra',
     'Small',
     12.9,
     15.9,
     18.8,
     25.0,
     31.0,
     'None',
     'Front',
     '4',
     1.8,
     140.0,
     6300.0,
     2890.0,
     'Yes',
     13.2,
     5.0,
     177.0,
     102.0,
     68.0,
     37.0,
     26.5,
     nan,
     2705.0,
     'non-USA',
     'Acura Integra']



We extract the first row and call the ```tolist``` function on the ```values``` function to cast the row to a list. It is also possible to cast the whole dataframe to a list. 

### Ex 38: How to extract the row and column number of a particular cell with a given criterion?

Q: Which manufacturer, model and type have the highest Price? What are the row and column number of the cell with the highest Price value?


```python
cars_dataset = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```

#### Desired output


```python
# Manufacturer with the highest price

# 'Mercedes-Benz'

# Model with the highest price

# '300E'

# Type with the highest price

# 'Midsize'

# row and column number of the cell with the highest Price value

# (array([58]), array([4]))
```

#### Solution

#### Manufacturer, model and midsize with the highest price


```python
cars_dataset.iloc[np.argmax(cars_dataset["Price"])]["Manufacturer"]
```




    'Mercedes-Benz'




```python
cars_dataset.iloc[np.argmax(cars_dataset["Price"])]["Model"]
```




    '300E'




```python
cars_dataset.iloc[np.argmax(cars_dataset["Price"])]["Type"]
```




    'Midsize'



We first find the row with the highest price using the NumPy ```argmax``` function by passing in the price column as an argument. With the index number of the row with the highest price, we use ```iloc``` to get all the columns of that row as Series and then use indexing on that Series to get the manufacturer, model and type.

#### Row and column with the highest price


```python
np.where(cars_dataset.values == np.max(cars_dataset["Price"]))
```




    (array([58]), array([4]))



We use the NumPy ```where``` function to compares all the values in the dataset with the highest price and returns a tuple with the row and column.

### Ex 39: How to rename a specific column in a dataframe?

Q: Rename the column Type as CarType in df and replace the ‘.’ in column names with ‘_’.


```python
cars_dataset = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```


```python
cars_dataset.columns
```




    Index(['Manufacturer', 'Model', 'Type', 'Min.Price', 'Price', 'Max.Price',
           'MPG.city', 'MPG.highway', 'AirBags', 'DriveTrain', 'Cylinders',
           'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 'Man.trans.avail',
           'Fuel.tank.capacity', 'Passengers', 'Length', 'Wheelbase', 'Width',
           'Turn.circle', 'Rear.seat.room', 'Luggage.room', 'Weight', 'Origin',
           'Make'],
          dtype='object')



#### Desired output


```python
# Index(['Manufacturer', 'Model', 'Type', 'Min_Price', 'Price', 'Max_Price',
#        'MPG_city', 'MPG_highway', 'AirBags', 'DriveTrain', 'Cylinders',
#        'EngineSize', 'Horsepower', 'RPM', 'Rev_per_mile', 'Man_trans_avail',
#        'Fuel_tank_capacity', 'Passengers', 'Length', 'Wheelbase', 'Width',
#        'Turn_circle', 'Rear_seat_room', 'Luggage_room', 'Weight', 'Origin',
#        'Make'],
#       dtype='object')
```

#### Solution


```python
cars_dataset.columns.str.replace(".","_")
```




    Index(['Manufacturer', 'Model', 'Type', 'Min_Price', 'Price', 'Max_Price',
           'MPG_city', 'MPG_highway', 'AirBags', 'DriveTrain', 'Cylinders',
           'EngineSize', 'Horsepower', 'RPM', 'Rev_per_mile', 'Man_trans_avail',
           'Fuel_tank_capacity', 'Passengers', 'Length', 'Wheelbase', 'Width',
           'Turn_circle', 'Rear_seat_room', 'Luggage_room', 'Weight', 'Origin',
           'Make'],
          dtype='object')



To replace all the occurrences of ```.``` by ```_``` we use the ```str``` function which captures the columns as string data type and uses the ```replace``` function to achieve the column's name change.

### Ex 40: How to check if a dataframe has any missing values?

Q: Check if ```cars_dataset``` has any missing values.


```python
cars_dataset = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```

#### Desired output


```python
# There are nan values in the dataset, so true has to be returned

# True
```

#### Solution


```python
cars_dataset.isnull().values.any()
```




    True



To check if there is any ```nan``` value in the whole dataset, we first use the ```isnull``` function to return booleans in each cell of the dataframe. ```true``` will be returned in a cell with a ```nan``` value and ```false``` will be a cell without a ```nan``` value. So to check if there any ```nan``` in the dataset, we use ```any``` on the values in the dataset. If there is at least one ```nan``` value ```true``` will be returned otherwise it will be```false```.

### Ex 41: How to count the number of missing values in each column?

Q: Count the number of missing values in each column of df. Which column has the maximum number of missing values?


```python
cars_dataset = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```

#### Desired output


```python
# Count of missing values in each column

# Manufacturer           4
# Model                  1
# Type                   3
# Min.Price              7
# Price                  2
# Max.Price              5
# MPG.city               9
# MPG.highway            2
# AirBags                6
# DriveTrain             7
# Cylinders              5
# EngineSize             2
# Horsepower             7
# RPM                    3
# Rev.per.mile           6
# Man.trans.avail        5
# Fuel.tank.capacity     8
# Passengers             2
# Length                 4
# Wheelbase              1
# Width                  6
# Turn.circle            5
# Rear.seat.room         4
# Luggage.room          19
# Weight                 7
# Origin                 5
# Make                   3
# dtype: int64

# column with the maximum number of missing values

# 'Luggage.room'
```

#### Solution


```python
nan_per_column = cars_dataset.isnull().sum()
```


```python
nan_per_column
```




    Manufacturer           4
    Model                  1
    Type                   3
    Min.Price              7
    Price                  2
    Max.Price              5
    MPG.city               9
    MPG.highway            2
    AirBags                6
    DriveTrain             7
    Cylinders              5
    EngineSize             2
    Horsepower             7
    RPM                    3
    Rev.per.mile           6
    Man.trans.avail        5
    Fuel.tank.capacity     8
    Passengers             2
    Length                 4
    Wheelbase              1
    Width                  6
    Turn.circle            5
    Rear.seat.room         4
    Luggage.room          19
    Weight                 7
    Origin                 5
    Make                   3
    dtype: int64




```python
cars_data.columns[nan_per_column.argmax()]
```




    'Luggage.room'



To get the count of ```nan``` values in each column, we use the ```isnull``` function. This function will return ```True``` in each cell where there is a ```nan``` value or ```False``` where there is any other value other than ```nan```. Finally, we apply the ```sum``` function, to get back a series where the indexes are the column names and the values are the count of ```nan``` value in each column.

To know which column has the highest number of ```nan```, we use the ```argmax``` function which will return the index (column name in this case) with the highest count.

### Ex 42: How to replace missing values of multiple numeric columns with the mean?

Q: Replace missing values in ```Min.Price``` and ```Max.Price``` columns with their respective mean.


```python
cars_dataset = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```

#### Desired output

![Pandas_ex42](/blog/assets/post_cont_image/pandas_ex42.png)

#### Solution

#### 1st Method


```python
min_price_mean = round(cars_dataset["Min.Price"].mean(),2)
```


```python
max_price_mean = round(cars_dataset["Max.Price"].mean(),2)
```


```python
cars_dataset["Min.Price"].replace(np.nan,min_price_mean,inplace=True)
```


```python
cars_dataset["Max.Price"].replace(np.nan,max_price_mean,inplace=True)
```


```python
cars_dataset.head()
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
      <th>Manufacturer</th>
      <th>Model</th>
      <th>Type</th>
      <th>Min.Price</th>
      <th>Price</th>
      <th>Max.Price</th>
      <th>MPG.city</th>
      <th>MPG.highway</th>
      <th>AirBags</th>
      <th>DriveTrain</th>
      <th>...</th>
      <th>Passengers</th>
      <th>Length</th>
      <th>Wheelbase</th>
      <th>Width</th>
      <th>Turn.circle</th>
      <th>Rear.seat.room</th>
      <th>Luggage.room</th>
      <th>Weight</th>
      <th>Origin</th>
      <th>Make</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Acura</td>
      <td>Integra</td>
      <td>Small</td>
      <td>12.90</td>
      <td>15.9</td>
      <td>18.80</td>
      <td>25.0</td>
      <td>31.0</td>
      <td>None</td>
      <td>Front</td>
      <td>...</td>
      <td>5.0</td>
      <td>177.0</td>
      <td>102.0</td>
      <td>68.0</td>
      <td>37.0</td>
      <td>26.5</td>
      <td>NaN</td>
      <td>2705.0</td>
      <td>non-USA</td>
      <td>Acura Integra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Legend</td>
      <td>Midsize</td>
      <td>29.20</td>
      <td>33.9</td>
      <td>38.70</td>
      <td>18.0</td>
      <td>25.0</td>
      <td>Driver &amp; Passenger</td>
      <td>Front</td>
      <td>...</td>
      <td>5.0</td>
      <td>195.0</td>
      <td>115.0</td>
      <td>71.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>15.0</td>
      <td>3560.0</td>
      <td>non-USA</td>
      <td>Acura Legend</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Audi</td>
      <td>90</td>
      <td>Compact</td>
      <td>25.90</td>
      <td>29.1</td>
      <td>32.30</td>
      <td>20.0</td>
      <td>26.0</td>
      <td>Driver only</td>
      <td>Front</td>
      <td>...</td>
      <td>5.0</td>
      <td>180.0</td>
      <td>102.0</td>
      <td>67.0</td>
      <td>37.0</td>
      <td>28.0</td>
      <td>14.0</td>
      <td>3375.0</td>
      <td>non-USA</td>
      <td>Audi 90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Audi</td>
      <td>100</td>
      <td>Midsize</td>
      <td>17.12</td>
      <td>37.7</td>
      <td>44.60</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>Driver &amp; Passenger</td>
      <td>NaN</td>
      <td>...</td>
      <td>6.0</td>
      <td>193.0</td>
      <td>106.0</td>
      <td>NaN</td>
      <td>37.0</td>
      <td>31.0</td>
      <td>17.0</td>
      <td>3405.0</td>
      <td>non-USA</td>
      <td>Audi 100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>535i</td>
      <td>Midsize</td>
      <td>17.12</td>
      <td>30.0</td>
      <td>21.46</td>
      <td>22.0</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>Rear</td>
      <td>...</td>
      <td>4.0</td>
      <td>186.0</td>
      <td>109.0</td>
      <td>69.0</td>
      <td>39.0</td>
      <td>27.0</td>
      <td>13.0</td>
      <td>3640.0</td>
      <td>non-USA</td>
      <td>BMW 535i</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



The first way to solve this problem is to use the ```replace``` function. First, we need to find the mean of the ```Min.Price``` and ```Max.Price``` columns and round it by two decimal places using ```round```. 

After that, we select the columns and use the ```replace``` function to replace all the occurrences of the ```nan``` value with the means previously found and finally set the ```inplace``` argument to ```True```, to apply the changes on the original dataset.

#### 2nd Method


```python
cars_dataset[["Min.Price","Max.Price"]] = cars_dataset[["Min.Price","Max.Price"]].apply(lambda x: x.fillna(round(x.mean(),2)))
```


```python
cars_dataset.head()
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
      <th>Manufacturer</th>
      <th>Model</th>
      <th>Type</th>
      <th>Min.Price</th>
      <th>Price</th>
      <th>Max.Price</th>
      <th>MPG.city</th>
      <th>MPG.highway</th>
      <th>AirBags</th>
      <th>DriveTrain</th>
      <th>...</th>
      <th>Passengers</th>
      <th>Length</th>
      <th>Wheelbase</th>
      <th>Width</th>
      <th>Turn.circle</th>
      <th>Rear.seat.room</th>
      <th>Luggage.room</th>
      <th>Weight</th>
      <th>Origin</th>
      <th>Make</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Acura</td>
      <td>Integra</td>
      <td>Small</td>
      <td>12.90</td>
      <td>15.9</td>
      <td>18.80</td>
      <td>25.0</td>
      <td>31.0</td>
      <td>None</td>
      <td>Front</td>
      <td>...</td>
      <td>5.0</td>
      <td>177.0</td>
      <td>102.0</td>
      <td>68.0</td>
      <td>37.0</td>
      <td>26.5</td>
      <td>NaN</td>
      <td>2705.0</td>
      <td>non-USA</td>
      <td>Acura Integra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Legend</td>
      <td>Midsize</td>
      <td>29.20</td>
      <td>33.9</td>
      <td>38.70</td>
      <td>18.0</td>
      <td>25.0</td>
      <td>Driver &amp; Passenger</td>
      <td>Front</td>
      <td>...</td>
      <td>5.0</td>
      <td>195.0</td>
      <td>115.0</td>
      <td>71.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>15.0</td>
      <td>3560.0</td>
      <td>non-USA</td>
      <td>Acura Legend</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Audi</td>
      <td>90</td>
      <td>Compact</td>
      <td>25.90</td>
      <td>29.1</td>
      <td>32.30</td>
      <td>20.0</td>
      <td>26.0</td>
      <td>Driver only</td>
      <td>Front</td>
      <td>...</td>
      <td>5.0</td>
      <td>180.0</td>
      <td>102.0</td>
      <td>67.0</td>
      <td>37.0</td>
      <td>28.0</td>
      <td>14.0</td>
      <td>3375.0</td>
      <td>non-USA</td>
      <td>Audi 90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Audi</td>
      <td>100</td>
      <td>Midsize</td>
      <td>17.12</td>
      <td>37.7</td>
      <td>44.60</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>Driver &amp; Passenger</td>
      <td>NaN</td>
      <td>...</td>
      <td>6.0</td>
      <td>193.0</td>
      <td>106.0</td>
      <td>NaN</td>
      <td>37.0</td>
      <td>31.0</td>
      <td>17.0</td>
      <td>3405.0</td>
      <td>non-USA</td>
      <td>Audi 100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>535i</td>
      <td>Midsize</td>
      <td>17.12</td>
      <td>30.0</td>
      <td>21.46</td>
      <td>22.0</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>Rear</td>
      <td>...</td>
      <td>4.0</td>
      <td>186.0</td>
      <td>109.0</td>
      <td>69.0</td>
      <td>39.0</td>
      <td>27.0</td>
      <td>13.0</td>
      <td>3640.0</td>
      <td>non-USA</td>
      <td>BMW 535i</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



The alternative way to solve this is to use the ```fillna``` function. We start by selecting the two columns and then use the apply function with the lambda expression and pass in the ```mean``` function of ```x``` (```x``` is the column selected). Then we assign the expression to the columns to make it in place.

### 43 : How to use the ```apply``` function on existing columns with global variables as additional arguments?

Q: In dataframe, use the ```apply``` method to replace the missing values in ```Min.Price``` with the column’s mean and those in ```Max.Price``` with the column’s median.


```python
cars_dataset = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```

#### Desired output

![Pandas_ex43](/blog/assets/post_cont_image/pandas_ex43.png)

#### Solution


```python
dict_fun = {"Min.Price":np.nanmean,"Max.Price":np.nanmedian}

cars_dataset[["Min.Price","Max.Price"]].apply(lambda x, dict_fun: x.fillna(dict_fun[x.name](x)),args=(dict_fun,))
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
      <th>Min.Price</th>
      <th>Max.Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.900000</td>
      <td>18.80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29.200000</td>
      <td>38.70</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25.900000</td>
      <td>32.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17.118605</td>
      <td>44.60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.118605</td>
      <td>19.15</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>16.600000</td>
      <td>22.70</td>
    </tr>
    <tr>
      <th>89</th>
      <td>17.600000</td>
      <td>22.40</td>
    </tr>
    <tr>
      <th>90</th>
      <td>22.900000</td>
      <td>23.70</td>
    </tr>
    <tr>
      <th>91</th>
      <td>21.800000</td>
      <td>23.50</td>
    </tr>
    <tr>
      <th>92</th>
      <td>24.800000</td>
      <td>28.50</td>
    </tr>
  </tbody>
</table>
<p>93 rows × 2 columns</p>
</div>



We first create a dictionary with ```Min.Price``` column as the key and value as the NumPy function to calculate the mean, then the second value has ```Max.Price``` column as the key and NumPy median function as the value.

We select the ```Min.Price``` and ```Max.Price``` columns, apply the lambda expression, and replace in each instance of ```nan``` in ```x```(representing the columns) by the mean or median of that column using the ```fillna``` function and pass in the dictionary's value.

__Note:__ Refer to [this]("https://stackoverflow.com/questions/32437435/passing-additional-arguments-to-python-pandas-dataframe-apply") StackOverflow question to learn more.

### 44: How to select a specific column from a dataframe as a dataframe instead of a series?

Q: Get the first column ```a``` in dataframe as a dataframe (rather than as a Series).


```python
tab = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))
```

#### Desired output

![Pandas_ex44](/blog/assets/post_cont_image/pandas_ex44.png)


```python
# Data type

# pandas.core.frame.DataFrame
```

#### Solution

#### 1st Method


```python
type(the_dataframe["a"].to_frame())
```




    pandas.core.frame.DataFrame



The first method is straight forward, we get column ```a``` as a series and then we cast it to a dataframe. 

#### 2nd Method


```python
type(the_dataframe.loc[:,["a"]])
```




    pandas.core.frame.DataFrame



We can get directly deduce a column as a dataframe by using the ```loc``` function and pass in __the name of the column with brackets around it__.

#### 3rd Method


```python
type(the_dataframe[['a']])
```




    pandas.core.frame.DataFrame



Something similar can be achieve using indexing.

#### 4th Method


```python
type(the_dataframe.iloc[:, [0]])
```




    pandas.core.frame.DataFrame



Same as ```iloc```.

__Note:__ If you want to understand the difference between ```loc``` and ```iloc```, read [this page from StackOverflow](https://stackoverflow.com/questions/31593201/how-are-iloc-and-loc-different)

### Ex 45: How to change the order of columns of a Dataframe?

Q:

1. In ```the_dataframe```, interchange columns ```a``` and ```c```.

2. Create a generic function to interchange two columns, without hardcoding column names.

3. Sort the columns in reverse alphabetical order, starting from column ```e``` first through column ```a``` last.


```python
import pandas as pd
import numpy as np
```


```python
the_dataframe = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))
```

#### Desired output

![Pandas_ex45](/blog/assets/post_cont_image/pandas_ex45.png)

#### Solution

#### Q1


```python
the_dataframe
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
the_columns = list(the_dataframe.columns)
```


```python
the_columns
```




    ['a', 'b', 'c', 'd', 'e']




```python
a, c = the_columns.index("a"), the_columns.index("c")
```


```python
the_columns[a], the_columns[c] = the_columns[c], the_columns[a]
```


```python
the_dataframe = the_dataframe[the_columns]
```


```python
the_dataframe
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
      <th>c</th>
      <th>b</th>
      <th>a</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>6</td>
      <td>5</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>11</td>
      <td>10</td>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17</td>
      <td>16</td>
      <td>15</td>
      <td>18</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>



To interchange column ```a``` by ```c```, we first get all the column name as a list and store it in ```the_columns``` variable. We then extract the indexes of ```a``` and ```c``` which are 0 and 2 respectively, then swap them in ```the_columns``` list using the indexes previously extracted. 

We finally use indexing in the original dataframe to now swap the columns name and values from column ```a``` and ```c```.

#### Q2


```python
def swap_col(col_1,col_2,df):
    all_col = list(df.columns)
    col_1_idx, col_2_idx = all_col.index(col_1), all_col.index(col_2)
    all_col[col_1_idx],all_col[col_2_idx] = all_col[col_2_idx], all_col[col_1_idx]
    return df[all_col]
    
    
print(swap_col("d","b",the_dataframe))
```

        c   d   a   b   e
    0   2   3   0   1   4
    1   7   8   5   6   9
    2  12  13  10  11  14
    3  17  18  15  16  19


This function is based on the same steps as Q1. Instead of using the names of the columns ```a``` and ```b```, we use generic names parameters ```col_1``` and ```col_2``` and pass in the original dataframe as ```df```. The rest is the same as Q1.

#### Q3


```python
the_columns_reversed = sorted(the_dataframe.columns,reverse=True)
```


```python
the_dataframe[the_columns_reversed]
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
      <th>e</th>
      <th>d</th>
      <th>c</th>
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>8</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>13</td>
      <td>12</td>
      <td>11</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19</td>
      <td>18</td>
      <td>17</td>
      <td>16</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



For this subquestion, we get first the reversed list of columns starting from ```e``` to ```a``` and store it in the ```the_columns_reversed``` variable. We use indexing on the original dataframe to get the columns aligned in reverse alphabetical order.

### Ex 46: How to set the number of rows and columns displayed in the output?

Q: Change the pandas display settings on printing the dataframe df it shows a maximum of 10 rows and 10 columns.




```python
the_dataframe = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```

#### Desired output

![Pandas_ex46](/blog/assets/post_cont_image/pandas_ex46.png)

#### Solution

#### 1st Method


```python
pd.options.display.max_columns = 10
pd.options.display.max_rows = 10
```


```python
the_dataframe
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
      <th>Manufacturer</th>
      <th>Model</th>
      <th>Type</th>
      <th>Min.Price</th>
      <th>Price</th>
      <th>...</th>
      <th>Rear.seat.room</th>
      <th>Luggage.room</th>
      <th>Weight</th>
      <th>Origin</th>
      <th>Make</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Acura</td>
      <td>Integra</td>
      <td>Small</td>
      <td>12.9</td>
      <td>15.9</td>
      <td>...</td>
      <td>26.5</td>
      <td>NaN</td>
      <td>2705.0</td>
      <td>non-USA</td>
      <td>Acura Integra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Legend</td>
      <td>Midsize</td>
      <td>29.2</td>
      <td>33.9</td>
      <td>...</td>
      <td>30.0</td>
      <td>15.0</td>
      <td>3560.0</td>
      <td>non-USA</td>
      <td>Acura Legend</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Audi</td>
      <td>90</td>
      <td>Compact</td>
      <td>25.9</td>
      <td>29.1</td>
      <td>...</td>
      <td>28.0</td>
      <td>14.0</td>
      <td>3375.0</td>
      <td>non-USA</td>
      <td>Audi 90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Audi</td>
      <td>100</td>
      <td>Midsize</td>
      <td>NaN</td>
      <td>37.7</td>
      <td>...</td>
      <td>31.0</td>
      <td>17.0</td>
      <td>3405.0</td>
      <td>non-USA</td>
      <td>Audi 100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>535i</td>
      <td>Midsize</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>27.0</td>
      <td>13.0</td>
      <td>3640.0</td>
      <td>non-USA</td>
      <td>BMW 535i</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Volkswagen</td>
      <td>Eurovan</td>
      <td>Van</td>
      <td>16.6</td>
      <td>19.7</td>
      <td>...</td>
      <td>34.0</td>
      <td>NaN</td>
      <td>3960.0</td>
      <td>NaN</td>
      <td>Volkswagen Eurovan</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Volkswagen</td>
      <td>Passat</td>
      <td>Compact</td>
      <td>17.6</td>
      <td>20.0</td>
      <td>...</td>
      <td>31.5</td>
      <td>14.0</td>
      <td>2985.0</td>
      <td>non-USA</td>
      <td>Volkswagen Passat</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Volkswagen</td>
      <td>Corrado</td>
      <td>Sporty</td>
      <td>22.9</td>
      <td>23.3</td>
      <td>...</td>
      <td>26.0</td>
      <td>15.0</td>
      <td>2810.0</td>
      <td>non-USA</td>
      <td>Volkswagen Corrado</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Volvo</td>
      <td>240</td>
      <td>Compact</td>
      <td>21.8</td>
      <td>22.7</td>
      <td>...</td>
      <td>29.5</td>
      <td>14.0</td>
      <td>2985.0</td>
      <td>non-USA</td>
      <td>Volvo 240</td>
    </tr>
    <tr>
      <th>92</th>
      <td>NaN</td>
      <td>850</td>
      <td>Midsize</td>
      <td>24.8</td>
      <td>26.7</td>
      <td>...</td>
      <td>30.0</td>
      <td>15.0</td>
      <td>3245.0</td>
      <td>non-USA</td>
      <td>Volvo 850</td>
    </tr>
  </tbody>
</table>
<p>93 rows × 27 columns</p>
</div>



#### 2nd Method


```python
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
```


```python
the_dataframe
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
      <th>Manufacturer</th>
      <th>Model</th>
      <th>Type</th>
      <th>Min.Price</th>
      <th>Price</th>
      <th>...</th>
      <th>Rear.seat.room</th>
      <th>Luggage.room</th>
      <th>Weight</th>
      <th>Origin</th>
      <th>Make</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Acura</td>
      <td>Integra</td>
      <td>Small</td>
      <td>12.9</td>
      <td>15.9</td>
      <td>...</td>
      <td>26.5</td>
      <td>NaN</td>
      <td>2705.0</td>
      <td>non-USA</td>
      <td>Acura Integra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Legend</td>
      <td>Midsize</td>
      <td>29.2</td>
      <td>33.9</td>
      <td>...</td>
      <td>30.0</td>
      <td>15.0</td>
      <td>3560.0</td>
      <td>non-USA</td>
      <td>Acura Legend</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Audi</td>
      <td>90</td>
      <td>Compact</td>
      <td>25.9</td>
      <td>29.1</td>
      <td>...</td>
      <td>28.0</td>
      <td>14.0</td>
      <td>3375.0</td>
      <td>non-USA</td>
      <td>Audi 90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Audi</td>
      <td>100</td>
      <td>Midsize</td>
      <td>NaN</td>
      <td>37.7</td>
      <td>...</td>
      <td>31.0</td>
      <td>17.0</td>
      <td>3405.0</td>
      <td>non-USA</td>
      <td>Audi 100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>535i</td>
      <td>Midsize</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>27.0</td>
      <td>13.0</td>
      <td>3640.0</td>
      <td>non-USA</td>
      <td>BMW 535i</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Volkswagen</td>
      <td>Eurovan</td>
      <td>Van</td>
      <td>16.6</td>
      <td>19.7</td>
      <td>...</td>
      <td>34.0</td>
      <td>NaN</td>
      <td>3960.0</td>
      <td>NaN</td>
      <td>Volkswagen Eurovan</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Volkswagen</td>
      <td>Passat</td>
      <td>Compact</td>
      <td>17.6</td>
      <td>20.0</td>
      <td>...</td>
      <td>31.5</td>
      <td>14.0</td>
      <td>2985.0</td>
      <td>non-USA</td>
      <td>Volkswagen Passat</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Volkswagen</td>
      <td>Corrado</td>
      <td>Sporty</td>
      <td>22.9</td>
      <td>23.3</td>
      <td>...</td>
      <td>26.0</td>
      <td>15.0</td>
      <td>2810.0</td>
      <td>non-USA</td>
      <td>Volkswagen Corrado</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Volvo</td>
      <td>240</td>
      <td>Compact</td>
      <td>21.8</td>
      <td>22.7</td>
      <td>...</td>
      <td>29.5</td>
      <td>14.0</td>
      <td>2985.0</td>
      <td>non-USA</td>
      <td>Volvo 240</td>
    </tr>
    <tr>
      <th>92</th>
      <td>NaN</td>
      <td>850</td>
      <td>Midsize</td>
      <td>24.8</td>
      <td>26.7</td>
      <td>...</td>
      <td>30.0</td>
      <td>15.0</td>
      <td>3245.0</td>
      <td>non-USA</td>
      <td>Volvo 850</td>
    </tr>
  </tbody>
</table>
<p>93 rows × 27 columns</p>
</div>



By change the ```max_columns``` or ```max_rows``` from the pandas' display option either by using the dot operator or by calling the ```set_option```, we can print only the five first and last rows/columns. All the rows/columns in between are abstracted by three dotes```...```

### Ex 47: How to format or suppress scientific notations in a pandas dataframe?

Q: Suppress scientific notations like ```e-03``` in ```the_dataframe``` and print up to 4 numbers after the decimal.


```python
the_dataframe = pd.DataFrame(np.random.random(4)**10, columns=['random'])
```


```python
the_dataframe
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
      <th>random</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.694142e-18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.132125e-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.673025e-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.668673e-01</td>
    </tr>
  </tbody>
</table>
</div>



#### Desired output


```python
# Your result will be different from mine as the numbers are randomly generated
```

![Pandas_ex47](/blog/assets/post_cont_image/pandas_ex47.png)

#### Solution

#### 1st Method


```python
pd.options.display.float_format = '{:.4f}'.format
```


```python
the_dataframe
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
      <th>random</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0213</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.3669</td>
    </tr>
  </tbody>
</table>
</div>



#### 2nd Method


```python
pd.set_option("display.float_format", '{:.4f}'.format)
```


```python
the_dataframe
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
      <th>random</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0213</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.3669</td>
    </tr>
  </tbody>
</table>
</div>



To suppress the scientific notation, we use ```display.float_format``` or the ```set_option``` function just like we did in the previous exercise and we use this time ```'{:.4f}'.format``` to tell Pandas that we want to display numbers with a four decimal points.

### Ex 48: How to format all the values in a Dataframe as percentages?

Q: Format the values in column ```random``` of ```the_dataframe``` as percentages.


```python
the_dataframe = pd.DataFrame(np.random.random(4), columns=['random'])
```


```python
the_dataframe
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
      <th>random</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.5170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0474</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.4878</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.9220</td>
    </tr>
  </tbody>
</table>
</div>



#### Desired output


```python
# Your result will be different from mine as the numbers are randomly generated
```

![Pandas_ex48](/blog/assets/post_cont_image/pandas_ex48.png)

#### Solution

#### 1st Method


```python
pd.set_option("display.float_format", '{:.2%}'.format)
```


```python
the_dataframe
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
      <th>random</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>51.70%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.74%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.78%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92.20%</td>
    </tr>
  </tbody>
</table>
</div>



#### 2nd Method


```python
pd.options.display.float_format = '{:.2%}'.format
```


```python
the_dataframe
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
      <th>random</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>51.70%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.74%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.78%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92.20%</td>
    </tr>
  </tbody>
</table>
</div>



Just as we did in the previous exercise, we use ```display.float_format``` or the ```set_option``` function but this time to display the values as percentages, we use ```'{:.2%}'.format```.

### Ex 49: How to filter every nth row in a Dataframe?

Q: From ```cars_dataset```, filter the ```Manufacturer```, ```Model``` and ```Type``` for every 20th row starting from 1st (row 0).


```python
cars_dataset = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```

#### Desired output

![Pandas_ex49](/blog/assets/post_cont_image/pandas_ex49.png)

#### Solution


```python
cars_dataset[["Manufacturer", "Model", "Type"]][::20]
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
      <th>Manufacturer</th>
      <th>Model</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Acura</td>
      <td>Integra</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Chrysler</td>
      <td>LeBaron</td>
      <td>Compact</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Honda</td>
      <td>Prelude</td>
      <td>Sporty</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Mercury</td>
      <td>Cougar</td>
      <td>Midsize</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Subaru</td>
      <td>Loyale</td>
      <td>Small</td>
    </tr>
  </tbody>
</table>
</div>



We start by selecting the 3 columns and step by 20 on each row using indexing.

### Ex 50: How to create a primary key index by combining relevant columns?

Q: In ```cars_dataset```, Replace NaNs with ```missing``` in columns ```Manufacturer```, ```Model``` and ```Type``` and create an index as a combination of these three columns and check if the index is a primary key.


```python
cars_dataset = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv', usecols=[0,1,2,3,5])
```

#### Desired output

![Pandas_ex50](/blog/assets/post_cont_image/pandas_ex50.png)

#### Solution


```python
replaced_nan = cars_dataset[["Manufacturer", "Model", "Type"]].fillna(value="missing")
```


```python
replaced_nan
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
      <th>Manufacturer</th>
      <th>Model</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Acura</td>
      <td>Integra</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>1</th>
      <td>missing</td>
      <td>Legend</td>
      <td>Midsize</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Audi</td>
      <td>90</td>
      <td>Compact</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Audi</td>
      <td>100</td>
      <td>Midsize</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>535i</td>
      <td>Midsize</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Volkswagen</td>
      <td>Eurovan</td>
      <td>Van</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Volkswagen</td>
      <td>Passat</td>
      <td>Compact</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Volkswagen</td>
      <td>Corrado</td>
      <td>Sporty</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Volvo</td>
      <td>240</td>
      <td>Compact</td>
    </tr>
    <tr>
      <th>92</th>
      <td>missing</td>
      <td>850</td>
      <td>Midsize</td>
    </tr>
  </tbody>
</table>
<p>93 rows × 3 columns</p>
</div>




```python
replaced_nan.set_index(replaced_nan["Manufacturer"]+"_"+replaced_nan["Model"]+"_"+replaced_nan["Type"])
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
      <th>Manufacturer</th>
      <th>Model</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acura_Integra_Small</th>
      <td>Acura</td>
      <td>Integra</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>missing_Legend_Midsize</th>
      <td>missing</td>
      <td>Legend</td>
      <td>Midsize</td>
    </tr>
    <tr>
      <th>Audi_90_Compact</th>
      <td>Audi</td>
      <td>90</td>
      <td>Compact</td>
    </tr>
    <tr>
      <th>Audi_100_Midsize</th>
      <td>Audi</td>
      <td>100</td>
      <td>Midsize</td>
    </tr>
    <tr>
      <th>BMW_535i_Midsize</th>
      <td>BMW</td>
      <td>535i</td>
      <td>Midsize</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Volkswagen_Eurovan_Van</th>
      <td>Volkswagen</td>
      <td>Eurovan</td>
      <td>Van</td>
    </tr>
    <tr>
      <th>Volkswagen_Passat_Compact</th>
      <td>Volkswagen</td>
      <td>Passat</td>
      <td>Compact</td>
    </tr>
    <tr>
      <th>Volkswagen_Corrado_Sporty</th>
      <td>Volkswagen</td>
      <td>Corrado</td>
      <td>Sporty</td>
    </tr>
    <tr>
      <th>Volvo_240_Compact</th>
      <td>Volvo</td>
      <td>240</td>
      <td>Compact</td>
    </tr>
    <tr>
      <th>missing_850_Midsize</th>
      <td>missing</td>
      <td>850</td>
      <td>Midsize</td>
    </tr>
  </tbody>
</table>
<p>93 rows × 3 columns</p>
</div>



We first select the 3 columns and use the ```fillna``` function to replace all the occurrences of ```nan``` by the string ```missing``` then store it the ```replace_nan``` variable. 

We call the ```set_index``` function on that dataframe to give it a new index column whereby each index value is a concatenation of the values from ```Manufacturer``` with the values in ```Model``` and lastly the values in ```Type```.

### Conclusion

In this part 2 of the pandas series, we have introduced the dataframe data structure which is the main data structure of Pandas. We have discovered how to import data from a CSV file, different ways to manipulate the data and many more techniques that you will be using on a daily basis if you will be working as a machine learning engineer or data scientist. 

Remember that 60% of the time spent on end-to-end machine learning project is dedicated to data cleaning and visualization. So Pandas, NumPy and Matplotlib (and Seaborn) are fantastic tools to learn and master. Practice makes perfect.

In the next post, we will explore more advanced Pandas exercises that I am sure you will enjoy. Find the jupyter notebook version of this post at my GitHub profile [here]("https://github.com/semasuka/blog/blob/gh-pages/ipynb/Pandas%20Exercise%20Part%202.ipynb").

Thank you again for doing these exercises with me. I hope you have learned one or two things. If you like this post, please subscribe to stay updated with new posts, and if you have a thought or a question, I would love to hear it by commenting below. Cheers, and keep learning!

