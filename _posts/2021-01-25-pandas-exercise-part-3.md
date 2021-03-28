---
title:  "Pandas Exercises Part 3"
image: /assets/post_images/pandas.jpg
excerpt_separator: <!-- more -->
tags:
- python
- exercises
- pandas
---



Great to see you again here! In this last post of the Pandas series, we will continue exploring advanced DataFrame exercises. Pandas is easer to learn than NumPy, in my opinion. Its documentation is well written, so don't be shy! Read its documentation throughout if you get stuck [here](https://pandas.pydata.org/pandas-docs/stable/index.html).<!-- more -->

Let's get started by importing NumPy and Pandas.


```python
import numpy as np
import pandas as pd 
```

### Ex 51: How to get the row number of the nth largest value in a column?

Q: Find the row position of the 5th largest value of column ```a``` of ```the_dataframe```.


```python
from numpy.random import default_rng
np.random.seed(42)
rng = default_rng()
the_dataframe = pd.DataFrame(rng.choice(30, size=30, replace=False).reshape(10,-1), columns=list('abc'))
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16</td>
      <td>11</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25</td>
      <td>22</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>18</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>17</td>
      <td>20</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>24</td>
      <td>28</td>
    </tr>
    <tr>
      <th>6</th>
      <td>23</td>
      <td>13</td>
      <td>21</td>
    </tr>
    <tr>
      <th>7</th>
      <td>29</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7</td>
      <td>10</td>
      <td>26</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6</td>
      <td>14</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



Note: We import and use ```default_rng``` to generated no duplicate values in the DataFrame and use ```random.seed``` to always generate the same numbers even on a different computer.

#### Desired out


```python
# The row with the 5th largest number is 6
```

#### 1st Method


```python
row_fifth_largest_num = the_dataframe["a"].sort_values(reversed)[::-1].index[4]
```


```python
print("The row with the 5th largest number is {}".format(row_fifth_largest_num))
```

    The row with the 5th largest number is 6


We first sort the values in column ```a```, and then reverse it. To get the index(the row number) of the 5th position, we pass in 4 in the ```index``` (indexes start from 0).

#### 2nd Method


```python
row_fifth_largest_num = the_dataframe["a"].argsort()[::-1][5]
```


```python
print("The row with the 5th largest number is {}".format(row_fifth_largest_num))
```

    The row with the 5th largest number is 6


Another way is by using ```argsort``` which will return sorted indexes according to the values in those indexes. Then we reverse those indexes using ```[::1]``` and get the fifth element using ```[5]```.

### Ex 52: How to find the position of the nth largest value greater than a given value?

Q: In ```the_serie``` find the position of the 2nd largest value greater than the mean.


```python
np.random.seed(42)
the_serie = pd.Series(np.random.randint(1, 100, 15))
```


```python
the_serie
```




    0     52
    1     93
    2     15
    3     72
    4     61
    5     21
    6     83
    7     87
    8     75
    9     75
    10    88
    11    24
    12     3
    13    22
    14    53
    dtype: int64



#### Desired output


```python
# The mean is 55.0 and the row of the second largest number is 3
```

#### Solution


```python
the_mean = np.mean(the_serie.values).round()
```


```python
the_mean
```




    55.0




```python
greater_than_mean_arr = the_serie.where(the_serie > the_mean).dropna().sort_values()
```


```python
row_second_largest_num = greater_than_mean_arr.index[1]
```


```python
row_second_largest_num
```




    3




```python
print("The mean is {} and the row of the second largest number is {}".format(the_mean, row_second_largest_num))
```

    The mean is 55.0 and the row of the second largest number is 3


We start by calculating the mean of the values in the series using ```np.mean``` and round it. Then we use ```where``` to get all rows with the values superior to the mean.

We drop NaN values (which are values inferior to the mean) and sort the remaining values to finally get the second value in the sorted series using ```.index[1]``` which correspond to the second largest number superior to the mean.

### Ex 53: How to get the last n rows of a DataFrame with row sum > 100?

Q: Get the last two rows of ```the_dataframe``` whose row sum is superior to 100.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16</td>
      <td>29</td>
      <td>38</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>17</td>
      <td>38</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>35</td>
      <td>28</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>20</td>
      <td>33</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>17</td>
      <td>33</td>
      <td>12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>31</td>
      <td>30</td>
      <td>11</td>
      <td>33</td>
    </tr>
    <tr>
      <th>6</th>
      <td>21</td>
      <td>39</td>
      <td>15</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>37</td>
      <td>30</td>
      <td>10</td>
      <td>21</td>
    </tr>
    <tr>
      <th>8</th>
      <td>35</td>
      <td>31</td>
      <td>38</td>
      <td>21</td>
    </tr>
    <tr>
      <th>9</th>
      <td>34</td>
      <td>26</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>10</th>
      <td>19</td>
      <td>37</td>
      <td>37</td>
      <td>25</td>
    </tr>
    <tr>
      <th>11</th>
      <td>24</td>
      <td>39</td>
      <td>39</td>
      <td>24</td>
    </tr>
    <tr>
      <th>12</th>
      <td>39</td>
      <td>28</td>
      <td>21</td>
      <td>32</td>
    </tr>
    <tr>
      <th>13</th>
      <td>29</td>
      <td>34</td>
      <td>12</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>28</td>
      <td>16</td>
      <td>30</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



#### Desired output

![Pandas_ex53](/blog/assets/post_cont_image/pandas_ex53.png)

#### Solution


```python
rows_sum = the_dataframe.sum(axis=1).sort_values()
```


```python
rows_sum
```




    4      75
    6      86
    13     89
    14     92
    7      98
    3     103
    1     105
    5     105
    0     107
    2     111
    10    118
    12    120
    8     125
    11    126
    9     132
    dtype: int64




```python
rows_sum_greater_100 = rows_sum.where(rows_sum > 100).dropna()
rows_sum_greater_100
```




    3     103.0
    1     105.0
    5     105.0
    0     107.0
    2     111.0
    10    118.0
    12    120.0
    8     125.0
    11    126.0
    9     132.0
    dtype: float64




```python
the_dataframe.iloc[rows_sum_greater_100[::-1][:2].index]
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>34</td>
      <td>26</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>11</th>
      <td>24</td>
      <td>39</td>
      <td>39</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



We calculate a series with of the sum of all the elements row-wise and sort it. We then use ```where``` function to get only the row with element greater than 100 and drop the rest using ```dropna``` function. 

Finally, we reverse that row using ```[::-1]``` and get the indexes first two rows using indexing from 0 to 2 (exclusive). We replace those indexes in the original dataframe to get the two rows using ```iloc```. 

### Ex 54: How to find and cap outliers from a series or DataFrame column?

Q: Replace all values of ```the_serie``` lower to the 5th percentile and greater than 95th percentile respectively with the 5th and 95th percentile value.


```python
the_serie = ser = pd.Series(np.logspace(-2, 2, 30))
```

#### Desired output


```python
# 0      0.016049
# 1      0.016049
# 2      0.018874
# 3      0.025929
# 4      0.035622
# 5      0.048939
# 6      0.067234
# 7      0.092367
# 8      0.126896
# 9      0.174333
# 10     0.239503
# 11     0.329034
# 12     0.452035
# 13     0.621017
# 14     0.853168
# 15     1.172102
# 16     1.610262
# 17     2.212216
# 18     3.039195
# 19     4.175319
# 20     5.736153
# 21     7.880463
# 22    10.826367
# 23    14.873521
# 24    20.433597
# 25    28.072162
# 26    38.566204
# 27    52.983169
# 28    63.876672
# 29    63.876672
# dtype: float64
```

#### Solution


```python
low_perc = the_serie.quantile(q=0.05)
low_perc
```




    0.016049294076965887




```python
high_perc = the_serie.quantile(q=0.95)
high_perc
```




    63.876672220183934




```python
the_serie.where(the_serie > low_perc, other=low_perc, inplace=True) 
```


```python
the_serie.where(the_serie < high_perc, other=high_perc, inplace=True) 
```


```python
the_serie
```




    0      0.016049
    1      0.016049
    2      0.018874
    3      0.025929
    4      0.035622
    5      0.048939
    6      0.067234
    7      0.092367
    8      0.126896
    9      0.174333
    10     0.239503
    11     0.329034
    12     0.452035
    13     0.621017
    14     0.853168
    15     1.172102
    16     1.610262
    17     2.212216
    18     3.039195
    19     4.175319
    20     5.736153
    21     7.880463
    22    10.826367
    23    14.873521
    24    20.433597
    25    28.072162
    26    38.566204
    27    52.983169
    28    63.876672
    29    63.876672
    dtype: float64



We first calculate the 5th and the 95th percentile using the ```quantile``` function and pass in ```q``` the number 0.05 and 0.95 respectively. Then we call the ```where``` function on the original series, pass in the condition as ```the_serie > low_perc```. 

This condition will target the elements superior to the 5th percentile and set ```other``` which is the remaining element (inferior to the 5th percentile) to be the 5th percentile. The assignment will replace all the values in the series lower than the 5th percentile by the 5th percentile value. Finally, we set ```inplace``` to ```True```.

We do the same for the 95th percentile, just that this time we are targeting elements inferior to the 95th percentile and set ```other``` to the value of the 95th percentile.

### Ex 55: How to reshape a DataFrame to the largest possible square after removing the negative values?

Q: Reshape ```the_dataframe``` to the largest possible square with negative values removed. Drop the smallest values if need be. The order of the positive numbers in the result should remain the same as the original.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame(np.random.randint(-20, 50, 100).reshape(10,-1))
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31</td>
      <td>-6</td>
      <td>40</td>
      <td>0</td>
      <td>3</td>
      <td>-18</td>
      <td>1</td>
      <td>32</td>
      <td>-19</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>-19</td>
      <td>43</td>
      <td>39</td>
      <td>0</td>
      <td>12</td>
      <td>37</td>
      <td>1</td>
      <td>28</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>39</td>
      <td>-6</td>
      <td>41</td>
      <td>41</td>
      <td>26</td>
      <td>41</td>
      <td>30</td>
      <td>34</td>
      <td>43</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-18</td>
      <td>30</td>
      <td>-14</td>
      <td>0</td>
      <td>18</td>
      <td>-3</td>
      <td>-17</td>
      <td>39</td>
      <td>-7</td>
      <td>-12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>-19</td>
      <td>39</td>
      <td>23</td>
      <td>-13</td>
      <td>26</td>
      <td>14</td>
      <td>15</td>
      <td>29</td>
      <td>-17</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-19</td>
      <td>-15</td>
      <td>33</td>
      <td>-17</td>
      <td>33</td>
      <td>42</td>
      <td>-3</td>
      <td>23</td>
      <td>13</td>
      <td>41</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-7</td>
      <td>27</td>
      <td>-6</td>
      <td>41</td>
      <td>19</td>
      <td>32</td>
      <td>3</td>
      <td>5</td>
      <td>39</td>
      <td>20</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>-6</td>
      <td>24</td>
      <td>44</td>
      <td>-12</td>
      <td>-20</td>
      <td>-13</td>
      <td>42</td>
      <td>-10</td>
      <td>-13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>14</td>
      <td>14</td>
      <td>12</td>
      <td>-16</td>
      <td>20</td>
      <td>7</td>
      <td>-14</td>
      <td>-9</td>
      <td>13</td>
      <td>12</td>
    </tr>
    <tr>
      <th>9</th>
      <td>27</td>
      <td>2</td>
      <td>41</td>
      <td>16</td>
      <td>23</td>
      <td>14</td>
      <td>44</td>
      <td>26</td>
      <td>-18</td>
      <td>-20</td>
    </tr>
  </tbody>
</table>
</div>



#### Desired output


```python
# array([[31., 40.,  3., 32.,  9., 17., 43., 39.],
#        [12., 37., 28., 38., 21., 39., 41., 41.],
#        [26., 41., 30., 34., 43., 30., 18., 39.],
#        [32., 39., 23., 26., 14., 15., 29., 33.],
#        [33., 42., 23., 13., 41., 27., 41., 19.],
#        [32.,  3.,  5., 39., 20.,  8., 24., 44.],
#        [42., 14., 14., 12., 20.,  7., 13., 12.],
#        [27.,  2., 41., 16., 23., 14., 44., 26.]])
```

#### Solution

#### Step 1: Remove the negatives


```python
the_arr = the_dataframe[the_dataframe > 0].values.flatten()
the_arr
```




    array([31., nan, 40., nan,  3., nan,  1., 32., nan,  9., 17., nan, 43.,
           39., nan, 12., 37.,  1., 28., 38., 21., 39., nan, 41., 41., 26.,
           41., 30., 34., 43., nan, 30., nan, nan, 18., nan, nan, 39., nan,
           nan, 32., nan, 39., 23., nan, 26., 14., 15., 29., nan, nan, nan,
           33., nan, 33., 42., nan, 23., 13., 41., nan, 27., nan, 41., 19.,
           32.,  3.,  5., 39., 20.,  8., nan, 24., 44., nan, nan, nan, 42.,
           nan, nan, 14., 14., 12., nan, 20.,  7., nan, nan, 13., 12., 27.,
            2., 41., 16., 23., 14., 44., 26., nan, nan])



We use indexing with ```[]``` to get all the positive elements in ```the_dataframe``` and reshaped them into a 1D array using ```flatten()``` function to finally store it into ```the_arr```.


```python
pos_arr = the_arr[~np.isnan(the_arr)]
```


```python
np.isnan(the_arr)
```




    array([False,  True, False,  True, False,  True, False, False,  True,
           False, False,  True, False, False,  True, False, False, False,
           False, False, False, False,  True, False, False, False, False,
           False, False, False,  True, False,  True,  True, False,  True,
            True, False,  True,  True, False,  True, False, False,  True,
           False, False, False, False,  True,  True,  True, False,  True,
           False, False,  True, False, False, False,  True, False,  True,
           False, False, False, False, False, False, False, False,  True,
           False, False,  True,  True,  True, False,  True,  True, False,
           False, False,  True, False, False,  True,  True, False, False,
           False, False, False, False, False, False, False, False,  True,
            True])




```python
pos_arr
```




    array([31., 40.,  3.,  1., 32.,  9., 17., 43., 39., 12., 37.,  1., 28.,
           38., 21., 39., 41., 41., 26., 41., 30., 34., 43., 30., 18., 39.,
           32., 39., 23., 26., 14., 15., 29., 33., 33., 42., 23., 13., 41.,
           27., 41., 19., 32.,  3.,  5., 39., 20.,  8., 24., 44., 42., 14.,
           14., 12., 20.,  7., 13., 12., 27.,  2., 41., 16., 23., 14., 44.,
           26.])



To drop the ```nan``` in the array, we use indexing and ```isnan``` function to return a boolean array where ```False``` represent a non ```nan``` value and ```True``` is a position of a ```nan``` value. We then ```~``` sign to inverse the boolean to get ```True``` in where there is non ```nan``` value. Now we get a new array with no ```nan``` values.

#### Step 2: Find side-length of largest possible square


```python
len(pos_arr)
```




    66




```python
n = int(np.floor(np.sqrt(pos_arr.shape[0])))
```


```python
n
```




    8



To search for the largest possible square, we get first the length of the array using ```shape``` function (we could also use ```len()``` function). We then find the square root of the number of elements in the array and remove the decimal using ```floor``` function cast it to an integer.

#### Step 3: Take top n^2 items without changing positions


```python
top_indexes = np.argsort(pos_arr)[::-1]
```


```python
top_indexes
```




    array([64, 49, 22,  7, 50, 35, 16, 40, 19, 17, 60, 38,  1, 15, 27,  8, 25,
           45, 13, 10, 21, 33, 34, 42, 26,  4,  0, 23, 20, 32, 12, 39, 58, 65,
           29, 18, 48, 62, 36, 28, 14, 54, 46, 41, 24,  6, 61, 31, 63, 51, 52,
           30, 37, 56,  9, 57, 53,  5, 47, 55, 44, 43,  2, 59, 11,  3])




```python
np.take(pos_arr, sorted(top_indexes[:n**2])).reshape(n,-1)
```




    array([[31., 40.,  3., 32.,  9., 17., 43., 39.],
           [12., 37., 28., 38., 21., 39., 41., 41.],
           [26., 41., 30., 34., 43., 30., 18., 39.],
           [32., 39., 23., 26., 14., 15., 29., 33.],
           [33., 42., 23., 13., 41., 27., 41., 19.],
           [32.,  3.,  5., 39., 20.,  8., 24., 44.],
           [42., 14., 14., 12., 20.,  7., 13., 12.],
           [27.,  2., 41., 16., 23., 14., 44., 26.]])



We then sort the element indexes using ```argsort``` and reverse the order into a descending order using slicing ```[::-1]``` and store it in ```top_indexes```.

Finally, we use ```take``` NumPy function that takes the ```pos_arr``` and as indices the sorted ```top_indexes``` (from the first indices up to ```n``` raised to the power of 2). Then we reshape the array using ```(n,-1)``` to let Pandas figure out the best reshape argument to use depending on ```n``` value.

### Ex 56: How to swap two rows of a DataFrame?

Q: Swap rows 1 and 2 in ```the_dataframe```


```python
the_dataframe = pd.DataFrame(np.arange(25).reshape(5, -1))
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
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
    <tr>
      <th>4</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



#### Desired solution

![Pandas_ex56](/blog/assets/post_cont_image/pandas_ex56.png)

#### Solution


```python
def row_swap(the_dataframe,row_index_1,row_index_2):
    row_1, row_2 = the_dataframe.iloc[row_index_1,:].copy(), the_dataframe.iloc[row_index_2,:].copy()
    the_dataframe.iloc[row_index_1,:], the_dataframe.iloc[row_index_2,:] = row_2, row_1
    return the_dataframe
```


```python
row_swap(the_dataframe,0,1)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
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
    <tr>
      <th>4</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



We create a function that performs the swap it takes in the dataframe and the two indexes of the rows that need to be swap. We then copy the rows using ```iloc``` and store them in ```row_1``` and ```row_2```.  

To do the swap, we do the opposite of what we did by assigning ```row_1``` and ```row_2``` to the equivalent row index we want to change to occur. So ```row_2``` will be assigned to ```row_index_1``` and ```row_1``` will be assigned to ```row_index_2```. Finally, we return ```the_dataframe```.

### Ex 57: How to reverse the rows of a DataFrame?

Q: Reverse all the rows of a DataFrame.


```python
the_dataframe = pd.DataFrame(np.arange(25).reshape(5, -1))
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
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
    <tr>
      <th>4</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



#### Desired output

![Pandas_ex57](/blog/assets/post_cont_image/pandas_ex57.png)

#### Solution

#### 1st method


```python
the_dataframe.iloc[::-1]
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
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
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



#### 2nd method


```python
the_dataframe[::-1]
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
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
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



To reverse the dataframe row-wise, we use indexing with ```[::-1]```. The second method is the short form of the first method we used ```iloc```. 

Note: In Exercise 75, we will see how to achieve the same result column-wise.

### Ex 58: How to create one-hot encodings of a categorical variable (dummy variables)?

Q: Get one-hot encodings for column ```a``` in the dataframe ```the_dataframe``` and append it as columns.


```python
the_dataframe = pd.DataFrame(np.arange(25).reshape(5,-1), columns=list('abcde'))
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
    <tr>
      <th>4</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



![Pandas_ex58](/blog/assets/post_cont_image/pandas_ex58.png)

#### Desired output

#### Solution


```python
one_hot = pd.get_dummies(the_dataframe["a"])
```


```python
one_hot
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
      <th>0</th>
      <th>5</th>
      <th>10</th>
      <th>15</th>
      <th>20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([one_hot, the_dataframe[["b","c","d","e"]]],axis=1)
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
      <th>0</th>
      <th>5</th>
      <th>10</th>
      <th>15</th>
      <th>20</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>12</td>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



We get the one-hot encoding of column ```a``` by using the ```get_dummies``` function and pass in the column ```a```.

To append the newly created ```one_hot```, we use ```concat``` and pass in ```one_hot``` and the remaining columns of ```dataframe``` (except column ```a```). Finally, we set the axis to 1 since we want to concatenate column-wise. 

### Ex 59: Which column contains the highest number of row-wise maximum values? 

Q: Obtain the column name with the highest number of row-wise in ```the_dataframe```.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1))
```

#### Desired output


```python
# The column name with the highest number of row-wise is 0
```

#### Solution


```python
row_high_num = the_dataframe.sum(axis=0).argmax()
```


```python
print("The column name with the highest number of row-wise is {}".format(row_high_num))
```

    The column name with the highest number of row-wise is 0


To get the row with the largest sum row-wise, we use the ```sum``` function and pass in the axis argument set to 0 (telling Pandas to calculate the sum of elements row-wise) and then use ```argmax``` to get the index with the highest value in the series.

### Ex 60: How to know the maximum possible correlation value of each column against other columns?

Q: Compute the maximum possible absolute correlation value of each column against other columns in ```the_dataframe```.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1), columns=list('pqrstuvwxy'), index=list('abcdefgh'))
```

#### Desired output


```python
# array([0.81941445, 0.84466639, 0.44944264, 0.44872809, 0.81941445,
#        0.80618428, 0.44944264, 0.5434561 , 0.84466639, 0.80618428])
```

#### Solution


```python
abs_corr = np.abs(the_dataframe.corr())
```


```python
abs_corr
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
      <th>p</th>
      <th>q</th>
      <th>r</th>
      <th>s</th>
      <th>t</th>
      <th>u</th>
      <th>v</th>
      <th>w</th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p</th>
      <td>1.000000</td>
      <td>0.019359</td>
      <td>0.088207</td>
      <td>0.087366</td>
      <td>0.819414</td>
      <td>0.736955</td>
      <td>0.070727</td>
      <td>0.338139</td>
      <td>0.163112</td>
      <td>0.665627</td>
    </tr>
    <tr>
      <th>q</th>
      <td>0.019359</td>
      <td>1.000000</td>
      <td>0.280799</td>
      <td>0.121217</td>
      <td>0.172215</td>
      <td>0.262234</td>
      <td>0.304781</td>
      <td>0.042748</td>
      <td>0.844666</td>
      <td>0.243317</td>
    </tr>
    <tr>
      <th>r</th>
      <td>0.088207</td>
      <td>0.280799</td>
      <td>1.000000</td>
      <td>0.184988</td>
      <td>0.223515</td>
      <td>0.017763</td>
      <td>0.449443</td>
      <td>0.127091</td>
      <td>0.228455</td>
      <td>0.018350</td>
    </tr>
    <tr>
      <th>s</th>
      <td>0.087366</td>
      <td>0.121217</td>
      <td>0.184988</td>
      <td>1.000000</td>
      <td>0.210879</td>
      <td>0.096695</td>
      <td>0.422290</td>
      <td>0.306141</td>
      <td>0.100174</td>
      <td>0.448728</td>
    </tr>
    <tr>
      <th>t</th>
      <td>0.819414</td>
      <td>0.172215</td>
      <td>0.223515</td>
      <td>0.210879</td>
      <td>1.000000</td>
      <td>0.576720</td>
      <td>0.334690</td>
      <td>0.543456</td>
      <td>0.047136</td>
      <td>0.273478</td>
    </tr>
    <tr>
      <th>u</th>
      <td>0.736955</td>
      <td>0.262234</td>
      <td>0.017763</td>
      <td>0.096695</td>
      <td>0.576720</td>
      <td>1.000000</td>
      <td>0.137836</td>
      <td>0.352145</td>
      <td>0.363597</td>
      <td>0.806184</td>
    </tr>
    <tr>
      <th>v</th>
      <td>0.070727</td>
      <td>0.304781</td>
      <td>0.449443</td>
      <td>0.422290</td>
      <td>0.334690</td>
      <td>0.137836</td>
      <td>1.000000</td>
      <td>0.158152</td>
      <td>0.188482</td>
      <td>0.033227</td>
    </tr>
    <tr>
      <th>w</th>
      <td>0.338139</td>
      <td>0.042748</td>
      <td>0.127091</td>
      <td>0.306141</td>
      <td>0.543456</td>
      <td>0.352145</td>
      <td>0.158152</td>
      <td>1.000000</td>
      <td>0.325939</td>
      <td>0.071704</td>
    </tr>
    <tr>
      <th>x</th>
      <td>0.163112</td>
      <td>0.844666</td>
      <td>0.228455</td>
      <td>0.100174</td>
      <td>0.047136</td>
      <td>0.363597</td>
      <td>0.188482</td>
      <td>0.325939</td>
      <td>1.000000</td>
      <td>0.338705</td>
    </tr>
    <tr>
      <th>y</th>
      <td>0.665627</td>
      <td>0.243317</td>
      <td>0.018350</td>
      <td>0.448728</td>
      <td>0.273478</td>
      <td>0.806184</td>
      <td>0.033227</td>
      <td>0.071704</td>
      <td>0.338705</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
max_abs_corr = abs_corr.apply(lambda x: sorted(x)[-2]).values
```


```python
max_abs_corr
```




    array([0.81941445, 0.84466639, 0.44944264, 0.44872809, 0.81941445,
           0.80618428, 0.44944264, 0.5434561 , 0.84466639, 0.80618428])



We first calculate the absolute correlation the whole dataset. We use the ```corr()``` function and pass it as the argument to the NumPy function ```abs``` to get the absolute values (non-negative values).

Now that we have the absolute correction, use lambda expression with the ```apply``` function to find the highest correlation value in each row.

We sorted first each row (represented by ```x``` in the lambda expression), secondly get the second last element in the row using indexing. The reason why we get the second-highest value instead of the last one is because the last value is ```1``` (calculated from the correlation of the same column). We then form an array with the highest correlation values in each row.

### Ex 61: How to create a column containing the minimum by the maximum of each row?

Q: Compute the minimum-by-maximum for every row of ```the_dataframe```.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52</td>
      <td>93</td>
      <td>15</td>
      <td>72</td>
      <td>61</td>
      <td>21</td>
      <td>83</td>
      <td>87</td>
      <td>75</td>
      <td>75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88</td>
      <td>24</td>
      <td>3</td>
      <td>22</td>
      <td>53</td>
      <td>2</td>
      <td>88</td>
      <td>30</td>
      <td>38</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64</td>
      <td>60</td>
      <td>21</td>
      <td>33</td>
      <td>76</td>
      <td>58</td>
      <td>22</td>
      <td>89</td>
      <td>49</td>
      <td>91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>59</td>
      <td>42</td>
      <td>92</td>
      <td>60</td>
      <td>80</td>
      <td>15</td>
      <td>62</td>
      <td>62</td>
      <td>47</td>
      <td>62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>55</td>
      <td>64</td>
      <td>3</td>
      <td>51</td>
      <td>7</td>
      <td>21</td>
      <td>73</td>
      <td>39</td>
      <td>18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>89</td>
      <td>60</td>
      <td>14</td>
      <td>9</td>
      <td>90</td>
      <td>53</td>
      <td>2</td>
      <td>84</td>
      <td>92</td>
    </tr>
    <tr>
      <th>6</th>
      <td>60</td>
      <td>71</td>
      <td>44</td>
      <td>8</td>
      <td>47</td>
      <td>35</td>
      <td>78</td>
      <td>81</td>
      <td>36</td>
      <td>50</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>54</td>
      <td>4</td>
      <td>54</td>
      <td>93</td>
      <td>63</td>
      <td>18</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
</div>



#### Desired output


```python
# 0    0.161290
# 1    0.022727
# 2    0.230769
# 3    0.163043
# 4    0.041096
# 5    0.021739
# 6    0.098765
# 7    0.021505
# dtype: float64
```

#### Solution

#### 1st Method


```python
the_min = the_dataframe.min(axis=1)
```


```python
the_max = the_dataframe.max(axis=1)
```


```python
min_by_max = the_min/the_max
```


```python
min_by_max
```




    0    0.161290
    1    0.022727
    2    0.230769
    3    0.163043
    4    0.041096
    5    0.021739
    6    0.098765
    7    0.021505
    dtype: float64



The easiest way to solve this problem is to find the minimum values in each column using the ```min``` function by setting the axis to 1. We do the same for the maximum to finally divide the minimum values by the maximum values.

#### 2nd Method


```python
the_dataframe.apply(lambda x: np.min(x)/np.max(x), axis=1)
```




    0    0.161290
    1    0.022727
    2    0.230769
    3    0.163043
    4    0.041096
    5    0.021739
    6    0.098765
    7    0.021505
    dtype: float64



The previous method uses three lines of codes we can write in one line of code.  We use the lambda expression to calculate the division of the minimum by the maximum of ```x``` and set axis to ```1```.

### Ex 62: How to create a column that contains the penultimate value in each row?

Q: Create a new column ```penultimate``` which has the second-largest value of each row of ```the_dataframe```.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))
```

#### Desire output

![Pandas_ex62](/blog/assets/post_cont_image/pandas_ex62.png)

#### Solution


```python
the_dataframe['penultimate'] = the_dataframe.apply(lambda x: x.sort_values().unique()[-2], axis=1)
```

As previously seen, to solve this type of challenge, we use a lambda expression with the ```apply``` function. We first set ```axis``` to ```1``` as we are calculating the values row-wise and in the lambda expression, we sort ```x``` ignore the duplicate with ```ignore``` function and return the second largest value using indexing ```[-2]```.

### Ex 63: How to normalize all columns in a dataframe?

Q1: Normalize all columns of ```the_dataframe``` by subtracting the column mean and divide by standard deviation.

Q2: Range all columns values of ```the_dataframe``` such that the minimum value in each column is 0 and max is 1.

Note: Don’t use external packages like sklearn.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))
```

#### Desired output

#### Q1

![Pandas_ex63_q1](/blog/assets/post_cont_image/pandas_ex6_q1.png)

#### Q2

![Pandas_ex63_q2](/blog/assets/post_cont_image/pandas_ex63_q2.png)

#### Solution

#### Q1


```python
the_dataframe.apply(lambda x: (x - x.mean()) / x.std(),axis=0)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.144954</td>
      <td>1.233620</td>
      <td>-0.722109</td>
      <td>1.495848</td>
      <td>0.478557</td>
      <td>-0.471883</td>
      <td>0.718777</td>
      <td>0.860589</td>
      <td>1.241169</td>
      <td>0.434515</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.372799</td>
      <td>-0.977283</td>
      <td>-1.096825</td>
      <td>-0.434278</td>
      <td>0.192317</td>
      <td>-1.101061</td>
      <td>0.894088</td>
      <td>-1.017060</td>
      <td>-0.475588</td>
      <td>-1.680126</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.554235</td>
      <td>0.176231</td>
      <td>-0.534751</td>
      <td>-0.009651</td>
      <td>1.015256</td>
      <td>0.753357</td>
      <td>-1.420023</td>
      <td>0.926472</td>
      <td>0.034799</td>
      <td>0.897999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.383701</td>
      <td>-0.400526</td>
      <td>1.682318</td>
      <td>1.032617</td>
      <td>1.158376</td>
      <td>-0.670571</td>
      <td>-0.017531</td>
      <td>0.037059</td>
      <td>-0.057999</td>
      <td>0.057935</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.110847</td>
      <td>0.016021</td>
      <td>0.807981</td>
      <td>-1.167726</td>
      <td>0.120757</td>
      <td>-0.935488</td>
      <td>-1.455085</td>
      <td>0.399412</td>
      <td>-0.429189</td>
      <td>-1.216643</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.492172</td>
      <td>1.105451</td>
      <td>0.683076</td>
      <td>-0.743098</td>
      <td>-1.382001</td>
      <td>1.813025</td>
      <td>-0.333092</td>
      <td>-1.939414</td>
      <td>1.658759</td>
      <td>0.926966</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.417808</td>
      <td>0.528694</td>
      <td>0.183455</td>
      <td>-0.974714</td>
      <td>-0.022362</td>
      <td>-0.008279</td>
      <td>0.543466</td>
      <td>0.662942</td>
      <td>-0.568386</td>
      <td>-0.289677</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.492172</td>
      <td>-1.682209</td>
      <td>-1.003146</td>
      <td>0.801002</td>
      <td>-1.560900</td>
      <td>0.620899</td>
      <td>1.069400</td>
      <td>0.070000</td>
      <td>-1.403565</td>
      <td>0.869031</td>
    </tr>
  </tbody>
</table>
</div>



To solve this issue, we need to know the formula for normalizing. The formulation is as follow: 

![Pandas_ex63_formular](/blog/assets/post_cont_image/pandas_ex63_formula.png)

So now we can proceed with the ```apply``` function and lambda expression. We set the ```axis``` to ```0``` since we are normalizing column-wise and with the lambda expression we do the calculation according to the formula above.

#### Q2


```python
the_dataframe.apply(lambda x: (x.max() - x)/(x.max()-x.min()),axis=0)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.428571</td>
      <td>0.000000</td>
      <td>0.865169</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.784091</td>
      <td>0.138889</td>
      <td>0.022989</td>
      <td>0.136364</td>
      <td>0.188889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.758242</td>
      <td>1.000000</td>
      <td>0.724638</td>
      <td>0.355263</td>
      <td>1.000000</td>
      <td>0.069444</td>
      <td>0.678161</td>
      <td>0.696970</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.285714</td>
      <td>0.362637</td>
      <td>0.797753</td>
      <td>0.565217</td>
      <td>0.052632</td>
      <td>0.363636</td>
      <td>0.986111</td>
      <td>0.000000</td>
      <td>0.530303</td>
      <td>0.011111</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.345238</td>
      <td>0.560440</td>
      <td>0.000000</td>
      <td>0.173913</td>
      <td>0.000000</td>
      <td>0.852273</td>
      <td>0.430556</td>
      <td>0.310345</td>
      <td>0.560606</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.440476</td>
      <td>0.417582</td>
      <td>0.314607</td>
      <td>1.000000</td>
      <td>0.381579</td>
      <td>0.943182</td>
      <td>1.000000</td>
      <td>0.183908</td>
      <td>0.681818</td>
      <td>0.822222</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000000</td>
      <td>0.043956</td>
      <td>0.359551</td>
      <td>0.840580</td>
      <td>0.934211</td>
      <td>0.000000</td>
      <td>0.555556</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.333333</td>
      <td>0.241758</td>
      <td>0.539326</td>
      <td>0.927536</td>
      <td>0.434211</td>
      <td>0.625000</td>
      <td>0.208333</td>
      <td>0.091954</td>
      <td>0.727273</td>
      <td>0.466667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.966292</td>
      <td>0.260870</td>
      <td>1.000000</td>
      <td>0.409091</td>
      <td>0.000000</td>
      <td>0.298851</td>
      <td>1.000000</td>
      <td>0.022222</td>
    </tr>
  </tbody>
</table>
</div>



To change the values in the dataframe to put them on a scale of ```0``` to ```1``` we use the following formula``` MAX - Z / MAX - MIN``` and do the same as we did in Q1.  

### Ex 64: How to compute the correlation of each row with the succeeding row?

Q: Compute the correlation of each row with its previous row, round the result by 2.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))
```

#### Desired output


```python
# [0.31, -0.14, -0.15, 0.47, -0.32, -0.07, 0.12]
```

#### Solution


```python
[the_dataframe.iloc[i].corr(the_dataframe.iloc[i+1]).round(2) for i in range(the_dataframe.shape[0]-1)]
```




    [0.31, -0.14, -0.15, 0.47, -0.32, -0.07, 0.12]



We first loop through the range of the number of rows in the dataframe using ```shape``` function (excluding the last row because we are comparing a pair of rows). We then call the ```corr``` function on each row using the ```iloc``` function and pass in the ```corr``` function the next row location by adding ```1``` to ```i```. Finally, we round the result by two decimal point and place it in a list comprehension.

### Ex 65:  How to replace both the diagonals of dataframe with 0?

Q: Replace both values in both diagonals of ```the_dataframe``` with 0.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame(np.random.randint(1,100, 100).reshape(10, -1))
```

#### Desired output

![Pandas_ex65](/blog/assets/post_cont_image/pandas_ex65.png)

#### Solution


```python
for i in range(the_dataframe.shape[0]):
    the_dataframe.iat[i,i] = 0
    the_dataframe.iat[the_dataframe.shape[0]-i-1,i] = 0
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>93</td>
      <td>15</td>
      <td>72</td>
      <td>61</td>
      <td>21</td>
      <td>83</td>
      <td>87</td>
      <td>75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88</td>
      <td>0</td>
      <td>3</td>
      <td>22</td>
      <td>53</td>
      <td>2</td>
      <td>88</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64</td>
      <td>60</td>
      <td>0</td>
      <td>33</td>
      <td>76</td>
      <td>58</td>
      <td>22</td>
      <td>0</td>
      <td>49</td>
      <td>91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>59</td>
      <td>42</td>
      <td>92</td>
      <td>0</td>
      <td>80</td>
      <td>15</td>
      <td>0</td>
      <td>62</td>
      <td>47</td>
      <td>62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>55</td>
      <td>64</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>39</td>
      <td>18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>89</td>
      <td>60</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>53</td>
      <td>2</td>
      <td>84</td>
      <td>92</td>
    </tr>
    <tr>
      <th>6</th>
      <td>60</td>
      <td>71</td>
      <td>44</td>
      <td>0</td>
      <td>47</td>
      <td>35</td>
      <td>0</td>
      <td>81</td>
      <td>36</td>
      <td>50</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>54</td>
      <td>4</td>
      <td>54</td>
      <td>93</td>
      <td>0</td>
      <td>18</td>
      <td>90</td>
    </tr>
    <tr>
      <th>8</th>
      <td>44</td>
      <td>0</td>
      <td>74</td>
      <td>62</td>
      <td>14</td>
      <td>95</td>
      <td>48</td>
      <td>15</td>
      <td>0</td>
      <td>78</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>62</td>
      <td>40</td>
      <td>85</td>
      <td>80</td>
      <td>82</td>
      <td>53</td>
      <td>24</td>
      <td>26</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We are going to fill both diagonal (right-to-left and left-to-right) with 0. There is a NumPy function called ```fill_diagnol``` to replace values on the left-to-right diagonal but the issue with this function is that it does not replace the right-to-left diagonal as well. We can't use this function, therefore.

To solve this challenge, we first loop through rows in the dataframe and then for each loop we are to replace two elements at a specific position with ```0``` row-wise.  For the left-to-right diagonal, we use the ```iat``` function which takes in at the first position index the row number and at the second position the column number, we use ```i``` for both positions. For the right-to-left diagonal, we use the ```iat``` function again but this time the first position we calculate the total number rows in the dataframe minus ```i``` (as ```i``` changes because of the loop) minus ```1``` because indexes start from 0 and for the second position corresponding to the columns we use ```i```.

### Ex 66: How to get the particular group of a groupby dataframe by key?

Q: This is a question related to the understanding of grouped dataframe. From ```df_grouped```, get the group belonging to ```apple``` as a dataframe.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame({'col1': ['apple', 'banana', 'orange'] * 3,
                   'col2': np.random.rand(9),
                   'col3': np.random.randint(0, 15, 9)})

df_grouped = the_dataframe.groupby(['col1'])
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>0.374540</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>banana</td>
      <td>0.950714</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>orange</td>
      <td>0.731994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>apple</td>
      <td>0.598658</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>banana</td>
      <td>0.156019</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>orange</td>
      <td>0.155995</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>apple</td>
      <td>0.058084</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>banana</td>
      <td>0.866176</td>
      <td>13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>orange</td>
      <td>0.601115</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_grouped
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7fd121fc30d0>



#### Desired output


```python
#     col1      col2  col3
# 0  apple  0.374540     7
# 3  apple  0.598658     4
# 6  apple  0.058084    11
```

#### Solution

#### 1st Method


```python
df_grouped.get_group("apple")
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>0.374540</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>apple</td>
      <td>0.598658</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>apple</td>
      <td>0.058084</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



To get the group belonging to ```apple``` we call ```get_group``` on the ```df_grouped```.

#### 2nd Method


```python
for i, grp_val in df_grouped:
    if i == "apple":
        print(grp_val)
```

        col1      col2  col3
    0  apple  0.374540     7
    3  apple  0.598658     4
    6  apple  0.058084    11


Alternatively, we loop through all the elements in ```df_grouped``` and use the if statement to print the columns in the ```apple``` group.

### Ex 67: How to get the nth largest value of a column when grouped by another column?

Q: In ```the_dataframe```, find the second largest value of ```taste``` for ```banana```.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                   'taste': np.random.rand(9),
                   'price': np.random.randint(0, 15, 9)})
```

#### Desired output


```python
# 0.8661761457749352
```

#### Solution


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
      <th>fruit</th>
      <th>taste</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>0.374540</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>banana</td>
      <td>0.950714</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>orange</td>
      <td>0.731994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>apple</td>
      <td>0.598658</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>banana</td>
      <td>0.156019</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>orange</td>
      <td>0.155995</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>apple</td>
      <td>0.058084</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>banana</td>
      <td>0.866176</td>
      <td>13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>orange</td>
      <td>0.601115</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_grouped = the_dataframe.groupby(by="fruit")
```


```python
sorted(df_grouped.get_group("banana")["taste"])[-2]
```




    0.8661761457749352



Just like we did in the previous exercise, we first group the dataframe using the values in the ```fruit``` column and store it in ```df_grouped```. Then we get ```taste``` column for ```banana``` using ```get_group``` function and sort it out. Finally, to get the second largest element, we use indexing ```[-2]```.

### Ex 68: How to compute grouped mean on pandas DataFrame and keep the grouped column as another column (not index)?

Q: In ```the_dataframe```, Compute the mean price of every fruit, while keeping the fruit as another column instead of an index.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                   'taste': np.random.rand(9),
                   'price': np.random.randint(0, 15, 9)})
```

#### Desired output

![Pandas_ex69](/blog/assets/post_cont_image/pandas_ex69.png)

#### Solution

#### 1st Method


```python
the_dataframe.groupby(by="fruit").mean()["price"].reset_index()
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
      <th>fruit</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>7.333333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>banana</td>
      <td>5.333333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>orange</td>
      <td>5.666667</td>
    </tr>
  </tbody>
</table>
</div>



The most straightforward way to go about this exercise is to group the dataframe by ```fruit``` and get the mean of the numerical columns grouped by ```price``` and reset the index using ```reset_index``` which will change the index from ```fruit``` column to regular ascending numerical index.

#### 2nd Method


```python
the_dataframe.groupby(by="fruit",as_index=False)["price"].mean()
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
      <th>fruit</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>7.333333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>banana</td>
      <td>5.333333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>orange</td>
      <td>5.666667</td>
    </tr>
  </tbody>
</table>
</div>



Alternatively, we can reset the index using the ```as_index``` parameter from the ```groupby``` function. 

### Ex 69: How to join two DataFrames by 2 columns so that they have only the common rows?

Q: Join dataframes ```the_dataframe_1``` and ```the_dataframe_2``` by ```fruit-pazham``` and ```weight-kilo```.


```python
the_dataframe_1 = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                    'weight': ['high', 'medium', 'low'] * 3,
                    'price': np.random.randint(0, 15, 9)})

the_dataframe_2 = pd.DataFrame({'pazham': ['apple', 'orange', 'pine'] * 2,
                    'kilo': ['high', 'low'] * 3,
                    'price': np.random.randint(0, 15, 6)})
```

#### Desire output

![Pandas_ex69](/blog/assets/post_cont_image/pandas_ex69.png)

#### Solution


```python
the_dataframe_1
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
      <th>fruit</th>
      <th>weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>high</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>banana</td>
      <td>medium</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>orange</td>
      <td>low</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>apple</td>
      <td>high</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>banana</td>
      <td>medium</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>orange</td>
      <td>low</td>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>apple</td>
      <td>high</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>banana</td>
      <td>medium</td>
      <td>12</td>
    </tr>
    <tr>
      <th>8</th>
      <td>orange</td>
      <td>low</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
the_dataframe_2
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
      <th>pazham</th>
      <th>kilo</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>high</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>low</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pine</td>
      <td>high</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>apple</td>
      <td>low</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>orange</td>
      <td>high</td>
      <td>14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pine</td>
      <td>low</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(the_dataframe_1, the_dataframe_2, how="inner", left_on=["fruit","weight"], right_on=["pazham","kilo"], suffixes=["_left","_right"])
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
      <th>fruit</th>
      <th>weight</th>
      <th>price_left</th>
      <th>pazham</th>
      <th>kilo</th>
      <th>price_right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>high</td>
      <td>1</td>
      <td>apple</td>
      <td>high</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>high</td>
      <td>0</td>
      <td>apple</td>
      <td>high</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>apple</td>
      <td>high</td>
      <td>5</td>
      <td>apple</td>
      <td>high</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>orange</td>
      <td>low</td>
      <td>4</td>
      <td>orange</td>
      <td>low</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>orange</td>
      <td>low</td>
      <td>9</td>
      <td>orange</td>
      <td>low</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>orange</td>
      <td>low</td>
      <td>11</td>
      <td>orange</td>
      <td>low</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We use the ```merge``` to combine the two dataframes, and set ```how``` parameter to ```inner``` which means that we are only interested in rows with the same value in ```fruit``` and ```weight``` column on the left and ```pazham``` and ```kilo``` column on the right. Finally, we add suffix "_left" and "_right" on those columns.

### Ex 70: How to remove rows from a DataFrame that are present in another DataFrame?

Q: From ```the_dataframe_1```, remove the rows present in ```the_dataframe_2```. All three columns values must be the same for the row to be drop.


```python
np.random.seed(42)
the_dataframe_1 = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                    'weight': ['high', 'medium', 'low'] * 3,
                    'price': np.random.randint(0, 15, 9)})

the_dataframe_2 = pd.DataFrame({'pazham': ['apple', 'orange', 'pine'] * 2,
                    'kilo': ['high', 'low'] * 3,
                    'price': np.random.randint(0, 15, 6)})
```

#### Desired output

![Pandas_ex70](/blog/assets/post_cont_image/pandas_ex70.png)

#### Solution


```python
the_dataframe_1
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
      <th>fruit</th>
      <th>weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>high</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>banana</td>
      <td>medium</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>orange</td>
      <td>low</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>apple</td>
      <td>high</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>banana</td>
      <td>medium</td>
      <td>10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>orange</td>
      <td>low</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>apple</td>
      <td>high</td>
      <td>12</td>
    </tr>
    <tr>
      <th>7</th>
      <td>banana</td>
      <td>medium</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>orange</td>
      <td>low</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
the_dataframe_2
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
      <th>pazham</th>
      <th>kilo</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>high</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>low</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pine</td>
      <td>high</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>apple</td>
      <td>low</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>orange</td>
      <td>high</td>
      <td>10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pine</td>
      <td>low</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
the_dataframe_1[~the_dataframe_1.isin(the_dataframe_2).all(axis=1)]
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
      <th>fruit</th>
      <th>weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>high</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>banana</td>
      <td>medium</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>orange</td>
      <td>low</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>apple</td>
      <td>high</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>banana</td>
      <td>medium</td>
      <td>10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>orange</td>
      <td>low</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>apple</td>
      <td>high</td>
      <td>12</td>
    </tr>
    <tr>
      <th>7</th>
      <td>banana</td>
      <td>medium</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>orange</td>
      <td>low</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



We first get the element in ```the_dataframe_1``` that are present in ```the_dataframe_2``` using the ```isin``` function. A new dataframe with boolean values will is return where ```True``` represent a similar value between ```the_dataframe_1``` and ```the_dataframe_2``` and ```False``` represent a different value. We use ```all``` to get an AND operator function between the boolean values row-wise(```axis``` set to ```1```). For example, in row 4, we'll have "False" AND "False" AND "True" = "False". 

Finally, we use ```~``` to reverse all the boolean value ("False" becomes "True" and "True" becomes "False") and use indexing into ```the_dataframe_1```. We find out that we are keeping all the rows in ```the_dataframe_1``` meaning that no row that is identical in ```the_dataframe_1``` and ```the_dataframe_2```.

### Ex 71: How to get the positions where values of two columns match?

Q: Get the positions where values of two columns match


```python
np.random.seed(42)
the_dataframe = pd.DataFrame({'fruit1': np.random.choice(['apple', 'orange', 'banana'], 10),
                    'fruit2': np.random.choice(['apple', 'orange', 'banana'], 10)})
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
      <th>fruit1</th>
      <th>fruit2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>banana</td>
      <td>banana</td>
    </tr>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>banana</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>3</th>
      <td>banana</td>
      <td>banana</td>
    </tr>
    <tr>
      <th>4</th>
      <td>apple</td>
      <td>orange</td>
    </tr>
    <tr>
      <th>5</th>
      <td>apple</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>6</th>
      <td>banana</td>
      <td>orange</td>
    </tr>
    <tr>
      <th>7</th>
      <td>orange</td>
      <td>orange</td>
    </tr>
    <tr>
      <th>8</th>
      <td>banana</td>
      <td>orange</td>
    </tr>
    <tr>
      <th>9</th>
      <td>banana</td>
      <td>orange</td>
    </tr>
  </tbody>
</table>
</div>



#### Desired output


```python
# [0, 3, 5, 7]
```

#### Solution

#### 1st Method


```python
the_dataframe.where(the_dataframe["fruit1"] == the_dataframe["fruit2"])
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
      <th>fruit1</th>
      <th>fruit2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>banana</td>
      <td>banana</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>banana</td>
      <td>banana</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>apple</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>orange</td>
      <td>orange</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
list(the_dataframe.where(the_dataframe["fruit1"] == the_dataframe["fruit2"]).dropna().index)
```




    [0, 3, 5, 7]



We first call the ```where``` function the ```the_dataframe``` with the condition that we need to the same fruit in column ```fruit1``` and ```fruit2```.  We get boolean values dataframe with the rows where the values are the same and ```NaN``` where the values are different. We drop ```NaN``` using ```dropna``` function and extract the indexes. Finally, place the array into a list et voila!

#### 2nd Method


```python
list(np.where(the_dataframe["fruit1"] == the_dataframe["fruit2"])[0])
```




    [0, 3, 5, 7]




```python
Alternatevely w
```


```python
list(np.where(the_dataframe["fruit1"] == the_dataframe["fruit2"])[0])
```




    [0, 3, 5, 7]




```python
np.where(the_dataframe["fruit1"] == the_dataframe["fruit2"])
```




    (array([0, 3, 5, 7]),)



Alternatively, we can use the ```where``` function which returns a tuple with first element an array of the indexes where the condition is satisfied. We extract that array and cast it into a list. I prefer this second method as it is more concise.

### Ex 72: How to create lags and leads of a column in a DataFrame?

Q: Create two new columns in ```the_dataframe```, one of which is a ```lag1``` (shift column a down by 1 row) of column ```a``` and the other is a ```lead1``` (shift column b up by 1 row).


```python
np.random.seed(42)
the_dataframe = pd.DataFrame(np.random.randint(1, 100, 20).reshape(-1, 4), columns = list('abcd'))
```

#### Desired output

![Pandas_ex72](/blog/assets/post_cont_image/pandas_ex72.png)

#### Solution


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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52</td>
      <td>93</td>
      <td>15</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>61</td>
      <td>21</td>
      <td>83</td>
      <td>87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75</td>
      <td>75</td>
      <td>88</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>22</td>
      <td>53</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>88</td>
      <td>30</td>
      <td>38</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
the_dataframe["lag1"] = the_dataframe["a"].shift(periods=1)
the_dataframe["lead1"] = the_dataframe["a"].shift(periods=-1)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>lag1</th>
      <th>lead1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52</td>
      <td>93</td>
      <td>15</td>
      <td>72</td>
      <td>NaN</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>61</td>
      <td>21</td>
      <td>83</td>
      <td>87</td>
      <td>52.0</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75</td>
      <td>75</td>
      <td>88</td>
      <td>24</td>
      <td>61.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>22</td>
      <td>53</td>
      <td>2</td>
      <td>75.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>88</td>
      <td>30</td>
      <td>38</td>
      <td>2</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



To create a shift of values in a column upward or downward, we use the ```shift``` function on the desired column and pass in as the ```periods``` number ```1``` to shift upward or ```-1``` to shift downward. 

### Ex 73: How to get the frequency of unique values in the entire DataFrame?

Q: Get the frequency of unique values in the entire DataFrame.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame(np.random.randint(1, 10, 20).reshape(-1, 4), columns = list('abcd'))
```

#### Desired output


```python
# 8    5
# 5    4
# 7    3
# 6    2
# 4    2
# 3    2
# 2    2
# dtype: int64
```

#### Solution


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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>4</td>
      <td>8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>3</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>4</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>6</td>
      <td>2</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.value_counts(the_dataframe.values.flatten())
```




    8    5
    5    4
    7    3
    6    2
    4    2
    3    2
    2    2
    dtype: int64



To get the frequency of unique values or how many time one value is repeated in the dataframe, we use ```value_counts``` and pass in the values of ```the_dataframe``` flattened which transform the dataframe from an n-dimensional dataframe into a 1D array.

### Ex 74: How to split a text column into two separate columns?

Q: Split the string column in ```the_dataframe``` to form a dataframe with 3 columns as shown.


```python
the_dataframe = pd.DataFrame(["Temperature, City    Province",
                              "33, Bujumbura    Bujumbura",
                              "30, Buganda    Cibitoke",
                              "25, Ncendajuru    Cankuzo",
                              "35, Giheta    Gitega"], 
                              columns=['row']
                            )
```

#### Desired output

![Pandas_ex74](/blog/assets/post_cont_image/pandas_ex74.png)

#### Solution

#### Step 1: split the string data


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
      <th>row</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Temperature, City    Province</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33, Bujumbura    Bujumbura</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30, Buganda    Cibitoke</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25, Ncendajuru    Cankuzo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35, Giheta    Gitega</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_splitted = the_dataframe.row.str.split(pat=",|\t|    ", expand=True)
```


```python
df_splitted
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Temperature</td>
      <td>City</td>
      <td>Province</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>Bujumbura</td>
      <td>Bujumbura</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>Buganda</td>
      <td>Cibitoke</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>Ncendajuru</td>
      <td>Cankuzo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>Giheta</td>
      <td>Gitega</td>
    </tr>
  </tbody>
</table>
</div>



In this first step, we split the strings in the one column into three different columns. We call the ```split``` function on the ```str``` function from the ```row``` dataframe. We pass in as the pattern a regular expression that targets ```,``` or ```\t``` (tab) and ```     ``` (4 spaces) and set ```expand``` to ```True``` which expand the split strings into separate columns.

#### Step 2: Rename the columns


```python
new_header = df_splitted.iloc[0]
```


```python
new_header
```




    0    Temperature
    1           City
    2       Province
    Name: 0, dtype: object




```python
df_splitted.columns = new_header
```


```python
df_splitted
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
      <th>Temperature</th>
      <th>City</th>
      <th>Province</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Temperature</td>
      <td>City</td>
      <td>Province</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>Bujumbura</td>
      <td>Bujumbura</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>Buganda</td>
      <td>Cibitoke</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>Ncendajuru</td>
      <td>Cankuzo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>Giheta</td>
      <td>Gitega</td>
    </tr>
  </tbody>
</table>
</div>



In step 2, we are going to use the first row strings as the column names. We first extract the row using ```iloc``` and store it in the ```new_header```, Then assign it to ```columns``` of the dataframe.

#### Step 3: Drop the first row


```python
df_splitted.drop(labels=0,axis="index",inplace=True)
```


```python
df_splitted
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
      <th>Temperature</th>
      <th>City</th>
      <th>Province</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>Bujumbura</td>
      <td>Bujumbura</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>Buganda</td>
      <td>Cibitoke</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>Ncendajuru</td>
      <td>Cankuzo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>Giheta</td>
      <td>Gitega</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have the column names all set, we no longer need that first row, so we are dropping it. We call the ```drop``` function on the dataframe and pass in as parameters ```label``` set to ```0``` to tell Pandas that we want to drop a row not a column, then set ```axis``` to ```index``` (We could also have used ```0```) to tell Pandas that we want to drop labels from the index. Finally set ```inplace``` to ```True``` to tell Pandas we don't want a copy of the dataframe that instead, we want the change to occur in the original dataframe.

### Ex 75: How to reverse the columns of a DataFrame?

Q: Reverse all the columns of a DataFrame.


```python
np.random.seed(42)
the_dataframe = pd.DataFrame(np.arange(25).reshape(5, -1))
```

#### Desired output

![Pandas_ex75](/blog/assets/post_cont_image/pandas_ex75.png)

#### Solution


```python
the_dataframe[the_dataframe.columns[::-1]]
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
      <th>4</th>
      <th>3</th>
      <th>2</th>
      <th>1</th>
      <th>0</th>
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
    <tr>
      <th>4</th>
      <td>24</td>
      <td>23</td>
      <td>22</td>
      <td>21</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



This exercise is similar to exercise 57 the difference is that this time we are reversing columns instead of rows. To do the reversal, we first extract the columns and reverse them using indexing ```[::-1]``` and place it into the original dataframe using again indexing.

### Conclusion

Yaaayy! We made it finally. In the last posts, we have explored more than 150 exercises on NumPy and Pandas. I am very confident that after going through all these exercises, you are ready to tackle the next step: Machine Learning with Scikit-learn. We will continue using NumPy in end-to-end Machine Learning projects coming in the next blog posts.


In the next post, we will introduce the common jargon used in Machine Learning and after we will start working on machine learning projects finally. I am super duper excited for the upcoming posts. Remember, practice makes perfect! Find the jupyter notebook version of this post at my GitHub profile [here](https://github.com/semasuka/blog/blob/gh-pages/ipynb/Pandas%20Exercise%20Part%203.ipynb).

Thank you again for doing these exercises with me. I hope you have learned one or two things. If you like this post, please subscribe to stay updated with new posts, and if you have a thought or a question, I would love to hear it by commenting below. Cheers, and keep learning!
