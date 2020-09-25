---
title:  "NumPy Exercises Part 2"
image: /assets/post_images/numpy.png
excerpt_separator: <!-- more -->
tags:
- python
- exercises
- numpy
---

For this second post of NumPy exercises series, we will be doing intermediate level exercises in NumPy and will go through the solution together as we did in the first part. Try to solve the exercises on your own then compare your answer with mine. Let's get started.<!-- more -->

We first import the NumPy.


```python
import numpy as np
```

### Ex 21: Create a two-dimensional array containing random floats between 5 and 10

Q: Let's create a two-dimensional array with a shape of 5x3 that contain random decimal between 5 and 10.

#### Desire output


```python
# Of course the numbers in the array will not be the same as mine, but your solution should have a similar format.
# [[9.60743436 9.93159453 5.13512998]
#  [9.12587012 9.52496391 5.38363015]
#  [6.78095565 9.78322155 8.06633546]
#  [6.52216894 9.34490397 7.42139115]
#  [7.22697926 8.63126859 7.72817642]]
```

#### Solution

#### 1st Method


```python
my_array = np.random.uniform(low=5,high=10,size=15)
resh_array = my_array.reshape(5,-1)
```


```python
print(resh_array)
```

    [[9.60743436 9.93159453 5.13512998]
     [9.12587012 9.52496391 5.38363015]
     [6.78095565 9.78322155 8.06633546]
     [6.52216894 9.34490397 7.42139115]
     [7.22697926 8.63126859 7.72817642]]


We use the uniform method on the random NumPy method and pass the lowest number, then the highest and finally the size.

Then use the reshape method to change it from a one-dimensional array to a two-dimensional array.

#### 2nd Method


```python
my_array = np.random.randint(low=5,high=10,size=(5,3)) + np.random.random((5,3))
```


```python
print(my_array)
```

    [9.60743436 9.93159453 5.13512998 9.12587012 9.52496391 5.38363015
     6.78095565 9.78322155 8.06633546 6.52216894 9.34490397 7.42139115
     7.22697926 8.63126859 7.72817642]


We can achieve the same result by randomly generating integers using the randint method then concatenate (using addition) with randomly generated decimal parts to finally get a number (formed from the random integer and decimal) which is placed in a variable.

Note: The shape of the two arrays concatenated must be the same.

### Ex 22: Print only one decimal place in an array

Q: Print the numbers in the array with only one decimal places.


```python
my_array = np.random.rand(3)
```


```python
print(my_array)
```

    [0.42306537 0.2529142  0.57457565]


#### Desire output


```python
# Of course the numbers in the array will not be the same as mine, but your solution should have a similar format.
#[0.4 0.3 0.6]
```

#### Solution

#### 1st Method


```python
my_array_decimal_place = np.around(my_array,decimals=1)
```


```python
print(my_array_decimal_place)
```

    [0.4 0.3 0.6]


We use the around method, pass as arguments the array itself and the decimal place. In our case, it is 1.

#### 2nd Method


```python
np.set_printoptions(precision=1)
print(my_array)
```

    [0.4 0.3 0.6]


We can also set to 1 the precision argument in the set_printoptions method to the same results.

Note: changing any argument in the set_printoption method affect all the notebook, which means that in our case, all the decimals in all the remaining cells will be printed with one decimal place.  It is crucial to keep this in mind.

### Ex 23: Remove the exponential notation in an array

Q: In this exercise, we want to change the elements written using the exponential scientific notation, to decimal notation.


```python
my_array = np.random.rand(16)/1.34e4
```


```python
print(my_array)
```

    [5.315786434e-05 6.261432178e-05 1.816219472e-05 6.605228344e-05
     5.613819546e-05 6.026712672e-05 1.729978859e-05 8.851793521e-06
     1.456005418e-05 5.940398704e-05 3.211098305e-05 7.301915937e-05
     3.377834308e-05 6.710361776e-05 4.399697976e-05 1.909041117e-06]


#### Desire output


```python
# Of course the numbers in the array will not be the same as mine, but your solution should have a similar format.
# array([0.000053158, 0.000062614, 0.000018162, 0.000066052, 0.000056138,
#        0.000060267, 0.0000173  , 0.000008852, 0.00001456 , 0.000059404,
#        0.000032111, 0.000073019, 0.000033778, 0.000067104, 0.000043997,
#        0.000001909])
```

#### Solution


```python
np.set_printoptions(suppress=True,precision=9)
my_array
```




    array([0.000053158, 0.000062614, 0.000018162, 0.000066052, 0.000056138,
           0.000060267, 0.0000173  , 0.000008852, 0.00001456 , 0.000059404,
           0.000032111, 0.000073019, 0.000033778, 0.000067104, 0.000043997,
           0.000001909])



We remove the exponential scientific by setting the suppress argument to True inside the set_printoptions method and set the precision to 9 because the numbers are too small to be printed using the default precision (which is 1) and it will make the whole array to be composed of 0.0.

### Ex 24: Generate the same random array using the random method

Q: Keep on generating the same random array composed of elements under 30, even on a different system using the random method.

#### Solution


```python
np.random.seed(2)
my_array = np.random.random(30)
```


```python
print(my_array)
```

    [0.435994902 0.025926232 0.549662478 0.435322393 0.420367802 0.330334821
     0.204648634 0.619270966 0.299654674 0.266827275 0.621133833 0.529142094
     0.134579945 0.513578121 0.184439866 0.785335148 0.853975293 0.494236837
     0.846561485 0.079645477 0.50524609  0.065286504 0.428122328 0.096530916
     0.127159972 0.596745309 0.226012001 0.106945684 0.220306207 0.349826285]


We can generate the same array each time by using the seed method. The numbers passed in will determine the array that will be created. In other words, if the number passed in the seed method remain the same, we will keep on generating the exact array.

### Ex 25: Limit the number of elements printed in an array

Q: Limit the number of items printed in an array to a maximum of 6 elements (The first 3 elements and the last 3).


```python
my_array = np.arange(0,100)
```


```python
print(my_array)
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
     48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
     72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
     96 97 98 99]


#### Desire output


```python
# [ 0  1  2 ... 97 98 99]
```

#### Solution


```python
np.set_printoptions(threshold=6)
```


```python
print(my_array)
```

    [ 0  1  2 ... 97 98 99]


Setting threshold argument to 6, we telling NumPy only to print the first three and last three elements in an array.

### Ex 26: Print the full array

Q: This time, we are going to do the opposite of what we did in the previous exercise.

#### Desire output


```python
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
#  48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
#  72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
#  96 97 98 99]
```

#### Solution


```python
np.set_printoptions(threshold=1000)
```


```python
print(my_array)
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
     48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
     72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
     96 97 98 99]


We set back the threshold to the default value, which is 1000. NumPy will print up to 1000 elements in an array before starting hiding the elements in the middle of the array.

### Ex 27: Import the text from a dataset without changing the text

Q: We are going to import [this](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) dataset and place it in NumPy array then print only the first three rows.

#### Desire output


```python
# [(5.1, 3.5, 1.4, 0.2, b'Iris-setosa') (4.9, 3. , 1.4, 0.2, b'Iris-setosa')
#  (4.7, 3.2, 1.3, 0.2, b'Iris-setosa')]
```

#### Solution

#### 1st Method


```python
the_data_1 = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",dtype={'names': ('sepal length', 'sepal width', 'petal length', 'petal width', 'species'),
          'formats': (np.float, np.float, np.float, np.float, '|S15')},delimiter=",")
```


```python
print(the_data_1[:3])
```

    [(5.1, 3.5, 1.4, 0.2, b'Iris-setosa') (4.9, 3. , 1.4, 0.2, b'Iris-setosa')
     (4.7, 3.2, 1.3, 0.2, b'Iris-setosa')]


So we can store the data in an array using the loadtxt method, the first argument is the path (online URL or path on a local machine) to the dataset, then a dictionary that will hold the name and the data type of each column and finally, we set the delimiter to be a comma since we are importing from a CSV file.

#### 2nd Method


```python
the_data_2 = np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",dtype={'names': ('sepal length', 'sepal width', 'petal length', 'petal width', 'species'),
          'formats': (np.float, np.float, np.float, np.float, '|S15')},delimiter=",")
```


```python
print(the_data_2[:3])
```

    [(5.1, 3.5, 1.4, 0.2, b'Iris-setosa') (4.9, 3. , 1.4, 0.2, b'Iris-setosa')
     (4.7, 3.2, 1.3, 0.2, b'Iris-setosa')]


We can achieve the same result by using the genfromtxt method, which does also takes the path (online or local) to the dataset, then the name and data type of each column in a key-value pair format (dictionaries) and finally the delimiter.

### Ex 28: Extract a specific column from the previous dataset

Q: get the sepal width column from the previous dataset.

#### Desire output


```python
#array([b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
#       b'Iris-setosa'], dtype='|S15')
```

#### Solution


```python
the_data_1['species'][:5]
```




    array([b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa'], dtype='|S15')



We are using the indexing on the_data_1 array to get back the specified name of the column then print the first five rows. 

### Ex 29: Import specific columns in the dataset as a two-dimensional array

Q: Import only the columns that contain numbers (as float) in the dataset as a two-dimensional array by omitting the species column. Then print the first five rows.

#### Desire output


```python
# array([[5.1, 3.5, 1.4, 0.2],
#        [4.9, 3. , 1.4, 0.2],
#        [4.7, 3.2, 1.3, 0.2],
#        [4.6, 3.1, 1.5, 0.2],
#        [5. , 3.6, 1.4, 0.2]])
```

#### Solution


```python
the_data = np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",delimiter=",",dtype="float",usecols=[0,1,2,3])
```


```python
the_data[:5]
```




    array([[5.1, 3.5, 1.4, 0.2],
           [4.9, 3. , 1.4, 0.2],
           [4.7, 3.2, 1.3, 0.2],
           [4.6, 3.1, 1.5, 0.2],
           [5. , 3.6, 1.4, 0.2]])



We can use the genfromtxt or loadtxt method, pass in the path to the dataset, the delimiter, set the data type of all the columns to be float and finally select the desired columns (column 0 to 3) which will exclude the species column. The returned array will be a two-dimensional array.

### Ex 30: Compute the mean, median, standard deviation of a NumPy column

Q: Find the mean, median, standard deviation of iris's sepal length (1st column).

#### Desire output


```python
# The mean is 5.84, the median is 5.80, the std is 0.8253012917851409
```

#### Solution


```python
sepal_lenght_mean = np.mean(the_data_1["sepal length"])
```


```python
sepal_lenght_std = np.std(the_data_1["sepal length"])
```


```python
sepal_lenght_median = np.median(the_data_1["sepal length"])
```


```python
print("The mean is {:.2f}, the median is {:.2f}, the std is {}".format(sepal_lenght_mean,sepal_lenght_median,sepal_lenght_std))
```

    The mean is 5.84, the median is 5.80, the std is 0.8253012917851409


We first used indexing to get the specified column and passed it as an argument to the NumPy mean, std and the median method.

### Ex 31: Normalize an array so that all the values range between 0 and 1?

Q: Create a normalized form of iris's sepal length on a scale of 0 and 1, where 0 is the lowest number, and 1 the highest number.

#### Desire output


```python
# [0.22222 0.16667 0.11111 0.08333 0.19444 0.30556 0.08333 0.19444 0.02778
#  0.16667 0.30556 0.13889 0.13889 0.      0.41667 0.38889 0.30556 0.22222
#  0.38889 0.22222 0.30556 0.22222 0.08333 0.22222 0.13889 0.19444 0.19444
#  0.25    0.25    0.11111 0.13889 0.30556 0.25    0.33333 0.16667 0.19444
#  0.33333 0.16667 0.02778 0.22222 0.19444 0.05556 0.02778 0.19444 0.22222
#  0.13889 0.22222 0.08333 0.27778 0.19444 0.75    0.58333 0.72222 0.33333
#  0.61111 0.38889 0.55556 0.16667 0.63889 0.25    0.19444 0.44444 0.47222
#  0.5     0.36111 0.66667 0.36111 0.41667 0.52778 0.36111 0.44444 0.5
#  0.55556 0.5     0.58333 0.63889 0.69444 0.66667 0.47222 0.38889 0.33333
#  0.33333 0.41667 0.47222 0.30556 0.47222 0.66667 0.55556 0.36111 0.33333
#  0.33333 0.5     0.41667 0.19444 0.36111 0.38889 0.38889 0.52778 0.22222
#  0.38889 0.55556 0.41667 0.77778 0.55556 0.61111 0.91667 0.16667 0.83333
#  0.66667 0.80556 0.61111 0.58333 0.69444 0.38889 0.41667 0.58333 0.61111
#  0.94444 0.94444 0.47222 0.72222 0.36111 0.94444 0.55556 0.66667 0.80556
#  0.52778 0.5     0.58333 0.80556 0.86111 1.      0.58333 0.55556 0.5
#  0.94444 0.55556 0.58333 0.47222 0.72222 0.66667 0.72222 0.41667 0.69444
#  0.66667 0.66667 0.55556 0.61111 0.52778 0.44444]
```

#### Solution


```python
the_min = the_data_1["sepal length"].min()
the_max = the_data_1["sepal length"].max()
```


```python
norm_data = (the_data_1["sepal length"] - the_min) / (the_max - the_min)
```


```python
np.set_printoptions(precision=5)
print(norm_data)
```

    [0.22222 0.16667 0.11111 0.08333 0.19444 0.30556 0.08333 0.19444 0.02778
     0.16667 0.30556 0.13889 0.13889 0.      0.41667 0.38889 0.30556 0.22222
     0.38889 0.22222 0.30556 0.22222 0.08333 0.22222 0.13889 0.19444 0.19444
     0.25    0.25    0.11111 0.13889 0.30556 0.25    0.33333 0.16667 0.19444
     0.33333 0.16667 0.02778 0.22222 0.19444 0.05556 0.02778 0.19444 0.22222
     0.13889 0.22222 0.08333 0.27778 0.19444 0.75    0.58333 0.72222 0.33333
     0.61111 0.38889 0.55556 0.16667 0.63889 0.25    0.19444 0.44444 0.47222
     0.5     0.36111 0.66667 0.36111 0.41667 0.52778 0.36111 0.44444 0.5
     0.55556 0.5     0.58333 0.63889 0.69444 0.66667 0.47222 0.38889 0.33333
     0.33333 0.41667 0.47222 0.30556 0.47222 0.66667 0.55556 0.36111 0.33333
     0.33333 0.5     0.41667 0.19444 0.36111 0.38889 0.38889 0.52778 0.22222
     0.38889 0.55556 0.41667 0.77778 0.55556 0.61111 0.91667 0.16667 0.83333
     0.66667 0.80556 0.61111 0.58333 0.69444 0.38889 0.41667 0.58333 0.61111
     0.94444 0.94444 0.47222 0.72222 0.36111 0.94444 0.55556 0.66667 0.80556
     0.52778 0.5     0.58333 0.80556 0.86111 1.      0.58333 0.55556 0.5
     0.94444 0.55556 0.58333 0.47222 0.72222 0.66667 0.72222 0.41667 0.69444
     0.66667 0.66667 0.55556 0.61111 0.52778 0.44444]


We first get the lowest and the highest number in the sepal length column.

Then calculate the norm by subtracting each element in the column by the minimum and dividing it by the maximum minus the minimum.

Finally, we set the precision to 5.

### Ex 32: Find the softmax score

Q: Compute the softmax score of the sepal length column and print only the first five values.

#### Desire output


```python
# [0.002219585 0.001817243 0.001487833 0.001346247 0.002008364]
```

#### Solution

#### 1st Method


```python
softmax_score = np.exp(the_data_1["sepal length"])/sum(np.exp(the_data_1["sepal length"]))
print(softmax_score[:5])
```

    [0.002219585 0.001817243 0.001487833 0.001346247 0.002008364]


The softmax score is obtained by getting the exponential of each element in the sepal length column then dividing it with the sum of all exponentials of elements in the sepal length column.

#### 2nd Method


```python
from scipy.special import softmax

softmax_score = softmax(the_data_1["sepal length"])
print(softmax_score[:5])
```

    [0.002219585 0.001817243 0.001487833 0.001346247 0.002008364]


We can avoid hardcoding the softmax by importing the softmax method from SciPy library.

### Ex 33: Find different percentile scores of a column

Q: Find the 5th and 95th percentile of iris's sepal length column.

#### Desire output


```python
# The 5th percentile of iris's sepal length column 4.6 and the 95th percentile is 7.25
```

#### Solution


```python
percentile_5 = np.percentile(the_data_1["sepal length"],5)
```


```python
percentile_95 = np.percentile(the_data_1["sepal length"],95)
```


```python
print("The 5th percentile of iris's sepal length column {} and the 95th percentile is {:.2f}".format(percentile_5,percentile_95))
```

    The 5th percentile of iris's sepal length column 4.6 and the 95th percentile is 7.25


We are using the percentile method from NumPy, pass in the sepal length column then we specify the percentile number as a second argument.

### Ex 34: Insert a nan value at a random position in an array

Q: Insert np.nan values at 20 random positions in dataset.


```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
the_data_2d = np.genfromtxt(url,dtype="object",delimiter=",")
```

#### Desire output


```python
# use random.seed(100) to choose the number so that you can compare your answer with mine
# array([[b'5.1', b'3.5', b'1.4', b'0.2', nan],
#        [b'4.9', b'3.0', b'1.4', b'0.2', b'Iris-setosa'],
#        [b'4.7', b'3.2', b'1.3', b'0.2', b'Iris-setosa'],
#        [b'4.6', b'3.1', b'1.5', b'0.2', b'Iris-setosa'],
#        [b'5.0', b'3.6', b'1.4', b'0.2', b'Iris-setosa'],
#        [b'5.4', b'3.9', b'1.7', b'0.4', b'Iris-setosa'],
#        [b'4.6', b'3.4', b'1.4', b'0.3', b'Iris-setosa'],
#        [b'5.0', b'3.4', b'1.5', b'0.2', b'Iris-setosa'],
#        [nan, b'2.9', b'1.4', b'0.2', b'Iris-setosa'],
#        [b'4.9', b'3.1', b'1.5', b'0.1', b'Iris-setosa']], dtype=object)
```

#### Solution


```python
np.random.seed(100)
the_data_2d[np.random.randint(150,size=20),np.random.randint(5,size=20)] = np.nan
```


```python
the_data_2d[:10]
```




    array([[b'5.1', b'3.5', b'1.4', b'0.2', b'Iris-setosa'],
           [b'4.9', b'3.0', b'1.4', b'0.2', b'Iris-setosa'],
           [b'4.7', b'3.2', b'1.3', b'0.2', b'Iris-setosa'],
           [b'4.6', b'3.1', b'1.5', b'0.2', b'Iris-setosa'],
           [b'5.0', b'3.6', b'1.4', b'0.2', b'Iris-setosa'],
           [b'5.4', b'3.9', b'1.7', b'0.4', b'Iris-setosa'],
           [b'4.6', b'3.4', b'1.4', b'0.3', b'Iris-setosa'],
           [b'5.0', b'3.4', b'1.5', b'0.2', b'Iris-setosa'],
           [b'4.4', b'2.9', nan, b'0.2', b'Iris-setosa'],
           [b'4.9', b'3.1', b'1.5', b'0.1', b'Iris-setosa']], dtype=object)



We have used indexing on this data set to insert np.nan randomly. The first part of indexing, we are randomly selecting 20 positions from 0 to 149, which correspond to the number of rows. 

Then the second part, we have also randomly chosen 20 positions between 0 to 4. In the end, we will have 20 pairs (rows and columns) of positions randomly generated. Nan will replace the element at those positions.

Finally, we print the first ten rows.

### Ex 35: Find the position of missing values (nan) in NumPy array

Q: Find the number of times nan occurs and which row is it in the sepal length column (1st column).


```python
np.random.seed(100)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
the_data_2d = np.genfromtxt(url, delimiter=',', dtype='float',usecols=[0,1,2,3])
the_data_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
```

#### Desire output


```python
# The total number of nan in the sepal length column is 4 and the position of nan is [14 34 79 87]
```

#### Solution


```python
nan_total = np.isnan(the_data_2d[:,0]).sum()
nan_position = np.where(np.isnan(the_data_2d[:,0]))[0]
```


```python
print("The total number of nan in the sepal length column is {} and the position of nan is {}".format(nan_total,nan_position))
```

    The total number of nan in the sepal length column is 4 and the position of nan is [14 34 79 87]


To get the total number of nan in the first column, we use indexing to get all the elements in the first column and apply isnan method only to get back all the occurrence of nan in the column, then use the sum to get the total number of nan.

To get the position, we start by extracting the column and apply the isnan method again, but this time we are using the where method to get back a boolean array composed where True is the position nan is found, and False is the position of any value other than nan.

The where method returns a tuple, but we are only interested in the first element that is why we used [0] at the end.

### Ex 36: Filter a NumPy array based on two or more conditions

Q: Filter the rows of the_data_2d dataset that have a petal length (3rd column) > 1.5 and sepal length (1st column) < 5.0.


```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
the_data_2d = np.genfromtxt(url, delimiter=',', dtype='float',usecols=[0,1,2,3])
```

#### Desire output


```python
# array([[4.8, 3.4, 1.6, 0.2],
#        [4.8, 3.4, 1.9, 0.2],
#        [4.7, 3.2, 1.6, 0.2],
#        [4.8, 3.1, 1.6, 0.2],
#        [4.9, 2.4, 3.3, 1. ],
#        [4.9, 2.5, 4.5, 1.7]])
```

#### Solution


```python
the_condition = (the_data_2d[:,2]>1.5) & (the_data_2d[:,0]<5.0)
the_data_2d[the_condition]
```




    array([[4.8, 3.4, 1.6, 0.2],
           [4.8, 3.4, 1.9, 0.2],
           [4.7, 3.2, 1.6, 0.2],
           [4.8, 3.1, 1.6, 0.2],
           [4.9, 2.4, 3.3, 1. ],
           [4.9, 2.5, 4.5, 1.7]])



We get the rows where the elements are greater than 1.5  in the third column and bigger than five the in the first column.

We are using indexing with the and operator (&) to get the elements that meet those two conditions in the two columns and store them in a variable.

Finally, we use the variable with indexing to get back all the rows that meet the two conditions.

### Ex 37: Drop rows that contain a missing value

Q: Select only the rows in the_data_2d that does not have any nan value.


```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
the_data_2d = np.genfromtxt(url,usecols=[0,1,2,3],delimiter=",",dtype="float")
np.random.seed(100)
#injecting nan value in the dataset
the_data_2d[np.random.randint(150,size=20),np.random.randint(4,size=20)] = np.nan
```

#### Desire output


```python
# The output should be the same by using the seed method with 
# array([[5.1, 3.5, 1.4, 0.2],
#        [4.9, 3. , 1.4, 0.2],
#        [4.7, 3.2, 1.3, 0.2],
#        [4.6, 3.1, 1.5, 0.2],
#        [5. , 3.6, 1.4, 0.2]])
```

#### Solution


```python
the_condition = [np.sum(np.isnan(the_data_2d), axis = 1)==0]
the_data_2d[tuple(the_condition)][:5]
```




    array([[5.1, 3.5, 1.4, 0.2],
           [4.9, 3. , 1.4, 0.2],
           [4.7, 3.2, 1.3, 0.2],
           [4.6, 3.1, 1.5, 0.2],
           [5. , 3.6, 1.4, 0.2]])



We use the isnan method to get all the occurrences of nan in the whole dataset. We get back a boolean array where True is a nan value and False represents any other value. This boolean is then placed in a sum method which will sum up all the occurrences of True value row-wise because we have set the axis to 1. 

After getting the total counts for each row, we compare it to 0 to get back a boolean array where True represents the rows that don't contain any nan value.

Now that we have this array, we can use indexing on the original array to get back the first five rows which don't contain any nan value.

### Ex 38: Find the correlation between two columns of a NumPy array?

Q: Find the correlation between sepal length(1st column) and petal length(3rd column) in the_data_2d dataset.


```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
the_data_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
```

#### Desired output


```python
# 0.8717541573048718
```

#### Solution

For this exercise, we are asked to calculate the correlation. But what is a correlation in the first place? Well, it is a dependence or association is any statistical relationship, whether causal or not, between two random variables or bivariate data. In simple words, correlation indicates the degree of a linear relationship between two numeric variables. Remember, correlation does not imply causation. Sometimes, a correlation is also called Pearson's correlation

Strictly speaking, Pearson's correlation requires a normally distributed dataset and not necessarily zero-mean. Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation. Correlations of -1 or +1 imply an exact linear relationship. Positive correlations indicate that as x increases, so
does y. Negative correlations suggest that as x increases, y decreases. 

#### 1st Method


```python
all_correlation = np.corrcoef((the_data_2d[:,0],the_data_2d[:,2]))
all_correlation
```




    array([[1.         , 0.871754157],
           [0.871754157, 1.         ]])




```python
print(all_correlation[0][1])
```

    0.8717541573048718


The first way, we can get the correlation in any row or column is by using the corrcoef method. We pass in a tuple composed of sepal length column (which is the first column) and petal length column (which is the second column).

We get back a two-dimensional array of all possible combination of the sepal length and petal length column including correlation of each column with itself (which is +1). Now we only have to extract the correlation of sepal length and petal length located in the first row (position 0), the second column (1).

#### 2nd Method


```python
from scipy.stats.stats import pearsonr
```


```python
corr,p_value = pearsonr(the_data_2d[:,0],the_data_2d[:,2])
```


```python
print("The correlation is {} and the p value is {}".format(corr,p_value))
```

    The correlation is 0.8717541573048712 and the p value is 1.0384540627941809e-47


We can import the Pearson's correlation method from SciPy and pass in the two columns. We get back two values, correlation and p-value. 

The p-value roughly indicates the probability of an uncorrelated system producing datasets that correlate at least as extreme as the one computed. The lower the p-value (<0.01), the stronger is the significance of the relationship. It is not an indicator of strength. The p-values are not entirely reliable but are probably reasonable for datasets larger than 500 or so.

### Ex 39: Find out if a given array has any null or nan values

Q: Find out if the_data_2d has any missing values.


```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
the_data_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
np.random.seed(100)
the_data_2d[np.random.randint(150,size=20),np.random.randint(4,size=20)] = np.nan
```

#### Desire output


```python
#True
```

#### Solution


```python
np.isnan(the_data_2d).any()
```




    True



We use the isnan method to get back an array of boolean where True is a nan value, and False is any other value other than nan. We apply the any method to the boolean array which will return True because there is at least one True value in the boolean array.

### Ex 40: Replace all nan values by 0 in the array

Q: Replace all the occurrences of nan by 0 in the array.


```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
the_data_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
np.random.seed(100)
#injecting nan values
the_data_2d[np.random.randint(150,size=20),np.random.randint(4,size=20)] = np.nan
```

#### Desire output


```python
#the_data_2d[:10]
# array([[5.1, 3.5, 1.4, 0.2],
#        [4.9, 3. , 1.4, 0.2],
#        [4.7, 3.2, 1.3, 0.2],
#        [4.6, 3.1, 1.5, 0.2],
#        [5. , 3.6, 1.4, 0.2],
#        [5.4, 3.9, 1.7, 0.4],
#        [4.6, 3.4, 1.4, 0.3],
#        [5. , 3.4, 1.5, 0.2],
#        [4.4, 0. , 1.4, 0.2],
#        [4.9, 3.1, 1.5, 0.1]])
```

#### Solution


```python
the_data_2d[np.isnan(the_data_2d)] = 0
```


```python
the_data_2d[:10]
```




    array([[5.1, 3.5, 1.4, 0.2],
           [4.9, 3. , 1.4, 0.2],
           [4.7, 3.2, 1.3, 0.2],
           [4.6, 3.1, 1.5, 0.2],
           [5. , 3.6, 1.4, 0.2],
           [5.4, 3.9, 1.7, 0.4],
           [4.6, 3.4, 1.4, 0.3],
           [5. , 3.4, 1.5, 0.2],
           [4.4, 0. , 1.4, 0.2],
           [4.9, 3.1, 1.5, 0.1]])



We used the isnan method again to get back all the instances of nan, then replace each of these instances by assigning 0 to the indexed array.

### Conclusion

By now, you should have a great understanding of NumPy and most importantly, how you can use Google and the NumPy's documentation to see how to perform a specific task in NumPy.

If you were able to do half of these exercises from these two posts, congratulation!! I am confident to tell you that you can start using NumPy into your ML projects. Most ML projects won't require advanced NumPy knowledge than this. The last post from these series is coming soon. It has some advanced NumPy exercises for those who want to master NumPy.

Find the jupyter notebook version of this post at my GitHub profile [here.](https://github.com/semasuka/blog/blob/gh-pages/ipynb/NumPy%20Exercises%20Part%202.ipynb)

Thank you for doing these exercises with me. I hope you have learned one or two things. If you like this post, please subscribe to stay updated with new posts, and if you have a thought or a question, I would love to hear it by commenting below. Remember keep learning!
