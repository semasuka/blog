---
title:  "NumPy Exercises Part 1"
image: /assets/post_images/numpy.png
excerpt_separator: <!-- more -->
tags:
- python
- exercises
- numpy
---

In this post, we will be solving 20 exercises in NumPy to sharpen what you have learnt from the NumPy introduction post. If you have not read the NumPy post, I highly encourage to go first through that post [on this link](https://semasuka.github.io/blog/2019/07/21/numpy-crash-course.html){:target="_blank"} and then come back to try out the exercises.<!-- more -->

Before we start, please allow me to give you some update about the blog. It is with an honour that I am announcing that MIB has been ranked among the top 40 Machine Learning blog to follow in 2019 alongside some very known Machine Learning blog like Google Machine Learning News, MIT News, Machine Learning Mastery and many more great blogs. To see the full list, please visit this [link](https://blog.feedspot.com/machine_learning_blogs/){:target="_blank"}.

As of the time of writing this post, MIB is just 8 months old and has been ranked already among the top 40 ML blogs. I want to thank all the readers of this blog and Feedspot because you are, indeed my motivation to keep on posting and learning ML. Many of you had asked me how they could support the blog; now you can donate by scrolling to the bottom of the about page [here](https://semasuka.github.io/blog/about/){:target="_blank"}.

I am also taking this occasion to tell you a little bit about Feedspot, Feedspot is for all of us who have busy lives and don't have the time to check the contents/news from each one of our favourite platforms like blog, RSS, YouTube channel, Podcast, Magazine and many more, Feedspot will take care of all the contents and compile them in one place for you and save you time. Please visit the website at [https://www.feedspot.com/](https://www.feedspot.com/){:target="_blank"}.

Now let get back to work; this first part of NumPy exercises posts series is composed of relatively easy exercise compared to part 2. Part 3 will be composed of intermediate and advanced level exercises. These exercises are inspired from [this](https://www.machinelearningplus.com/python/101-numpy-exercises-python/){:target="_blank"} fantastic blog post and I want to give credit to the author of their amazing work, but unfortunately, there are no explanations of the "why" which is the most important thing to understand when trying to grasp a new concept, so I will try to explain in my solutions.

Here is how we will proceed for these series of exercises, I will first post the exercise then you will go and try on your own, only after trying you shall come back and compare your result with mine. Do not cheat ;p

If you can't solve the challenge after giving all your best, it is ok to come back and check the solution. Don't feel bad about it; that is how everyone learns, yes! by failling first and figure it out after but most importantly try to understand how the exercise is solved. If you still don't understand, jump at the comments section below and ask me the question, I will happy to assist you as much as I can. 

Note: You will also need to do a little bit of research online for some of the exercises, which is a good thing because googling is an art you must learn and practice as programmer. Use also the handy [NumPy's documentation](https://numpy.org/devdocs/){:target="_blank"}. An exercise will most likely have more than one way to solve it, as long as the desired output is the same as the solution. You're good!

### Ex 1: Import NumPy and check its version

Q: For our first exercise, we will import NumPy library simple right? also, print its version

#### Solution


```python
import numpy as np
```

Before using NumPy anywhere, you must import it.

I hope this one, everyone got it right and you could have called NumPy anything else, but np is what is commonly accepted.


```python
print(np.version.version)
```

    1.16.4


Now to see which version of NumPy we are using, we the version method twice on the NumPy.

### Ex 2: Create a one-dimensional NumPy array 

Q: Now let's create a one-dimensional NumPy array from 0 to 9.

#### Desired output:


```python
# [0 1 2 3 4 5 6 7 8 9]
```

#### Solution


```python
my_arr = np.arange(10)
print(my_arr)
```

    [0 1 2 3 4 5 6 7 8 9]


We use the arange method to generate a sequence from 0 to 9.

### Ex 3: Create a boolean NumPy array

Q: Let's create a three-dimensional array composed each of three boolean True value elements (3X3).

#### Desired output:


```python
# [[ True  True  True]
# [ True  True  True]
# [ True  True  True]]
```

#### Solution


```python
my_arr = np.full((3,3),True)
print(my_arr)
```

    [[ True  True  True]
     [ True  True  True]
     [ True  True  True]]


We used the full method and passed in as the first argument, the shape of the array, and as the second argument, the value that will populate the array.

### Ex 4: Extract elements in a one-dimension array given specific condition

Q: Let's extract only even numbers from the following array.


```python
my_arr = np.arange(1,21)
print(my_arr)
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]


#### Desired output:


```python
# [ 2  4  6  8 10 12 14 16 18 20]
```

#### Solution


```python
even_array = my_arr[my_arr%2==0]
```


```python
print(even_array)
```

    [ 2  4  6  8 10 12 14 16 18 20]


We use indexing with a condition as the argument. For the condition to be True, it is required that for each element modulo 2 to be equal to 0. Modulo is the remainder(the rest) from the division of each element in the array with the divisor. In our case, the divisor is 2, and if the rest is 0, then we know that it is True because only even numbers divided by 2 return 0. Using this technique, we were able to get back all the even numbers in the array.

### Ex 5: Replace elements in a one-dimension array given specific condition

Q: Let's replace each odd number from the following array with a -1.


```python
my_arr = np.arange(1,21)
print(my_arr)
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]


#### Desired output


```python
# [-1  2 -1  4 -1  6 -1  8 -1 10 -1 12 -1 14 -1 16 -1 18 -1 20]
```

#### Solution


```python
odd_array = (my_arr%2!=0)
```


```python
my_arr[odd_array] = -1
```


```python
print(my_arr)
```

    [-1  2 -1  4 -1  6 -1  8 -1 10 -1 12 -1 14 -1 16 -1 18 -1 20]


We first get a boolean array that returns True if the element is odd or return False if the element is even using modulus by 2. After getting the boolean array, we use indexing on the original array and set it to -1 which will replace all the position where there is a True value by -1. 

### Ex 6: Replace elements in a one-dimension array given particular condition without affecting the original array

Q: Same question as to the previous one, just that this time the original array should not be changed.


```python
my_arr = np.arange(1,21)
print(my_arr)
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]


#### Desired output


```python
# new_arr
# [-1  2 -1  4 -1  6 -1  8 -1 10 -1 12 -1 14 -1 16 -1 18 -1 20]
# my_arr
# [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
```

#### Solution

#### 1st Method


```python
odd_array = (my_arr%2!=0)
```


```python
new_array = my_arr.copy()
new_array[odd_array] = -1
```


```python
print(new_array)
```

    [-1  2 -1  4 -1  6 -1  8 -1 10 -1 12 -1 14 -1 16 -1 18 -1 20]



```python
print(my_arr)
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]


This example is very similar to the previous one, the only difference is that instead of using the indexing on the original array, we are using a copy of the original array which won't be affected at all by the changes.

#### 2nd Method


```python
new_array = np.where(my_arr%2 != 0,-1,my_arr)
```


```python
print(new_array)
```

    [-1  2 -1  4 -1  6 -1  8 -1 10 -1 12 -1 14 -1 16 -1 18 -1 20]


There is a handy where method, which can is more concise than the first method. We first pass the condition, then the value to use where the condition was evaluated to True and finally, the original array. 

### Ex 7: Reshape an array

Q: Let's change a one-dimensional array into a two-dimensional array with two rows.


```python
my_arr = np.arange(1,41)
```


```python
print(my_arr)
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40]


#### Desired output


```python
# my_array
# [[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
# [21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40]]
```

#### Solution

#### 1st Method


```python
resh_arr = my_arr.reshape((2,20))
```


```python
print(resh_arr)
```

    [[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
     [21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40]]


We have applied the reshape method to the original array, pass in as argument a tuple with a first number being the number of rows and the second number being the number of elements in each row. 

Note: Remember that the number of elements must fit in the rows. Here is a formula that can help you reshape an array and avoid errors. The number of elements in 1D = number of row in nD X number of elements in each row.

#### 2nd Method

We can avoid all the hustle of figuring out how to fit the elements in the row with the correct numbers of row and column, we can use just -1 as the second argument in the tuple. Using -1, NumPy will figure out how many elements need to be placed in each array depending on the number of rows.


```python
resh_arr = my_arr.reshape((2,-1))
```


```python
print(resh_arr)
```

    [[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
     [21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40]]


Let's do another example.


```python
resh_arr = my_arr.reshape((4,-1))
```


```python
print(resh_arr)
```

    [[ 1  2  3  4  5  6  7  8  9 10]
     [11 12 13 14 15 16 17 18 19 20]
     [21 22 23 24 25 26 27 28 29 30]
     [31 32 33 34 35 36 37 38 39 40]]


Using -1 as the second parameter in the tuple, NumPy has figured out that to have a four-dimensional array; each array needs to have 10 elements each.

### Ex 8: Stack arrays together vertically

Q: Let's stack two arrays together vertically (row-wise) to form one array.


```python
my_arr_1 = np.arange(1,21).reshape(2,-1)
my_arr_2 = np.arange(21,41).reshape(2,-1)
```

#### Desired output


```python
# [[ 1  2  3  4  5  6  7  8  9 10]
#  [11 12 13 14 15 16 17 18 19 20]
#  [21 22 23 24 25 26 27 28 29 30]
#  [31 32 33 34 35 36 37 38 39 40]]
```

#### Solution

#### 1st Method


```python
my_arr_3 = np.vstack((my_arr_1,my_arr_2))
```


```python
print(my_arr_3)
```

    [[ 1  2  3  4  5  6  7  8  9 10]
     [11 12 13 14 15 16 17 18 19 20]
     [21 22 23 24 25 26 27 28 29 30]
     [31 32 33 34 35 36 37 38 39 40]]


the vstack method stack together arrays vertically. 

Note: if two arrays don't have the same number of columns, they can not stack together vertically.


```python
my_arr_4 = np.arange(1,16).reshape(3,-1)
```




    array([[ 1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10],
           [11, 12, 13, 14, 15]])




```python
my_arr_5 = np.vstack((my_arr_1,my_arr_4))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-189-f8c619030909> in <module>
    ----> 1 my_arr_5 = np.vstack((my_arr_1,my_arr_4))
    

    /anaconda3/lib/python3.7/site-packages/numpy/core/shape_base.py in vstack(tup)
        281     """
        282     _warn_for_nonsequence(tup)
    --> 283     return _nx.concatenate([atleast_2d(_m) for _m in tup], 0)
        284 
        285 


    ValueError: all the input array dimensions except for the concatenation axis must match exactly


my_arr_4 with 5 columns can not be concatenated with my_arr_1 which has 10 columns.

#### 2nd Method


```python
my_arr_3 = np.concatenate((my_arr_1,my_arr_2),axis=0)
```


```python
print(my_arr_3)
```

    [[ 1  2  3  4  5  6  7  8  9 10]
     [11 12 13 14 15 16 17 18 19 20]
     [21 22 23 24 25 26 27 28 29 30]
     [31 32 33 34 35 36 37 38 39 40]]


We can achieve the same result using the concatenate method; we must pass another argument, which is the axis. If we set the axis to 0, this means that we are stacking the array vertically (row-wise) and if we set it to 1, this means that we are stacking the array horizontally (column-wise).

### Ex 9: Stack arrays together horizontally

Q: Let's stack two arrays together horizontally (column-wise) to form one array.


```python
my_arr_1 = np.arange(1,21).reshape(2,-1)
my_arr_2 = np.arange(21,41).reshape(2,-1)
```

#### Desired output


```python
# [[ 1  2  3  4  5  6  7  8  9 10 21 22 23 24 25 26 27 28 29 30]
#  [11 12 13 14 15 16 17 18 19 20 31 32 33 34 35 36 37 38 39 40]]
```

#### Solution

#### 1st Method


```python
my_arr_3 = np.hstack((my_arr_1,my_arr_2))
```


```python
print(my_arr_3)
```

    [[ 1  2  3  4  5  6  7  8  9 10 21 22 23 24 25 26 27 28 29 30]
     [11 12 13 14 15 16 17 18 19 20 31 32 33 34 35 36 37 38 39 40]]


Same as we did with the vstack method, we can do the same using the hstack method this time to stack horizontally.

#### 2nd Method


```python
my_arr_3 = np.concatenate((my_arr_1,my_arr_2),axis=1)
```


```python
print(my_arr_3)
```

    [[ 1  2  3  4  5  6  7  8  9 10 21 22 23 24 25 26 27 28 29 30]
     [11 12 13 14 15 16 17 18 19 20 31 32 33 34 35 36 37 38 39 40]]


We can use concatenate; we need only to set the axis to 1.

### Ex 10: Generate a sequence without hardcoding

Q: Create the following sequence given my_arr array with only numpy methods. No hardcoding.


```python
my_arr = np.array([11,22,33])
```

#### Desired output


```python
#[11 11 11 22 22 22 33 33 33 11 22 33 11 22 33]
```

#### Solution


```python
trip_each = np.repeat(my_arr,3)
```


```python
double_array = np.tile(my_arr,2)
```


```python
final_arr = np.concatenate((trip_each,double_array),axis=0)
```


```python
print(final_arr)
```

    [11 11 11 22 22 22 33 33 33 11 22 33 11 22 33]


In this example, we have used 3 different methods to achieve the desired result. 

First, we used the repeat method to repeat each element in the array. The first argument in this method is the array itself, and the second argument is the number of repeats to be executed then store it in a variable.

Second is the tile method which will duplicate the entire array. The first argument will be the array, and the second will be how many times we repeat the array then store it in another variable.

The third method is the concatenate method, which does a concatenation of trip_each with double_array horizontally by setting the axis to 0 (column-wise).

### Ex 11: Get the common elements in arrays

Q: Get common elements in two arrays


```python
array_1 = np.arange(0,31)
array_2 = np.arange(20,51)
```

#### Desire output


```python
# [20 21 22 23 24 25 26 27 28 29 30]
```

#### Solution


```python
inters_arr = np.intersect1d(array_1,array_2)
```


```python
print(inters_arr)
```

    [20 21 22 23 24 25 26 27 28 29 30]


The intersect1d method will return a new array composed of elements both present in the first and second array. We stored it in the inters_arr variable then printed it.

### Ex 12: Remove the elements that are found in the first arrays but not the second

Q: Delete the elements that are present in the first array but not in the second array. In another world, we want to keep elements in the first array that are not present in the second array.


```python
array_1 = np.arange(0,31)
array_2 = np.arange(20,51)
```

#### Desire output


```python
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
```

#### Solution


```python
array_1 = np.setdiff1d(array_1,array_2)
```


```python
print(array_1)
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]


the setdiff1d method will return elements present in the first array but not found in the second array. We set it to array_1 and override the original array_1.

Note: If this time, we want elements found in the second array only we inverse the arguments like this


```python
uniq_in_2nd_arr = np.setdiff1d(array_2,array_1)
```


```python
print(uniq_in_2nd_arr)
```

    [31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50]


### Ex 13: Get the elements that are unique in an array

Q: Get only elements that are not duplicated in an array.


```python
my_arr = np.array([5,2,1,2,89,44,5,89,5,90,104,44,88,1])
```

#### Desired output


```python
# [  1,   2,   5,  44,  88,  89,  90, 104]
```

#### Solution


```python
uniq_arr = np.unique(my_arr)
```


```python
print(uniq_arr)
```

    [  1   2   5  44  88  89  90 104]


The unique method will return an array composed of only non-repeated elements in a given array.

### Ex 14: Get the position where two arrays have common elements

Q: Find all the position where elements in array_1 match the elements in array_2.


```python
array_1 = np.arange(100,121)
array_2 = np.arange(115,151)
```

#### Desire output


```python
# (array([15, 16, 17, 18, 19, 20]),)
```

#### Solution

#### 1st Method


```python
bool_array = np.in1d(array_1,array_2)
```


```python
repeat_idx_arr = np.where(bool_array)
```


```python
print(repeat_idx_arr)
```

    (array([15, 16, 17, 18, 19, 20]),)


For this example, we have used two different NumPy's methods

The first method is in1d, which will return a boolean array where True on the position where there is a common element in the first and second array.

The second method is the where method which takes the boolean array as an argument and then returns the index position of the True values.

#### 2nd Method


```python
bool_array = np.in1d(array_1,array_2)
```


```python
repeat_idx_arr = np.nonzero(bool_array)
```


```python
print(repeat_idx_arr)
```

    (array([15, 16, 17, 18, 19, 20]),)


Here, we achieved the same result using the nonzero method.

### Ex 15: Extract all the elements within a given range

Q: Get all the numbers between 50 and 110 in the following randomly generated array.


```python
my_arr = np.random.randint(0,150,size=200)
```


```python
print(my_arr)
```

    [ 43  74  68  78  47  21  37 112  82  76  75  63 108 115 143 137 100  21
      32 100 105   6  48 110  44 106 113  19  95  81  57  44 137 137  53   2
      67  93 139 114 130  18 118  74  72   8  52  63   4 149 125  39   2  69
     120 111  84   0 100  80   7 112  94  16  20 138  73 145 114  41 145  32
      91  10  51 106  90 128  93 135  67  63  54  86 128  76   7 146 104  83
      88 103  43 134  98  68  90 108  86  30 142  41  30  67 104  81  10  93
      68  82  34  26  40 147  26  89  18 119  15  17  93 145 123  62  32   1
       5  95  22  64   3 111  62  84  29 143  94  31  93 123 149  92 142  10
     112  80  19  77 142  50  78  52  22  16   9  93  56  66  61  69  70  11
      13 119  87 141 102  12  97  66 107  99  51  30  36  78  99  76 115 141
     134  71 112  98  11  85 112 148 119  20  36  19 120  27 137 101  31 149
       3  82]


#### Desired output


```python
# Can't give the exact array because obviously it is randomly generated. Your array will be most likely different from mine
# We will see how we can fix this in the upcoming exercises.
```

#### Solution

#### 1st Method


```python
extract_arr = my_arr[(my_arr >= 50) & (my_arr <=110)]
```


```python
print(extract_arr)
```

    [ 74  68  78  82  76  75  63 108 100 100 105 110 106  95  81  57  53  67
      93  74  72  52  63  69  84 100  80  94  73  91  51 106  90  93  67  63
      54  86  76 104  83  88 103  98  68  90 108  86  67 104  81  93  68  82
      89  93  62  95  64  62  84  94  93  92  80  77  50  78  52  93  56  66
      61  69  70  87 102  97  66 107  99  51  78  99  76  71  98  85 101  82]


We use indexing on the array and pass in the condition which will only return elements greater or equal to 50.

#### 2nd Method


```python
extract_arr = np.where((my_arr >= 50) & (my_arr <=110))
```


```python
print(my_arr[extract_arr])
```

    [ 74  68  78  82  76  75  63 108 100 100 105 110 106  95  81  57  53  67
      93  74  72  52  63  69  84 100  80  94  73  91  51 106  90  93  67  63
      54  86  76 104  83  88 103  98  68  90 108  86  67 104  81  93  68  82
      89  93  62  95  64  62  84  94  93  92  80  77  50  78  52  93  56  66
      61  69  70  87 102  97  66 107  99  51  78  99  76  71  98  85 101  82]


Where method will evaluate the condition given to it and then return all the position place where the conditions are met, we stored these positions into a variable then use indexing technique to retrieve the numbers in my_arr.

#### 3rd Method


```python
extract_arr = np.where(np.logical_and((my_arr >= 50),(my_arr <=110)))
```


```python
print(my_arr[extract_arr])
```

    [ 74  68  78  82  76  75  63 108 100 100 105 110 106  95  81  57  53  67
      93  74  72  52  63  69  84 100  80  94  73  91  51 106  90  93  67  63
      54  86  76 104  83  88 103  98  68  90 108  86  67 104  81  93  68  82
      89  93  62  95  64  62  84  94  93  92  80  77  50  78  52  93  56  66
      61  69  70  87 102  97  66 107  99  51  78  99  76  71  98  85 101  82]


Instead of explicitly use the & symbol, we can use the built-in NumPy equivalent using logical_and method then pass the two condition and use the where method as we did before.

### Ex 16: Create a function that compares elements wise two arrays.

Q: Compare the corresponding elements in two arrays and return a new arrays with the maximum number.


```python
array_1 = np.random.randint(0,100,size=10)
array_2 = np.random.randint(0,100,size=10)
print(array_1)
print(array_2)
```

    [44 47 27 22 39 98 55 74 69 79]
    [56 53 78 48 12  4 84 50 47 61]



```python
# The function to use
def comparison(x,y):
    if x > y:
        return x
    else:
        return y
```

#### Desire output


```python
# Your result will most likely be different because the two arrays are randomly generated
# [56 53 78 48 39 98 84 74 69 79]
```

#### Solution


```python
arr_comparisonr = np.vectorize(comparison)
print(arr_comparison(array_1,array_2))
```

    [56 53 78 48 39 98 84 74 69 79]


Think of the vectorize method as the map function of Python but this time for NumPy arrays. This method will take as argument the function which will be applied on all the corresponding elements of the two arrays. Then we store it in a variable which will be used to call the function by passing in the two arrays.

### Ex 17: Swap two columns in a two-dimensional array

Q: We want to swap the 4th column with the 7th column in the following array.


```python
my_arr = np.arange(27).reshape(3,-1)
```


```python
print(my_arr)
```

    [[ 0  1  2  3  4  5  6  7  8]
     [ 9 10 11 12 13 14 15 16 17]
     [18 19 20 21 22 23 24 25 26]]


#### Desire output


```python
# [[ 0  1  2  6  4  5  3  7  8]
#  [ 9 10 11 15 13 14 12 16 17]
#  [18 19 20 24 22 23 21 25 26]]
```

#### Solution


```python
my_arr[:,[6,3]] = my_arr[:,[3,6]]
```


```python
print(my_arr)
```

    [[ 0  1  2  6  4  5  3  7  8]
     [ 9 10 11 15 13 14 12 16 17]
     [18 19 20 24 22 23 21 25 26]]


We use indexing to swap two columns. A comma separates the two arguments inside the square bracket [ ], the first argument is the row, and we use a colon to get all the elements in the row then the second argument is another square bracket which selects the 7th and 4th columns respectively. 

Now comes the following expression on the right of the equal sign, we select all the elements in the row using a colon, and then we select the 4th and 7th column. You see this time we have started with the 4th and then the 7th column. 

We have understood what both sides are doing, now comes the magic moment where we set the two sides using the equal sign and keep in mind that the operator precedence of the equal sign is from left to right, so we are changing the 4th column to be the 7th at the same time setting the 7th column to be equal to the 4th. There you go! The swap just happened.

### Ex 18: Swap two rows in a two-dimensional array

Q: Now let's swap the second row with the third row.


```python
my_arr = np.arange(27).reshape(3,-1)
```


```python
print(my_arr)
```

    [[ 0  1  2  3  4  5  6  7  8]
     [ 9 10 11 12 13 14 15 16 17]
     [18 19 20 21 22 23 24 25 26]]


#### Desired output


```python
# [[ 0  1  2  3  4  5  6  7  8]
#  [18 19 20 21 22 23 24 25 26]
#  [ 9 10 11 12 13 14 15 16 17]]
```

#### Solution


```python
my_arr[[1,2],:] = my_arr[[2,1],:]
```


```python
print(my_arr)
```

    [[ 0  1  2  3  4  5  6  7  8]
     [18 19 20 21 22 23 24 25 26]
     [ 9 10 11 12 13 14 15 16 17]]


This exercise is almost identical to the previous one; this time, we are executing the same code row-wise instead of column-wise.

### Ex 19: Reverse rows in a two-dimensional array

Q: Let's reverse rows in a two-dimensional array.


```python
my_arr = np.arange(27).reshape(3,-1)
```


```python
print(my_arr)
```

    [[ 0  1  2  3  4  5  6  7  8]
     [ 9 10 11 12 13 14 15 16 17]
     [18 19 20 21 22 23 24 25 26]]


#### Desired output


```python
# [[18 19 20 21 22 23 24 25 26]
#  [ 9 10 11 12 13 14 15 16 17]
#  [ 0  1  2  3  4  5  6  7  8]]
```

#### Solution


```python
rev_row_arr = np.flip(my_arr,axis=0)
```


```python
print(rev_row_arr)
```

    [[18 19 20 21 22 23 24 25 26]
     [ 9 10 11 12 13 14 15 16 17]
     [ 0  1  2  3  4  5  6  7  8]]


We used the flip method to reverse the order in the row by setting the axis to 0, which corresponds to the row.

### Ex 20: Reverse columns in two-dimensional array

Q: Let's reverse columns in a two-dimensional array.


```python
my_arr = np.arange(27).reshape(3,-1)
```


```python
print(my_arr)
```

    [[ 0  1  2  3  4  5  6  7  8]
     [ 9 10 11 12 13 14 15 16 17]
     [18 19 20 21 22 23 24 25 26]]


#### Desired output


```python
# [[ 8  7  6  5  4  3  2  1  0]
#  [17 16 15 14 13 12 11 10  9]
#  [26 25 24 23 22 21 20 19 18]]
```

#### Solution


```python
rev_col_arr = np.flip(my_arr,axis=1)
```


```python
print(rev_col_arr)
```

    [[ 8  7  6  5  4  3  2  1  0]
     [17 16 15 14 13 12 11 10  9]
     [26 25 24 23 22 21 20 19 18]]


Same as the previous exercise, the only difference is that we have set the axis argument to 1 to refer to the columns.

### Conclusion

Hope by now you are starting to gain confidence in your NumPy skills, this is was the first part in a series of three posts entirely dedicated to exercise in NumPy. I am pretty sure that at the end of these series, you will have a strong understanding of NumPy and you start using it in your projects. Stay tuned for part 2!!

Find the jupyter notebook version of this post on my GitHub profile [here.](Find the jupyter notebook version of this post on my GitHub profile [here.]())

Thank you for doing these exercises with me. I hope you have learned one or two things. If you like this post, please subscribe to stay updated with new posts, and if you have a thought or a question, I would love to hear it by commenting below. Remember keep learning!
