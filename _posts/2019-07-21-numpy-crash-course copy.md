---
title:  "NumPy Crash Course"
image: /assets/post_images/numpy.png
excerpt_separator: <!-- more -->
tags:
- python
- tutorial
- numpy
---

One of the most used scientific computing library for python is without a doubt NumPy, Numpy, which is an abbreviation of Numerical Python, is very fast at computing arrays since it is mostly written in C programming. NumPy adds support for large, multi-dimensional arrays and matrices, along with an extensive collection of high-level mathematical functions (for linear algebra) to operate on these arrays.<!-- more --> Pandas, which is a library for data manipulation, is written based on NumPy. In this post, we will discuss NumPy and how to use it, and by the end of the post, you will see why it is one of the most famous Python libraries for data science and machine learning.

Let's jump into Anaconda and import the NumPy library.


```python
import numpy as np
```

Note: if you do not have NumPy installed on your anaconda environment, read [this](https://semasuka.github.io/blog/2019/01/06/introduction-to-jupyter-notebook.html) post where I explain how to install NumPy.

Now that we all set let's start.

The primary data structures of NumPy is Array. An array comes into two flavors, which are Vector (1D array) and Matrice (2D or more dimension array). 

Now let's create a new list and cast it into a NumPy array.


```python
my_list = [2,2,1,36,62,1]
my_vector = np.array(my_list)
```

We use a method called array from the NumPy package to cast the list into a vector array. We can check that my_vector is a NumPy array by using the type function from the standard Python libraries. 


```python
print(type(my_vector))
```

    <class 'numpy.ndarray'>


We see that my_vector belongs to the numpy class. To check precisely what data type the array is, we write it like this.


```python
print(my_vector.dtype)
```

    int64


The NumPy array is of type int64.

What if we want a two-dimensional array, we cast a list of lists like this.


```python
my_list_2d = [[1,2,3],[4,5,6]]
my_vector_2d = np.array(my_list_2d) 
```


```python
print(my_vector_2d)
```

    [[1 2 3]
     [4 5 6]]


Now this time, we have a matrice (multidimensional array) composed of two arrays.

### Numpy Methods

NumPy has several built-in methods to create or change arrays.

#### Arange

The arange method is used to create a new NumPy array if you want an array that has consecutive elements. We have to pass in as arguments the start number, the stop number, and optionally the step number. This method is very similar to the range Python built-in function.


```python
seq_array = np.arange(3,8)
```


```python
print(seq_array)
```

    [3 4 5 6 7]



```python
print(type(seq_array))
```

    <class 'numpy.ndarray'>


We have passed as the first argument 3, which is the starting point of our array, then the stopping point. Remember, the stopping number is not included so for example if the stopping number is 8, the array counts up to 7.

We can also use the step to skip every number at a specific step.


```python
seq_array_step = np.arange(1,21,2)
```


```python
print(seq_array_step)
```

    [ 1  3  5  7  9 11 13 15 17 19]


In the example above, we have started at 1 then stopped at 21 with 2 as the step, which means that we have added to the array only the second number in the sequence which corresponds to odds numbers. Let's see another example.


```python
seq_array_step_2 = np.arange(0,501,10)
```


```python
print(seq_array_step_2)
```

    [  0  10  20  30  40  50  60  70  80  90 100 110 120 130 140 150 160 170
     180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350
     360 370 380 390 400 410 420 430 440 450 460 470 480 490 500]


In this example, you can see that it started at 0 and then stop at 501 and stepped 10. It means that we have started counting from zero up to 501 and only considered each 10th number.

#### Zero

We generate an array composed only of zeros using the zeros method.


```python
zeros_arr = np.zeros(9)
```


```python
print(zeros_arr)
```

    [0. 0. 0. 0. 0. 0. 0. 0. 0.]


#### Ones

We also generate an array of only 1's.


```python
ones_arr = np.ones(15)
```


```python
print(ones_arr)
```

    [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]


#### Full

The same way we did with zeros and ones, we can generate an array of any number using the full method. This time we use two arguments, the first argument is how many times we want to generate the number, and the second argument is the actual number.


```python
fives_arr = np.full(10,5)
```


```python
print(fives_arr)
```

    [5 5 5 5 5 5 5 5 5 5]


#### Linspace

There is this useful method called linspace. Linspace is very similar to arrange in the sense that they both have a start and a stop number, the only difference is that linspace's thrid argument corresponds to the number of elements evenly spaced in the array.


```python
interv_arr = np.linspace(0,100,34)
```


```python
print(interv_arr)
```

    [  0.           3.03030303   6.06060606   9.09090909  12.12121212
      15.15151515  18.18181818  21.21212121  24.24242424  27.27272727
      30.3030303   33.33333333  36.36363636  39.39393939  42.42424242
      45.45454545  48.48484848  51.51515152  54.54545455  57.57575758
      60.60606061  63.63636364  66.66666667  69.6969697   72.72727273
      75.75757576  78.78787879  81.81818182  84.84848485  87.87878788
      90.90909091  93.93939394  96.96969697 100.        ]


In the example above, the count started from 0 and stopped at 100 (this time 100 is included), and the 34 is the count of numbers between 0 and 100, which are evenly spaced. It means that the interval between 0 and 3.03030303 is the same as the interval between 3.03030303 and 6.06060606 until we reach 34 numbers. The last number will be 100.

#### Eye

An identity matrix is a handy matrix used mainly in linear algebra. It is a multidimensional matrix with a diagonal line of 1's, while everything else is zeros in the arrays.


```python
identity_arr = np.eye(10)
```


```python
print(identity_arr)
```

    [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]


We have 10 arrays, and 1's on the diagonal.

#### Random

We generate random numbers using the random function


```python
rand_num = np.random.rand(5)
```


```python
print(rand_num)
```

    [0.3284542  0.12707841 0.43429692 0.62579159 0.94835326]


Here, we have an array of 5 randomly generated numbers. Remember the numbers generated range from 0 to 1. If you want to generate a number, for example between 0 and 100, you could multiply each element in the array with 100 like this.


```python
print(rand_num * 100)
```

    [32.84541984 12.70784146 43.42969158 62.57915874 94.83532581]


We will discuss more this when we will be talking about the operations on the NumPy array.

If we want to generate a sample from a standard normal distribution commonly used in statistic, we use randn() method


```python
snd_arr = np.random.randn(4)
```


```python
print(snd_arr)
```

    [-1.00599586 -0.23881844  0.25583286  1.49561263]


We have generated randomly four values using the standard normal distribution. We can generate a multidimensional array the same way.


```python
snd_arr_multi = np.random.randn(5,3)
```


```python
print(snd_arr_multi)
```

    [[ 0.25959845  2.21015609 -1.55850209]
     [ 0.81583828 -1.91246958  0.7453705 ]
     [ 0.55490309 -0.35774245 -0.24781665]
     [ 0.53919982 -0.03549705 -1.47300085]
     [-1.12503957 -1.38798392  0.78566072]]


We can generate integers and pass as arguments low, high, and the size of the array to generate.


```python
np.random.randint(2,10010,50)
```




    array([5176, 2232, 9839, 5948, 6497, 3196, 3114, 2990, 1831, 4286, 9205,
           9128, 8280, 2003, 3586, 3030, 4834, 1252, 8876, 9600, 5123, 9166,
           6768, 4541, 8728, 6684, 3928, 2300, 6073, 5876, 2802, 8426, 4545,
           2667, 9668, 8287, 1281, 5363, 4003, 1961, 3602, 8699, 7767, 7175,
           3378, 8568,  915,  396, 7047, 6718])



Here we have an array of 50 numbers randomly selected starting from 2 up to 10010.

#### Reshape

We use the reshape method to transform an array from 1 dimension to other dimensions or vice versa.


```python
oneD_arr = np.arange(0,8,2)
```


```python
print(oneD_arr)
```

    [0 2 4 6]


We have oneD_arr a one-dimensional array if we want to change it to a two-dimensional array, we use the reshape method to change it.


```python
twoD_arr = oneD_arr.reshape(2,2)
```


```python
print(twoD_arr)
```

    [[0 2]
     [4 6]]


We have transformed the one-dimension array into a two-dimension array and stored it in a new variable.

Careful here, we can not transform an array into a multidimensional array that does not fit. for example


```python
twoD_arr = oneD_arr.reshape(2,3)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-109-c90c7d9b6f76> in <module>
    ----> 1 twoD_arr = oneD_arr.reshape(2,3)
    

    ValueError: cannot reshape array of size 4 into shape (2,3)


Here we are trying to fit oneD_arr into a two-dimension array with three elements each. The interpreter is throwing an error because we are trying to fit oneD_arr into two arrays with three elements each. We are missing two more elements to create two arrays with three elements.

Let's see another example.


```python
fiveD_array = np.arange(0,1040,15)
```


```python
print(fiveD_array)
```

    [   0   15   30   45   60   75   90  105  120  135  150  165  180  195
      210  225  240  255  270  285  300  315  330  345  360  375  390  405
      420  435  450  465  480  495  510  525  540  555  570  585  600  615
      630  645  660  675  690  705  720  735  750  765  780  795  810  825
      840  855  870  885  900  915  930  945  960  975  990 1005 1020 1035]


Now we can reshape the array into 10 arrays with 7 elements.


```python
print(fiveD_array.reshape(10,7))
```

    [[   0   15   30   45   60   75   90]
     [ 105  120  135  150  165  180  195]
     [ 210  225  240  255  270  285  300]
     [ 315  330  345  360  375  390  405]
     [ 420  435  450  465  480  495  510]
     [ 525  540  555  570  585  600  615]
     [ 630  645  660  675  690  705  720]
     [ 735  750  765  780  795  810  825]
     [ 840  855  870  885  900  915  930]
     [ 945  960  975  990 1005 1020 1035]]


#### Shape

Now if we want to know which dimension is an array, we use the shape method on any array.


```python
multi_arr = np.random.rand(4,5)
```


```python
print(multi_arr)
```

    [[0.29672388 0.30382348 0.63858171 0.04452958 0.42704145]
     [0.36501907 0.96622299 0.84038868 0.61598943 0.77640899]
     [0.59761916 0.93173121 0.23362733 0.1241686  0.37988366]
     [0.6542785  0.28888605 0.91213121 0.78753272 0.52377276]]


We have created a multidimensional array randomly with the random method. Now let's verify the shape of the array by calling the shape method on the array.


```python
multi_arr.shape
```




    (4, 5)



#### Max

The max method returns the highest number in an array.


```python
some_arr = np.random.randint(0,10000,size=500)
```


```python
print(some_arr)
```

    [4174 3014 7538 5104 2044 6475 9522 6095 5399 4183  587 2049 3583  891
     4249 5871  222 9899 2115 2838 1454 8477 9462 8688 8820 8563 7787 6855
     4222 1583 7420 3261 2926 6972 6518 3374 8546 5285 2138 7135 9522 4794
      737  356 4037 3956 4313 8895 8411 7151 5577 3750 6538 9281 4723  218
     1396 8270 8603 9538 1458 6722 2007 6003 2545 6172  667 5687 7733  823
     6420  392 8123 4466 4725 1507 7450 3067 9035 4819 4123 2904 5058 8977
     6840 4094 9976 7208 4847  782 5157 5869 4338 2723 4299 9076 2350 6215
     2536 9875 2660 9415 3954 7371 9918 6908 6843  635 9867 5156 1747  389
      723 6645 7143 2258 7851 8436 9788 9851 1350 9688 5100 7621 8857 2605
     9706 1578 4838 9254  769 1503 5862 5590  970 9115 3794 4021 4555  414
     9722  389 8498 5542 1532 2975 7255 3622 2213 3407 3386 7277 5692 8348
     8328  257 5763 6974 6751 5141 2099 4528 4801 6631 9874 8595 8658 4862
     5049 1017 5050 2811 2730 9770 7428 6948 6993 4263  721 2036 1737 1403
     9454 2696 4577  876 3283 2694 5552 4392 9284 3520 6721 8348 7157 7443
     9116 5750 5613  901 2575  114 7047   18 1727 4713 2163 7447 4750 1519
     2361  336 6824 2563 6458 5233 6083  370 4454 5712 6403 2085 6377 4992
     9416 4225 1938 1758 8584  244 9112 2690 3281 5939 1098  597 6082 9659
     5359 7650 7784 4010 3858 8431 8440 6055 2678 7800 9156 9348 3446 8292
     4882 3321 8045  309 9927 8834 9324 3160 9903 8479 4314 4391 5184 4647
     1941 4310 1010 2877 9748 8743 1869 3710 4636 2887 9936 7555 9350 7095
      723 7353 3256 8871 8517 8018 7252 6276 3852 5653 1915 8004 7957 7605
     7308 3787 5259 9637 9199 3232 2386 1772 4443 2683 8617 1771 1437 8576
     4243 8950 4246  999 8168 4695 1420 1724 3329 1432 3286  508 3784 7066
     4361 9328 6695 6551 9749  476 7128 3410 2479 5586 5321 4842 7136 4302
     7881 5866 4193 8398 6298 5489 1526 9513  481 3216 3679 9534 5316 1390
     4763 5921 5360 6299 8271 6901  305 7171 9487 8145 9625 5280 4250 7478
     4532 2728 7669 7950 5678 2931 4515 8418  116 8335 9809 9084 9600 9225
     3945 2663 3169 8796 1167 8822 5570 2234 4792 5481 8909 4346 3107  770
     2568 1119 3762 3934 3489 2142 8797  154 8300 8618  343 5700 4279 3471
     7830 4936 6680 8843 6992 6924  133  748  482 6494 5879 7219 1462 4675
     2861 3406 5697 4873  792 2453 8257  317 8648 6538 3017 4965 9983 6006
     2617 9017 2461 3921 1118 4978 9979 4445  901 1101 6919 4891 2900 3636
     7353  834 5165 7775 1465 9477 8939 1061 7481 5913 1321 3723 1410 8456
     6969 2236 3401 4621 6194 4886 5343 2171 5075  131  526 4976 2721 7042
     3892 4661 6540 3706 1312 8319 9070 9792 4388 3797 7215 7964 5419 4331
     8503 4485   38 1829 3955 1953 1916 4390 7835 1512]



```python
print(some_arr.max())
```

    9983


The highest number in this array is 9983.

We can also rewrite it like this.


```python
print(np.max(some_arr))
```

    9983


#### Min

We can do the same to get the lowest number.


```python
print(some_arr.min())
```

    0


In this array, the lowest number is 0.

We can also rewrite it like this.


```python
print(np.min(some_arr))
```

    0


#### Argmax

We can also get the index at which the highest number is found using the argmax method.


```python
print(some_arr.argmax())
```

    257


The highest number in this array (which is 9970) is located at index 257. 

#### Argmin

The same applies to the minimum


```python
print(some_arr.argmin())
```

    286


### Numpy Array Indexing

Now that we took a look at some of the most useful NumPy methods, it is time to talk about array indexing.

So what is indexing in the first place? Well, indexing is the act of extracting a small portion of the array. It is very similar to the Python list's indexing.


```python
seq_array_1 = np.arange(0,90,2)
```


```python
print(seq_array_1)
```

    [ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46
     48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88]


If we want only to create a new array that contains numbers from 30 to 44 from the seq_array_1 array, we use the indexing at position 15 (which correspond to number 30) up to 23 (which correspond to number 46 so that 44 is included).


```python
new_array = seq_array_1[15:23]
```


```python
print(new_array)
```

    [30 32 34 36 38 40 42 44]


We used the [ ] and passed as the first argument the starting point and the last argument the stopping point(remember that the stopping point is not included).

If we want to start from the first number, the start will be 0.


```python
print(seq_array_1[0:16])
```

    [ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30]


We can achieve the same result by omitting the zero to save a lit bit of typing, like this.


```python
print(seq_array_1[:16])
```

    [ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30]


We can also do the same if we want to go until the end of the array.


```python
print(seq_array_1[16:])
```

    [32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78
     80 82 84 86 88]


We grabbed every element starting from index 16 up to the last one.

We can also use indexing to change the array like this


```python
seq_array_1[16:20] = 100
```


```python
seq_array_1
```




    array([  0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,
            26,  28,  30, 100, 100, 100, 100,  40,  42,  44,  46,  48,  50,
            52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,
            78,  80,  82,  84,  86,  88])



We select the elements from index 16 up to 20 and set those to the value 100.

We can set a whole array to a specific value, like this.


```python
sub_array = seq_array_1[:5]
```


```python
sub_array[:] = 1
```


```python
print(sub_array)
```

    [1 1 1 1 1]


We have extracted a sub-array, then selected the whole sub-array and changed it to 1. However, here is the catch; what about the original array?


```python
print(seq_array_1)
```

    [  1   1   1   1   1  10  12  14  16  18  20  22  24  26  28  30 100 100
     100 100  40  42  44  46  48  50  52  54  56  58  60  62  64  66  68  70
      72  74  76  78  80  82  84  86  88]


Surprise? Huh? Yeah, it is somewhat weird that the changes took place also in the seq_array_1. It means that seq_array_1 and sub_array refer to the same array in memory.

So logically, how can we fix this? Well, we create a copy of the original array then apply the changes only on the copied sub-array without affecting the original array.


```python
sub_array_cop = seq_array_1.copy()
```


```python
sub_array_cop[:] = 55
```


```python
print(sub_array_cop)
```

    [55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55
     55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55 55]



```python
print(seq_array_1)
```

    [  1   1   1   1   1  10  12  14  16  18  20  22  24  26  28  30 100 100
     100 100  40  42  44  46  48  50  52  54  56  58  60  62  64  66  68  70
      72  74  76  78  80  82  84  86  88]


This time you can see that the original array is unchanged because we created a new copy of it that does not affect the original array.

In a multidimensional array, we can select a specific array using indexing like this.


```python
nD_array = np.random.randint(0,high=501,size=300).reshape(30,10)
```


```python
print(nD_array)
```

    [[275 416 230 367 396 386  94  25 307 421]
     [312  34 243  55 284 461 113 384 285 174]
     [274 132 319 103 477 145 174 467  91 413]
     [ 54 469  73 143 488 310 420 424 202 300]
     [255  97 106 316 115  69 305 360 469 377]
     [338 450 479 193 249  64 452 306 468 338]
     [306 170 224 380 141 104 149 294 333 323]
     [382 252  49 335 283 256  71 227 282 319]
     [424 363 413 487   8 460 155 199 387 476]
     [ 36 470 335  68 409 473  16 361 330 206]
     [324  54 405 182 463 135 196 175  91 288]
     [446  71 358 482 480 151  27  92 105 398]
     [244  92  65  11 403 131 238 323 194 240]
     [332 319  78 363 317 344 172 400 274 293]
     [234 182  43 422 441 196 348 248 172 370]
     [283  74 413 336 183 421 170  34  15 442]
     [210 395 335 390  41 444 285 354 290  18]
     [214 305 292 469 484   8 359 434 464  62]
     [ 66 235 120 375  44 112 145 412 281 267]
     [498 446 212  11  74 190 299  65 487 365]
     [366 395 107 288 330 498 192 443 352 354]
     [401 150 390  34 220 186 319 483 344 241]
     [  2 325 476 293 485 417  85 432 240 235]
     [352 483  47  86 308 417 491 144 488 312]
     [325 359 499  41 203  70 485  33 273 295]
     [317 450 216  72 192 447  98 393 115 363]
     [255  75 178 269 150 233 458 202 434 166]
     [212 472 289 219  66 422  62  70 145 496]
     [102  71  30 228 220 351  92 100 259 237]
     [241 394 355 495 419  68 358  50 385  13]]


We created a random multidimensional array of 300 numbers with 0 as the lowest number, 501 as the highest number then reshaped it into 30 arrays with 10 numbers each.

If we want to only print, for example, the 10th array, we can pass 9 as the argument which is the index of the 10th array in the multidimensional array.


```python
print(nD_array[9])
```

    [ 36 470 335  68 409 473  16 361 330 206]


How about we get the 4th number in the 24th array? Well, there are two ways of doing this.

The first method is by using a double square bracket [ ], where the number in the first bracket represents the position of the array and the second is the position of the element we are looking for in the array.


```python
print(nD_array[23][3])
```

    86


The second method uses only one square bracket [ ], where we pass two numbers separated by a comma. The first number is the array, and the second is the position of the number we are looking for in the array.


```python
print(nD_array[23,3])
```

    86


We can use the method that makes more sense to you; I generally use the first method.

We can also choose a portion of arrays in a multidimensional array.


```python
print(nD_array[23:][3:])
```

    [[255  75 178 269 150 233 458 202 434 166]
     [212 472 289 219  66 422  62  70 145 496]
     [102  71  30 228 220 351  92 100 259 237]
     [241 394 355 495 419  68 358  50 385  13]]


Here we have printed all the arrays from position 23 up to the end at the same time selected only elements from index 3 up to the end in each array.

Here is another example.


```python
print(nD_array[15:23,4:9])
```

    [[183 421 170  34  15]
     [ 41 444 285 354 290]
     [484   8 359 434 464]
     [ 44 112 145 412 281]
     [ 74 190 299  65 487]
     [330 498 192 443 352]
     [220 186 319 483 344]
     [485 417  85 432 240]]


In the example above, we have only selected starting from array 15 up to array 22, then starting from elements 4 up to 9 within each array.

Last example


```python
print(nD_array[:5,8:])
```

    [[307 421]
     [285 174]
     [ 91 413]
     [202 300]
     [469 377]]


Here, we selected all the array from the beginning up to array number 4 then all the elements from index 8 up to the end.

Now let's see an interesting way to use NumPy array with boolean.


```python
some_arr = np.arange(0,30,2)
```


```python
print(some_arr)
```

    [ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28]



```python
bool_arr = some_arr < 15
```


```python
print(bool_arr)
```

    [ True  True  True  True  True  True  True  True False False False False
     False False False]


We use >, < or = directly on the array which will return another array composed of boolean values True or False. Internally, each number in the array is evaluated and compared to 15 if it is inferior to 15, then True is placed in the new array else False is used.

We can use the boolean array to get all elements in the original array that are inferior to 15.


```python
print(some_arr[bool_arr])
```

    [ 0  2  4  6  8 10 12 14]


We could also have written it like this.


```python
print(some_arr[some_arr < 15])
```

    [ 0  2  4  6  8 10 12 14]


### Operations on Numpy's Array

We can apply different operations on a NumPy array as we have already seen with the multiplication operation. In this section, we will discuss how we can apply other operations on the array.

#### Addition


```python
some_array = np.arange(0,20)
```


```python
print(some_array)
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]



```python
print(some_array + 5)
```

    [ 5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]


For each element in the array, we have added 5.


```python
print(some_array + some_array)
```

    [ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38]


This time, we have added each element in the array with itself.

#### Subtraction


```python
print(some_array - 2)
```

    [-2 -1  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17]


We have subtracted each element by 2.


```python
print(some_array - some_array)
```

    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]


We have removed each element in the array by itself; we got 0 for each element.

#### Multiplication


```python
print(some_array * 5)
```

    [ 0  5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95]



```python
print(some_array * some_array)
```

    [  0   1   4   9  16  25  36  49  64  81 100 121 144 169 196 225 256 289
     324 361]


We have to multiply each element in the array by 5 and then by itself.

#### Division


```python
print(some_array / 2)
```

    [0.  0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.  5.5 6.  6.5 7.  7.5 8.  8.5
     9.  9.5]



```python
print(some_array / some_array)
```

    [nan  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
      1.  1.]


    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: RuntimeWarning: invalid value encountered in true_divide
      """Entry point for launching an IPython kernel.


We have divided each element with 2 at first and then with itself. The first element is 0; we tried to divide 0 by 0 in the second example, which is impossible, that is why we have a warning label and got back nan as the first element in the array.


```python
print(1 / some_array)
```

    [       inf 1.         0.5        0.33333333 0.25       0.2
     0.16666667 0.14285714 0.125      0.11111111 0.1        0.09090909
     0.08333333 0.07692308 0.07142857 0.06666667 0.0625     0.05882353
     0.05555556 0.05263158]


    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: RuntimeWarning: divide by zero encountered in true_divide
      """Entry point for launching an IPython kernel.


When we try to divide 1 by 0, we get back infinity.

#### Power


```python
print(some_array ** 2)
```

    [  0   1   4   9  16  25  36  49  64  81 100 121 144 169 196 225 256 289
     324 361]


We applied the power of 2 on each element in the array.

#### Square Root


```python
print(np.sqrt(some_array))
```

    [0.         1.         1.41421356 1.73205081 2.         2.23606798
     2.44948974 2.64575131 2.82842712 3.         3.16227766 3.31662479
     3.46410162 3.60555128 3.74165739 3.87298335 4.         4.12310563
     4.24264069 4.35889894]


NumPy has a built-in square root function. We got back the square root of each element in the array.

#### Exponential


```python
print(np.exp(some_array))
```

    [1.00000000e+00 2.71828183e+00 7.38905610e+00 2.00855369e+01
     5.45981500e+01 1.48413159e+02 4.03428793e+02 1.09663316e+03
     2.98095799e+03 8.10308393e+03 2.20264658e+04 5.98741417e+04
     1.62754791e+05 4.42413392e+05 1.20260428e+06 3.26901737e+06
     8.88611052e+06 2.41549528e+07 6.56599691e+07 1.78482301e+08]


We got back the exponential of each element in the array.

#### Cosine & Sine


```python
print(np.cos(some_array))
```

    [ 1.          0.54030231 -0.41614684 -0.9899925  -0.65364362  0.28366219
      0.96017029  0.75390225 -0.14550003 -0.91113026 -0.83907153  0.0044257
      0.84385396  0.90744678  0.13673722 -0.75968791 -0.95765948 -0.27516334
      0.66031671  0.98870462]



```python
print(np.sin(some_array))
```

    [ 0.          0.84147098  0.90929743  0.14112001 -0.7568025  -0.95892427
     -0.2794155   0.6569866   0.98935825  0.41211849 -0.54402111 -0.99999021
     -0.53657292  0.42016704  0.99060736  0.65028784 -0.28790332 -0.96139749
     -0.75098725  0.14987721]


We calculated the cosine and the sine of each element in the array.

#### Logarithm


```python
print(np.log(some_array))
```

    [      -inf 0.         0.69314718 1.09861229 1.38629436 1.60943791
     1.79175947 1.94591015 2.07944154 2.19722458 2.30258509 2.39789527
     2.48490665 2.56494936 2.63905733 2.7080502  2.77258872 2.83321334
     2.89037176 2.94443898]


    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: RuntimeWarning: divide by zero encountered in log
      """Entry point for launching an IPython kernel.


We can also get the logarithm of each element, except the first element, which is 0. We can not compute Log 0, that is why infinity was returned.

### Conclusion

NumPy is one of those libraries in Machine Learning you can not ignore. It is pretty easy to get started with especially if you understand lists in Python, and we have seen some of its most used methods. However, this is just the tips of the iceberg because there is so much to learn NumPy that I can not cover it all. That is why I invite you to visit its documentation, which is pretty well written [here](https://numpy.org/devdocs/). You will find all the references, methods, and tricks of NumPy.

This post was a quick introduction to NumPy. In an upcoming post, we will work on some exciting exercises using NumPy to sharpen your skills, so stay tuned by subscribing to our mailing list.

Thank you for reading this tutorial. I hope you have learned one or two things. If you like this post, please subscribe to stay updated with new posts, and if you have a thought or a question, I would love to hear it by commenting below. Cheers, and keep learning!
