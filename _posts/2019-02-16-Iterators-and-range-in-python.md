---
title:  "Iterators and range in Python"
image: /assets/post_images/py_tuto.jpeg
excerpt_separator: <!-- more -->
tags:
- python
- tutorial
- programming
---


In this post, we will mainly discuss iterator and range function. We have been using iterator and range function; however, this time, we'll go in-depth. Let's first talk about iterator and iterable object in Python.<!-- more -->

### Iterator and Iterable object

An iterator is an object that represents a stream of data in order words an iterator is an object that you can traverse one element at a time. Any object that supports iteration is called iterable, and so far, we have seen some iterable objects like string and list.


```python
some_string = "0123456789"
some_list = ["burundi","kenya","rwanda","uganda"]
```


```python
for char in some_string:
    print(char)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9



```python
for country in some_list:
    print(country)
```

    burundi
    kenya
    rwanda
    uganda


As we have already seen this, in these examples above we are going through all the elements in the string and in the list one-by-one. What is actually happening is that there is an iterator that is created behind the scene by the for loop and uses it to return each element in the string or list (iterable) and when there is no more element left. An error will be raised and handled by the for loop, and the loop will terminate.

To see what is going on here, let's create our own iterable using the iter( ) function.


```python
interable_string = iter(some_string)
print(interable_string)
```

    <str_iterator object at 0x1083c1f60>


The result shows that we have created an iterable object and the numbers show the address where that iterable is located in memory.

Now how can we print the elements found in the iterable? We use the next( ) function to print each element one at the time in sequential order.


```python
print(next(interable_string))
```

    0


We have printed the first element in the string, to print the following element, we have to re-write it again.


```python
print(next(interable_string))
```

    1


Basically, the iter() goes to each element one at a time.


```python
print(next(interable_string))
print(next(interable_string))
print(next(interable_string))
print(next(interable_string))
print(next(interable_string))
print(next(interable_string))
print(next(interable_string))
print(next(interable_string))
```

    2
    3
    4
    5
    6
    7
    8
    9


This means that each time we write the next function, it remembers which element was returned by the previous next function.


```python
print(next(interable_string))
```


    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    <ipython-input-35-0070de267b2a> in <module>
    ----> 1 print(next(interable_string))
    

    StopIteration: 


After exhausting all the elements in the string will cause the following use of the next function to result in an error, StopIteration error will be raised which means that we have reached the end of the string, there is no more element in the iterable object. 

When we were using the for loop to iterate, StopIteration error was raised behind the scene, and the for loop took care of it for us and exited the loop.

### Challenge

For this challenge, we will create a sequence of numerical characters using either string or a list using iter( ) function with the next( ) function to print all the elements in the list/string.

We should use for loop to loop "n" times through the string/list. "n" corresponding to the number of elements found in the list or string.

hit: use the len( ) function to get "n".

Now go ahead and try this challenge on your own and only after you have attempted to, come back and compare your solution with mine.

### Challenge Solution

Remember for each solution, there is a multitude of implementation, so your code might look very different than mine but as long as the results are correct, then you good.


```python
ML_algorithms = ["CNN","GANs","linear regression","RNN","Random forest"]
ML_algorithms_iter = iter(ML_algorithms)
```


```python
for _ in range(len(ML_algorithms)):
    print(next(ML_algorithms_iter))
```

    CNN
    GANs
    linear regression
    RNN
    Random forest


we have a list of different machine learning algorithms strings stored in a list, we create its iterator object using iter( ) function and store it in the ML_algorithms_iter variable.

Now comes the for loop that will loop as long as we have not exceeded the number of elements found in the list using the len( ) and the range( ) function. We will print each element of the list one by one using the next( ) function. No StopIteration error will be raised(encountered) because we have not gone beyond the number of elements found in the list thanks to the range function.

Another thing here to mention is the use if _ as count variable of the for the loop. As a convention in Python, we use _ instead of "i" or "j" when we ain't using this variable inside the loop. The _ variable acts like a placeholder.

Note: the point here is to show you how iterator works and what is really going on behind the scene when we are using. This is by no mean "the" way of writing a for loop, you should write it without the iter( ) and next( ) as we did before since everything is being taken care for us. We will see how iterator are useful when we will be creating our own iterator classes and using generators in Python.

### Range

We have been using range in the for loop, let's take a deep dive into it.

As we have already seen, a range function has a start, stop and step value as parameters in this format: range(start, stop, step). If only one value is passed as a parameter, it will be considered as the stop value. Remember the range function is an exclusive function; this means that if we write range(21), the count will stop at value 20.

We can use a list constructor to generate a range of elements like this.


```python
list_num = list(range(21))
```


```python
print(list_num)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


Let's see another example.


```python
even = list(range(0,20,2))
odd = list(range(1,20,2))
```


```python
print("the even numbers are: {}\nthe odd numbers are: {}".format(even,odd))
```

    the even numbers are: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    the odd numbers are: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


Here we are creating two lists, one that contains all the even numbers from 0 to 20 with 2 as step and another one that contains odd numbers starting from 1, stepping by 2 up to 20.

#### index and slicing with range function

To get an index of a particular character and a character at a specific index in a string/list, we use slicing an index function.


```python
alpha = "abcdefghijklmnopqrstuvwxyz"
```


```python
print(alpha.index("g"))
```

    6


Using the index function on the string or list, we can get the index at which a particular character is located in the string. If the character is not found, an error will be raised.


```python
print(alpha.index("*"))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-5-a2e48dc282f8> in <module>
    ----> 1 print(alpha.index("*"))
    

    ValueError: substring not found


When a substring of more than one character is used as a parameter, only the first character index is returned.


```python
print(alpha.index("ghijklmnopq"))
```

    6


So now, what if we have an index, but we want to get the character found at that index. We will use slicing.


```python
print(alpha[6])
```

    g



```python
print(alpha[0])
```

    a


We can use the range function with slicing and index function.

Let's illustrate this by first creating a range without casting it to a list.


```python
my_nums = range(1,1000)
```


```python
print(my_nums)
```

    range(1, 1000)


Not what you expected? Some might expect to return all elements for 1 to 1000 but remember, we have not cast the range to a list. This is just a range class.


```python
print(type(my_nums))
```

    <class 'range'>


Now let's use the index and the slicing on range.

we can access to an index of a particular element.


```python
print(my_nums.index(3))
```

    2



```python
print(my_nums.index(20))
```

    19


In the examples above, we check at what position(index) the value 3 and 20 are found in the range.

we can access an element at a particular index.


```python
print(my_nums[0])
```

    1



```python
print(my_nums[7])
```

    8


In these examples above, we are returning the elements at index 0 and 7.

This is a useful case of using the range function without casting it to a list.
Casting it to a list would have been memory inefficient. Let's see another example.


```python
mupli_of_seven = range(7,1000,7)
```


```python
num_inputted = 0
while num_inputted not in range(7,1001):
    num_inputted = int(input("Please enter a number between 7 and 1000: "))
```

    Please enter a number between 7 and 1000: 0
    Please enter a number between 7 and 1000: 44



```python
if num_inputted in mupli_of_seven:
    print("{} is a multiple of 7".format(num_inputted))
else:
    print("{} is not a multiple of 7".format(num_inputted))
```

    44 is not a multiple of 7


We have a range from 7 to 1000 with 7 as the step, this means that we have a range of only multiples of 7 since we are starting from 7

After that, we ask the user to input a number between 7 and 1000. The while loop has been added as a test to check if the inputted number is within the range. The inputted number will be cast to int and stored in the variable num_inputted.

Now comes the conditional statement, these will check if num_inputted is found in the mupli_of_seven range, if it is or not, the appropriate message will be printed.

We can also use range at the same time with slicing.


```python
my_nums = range(1,100)
```


```python
print(my_nums[::2])
my_nums_step = my_nums[::2]
```

    range(1, 100, 2)



```python
print(list(my_nums[::2]))
```

    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]


We can compare the two ranges.


```python
if my_nums_step == range(1,100,2):
    print("The two range are equal")
```

    The two range are equal.


Now let's compare the two ranges.


```python

if range(0,5,2) == range(0,6,2):
    print("The two range are equal")
```

    The two range are equal.


The two ranges are equal because the two range will return 0,2,4. In the second range, 6 will not be included because as we have already seen the range function is an exclusive function.

Let's cast and print them to see what's going on.


```python
print(list(range(0,5,2)))
```

    [0, 2, 4]



```python
print(list(range(0,6,2)))
```

    [0, 2, 4]


We can use range and slicing in reverse order too.


```python
for i in range(20,-1,-2):
    print(i)
```

    20
    18
    16
    14
    12
    10
    8
    6
    4
    2
    0


To print in reverse order using the range function, the start will be higher than the end, and the step will be a negative number.


```python
the_range = range(0,21)
```


```python
for i in the_range[::-2]:
    print(i)
```

    20
    18
    16
    14
    12
    10
    8
    6
    4
    2
    0


In the first example, we have reversed the order starting from 20 and ending at 0 (that is why we use -1 as stop) and used -2 as a step which countdown by 2.

In the second example, we are doing the same thing just that this time we are using slicing at the same time with the range function.

Let's compare the two notation.


```python
if range(20,-1,-2) == range(0,21)[::-2]:
    print("The two range are equal")
```

    The two range are equal.


we can clearly see that the two ranges are equal, but pay attention to how we wrote the following expression, we started by writing the range function using only the start and the end parameters, and then we add a step using slicing. 

Now the question that comes next would be when do we need to use slicing in reversed order?

Well, let's illustrate this with an example.


```python
reversed_string = "idnuruB fo latipac eht si arubmujuB"
```


```python
print(reversed_string[::-1])
```

    Bujumbura is the capital of Burundi


Using slicing, we have managed to write a reversed ordered string into its correct order by using -1 as a step.


```python
the_nums = range(0,11)
for i in the_nums[::-1]:
    print(i)
```

    10
    9
    8
    7
    6
    5
    4
    3
    2
    1
    0


We can do the same with numbers too.

### Challenge

This challenge has two main parts
- The first part is to print all the multiple to 11 in a reversed order form 100 to 0 using range and slicing at the same time
- The second part should ask the user to input a number and check if that number is a multiple of 11

Now go ahead and try this challenge on your own and only after you have attempted to, come back and compare your solution with mine.

### Challenge Solution


```python
#First part
multi_eleven_count = list(range(1,100)[::-11])
print(multi_eleven_count)
```

    [99, 88, 77, 66, 55, 44, 33, 22, 11]



```python
#Second part
user_in = int(input("Please enter a number between 11 and 100: "))
if user_in in multi_eleven_count:
    print("The number inputted is a multiple of 11")
else:
    print("The number inputted is not a multiple of 11")
```

    Please enter a number between 11 and 100: 22
    The number inputted is a multiple of 11


multi_eleven_count is calculated using the range function that will get back all the number between 1 and 99 and use slicing to step by 11 on all the elements from 1 to 99 which will return all the multiple of 11. After getting the range, we will cast it to a list.

Now we will compare and see if the user_in is found in the multi_eleven_count list and print the appropriate message accordingly.

### Conclusion

Iterable objects are very crucial parts of Python because they are the ones that make a for and while loop through a sequential data type possible. Understanding how they work behind the scene is also essential because this will help us to understand how to create our own iterable class in the upcoming post. 

The range function generates a range of number between a given interval, we can use this range without casting it to a list to save up memory.

Thank you for reading this tutorial. I hope you have learned one or two things. If you like this post, please subscribe to stay updated with new posts, and if you have a thought or a question, I would love to hear it by commenting below. Cheers, and keep learning!
