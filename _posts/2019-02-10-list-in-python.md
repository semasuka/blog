---
title:  "List in Python"
image: /assets/post_images/py_tuto.jpeg
excerpt_separator: <!-- more -->
tags:
- python
- tutorial
- programming
---

So far, we have seen one sequence data type, which is the string data type. Python has 6 additional built-in sequence data type. In this and the upcoming posts, we will be discussing 3 of them which are the lists, tuple and range and see their functions. In this post, we will focus on the list only<!-- more -->

let's see how we can use a built-in function to a sequence data type to make our code more readable.

we will use the example of the valid IP address checker from the previous post, this time instead of having a counter of the dot character we can use a built-in operator that does the same.


```python
ip_address = input("Please enter an IP address: ")
```

    Please enter an IP address: 127.0.0.1



```python
dot_count = ip_address.count(".")
print("We have {} dots in this ip address".format(dot_count))
```

    We have 3 dots in this ip address


Count function returns the number of dots found in the ip_address variable. This function has done all the work of traversing the string, count one by one all the dot characters found and stored the count in the dot_count variable. This is a concise and readable way to write code.

### List

We have been introduced to list in the previous posts. Let's explain it in details; a list is a collection of objects like variables, functions and classes placed in a container. A list has many functions that act upon the elements of a list.

Remember, in Python, everything you write is an object and think of a list as a container of any object. Since a list can contain a string which is by itself a sequence data type this means that a list can also contain other lists, in other words, we can have a container of containers.

let's illustrate this


```python
burundi_provinces = ["Karuzi","Rutana","Bururi","Makamba","Ruyigi"]
for province in burundi_provinces:
    print("Now we are in {}".format(province))
```

    Now we are in Karuzi
    Now we are in Rutana
    Now we are in Bururi
    Now we are in Makamba
    Now we are in Ruyigi


We have a sequence of provinces found in Burundi and are stored in a list. The for loop traverses the list and print each element in the list one by one.

Now let's see when we have a list of lists.


```python
world_countries = [["Angola","DRC","Burundi","Rwanda"],["Spain","Italy","France","UK"],["China","India","Korea"]]
```

We have a list that contains different countries around the world. These countries are grouped in another list depending on which continent they belong to


```python
for continent in world_countries:
    print(continent)
```

    ['Angola', 'DRC', 'Burundi', 'Rwanda']
    ['Spain', 'Italy', 'France', 'UK']
    ['China', 'India', 'Korea']


looping through the list world_countries, all the lists found will be printed.

We can print the countries one by one using a nested for loop and separate the countries that belong to the same continent with space.


```python
for continent in world_countries:
    for country in continent:
        print(country)
    print()
```

    Angola
    DRC
    Burundi
    Rwanda
    
    Spain
    Italy
    France
    UK
    
    China
    India
    Korea
    


we can concatenate two lists together using the addition operator.


```python
even = [2,4,6,8]
odd = [1,3,5,7]
numbers = even + odd
```


```python
print(numbers)
```

    [2, 4, 6, 8, 1, 3, 5, 7]


### List Functions

The same way we have the format function for the string data type, we have different functions
for the list data type as well.

#### Sort function

The sort function will sort a list in numerical or alphabetical order. 


```python
numbers.sort()
print(numbers)
```

    [1, 2, 3, 4, 5, 6, 7, 8]



```python
print(numbers.sort())
```

    None


sort( ) function does not create a new list; it acts on the current list permanently. Therefore, it will return None. These kinds of functions are called in-place functions.

There is another function that sorts a list, but instead of returning None, it will create a new sorted list and keep the original list untouched.


```python
numbers_2 = [99,22,1,3,77,3,5,23,51,35,133]
ordered_list = sorted(numbers_2)
```


```python
print(ordered_list)
print(numbers_2)
```

    [1, 3, 3, 5, 22, 23, 35, 51, 77, 99, 133]
    [99, 22, 1, 3, 77, 3, 5, 23, 51, 35, 133]


sorted( ) function will sort the list, but the difference between the sort and the sorted function is that sorted function will create a new list that can be stored in a variable or directly printed. The original list(numbers_2 in this example) will remain unchanged.

We can compare the two lists using the two type of sorting and verify if they are equal.


```python
numbers_2.sort()
if ordered_list == numbers_2:
    print("They are equal")
else:
    print("They are not equal")
```

    They are equal


#### Append function

we add an element at the end of a list using the append function.


```python
burundi_provinces = ["Karuzi","Rutana","Bururi","Makamba","Ruyigi"]
burundi_provinces.append("Cibitoke")
for province in burundi_provinces:
    print("Now we are in {}".format(province))
```

    Now we are in Karuzi
    Now we are in Rutana
    Now we are in Bururi
    Now we are in Makamba
    Now we are in Ruyigi
    Now we are in Cibitoke


"Cibitoke" was added at the end of the list, which means that when we loop through the list "Cibitoke" shall be printed at the end.

#### Insert function

We have seen that the append function add an element at the end of a list, but what if we don't want the element to be added not at the end but instead somewhere else in the list. That is when the Insert function comes in handy. Insert is like the append function, but the only difference is that we can insert an element at a specific index of our choice in the list.


```python
burundi_provinces.insert(0,"Makamba")
for province in burundi_provinces:
    print("Now we are in {}".format(province))
```

    Now we are in Makamba
    Now we are in Karuzi
    Now we are in Rutana
    Now we are in Bururi
    Now we are in Makamba
    Now we are in Ruyigi
    Now we are in Cibitoke


"Makamba" has been added to the list at the beginning of the list because of the first parameter (input in function) in the insert function was 0, which corresponds to the first index in a list.

Now let's say we want to add a new element to the list, this time the element will be placed between "Rutana" which is at index 2 and "Bururi" which is at index 3.


```python
burundi_provinces.insert(3,"Cankuzo")
for province in burundi_provinces:
    print("Now we are in {}".format(province))
```

    Now we are in Makamba
    Now we are in Karuzi
    Now we are in Rutana
    Now we are in Cankuzo
    Now we are in Bururi
    Now we are in Makamba
    Now we are in Ruyigi
    Now we are in Cibitoke


"Cankuzo" has taken the place of "Bururi" and "Bururi" has been shifted to index 4.

#### extend function

As we have already seen, append and insert add a new element to the list. But we can also add a list to a list and have a container of containers.


```python
northern_provinces = ["Kirundo","Ngozi","Kayanza"]
burundi_provinces.insert(3,northern_provinces)
for province in burundi_provinces:
    print("Now we are in {}".format(province))
```

    Now we are in Makamba
    Now we are in Karuzi
    Now we are in Rutana
    Now we are in ['Kirundo', 'Ngozi', 'Kayanza']
    Now we are in Cankuzo
    Now we are in Bururi
    Now we are in Makamba
    Now we are in Ruyigi
    Now we are in Cibitoke



```python
print(burundi_provinces)
```

    ['Makamba', 'Karuzi', 'Rutana', ['Kirundo', 'Ngozi', 'Kayanza'], 'Cankuzo', 'Bururi', 'Makamba', 'Ruyigi', 'Cibitoke']


We can see that the list of the northern provinces has been added to burundi_provinces. However, now what if instead for adding the list, there was a merge between the two lists into one.


```python
northern_provinces = ["Kirundo","Ngozi","Kayanza"]
burundi_provinces.extend(northern_provinces)
for province in burundi_provinces:
    print("Now we are in {}".format(province))
```

    Now we are in Makamba
    Now we are in Karuzi
    Now we are in Rutana
    Now we are in Cankuzo
    Now we are in Bururi
    Now we are in Makamba
    Now we are in Ruyigi
    Now we are in Cibitoke
    Now we are in Kirundo
    Now we are in Ngozi
    Now we are in Kayanza



```python
print(burundi_provinces)
```

    ['Makamba', 'Karuzi', 'Rutana', 'Cankuzo', 'Bururi', 'Makamba', 'Ruyigi', 'Cibitoke', 'Kirundo', 'Ngozi', 'Kayanza']


For this round, we have merged the elements from northern_provinces into burundi_provinces using the extend function.

#### delete keyword

Now that we have seen how to add an element to a list let's how we can delete it from a list.

let's say we want to remove "Rutana" from the list which is at index 2


```python
del burundi_provinces[2]
```


```python
print(burundi_provinces)
```

    ['Makamba', 'Karuzi', 'Cankuzo', 'Bururi', 'Makamba', 'Ruyigi', 'Cibitoke', 'Kirundo', 'Ngozi', 'Kayanza']


"Rutana" has been removed from the list using the del keyword which stands for delete followed by burundi_province[2] which will return "Rutana" since it is the element at that specific index and delete it.

#### remove function

Sometimes we might have an occurrence of element multiple times in a list. To delete the first occurrence of that element, we use the remove function.

let's append "Karuzi" again in the list.


```python
burundi_provinces.append("Karuzi")
```


```python
print(burundi_provinces)
```

    ['Makamba', 'Karuzi', 'Cankuzo', 'Bururi', 'Makamba', 'Ruyigi', 'Cibitoke', 'Kirundo', 'Ngozi', 'Kayanza', 'Karuzi']


"Karuzi" is at index 1 and index 10, then remove function will delete the first occurrence of "Karuzi" in the list.


```python
burundi_provinces.remove("Karuzi")
```


```python
print(burundi_provinces)
```

    ['Makamba', 'Cankuzo', 'Bururi', 'Makamba', 'Ruyigi', 'Cibitoke', 'Kirundo', 'Ngozi', 'Kayanza', 'Karuzi']


Now we can see that Karuzi at index 1 has been removed, and if we execute the remove function again with "Karuzi" as the parameter, it will search in the list from left to right and delete any occurrence of the element in the list.

#### pop function

the pop function takes as a parameter the index of the element to delete and return the deleted element.


```python
province_deleted = burundi_provinces.pop(4)
```


```python
print(province_deleted)
```

    Ruyigi



```python
print(burundi_provinces)
```

    ['Makamba', 'Cankuzo', 'Bururi', 'Makamba', 'Cibitoke', 'Kirundo', 'Ngozi', 'Kayanza', 'Karuzi']


The element at index 4, which is "Ruyigi", has been deleted from the list, returned and stored in the province_deleted variable that we can print.

We could also print it straight away after deleting and returning the element.


```python
print(burundi_provinces.pop(5))
```

    Kirundo


"Kirundo" was deleted from the list and printed.

### List features

We can initialize an empty in two different ways. The purpose of initializing an empty list is to contain elements that we don't know initially.


```python
list_1 = []
list_2 = list()
```


```python
print("list 1 {}".format(list_1))
print("list 2 {}".format(list_2))
```

    list 1 []
    list 2 []


let's compare them and see if they are equal.


```python
if list_1 == list_2:
    print("They are equal")
else:
    print("They are not equal")
```

    They are equal


Yep! they are equal

We can either declare a list using [ ] or using list( ) which is called a constructor. A constructor means that we are initializing a list object in memory. We will go in details about constructor when we will learn about classes.

Now let's check if the lists occupy the same spot in memory.


```python
if list_1 is list_2:
    print("They are located at the same place in memory")
else:
    print("They are not located at the same place in memory")
```

    They are not located at the same place in memory.


Well, using the "is" keyword, we can see that the two lists are stored into two different places in the computer memory.

So now the question is when to use the constructor form or the bracket form. Well, choose any form you feel comfortable with since both forms do precisely the same thing.

However, there is one advantage of using the constructor form. This form helps us to traverse and print any sequential data type (also called iterable) without using a loop statement like this


```python
print(list("This will print all the characters one by one"))
```

    ['T', 'h', 'i', 's', ' ', 'w', 'i', 'l', 'l', ' ', 'p', 'r', 'i', 'n', 't', ' ', 'a', 'l', 'l', ' ', 't', 'h', 'e', ' ', 'c', 'h', 'a', 'r', 'a', 'c', 't', 'e', 'r', 's', ' ', 'o', 'n', 'e', ' ', 'b', 'y', ' ', 'o', 'n', 'e']


Think of this notation as casting a string data type into a list data type.

Let's see another example.


```python
random_num = [33,17,44,6,244,60,55,23,44,13,22,66]
random_num_2= random_num
random_num_2.sort(reverse=True)
```


```python
print(random_num_2)
print(random_num)
```

    [244, 66, 60, 55, 44, 44, 33, 23, 22, 17, 13, 6]
    [244, 66, 60, 55, 44, 44, 33, 23, 22, 17, 13, 6]


Surprised? Huh? Let me explain what is happening here, we have assigned random_num to random_num_2, and then we have sorted random_num_2 in a reversed order by using reverse=True as a parameter in the sort function. Now comes the confusing part, when we print we print the random_num_2 we can see that the list is now reversed as expected, but when we print random_num we also found out that it is also reversed? But why?

well, the answer is that random_num and random_num_2 are pointing to the same list in computer memory, so any function that is applied to the list affects random_num and random_num_2 since they refer to the same object. 

Let's test again using the "is" keyword to see they refer to the same list in memory.


```python
if random_num is random_num_2:
    print("The two variables refer to the same objects in memory")
else:
    print("The two variables do not refer to the same objects in memory")
```

    The two variables refer to the same objects in memory.


Now lets test again the two lists but this time we will assign random_num_2 using a constructor.


```python
random_num_2 = list(random_num)
print(random_num_2)
```

    [244, 66, 60, 55, 44, 44, 33, 23, 22, 17, 13, 6]



```python
if random_num is random_num_2:
    print("The two variables refer to the same objects in memory")
else:
    print("The two variables do not refer to the same objects in memory")
```

    The two variables do not refer to the same objects in memory.


Even though the two lists have the same elements arranged in the same reversed order, the two lists don't refer to the same list in memory. random_num_2 points to its list and random_num points to another one.


```python
if random_num == random_num_2:
    print("The two variables have the same elements and arranged in the same order")
else:
    print("The two variables don't have the same elements and are not arranged in the same order")
```

    The two variables have the same elements and arranged in the same order


The same thing will happen when using the sorted function.


```python
random_num_2 = sorted(random_num,reverse=True)
```


```python
if random_num is random_num_2:
    print("The two variables refer to the same objects in memory")
else:
    print("The two variables do not refer to the same objects in memory")
```

    The two variables do not refer to the same objects in memory.



```python
if random_num == random_num_2:
    print("The two variables have the same elements and arranged in the same order")
else:
    print("The two variables don't have the same elements and are not arranged in the same order")
```

    The two variables have the same elements and arranged in the same order


No surprise here since the two lists have the same elements arranged in the same manner.

### Challenge

For this challenge, we will have a list that contains lists of ingredients. These lists of ingredients will be added to the list then print all the ingredients that don't contain a specific unwanted ingredient. For example, let's say that we don't want "nuts" as an ingredient; the program shall print all the list elements that do not have "nuts" in it. While printing, we shall have a count of each element.

Now go ahead and try this challenge on your own and only after you have tried, come back and compare your solution with mine.

### Challenge solution


```python
meals = []
meals.append(["beans","egg","nuts","rice"])
meals.append(["cabbage","bread","banana","carots"])
meals.append(["bacon","beaf","steak","nuts"])
meals.append(["tomatos","burger","mustard","nuts"])
meals.append(["fries","chicken","nuts","avocado"])
meals.append(["meat","sukuma","ugali"])
for meal in meals:
    if "nuts" not in meal:
        for index,element in enumerate(meal,1):
            print("element {} is {}".format(index,element))
        print()
```

    element 1 is cabbage
    element 2 is bread
    element 3 is banana
    element 4 is carots
    
    element 1 is meat
    element 2 is sukuma
    element 3 is ugali
    


We first append to the list all the ingredients and loop through the elements of each list and check if any the element is "nuts" if there is this list will be ignored until we get a list that does not contain "nuts". 

After this finding,  we will loop through the list and enumerate all the elements using the enumerate function. There the second parameter in the enumerate function; this parameter corresponds to the starting count. By default, this parameter is not written and is set to 0; we can overwrite this by placing any number which will be the starting point of the count. There is an empty space after all the elements in the list are done printing.

### Conclusion

A list is a clean way of storing data in a container that allow us to iterate over on its elements and perform a useful operation on it like adding and removing new elements. We will be using lists a lot. In the upcoming post, we will discuss tuples and ranges.

Thank you for reading this tutorial. I hope you have learned one or two things. If you like this post, please subscribe to stay updated with new posts, and if you have a thought or a question, I would love to hear it by commenting below. Cheers, and keep on learning!

