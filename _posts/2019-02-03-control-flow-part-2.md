---
title:  "Control flow part 2 - loop statement"
image: /assets/post_images/py_tuto.jpeg
excerpt_separator: <!-- more -->
tags:
- python
- tutorial
- programming
---

In this second part of the control flow series, we will discuss the for and while loop statement which is one of the most important concepts in programming. This allows repetition of a certain block of code without rewriting the code multiple times, this is the reasons why computer are so efficient. They can repeat an operation so many times, very quickly by removing the need for a human to code the same repetitive task.<!-- more -->

### For loop statement


```python
for i in range(1,25):
    print("i is now {}".format(i))
```

    i is now 1
    i is now 2
    i is now 3
    i is now 4
    i is now 5
    i is now 6
    i is now 7
    i is now 8
    i is now 9
    i is now 10
    i is now 11
    i is now 12
    i is now 13
    i is now 14
    i is now 15
    i is now 16
    i is now 17
    i is now 18
    i is now 19
    i is now 20
    i is now 21
    i is now 22
    i is now 23
    i is now 24


As we have already seen in the previous post, the for loop statement count from 1 to 24 and assign that value 
to i and execute the block of code once for each value.

the range is an exclusive function which means that the last number will not be included. 
If we want i to go up to 25, we'll rewrite it like this


```python
for i in range(1,26):
    print("i is now {}".format(i)) #print 1
```

    i is now 1
    i is now 2
    i is now 3
    i is now 4
    i is now 5
    i is now 6
    i is now 7
    i is now 8
    i is now 9
    i is now 10
    i is now 11
    i is now 12
    i is now 13
    i is now 14
    i is now 15
    i is now 16
    i is now 17
    i is now 18
    i is now 19
    i is now 20
    i is now 21
    i is now 22
    i is now 23
    i is now 24
    i is now 25


what is happening here, is that Python is executing the print function 25 times starting from 1. 
Another thing to mention is on the naming of the variable in the for loop statement, in the previous posts. 
I have said that we should give a meaningful name to our variables, but in this case, we have an exception 

"i" which stands for index and "j" are generally accepted to be the variable's names in a for a loop statement.

let's see another example


```python
numbers = "2,33,112,3,55,32,11,909,3"
for i in range(0,len(numbers)):
    print(numbers[i])
```

    2
    ,
    3
    3
    ,
    1
    1
    2
    ,
    3
    ,
    5
    5
    ,
    3
    2
    ,
    1
    1
    ,
    9
    0
    9
    ,
    3


here we have a sequence of characters separated by a comma and stored in a string variable called numbers. We have a for loop that will loop from 0 the length of numbers, this length is calculated using the len() function, this function will count starting from 1 to the number of characters found in a string. In our example, numbers variable has 25 characters.

so the range function will count from 0 to 24 and assign each count to the variable i. The value assigned to i is an integer and for each count i is incremented by one

Now inside the print function, the value of i will be used as the index of numbers and will print the character at the index corresponding to the value of i, this will print all the character from the character at index 0 (which is 2) to the last character at index 25 (which is 3).

Now, what if we want to print only the number and exclude the commas. We can easily do this by using a for loop with the conditional statement


```python
numbers = "2,33,112,3,55,32,11,909,3"
for i in range(0,len(numbers)):
    if numbers[i] in "0123456789":
        print(numbers[i])
```

    2
    3
    3
    1
    1
    2
    3
    5
    5
    3
    2
    1
    1
    9
    0
    9
    3


We have introduced a conditional statement before printing the character, We are testing if the character at that particular index is found in "0123456789" if it is then the program shall proceed and print that character, else the program shall skip that index to the next one. That is why the commas are not printed since they are not part of "0123456789"

By default, Python will print each character on a new line. This is due to a keyword in the print function named "end", even though it is not written, by default the "end" is hidden and set to "\n" like this


```python
numbers = "2,33,112,3,55,32,11,909,3"
for i in range(0,len(numbers)):
    if numbers[i] in "0123456789":
        print(numbers[i],end="\n")
```

    2
    3
    3
    1
    1
    2
    3
    5
    5
    3
    2
    1
    1
    9
    0
    9
    3


We overwrite this by changing the value of end, let's say for example instead of print the character on a new line we want to print the character on the same line with space between, we are going to change "end" and assign it to an empty string "" like this


```python
numbers = "2,33,112,3,55,32,11,909,3"
for i in range(0,len(numbers)):
    if numbers[i] in "0123456789":
        print(numbers[i],end="")
```

    23311235532119093

As we can see, now all the characters are printed and joined together without the comma on the same line.

We can also separate the string with any character of our choice.


```python
numbers = "2,33,112,3,55,32,11,909,3"
for i in range(0,len(numbers)):
    if numbers[i] in "0123456789":
        print(numbers[i],end="*")
```

    2*3*3*1*1*2*3*5*5*3*2*1*1*9*0*9*3*

what if we want to cast each character to an actual integer and ignore the comma, we can easily do that too


```python
numbers = "2,33,112,3,55,32,11,909,3"
only_numbers = ""
for i in range(0,len(numbers)):
    if numbers[i] in "0123456789":
        only_numbers += numbers[i]
int_numbers = int(only_numbers)
print(int_numbers)
```

    23311235532119093



```python
print(type(int_numbers))
```

    <class 'int'>


we have created an empty string variable named only_numbers that will store each numerical character. After checking if the character is found in "0123456789", we used augmented assignment operator += and kept on storing each numerical character to only_numbers. After the for loop has exhausted all its count, all the numerical characters are stored in the only_numbers variables. Now we cast only_numbers to integer and store it to the int_numbers and print it

let's see another similar example


```python
numbers = "2,33,112,3,55,32,11,909,3"
only_numbers = ""

for char in numbers:
    if char in "0123456789":
        only_numbers += char
int_numbers = int(only_numbers)
print(int_numbers)
```

    23311235532119093


since now we ain't dealing with indexes and range function, we are going to name the variable in the for loop "char", so this loop will be interpreted as "for each char found in numbers do something with it", and we shall get the same answer as we did before. This much more concise and readable code than using the range function


```python
countries = ["Burundi","Rwanda","Tanzania","Kenya","Uganda"]
for country in countries:
    print("The country now we are printing is {}".format(country))
```

    The country now we are printing is Burundi
    The country now we are printing is Rwanda
    The country now we are printing is Tanzania
    The country now we are printing is Kenya
    The country now we are printing is Uganda


We can also loop through a list. In the example above we have a list of countries, we'll see list in more detail in the upcoming post but as I have already said, a list is just container and we can traverse through all the elements in the container using the for loop statement. Each country will be printed on its turn up to the last country in the list.

We can also use concatenation since we are dealing with string data type


```python
countries = ["Burundi","Rwanda","Tanzania","Kenya","Uganda"]
for country in countries:
    print("The country now we are printing is "+ country)
```

    The country now we are printing is Burundi
    The country now we are printing is Rwanda
    The country now we are printing is Tanzania
    The country now we are printing is Kenya
    The country now we are printing is Uganda


Now let's go back to the range function


```python
for i in range(0, 20, 2):
    print("i is now {}".format(i))
```

    i is now 0
    i is now 2
    i is now 4
    i is now 6
    i is now 8
    i is now 10
    i is now 12
    i is now 14
    i is now 16
    i is now 18


we have introduced a third value in the range function, the general format of the range function is like this: range(start, end, step). So in this example, i will start counting from 0(start) up to 19(end), for each iteration we going to increment i by 2(step).

let's see another example


```python
for i in range(0, 50, 5):
    print("i is now {}".format(i))
```

    i is now 0
    i is now 5
    i is now 10
    i is now 15
    i is now 20
    i is now 25
    i is now 30
    i is now 35
    i is now 40
    i is now 45


Here we are printing from 0 to 49 with a step of 5

we can use nested for loop to get very meaningful results

let's illustrate this by printing the multiplication tables from 1 to 12


```python
for i in range(1,13):
    for j in range(1,13):
        print("{} x {} = {}".format(i,j,i*j))
    print()
```

    1 x 1 = 1
    1 x 2 = 2
    1 x 3 = 3
    1 x 4 = 4
    1 x 5 = 5
    1 x 6 = 6
    1 x 7 = 7
    1 x 8 = 8
    1 x 9 = 9
    1 x 10 = 10
    1 x 11 = 11
    1 x 12 = 12
    
    2 x 1 = 2
    2 x 2 = 4
    2 x 3 = 6
    2 x 4 = 8
    2 x 5 = 10
    2 x 6 = 12
    2 x 7 = 14
    2 x 8 = 16
    2 x 9 = 18
    2 x 10 = 20
    2 x 11 = 22
    2 x 12 = 24
    
    3 x 1 = 3
    3 x 2 = 6
    3 x 3 = 9
    3 x 4 = 12
    3 x 5 = 15
    3 x 6 = 18
    3 x 7 = 21
    3 x 8 = 24
    3 x 9 = 27
    3 x 10 = 30
    3 x 11 = 33
    3 x 12 = 36
    
    4 x 1 = 4
    4 x 2 = 8
    4 x 3 = 12
    4 x 4 = 16
    4 x 5 = 20
    4 x 6 = 24
    4 x 7 = 28
    4 x 8 = 32
    4 x 9 = 36
    4 x 10 = 40
    4 x 11 = 44
    4 x 12 = 48
    
    5 x 1 = 5
    5 x 2 = 10
    5 x 3 = 15
    5 x 4 = 20
    5 x 5 = 25
    5 x 6 = 30
    5 x 7 = 35
    5 x 8 = 40
    5 x 9 = 45
    5 x 10 = 50
    5 x 11 = 55
    5 x 12 = 60
    
    6 x 1 = 6
    6 x 2 = 12
    6 x 3 = 18
    6 x 4 = 24
    6 x 5 = 30
    6 x 6 = 36
    6 x 7 = 42
    6 x 8 = 48
    6 x 9 = 54
    6 x 10 = 60
    6 x 11 = 66
    6 x 12 = 72
    
    7 x 1 = 7
    7 x 2 = 14
    7 x 3 = 21
    7 x 4 = 28
    7 x 5 = 35
    7 x 6 = 42
    7 x 7 = 49
    7 x 8 = 56
    7 x 9 = 63
    7 x 10 = 70
    7 x 11 = 77
    7 x 12 = 84
    
    8 x 1 = 8
    8 x 2 = 16
    8 x 3 = 24
    8 x 4 = 32
    8 x 5 = 40
    8 x 6 = 48
    8 x 7 = 56
    8 x 8 = 64
    8 x 9 = 72
    8 x 10 = 80
    8 x 11 = 88
    8 x 12 = 96
    
    9 x 1 = 9
    9 x 2 = 18
    9 x 3 = 27
    9 x 4 = 36
    9 x 5 = 45
    9 x 6 = 54
    9 x 7 = 63
    9 x 8 = 72
    9 x 9 = 81
    9 x 10 = 90
    9 x 11 = 99
    9 x 12 = 108
    
    10 x 1 = 10
    10 x 2 = 20
    10 x 3 = 30
    10 x 4 = 40
    10 x 5 = 50
    10 x 6 = 60
    10 x 7 = 70
    10 x 8 = 80
    10 x 9 = 90
    10 x 10 = 100
    10 x 11 = 110
    10 x 12 = 120
    
    11 x 1 = 11
    11 x 2 = 22
    11 x 3 = 33
    11 x 4 = 44
    11 x 5 = 55
    11 x 6 = 66
    11 x 7 = 77
    11 x 8 = 88
    11 x 9 = 99
    11 x 10 = 110
    11 x 11 = 121
    11 x 12 = 132
    
    12 x 1 = 12
    12 x 2 = 24
    12 x 3 = 36
    12 x 4 = 48
    12 x 5 = 60
    12 x 6 = 72
    12 x 7 = 84
    12 x 8 = 96
    12 x 9 = 108
    12 x 10 = 120
    12 x 11 = 132
    12 x 12 = 144
    


We start by counting the first index i,  i will start from 1 to 12 and will correspond to the multiplication tables from 1 to 12. The nested for loop has an index of j, which will count from 1 to 12 and will correspond to the factor in the table. The print function will print the table number (i) times the factor (j) and return the result. After each table has exhausted all its factors,a print function without anything inside is placed out the nested for loop. The purpose of this is to add a space after each table and it should be placed at the same level of indentation as the nested for loop statement in order for it to print whitespace after each table.

Sometime when we are coding we might need to stop the flow of a loop when a certain condition occurs. We may want to jump out the ongoing iteration and go to the next or stop completely the loop.  That is when keyword like continue and break come in.

### Continue keyword


```python
available_cars = ["bmw","porshes","benz","audi","mazda","tesla"]
for car in available_cars:
    print("Buy " + car)
```

    Buy bmw
    Buy porshes
    Buy benz
    Buy audi
    Buy mazda
    Buy tesla


There is nothing new here that we have not seen, what's happening is that we are going through all the car names in the list, and for each car, we are concatenating the string "Buy " with the name of the car.

But what if we are not interested in a particular car, let's say we don't want to buy a Mazda for instance, we want to skip Mazda each time it is its turn. That's when the continue keyword comes into place.


```python
available_cars = ["Bmw","Porshes","Benz","Audi","Mazda","Tesla","Toyota"]
for car in available_cars:
    if car == "Mazda":
        continue
    print("Buy " + car)
```

    Buy Bmw
    Buy Porshes
    Buy Benz
    Buy Audi
    Buy Tesla
    Buy Toyota


We can see that "buy Mazda" is now missing. what happened? well, we have introduced a test using a conditional statement to check if the name from the list is "Mazda" if it not, the test will fail and the print function will execute. if it is, python will read the continue keyword and will interpret it like this: "skip this item to the next item in the list". the print function will be bypassed.


To make things more clear, we can rewrite it like this


```python
available_cars = ["Bmw","Porshes","Benz","Audi","Mazda","Tesla","Toyota"]
for car in available_cars:
    if car == "Mazda":
        print("Ignoring "+car)
        continue
    print("Buy " + car)
```

    Buy Bmw
    Buy Porshes
    Buy Benz
    Buy Audi
    Ignoring Mazda
    Buy Tesla
    Buy Toyota


### Break keyword

On the other hand, break keyword will completely terminate the loop.


```python
available_cars = ["Bmw","Porshes","Benz","Audi","Mazda","Tesla","Toyota"]
for car in available_cars:
    if car == "Mazda":
        break
    print("Buy " + car)
print("The for loop have pre-maturity stop. mazda ,tesla and toyota will not be printed")
```

    Buy Bmw
    Buy Porshes
    Buy Benz
    Buy Audi
    The for loop have pre-maturity stop. mazda ,tesla and toyota will not be printed


All the strings after Mazda will not be printed. why? because we have terminated the for loop as soon as the conditional statement has been evaluated to True when compared the car variable to "Mazda", the break keyword will exit out of the for loop and will execute the code found after the for loop. "Tesla" and "Toyota" are not printed too because the for loop has exited before their turns. 

let's see another example


```python
meal_ingredients = ["bacon","egg","tomato","milk","nuts","mangoe","rice","beans"]
allergic_food = ""
for ingredient in meal_ingredients:
    if ingredient == "nuts":
        allergic_food = ingredient
        break
if allergic_food:
    print("Sorry but I can't eat that, I am allergic to nuts")
```

    Sorry but I can't eat that, I am allergic to nuts


In this example, we don't want to eat any meal that contains allergic food like nuts in this instance. We can see how helpful the break keyword is, in a sense that we don't have to traverse the whole list instead as soon as we find a match to an allergic ingredient, the program shall stop right away and exit the loop. This is done by first testing if any ingredient equal to "nuts" if it is we will store the ingredient value in the allergic_food variable then break out of the loop. After the loop, we are going to test if allergic_food is a none empty string. In this example, allergic_food is not empty, which means that it will be evaluated to True and display the message that the food can't be eaten.

let's see the same example but this time nuts is removed from the ingredients list.


```python
meal_ingredients = ["bacon","egg","tomato","milk","mangoe","rice","beans"]
allergic_food = ""
for ingredient in meal_ingredients:
    if ingredient == "nuts":
        allergic_food = ingredient
        break 
if allergic_food:
    print("Sorry but I can't eat that, I am allergic to nuts")
```

As we can see "nuts" ain't in the list, so allergic_food will remain an empty string, which will be evaluated as False when using the conditional statement. After running the program nothing will happen.

### Else in a for loop statement

Python unlike most of the other programming language can have an else statement for a for loop statement.


```python
meal_ingredients = ["bacon","egg","tomato","milk","mangoe","rice","beans"]
allergic_food = ""
for ingredient in meal_ingredients:
    if ingredient == "nuts":
        allergic_food = ingredient
        break
else:
    print("Hummm Yummiee I want to eat!!")

if allergic_food:
    print("Sorry but I can't eat that, I am allergic to nuts")
```

    Hummm Yummiee I want to eat!!


The else statement in a for loop will be executed only if the for loop has not broken out, in our case only when the nuts are not found in the meal_ingredients.


```python
meal_ingredients = ["bacon","egg","tomato","milk","nuts","mangoe","rice","beans"]
allergic_food = ""
for ingredient in meal_ingredients:
    if ingredient == "nuts":
        allergic_food = ingredient
        break
else:
    print("Hummm Yummiee I want to eat!!")

if allergic_food:
    print("Sorry but I can't eat that, I am allergic to nuts")
```

    Sorry but I can't eat that, I am allergic to nuts


If the break keyword has been executed, then else statement will be ignored. Careful here with the indentation of else statement should be aligned on the same level as the for loop, in order for it to be considered as part of the for loop statement.

We can also rewrite this code by initializing allergic_food to an empty string when the else statement of the for loop, which will make the conditional evaluation after the for loop equal to False because allergic_food was not initialized in the first place.


```python
meal_ingredients = ["bacon","egg","tomato","milk","nuts","mangoe","rice","beans"]
for ingredient in meal_ingredients:
    if ingredient == "nuts":
        allergic_food = ingredient
        break
else:
    allergic_food = ""
    print("Hummm Yummiee I want to eat!!")

if allergic_food:
    print("Sorry but I can't eat that, I am allergic to nuts")
```

    Sorry but I can't eat that, I am allergic to nuts


This is a matter of preferences, some people may declare allergic_food before it is being used or might just like it this way, Python is flexible with either way. Personally, I like to initialize a variable before it is being used.

### While loop statement

Python provides two different ways to loop around a block of code, using the for loop as we have already seen and using the while loop.

let's see the same example using first the for loop and later the while loop.


```python
for i in range(21):
    print("i is now {}".format(i))
```

    i is now 0
    i is now 1
    i is now 2
    i is now 3
    i is now 4
    i is now 5
    i is now 6
    i is now 7
    i is now 8
    i is now 9
    i is now 10
    i is now 11
    i is now 12
    i is now 13
    i is now 14
    i is now 15
    i is now 16
    i is now 17
    i is now 18
    i is now 19
    i is now 20


now let's use while loop.


```python
i = 0
while i <= 20:
    print("i is now {}".format(i))
    i += 1
```

    i is now 0
    i is now 1
    i is now 2
    i is now 3
    i is now 4
    i is now 5
    i is now 6
    i is now 7
    i is now 8
    i is now 9
    i is now 10
    i is now 11
    i is now 12
    i is now 13
    i is now 14
    i is now 15
    i is now 16
    i is now 17
    i is now 18
    i is now 19
    i is now 20


Let's see how the while loop works, we first initialize the variable i to 0 which is the starting point of our loop, then we will write the while loop statement and specify the stop of the loop. In this example, we are stating
that i should not go beyond 20 else skip the while loop. Now that we are inside the loop, Python will start by executing the print function for the first time after that there is an increment by 1 of the i variable using augmented assignment at the last line of code. 

Python will go back to the while statement and check if i is still less or equal to 20, if it is then we will execute the print function and so on up the point when the while loop evaluates to False, this is the case when i will be incremented to 21, since 21 is greater than 20, the execution will jump out of the while loop. so in some way, we have to make our while loop statement to evaluate to False in order for it to escape from the loop.

lets see what will happen if we don't make our while loop statement evaluate to False.


```python
i = 0
while i <= 20:
    print("i is now {}".format(i))
    #i += 1
```

I have not run this code because this will result in an infinite loop. Let's see what we did, we first specify 0 as the start of i and evaluate i in a while loop against the limit which is 20, but here is the important thing we did not change the value of i, meaning that all throughout the while loop, i has remained 0 and has not increased in order to make the evaluation of while statement False. this means that it will never reach 21 in order to escape the loop.

The main thing to keep in mind is that a while loop must have a start, and end and an incremental step value, if one of these is missing, it will result in an infinite loop or an error. 

Another way to write an infinite loop would be the use of True boolean value.


```python
i = 0
the_condition = True
while the_condition:
    print("i is now {}".format(i))
    #i += 1
    the_condition = False
```

    i is now 0


You will see many while loop statements are written using True boolean value, which mean that the block of code inside will be executed at least once whatever the situation. We have to make sure that the condition has a way to break out of the while else we will end up with an infinite loop that is why we are setting the_condition variable to False on the last line of code.

Let's see another example when actually while is useful.


```python
brand_available = ["lenovo","dell","apple","HP","asus","acer"]

choosen_brand = ""

while choosen_brand not in brand_available:
    choosen_brand = input("Please enter the brand of your choice: ")
print("Thank you for inputing {} as your brand of choice...".format(choosen_brand))
```

    Please enter the brand of your choice: 
    Please enter the brand of your choice: imaginary brand
    Please enter the brand of your choice: lenovo
    Thank you for inputing lenovo as your brand of choice...


In this example, we have a list of different brands of laptop available in a store and this program give a choice to the user to choose one brand among the brands available. we initialize the variable choosen_brand to an empty string, we do this because we can't use this variable in the while loop before it is being declared.

This while loop will check if the choose_brand is not in the brand_available list if it is, it will break out of the loop, else if it is not it will keep on looping and prompting to input till an input of a brand found in the brand_available is inputted. If the input is not found in the list, it will keep on looping.
After the while loop has exited, the print function on the last line will be printed.


```python
brand_available = ["lenovo","dell","apple","HP","asus","acer"]

choosen_brand = ""

while choosen_brand not in brand_available:
    choosen_brand = input("Please enter the brand of your choice: ")
    if choosen_brand == "none" or choosen_brand == "None" or not choosen_brand:
        print("No brand choosen")
        continue
else:
    print("Thank you for inputing {} as your brand of choice...".format(choosen_brand))
```

    Please enter the brand of your choice: 
    No brand choosen
    Please enter the brand of your choice: none
    No brand choosen
    Please enter the brand of your choice: None
    No brand choosen
    Please enter the brand of your choice: not a brand
    Please enter the brand of your choice: lenovo
    Thank you for inputing lenovo as your brand of choice...


we have added a test in the while loop, this will test if choosen_brand is "none" or "None" string or an empty string. if it is then the loop will keep on looping using the continue keyword. While loop can also have the else statement which behaves exactly the same as the for loop statement.

### For loop VS While loop statement

Even though the main purpose of the two loops is to iterate, in order to know when to use one instead of the other is that the for loop is used when we already know ahead of time how many times we are going to loop but on the other hand, while loop is used when we don't know ahead of time how many times we'll loop.

A particular use case of the while loop would be reading data from a file since most of the time we don't know how much data we are dealing with in advance. The while loop will keep on repeating till there is no more data left to process. We will talk more on this when we will see about input/output.

### Challenge 1

Challenge time!! For this first challenge, let's create a program that checks if an inputted IP address is valid or not. A message saying "valid IP address" or "Invalid IP address" should be returned. A valid IP address consists of 4 numbers separated by a dot, and those numbers must be between 0 and 255
examples of a valid IP address:
    	- 127.0.0.1
    	- 255.255.255.255
    	- 192.198.22.1
    	- 0.0.0.0
example of an invalid IP address:
    	- 342.11.333.1
        - .23.11.2.2
    	- 122.11.54.2.
    	- 123.145.22.34343343
    	- 192.184.22.1.5
    	- empty input eg: ""

Go ahead and try this challenge.

### Challenge 1 Solution

This is just one of the different way you could have approached this challenge there are many ways to do the same challenge, the important is to get the right answer. Bear with me here there are better ways to write this code but since we have not seen most advanced techniques like sets data type or error handling we will suppose that all the input are only numerical strings. The main goal here is to use the concepts that we have already seen.


```python
ip_address = ""
while not ip_address:
    ip_address = input("Please enter an IP address: ")
```

    Please enter an IP address: 
    Please enter an IP address: 
    Please enter an IP address: 
    Please enter an IP address: 127.3.1.99



```python
dot_count = 0
dot_index = []
num1 = 0
num2 = 0
num3 = 0
num4 = 0
if len(ip_address) <= 15:
    if ip_address[0] != "." and ip_address[-1] != ".":
        for char in ip_address:
            if char == ".":
                dot_count += 1
        if dot_count == 3:
            for index,char in enumerate(ip_address):
                if char == ".":
                    dot_index.append(index)
            num1 = int(ip_address[:dot_index[0]])
            num2 = int(ip_address[dot_index[0]+1:dot_index[1]])
            num3 = int(ip_address[dot_index[1]+1:dot_index[2]])
            num4 = int(ip_address[dot_index[2]+1:])
            if num1 and num2 and num3 and num4 in range(256):
                print("Valid IP address")
            else:
                print("Invalid IP address")
        else:
            print("Invalid IP address")
    else:
        print("Invalid IP address")
else:
    print("Invalid IP address")
            
```

    Valid IP address


We first initialize ip_address to an empty string and test it using the while loop to check if we have not inputted an empty string. The while loop will keep on looping if an empty string is inputted.

We then initialize the dot_count to 0, this variable will count the dots in the IP address variable since we know that a valid IP address has 3 and only 3 dots separating the 4 numbers. dox_index will be initialized as an empty list. this list will contain the indexes (positions) where the dot characters are found in the string. We also initialize to zero variables that will hold all the numerical character in the string.

We will first check if the length of ip_address is lower or equal to 15 because the length of the string that forms of a valid IP address can't be greater than 15 else we are dealing with an invalid IP address.

After that, we will check if the first or last character in the ip_address string is not a dot if it is then we are dealing again with an invalid IP address.

Then we will perform a for loop that will traverse the whole string and test if a character is a dot character. If it is a dot character, the dot_count will be incremented by 1 using augmented assignment. A valid IP address has 3 and only 3 dots so if we have less or more than 3, then we are dealing with an Invalid IP address.

After it has passed this test, we will use a new function called enumerate, which will traverse each the character in a string and will also count its position at the same time as it is traversing the string and store it in the index variable.

For each loop, we will test if the character is a dot character if it is its position index will be added to the dot_index list using the append function which adds each value at the end of a list container. The dot_index will hold all the indexes where the dot character had occur. For example, if the IP address 127.0.0.1, the dot_index will [3,5,7] which correspond to the positions where the dot are found in the string.

After we have stored all the indexes in the list, we use slicing in order to extract the exact position that corresponds to the number and cast it to the integer. The slicings are delimited by the indexes found in the dot_index list.

Now that we have gotten all the numbers, we have to check that all the numbers are in a range between 0 and 255. Only if all the numbers are in that range then we have a valid IP address.

### Challenge 2

In this second challenge, we will implement upon what we have already seen from the previous post. we have built a guessing number program but that program had two major drawbacks:
	- first, it was always guessing the same number (5)
    - secondly, we were only given two chance to guess the number. the more guesses we add the more conditional statement we would have to write which is not inefficient.

So for your second challenge, we will rewrite that program in a more efficient way. We will use the while loop with a conditional statement in order to guess the number. We will also have to guess the number within 3 attempts and guide the user to guess lower or higher.

But before you start let's see how to generate a random number between a specific range by importing the random module.


```python
import random

random_num = random.randint(1,10)
print(random_num)
```

    1


The random module is a built-in function in python, but not used in the normal instance because we want to keep python lightweight so we have to import it to our code and to only use it when needed. After importing it now we can start using it. We use the dot operator the same way we did when we are using the format() function on a string, to access to the functionalities that belong to the modules. In our case, we are using randint function from the random module. The general format of randint is like this randint(start_num,end_num) and will generate randomly a number between the value of start_num and end_num. This function is not exclusive which mean that the end_num will be part of the numbers randomly generated.

Now that we have seen the random module, you can go ahead and attempt the challenge.

### Challenge 2 Solution


```python
import random

highest_num = 10
answer_to_find = random.randint(1,highest_num)
attempt_count = 0
input_num = 0
while not input_num or input_num >= 10 or attempt_count < 3:
    input_num = int(input("Please guess a number between 1 to 10: "))
    if input_num == answer_to_find:
        print("\nYou have won the game, The number to guess was {}".format(answer_to_find))
        break
    elif input_num > 11 or input_num < 1:
        print("\nPlease guess between 1 and 10")
        continue
    else:
        attempt_count += 1
        if input_num > answer_to_find:
            print("\nGuess lower. You are left with {} guesse(s)".format(3-attempt_count))
        elif attempt_count == 3:
            print("\nGame over...you have exhausted all your guesse(s)")
            break
        else:
            print("\nGuess higher. You are left with {} guesse(s)".format(3-attempt_count))
            

        
```

    Please guess a number between 1 to 10: 10
    
    Guess lower. You are left with 2 guesse(s)
    Please guess a number between 1 to 10: 4
    
    You have won the game, The number to guess was 4


so we started by importing the random module as we have already seen and set the highest number to be 10, this means that the number that will be randomly generated will range from 1 to 10, including 10.

Now we will initialize attempt_count which will count how many time we are attempting to guess the number and input_num which will hold the number that the user is inputting. We will assume that all the inputs will be only in numerical characters.

Now we are entering the while statement, this while loop will continue to loop as long as these 3 conditions are True:
	- input_num is a zero integer
	- input_num is greater than 10
	- attemmpt_count is less than 3

Now we are entering the block code inside the while loop, and input the number the user is guessing and cast it to an integer.

We use the conditional statement to compare the input with the randomly generated number

If the two numbers are equal, then we have won. We will print the message with the number we were guessing and break out of the loop

Else if the number inputted is greater than 11 or smaller than 1, a message will be print as a reminder to choose a number between 1 and 10. The continue keyword will be executed and the loop will resume to the next count.

The last else statement will be True only if the number inputted is between 1 and 10 and different from answer_to_find value which is the range we are looking for. In case we will increment the attempt count by 1 we will create a new conditional statement that will help to guide the user to either guess higher or lower depending on the 
inputted number at the same time displaying the number of guesses left by subtracting the maximum number of guess ( which is 3 ) to the guesses count already performed. We have an else if statement that check if we have reached the maximum guesses. If we have, then we have lost the game and then break out the loop.

### Conclusion

In this tutorial post, we have seen the loops in Python. The loops, in general, are very useful in a way that they help us execute the same code multiple times without writing more codes. They also can be used with a conditional statement to create complex codes that can perform various operations. In the upcoming post, we will discuss lists, tuples and sets in python. Stay tuned!

Thank you for reading this tutorial. If you like this post, please subscribe to stay updated with new posts and if you have a thought or a question, I would love to hear it by commenting below.
