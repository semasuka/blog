---
title:  "Control flow part 1 - conditional statement"
image: /assets/post_images/py_tuto.jpeg
excerpt_separator: <!-- more -->
tags:
- python
- tutorial
- programming
---
In this series of posts, we going to talk about the control flow in Python programming. A program’s control flow is the order in which the program’s code executes. In this first post of the series, we'll talk about
the conditional and pass statement. But before we start, let's discuss on the indentation in python and the difference between a statement and an expression. Again, this post is written entirely in Jupyter notebook and exported as a markdown file to Jekyll.<!-- more -->

### statement vs expression

From this point, I will be using the term expression and statement quite often so let me explain right away
the difference. The general rule of thumb is that If you can print it, or assign it to a variable,
it’s an expression and if you can’t then it’s a statement.


```python
2 + 2
3 * 7
1 + 2 + 3 * (8 ** 9) - sqrt(4.0)
min(2, 22)
max(3, 94)
round(81.5)
"foo"
"bar"
"foo" + "bar"
None
True
False
2
3
4.0
```

All the above codes can be printed or assigned to a variable. These are called expressions.


```python
if CONDITION:
elif CONDITION:
else:
for VARIABLE in SEQUENCE:
while CONDITION:
try:
except EXCEPTION as e:
class MYCLASS:
def MYFUNCTION():
return SOMETHING
raise SOMETHING
with SOMETHING:
```

None of the above codes can be assigned to a variable. They are syntactic elements that serve a purpose but do not themselves have any intrinsic “value”. In other words, these codes don’t “evaluate” to anything.
Most of these codes we have not seen them yet but I will be referring them as a statement.


### indentation in python

python, unlike other C-family programming languages, does not use delimiter or block of code using the traditional {} but instead uses space and tab to avoid cluttering code and improve readability.


```python
for i in range(1,12):
    print("i is {}, i**2 is {}, i**3 is {}".format(i,i**2,i**3))
```

    i is 1, i**2 is 1, i**3 is 1
    i is 2, i**2 is 4, i**3 is 8
    i is 3, i**2 is 9, i**3 is 27
    i is 4, i**2 is 16, i**3 is 64
    i is 5, i**2 is 25, i**3 is 125
    i is 6, i**2 is 36, i**3 is 216
    i is 7, i**2 is 49, i**3 is 343
    i is 8, i**2 is 64, i**3 is 512
    i is 9, i**2 is 81, i**3 is 729
    i is 10, i**2 is 100, i**3 is 1000
    i is 11, i**2 is 121, i**3 is 1331


Bear with me, I know we have not seen the for loop, we'll go in depth later but consider a loop as a way to repeat the same code multiple time without rewritting the code.

we are using the range function that counts starting from 1 to 12. for each round, the count is increment by one and stored in the variable i. if you are keen, you may have noticed that the count started from 1 but stopped at 11 instead of 12. You may ask why is it like this? well this is because the range is exclusive so 12 will not be considered but instead the count will limit to 11.

Now let's talk about the indentation, we can clearly see that there is a tab ( or 4 space character ) just before the print function this means that it is part of the for loop statement. Each code on the same level of indentation is part of the same block code. In Python, we delimit a block code with spaces or tab and to add more code to this block, we simply press ENTER.


```python
for i in range(1,12):
    print("i is {}, i**2 is {}, i**3 is {}".format(i,i**2,i**3))
print("this will not be in the part of the loop")
```

    i is 1, i**2 is 1, i**3 is 1
    i is 2, i**2 is 4, i**3 is 8
    i is 3, i**2 is 9, i**3 is 27
    i is 4, i**2 is 16, i**3 is 64
    i is 5, i**2 is 25, i**3 is 125
    i is 6, i**2 is 36, i**3 is 216
    i is 7, i**2 is 49, i**3 is 343
    i is 8, i**2 is 64, i**3 is 512
    i is 9, i**2 is 81, i**3 is 729
    i is 10, i**2 is 100, i**3 is 1000
    i is 11, i**2 is 121, i**3 is 1331
    this will not be in the part of the loop


In this example, we have added a print function on the last line which is not part of the for loop statement and will only be executed at last after the for loop has finished executing.


```python
for i in range(1,12):
print("i is {}, i**2 is {}, i**3 is {}".format(i,i**2,i**3))
```


      File "<ipython-input-3-9559353d8ff3>", line 2
        print("i is {}, i**2 is {}, i**3 is {}".format(i,i**2,i**3))
            ^
    IndentationError: expected an indented block



In example, we are having IndentationError. This error is due to the fact that we have informed Python
that we going to add some code inside the block code of the for loop statement ( Python knows this because of the : after the expression in the for loop ) to find out that the print function
is not part of the for loop block code. To fix this error, we have to add tab or 4 spaces in front of the print 
function.


```python
for i in range(1,12):
    print("i is {}, i**2 is {}, i**3 is {}".format(i,i**2,i**3)) #print 1
    print("individual calculation complete") #print 2
```

    i is 1, i**2 is 1, i**3 is 1
    individual calculation complete
    i is 2, i**2 is 4, i**3 is 8
    individual calculation complete
    i is 3, i**2 is 9, i**3 is 27
    individual calculation complete
    i is 4, i**2 is 16, i**3 is 64
    individual calculation complete
    i is 5, i**2 is 25, i**3 is 125
    individual calculation complete
    i is 6, i**2 is 36, i**3 is 216
    individual calculation complete
    i is 7, i**2 is 49, i**3 is 343
    individual calculation complete
    i is 8, i**2 is 64, i**3 is 512
    individual calculation complete
    i is 9, i**2 is 81, i**3 is 729
    individual calculation complete
    i is 10, i**2 is 100, i**3 is 1000
    individual calculation complete
    i is 11, i**2 is 121, i**3 is 1331
    individual calculation complete


In this example, the #print 2 is part of the same block of code as #print 1. #print 1 and #print 2 functions will be printed on each time the for loop is counting.


```python
for i in range(1,12):
    print("i is {}, i**2 is {}, i**3 is {}".format(i,i**2,i**3)) #print 1
    print("individual calculation complete") #print 2
    
print("All the calculation are done") #print 3
```

    i is 1, i**2 is 1, i**3 is 1
    individual calculation complete
    i is 2, i**2 is 4, i**3 is 8
    individual calculation complete
    i is 3, i**2 is 9, i**3 is 27
    individual calculation complete
    i is 4, i**2 is 16, i**3 is 64
    individual calculation complete
    i is 5, i**2 is 25, i**3 is 125
    individual calculation complete
    i is 6, i**2 is 36, i**3 is 216
    individual calculation complete
    i is 7, i**2 is 49, i**3 is 343
    individual calculation complete
    i is 8, i**2 is 64, i**3 is 512
    individual calculation complete
    i is 9, i**2 is 81, i**3 is 729
    individual calculation complete
    i is 10, i**2 is 100, i**3 is 1000
    individual calculation complete
    i is 11, i**2 is 121, i**3 is 1331
    individual calculation complete
    All the calculation are done


In this example, #print 1 and #print 2 will be printed each time the for loop is running and #print 3 will be only
printed at last when the loop is over.

NB: Do not mix up spaces and tab, I repeat do not mix them up. The reason for this is that sometimes you can get into nasty bugs(errors) that are very hard to debug(to fix), so my advice is either go once for all with 4 spaces or a tab.

### Conditional Statement 

#### if and else statement

Everything we have seen so far has consisted of sequential execution which means that codes are always executed one after the next, in exactly the order specified. But the world is often more complicated than that. 
Frequently, a program needs to skip over some lines of code, execute a series of codes repetitively, or choose between alternate sets of codes to execute. That is where control structures come in. A control structure directs the order of execution of the statements in a program (referred to as the program’s control flow).

In the real world, we commonly must evaluate information around us and then choose one course of action or another based on what we observe and we often use a conditional statement in our day-to-day life like in this example: If the weather is nice, then I’ll wash my clothes. (It’s implied that if the weather isn’t nice, then I won’t wash my clothes.)

In a Python program, the if statement is how you perform this sort of decision-making. It allows for conditional 
execution of a statement or group of statements based on the value of an expression.

Introduction to the if Statement
We’ll start by looking at the most basic type of if statement. In its simplest form, it looks like this:

if "expression":
    "statement"
In the form shown above:

* "expression" is an expression evaluated in Boolean context, as discussed in the section on Logical Operators in the Operators and Expressions in Python tutorial.
  
* "statement" is a valid Python statement, which must be indented.

If "expression" is true, then "statement" is executed. If "expression" is false, then "statement" is skipped over and not executed.

N.B: the colon (:) following "expression" is required. Some programming languages require "expression" to be enclosed in parenthesis, but Python does not.

Let's recap a little from what we have seen in the last post


```python
name = input("What is you name: ")
age = int(input("How old are you {} ".format(name)))
```

    What is you name: stern
    How old are you stern 44



```python
print("Your name is {} and you are {} years old".format(name, age))
```

    Your name is stern and you are 44 years old


Nothing here new except the fact that we are using the format() function inside the input function which is very much possible since the format function a string's function.

Another thing we are doing here is to cast the age data type from string to integer directly when getting the value input. If we try to input a string literal, this will result in an error.

now let's do the real thing since age is cast to an integer data type, we can compare it with another integer.


```python
if age >= 21:
    print("{}, your are a grown up adult now".format(name)) #print 1
else:
    print("{}, your are still a young boy".format(name)) #print 2
```

    stern, your are a grown up adult now


age >= 21 will be evaluated first, and if the expression is True then the block of code indented will be executed ( #print 1 ) else if the expression is False, then the code indented in the else statement will be executed ( #print 2 ).

If we don't specify the else statement, this won't result in an error. Nothing will happen


```python
if age >= 21:
    print("{}, your are a grown up adult now".format(name))
```

    stern, your are a grown up adult now


If we change the age to be smaller than 21 and then evaluate age without specifying the else statement, nothing will happen, because we have not specified what python should do in the case when age is smaller than 21.


```python
age = 13
if age >= 21:
    print("{}, your are a grown up adult now".format(name))
```

let's look at another example

For this program, we'll check if someone has the minimum age to have a driving license.

We first start by getting the name and age using the input( ) function


```python
name = input("Please enter your name :")
```

    Please enter your name :stern



```python
age = int(input("Please {}, enter your age :".format(name)))
```

    Please stern, enter your age :12



```python
if age >= 16:
    print("{} you can have a driving license".format(name)) #print 1
else:
    print("Unfortunately {}, you can't have a driving license for now come back in {} years".format(name,16-age)) #print 2
print("Execution of the program has ended...") #print 3
```

    Unfortunately stern, you can't have a driving license for now come back in 4 years
    Execution of the program has ended...


In this example the age that we have input is 12, python will check if 12 >= 16 which is False, this means that the block code (#print 1) indented right after the if statement will not be executed at all, so the execution stack (the order each code is executed in a program) will jump up to the else, and internally check if 12 >= 16, this will return True, the bock of code (#print 2)inside the else statement will be executed.

The first placeholder will obviously display the name and the second will display the remaining age in order to obtain the driving license by subtracting 16 (which is the minimum age to obtain the driving license) to the current age. If the age inputted would have been greater than 16, then the first condition would have been evaluated and the block code inside it would have been executed, the else condition with its block code will be ignored.

We have a third print (#print 3) at the end of the if-else statement, this print function will execute and printed after the if-else statement has finish its execution.

#### elif statement

sometimes we might need to evaluate more than two conditions, in this case, we added we use elif statement which stands for "else if".

To demonstrate the usage of elif statement, we are going to create a small guessing number program.

we are going to start by getting the input from the user


```python
your_number = int(input("Please guess a number between 1-10: ")) #input 1
```

    Please guess a number between 1-10: 9



```python
if your_number < 5:
    new_number = int(input("Please guess a higher number: ")) #input 2
    if new_number == 5:
        print("Well this round, you got it right congratulation!") #print 1
    else:
        print("Sorry you missed it again") #print 2
elif your_number > 5:
    new_number = int(input("Please guess a lower number: ")) #input 3
    if new_number == 5:
        print("Well this round, you got it right congratulation!") #print 3
    else:
        print("Sorry you missed it again") #print 4
else:
    print("you got it right on the first trial congratulation!") #print 5
print("Execution of the program has ended...") #print 6
```

    Please guess a lower number: 2
    Sorry you missed it again
    Execution of the program has ended...


The number we are trying to guess is 5 but we'll assume that we don't know the number, we first input a number from the user (#input 1) and store it in the variable your_number.

If your_number is less than 5, this condition is being evaluated as True the block code indented after "if your_number < 5:" will be executed. We will be immediately prompted to guess higher, after inputting and storing the number ( #input 2 ) another if and else statement will be used to evaluate the new_number ( this if-else statement is called a nested if-else statement ). new_number will be evaluated and compared to 5 using == operator (not to confused with assignment operator =), if it is equal to 5 then you won ( #print 1 ) else you have lost ( #print 2 ).

Else if your_number is greater than 5, this condition will be evaluated as True the block of code indented after  " elif your_number > 5 " will be executed. We will be prompt to guess lower ( #input 3 ), after inputting new_number another nested if and else statement will be used to evaluate new_number checking if it is equal to 5 or not, if it is then you have won ( #print 3 ) else you have lost ( #print 4 ).

lastly, if none of the two conditions( if and elif statement ) is evaluated to True, this means that we are only left with the condition that your_number is equal to 5, which is the number we are guessing consequently this means that we have won the game.

N.B: 
* Make sure that the indentation is correct because the control flow of any program written in Python relies on that.
* we can add as many elif statements as we wish depending on the conditions we have.
* nested conditional statements can get deeper and deeper. the deeper they go, the more complex they become which is not necessarily a good thing because clean code must be readable. I would suggest not to go beyond than 3 nested conditional statement. I have more than 3, consider refactoring your codes.

this example can be written in a much more efficient and concise way because they were a lot of duplicate code.


```python
your_number = int(input("Please guess a number between 1-10: "))
```

    Please guess a number between 1-10: 3



```python
if your_number == 5:
    print("you got it right on the first trial congratulation!")
else:
    if your_number > 5:
        new_number = int(input("Please guess lower"))
    else:   # if your_number < 5
        new_number = int(input("Please guess higher"))
    if new_number == 5:
        print("Well this round, you got it right congratulation!")
    else:
        print("Sorry you missed it again")
print("Execution of the program has ended...")
```

    Please guess higher8
    Sorry you missed it again
    Execution of the program has ended...


this a more concise way of writing the same program, we first start by checking if your_number is equal or different from 5 right away if it is equal we have won else we are going to check if your_number is greater or smaller than 5 to direct the user to guess lower or higher. We store the new input value in the new_number variable and straight away we check if it is equal to 5 or different to 5, if it is equal then we have won, else we have missed again. After all, the if and else statement is done executing, the last print statement will be executed.

#### conditional statement with logical operator

##### and

Python gives you the flexibility to use at the same time conditional statement with a logical operator in an expression, let's illustrate this by an example.


```python
age = int(input("Please enter your age: "))
```

    Please enter your age: 55



```python
if age >= 23 and age <= 85:
    print("You are eligible to become president of the republic of Burundi") #print 1
else:
    print("You can't be the president of the republic of Burundi") #print 2
```

    You are eligible to become president of the republic of Burundi


Here we are using 3 operators ( ">="," <=", "and" ). the operator with the highest precedence ( in this case ">=","<=" ) will be evaluated first then the operators with lower precedence ( "and" operator ) will be executed last. age >= 23 and age <= 85 will be executed first, the values from this comparison ( which are either False or True ) will be compared on their turn using the "and" operator. The expression will return True if True and True is compared, will return False if True and False, False and True or False and False is compared.

Coming back to our example, let's say we have inputted 55 as the value of age, 55 will be compared with 23 like this 55 >= 23 which will return True, then 55 will be compared to 85 as 55 <= 85 which will also return True. Now that we are done with the comparison of operators with high precedence, we are going to compare the two boolean value True and True which will return True, now the blog code indented will be executed ( #print 1 ).


```python
if (age >= 23) and (age <= 85):
    print("You are eligible to become president of the republic of Burundi")
else:
    print("You can't be the president of the republic of Burundi")
```

    You are eligible to become president of the republic of Burundi


To be clear and explicit about what is happening, we can rewrite the expression using parenthesis and the result should be exactly the same. Again, the parenthesis is unnecessary, it used to make the expression easier to read.

Another way to write the same example


```python
if 23 <= age <= 85:
    print("You are eligible to become president of the republic of Burundi")
else:
    print("You can't be the president of the republic of Burundi")
```

    You can't be the president of the republic of Burundi


This is the same example written differently without using the "and" operator. Here we are checking if the age is between the range of 23 included and 85 included. It will either return True or False depending on 
value of age.

if we don't want use <= or >= but instead to use < or >, we can write


```python
if 22 < age < 86:
    print("You are eligible to become president of the republic of Burundi")
else:
    print("You can't be the president of the republic of Burundi")
```

    You are eligible to become president of the republic of Burundi


By incrementing and decrementing the values of the variable (changed from 23 to 22 and 85 to 86) evaluated against age, we can avoid using >= or <= and still get the same result.

N.B: when python is comparing two operands using "and" operator, it's gonna stop the comparison as soon as one of the operands is False and return False.

##### or

Let's do the same example but this time using "or" operator instead of "and" operator.


```python
age = int(input("Please enter your age: "))
```

    Please enter your age: 12



```python
if (age < 23) or (age > 85):
    print("You can't be the president of the republic of Burundi")
else:
    print("You are eligible to become president of the republic of Burundi")
```

    You can't be the president of the republic of Burundi


If any of two expressions ( age < 23 ) ( age > 85 ) result in True, the whole statement will be True because remember for the "or" logical operator, if any of the operand being evaluated is True then the result will be True, if both are True then the result will be True also. But if both are False only then the result will be False.

N.B: when python is comparing two operands using "or" operator, it's gonna stop the comparison as soon as one of the operands is True and return True.

#### Evaluating empty strings and zero based numbers

In Python a non-empty string or a numerical number else than 0, will be evaluated as True.


```python
x = "x"
if x:
    print("hehe interestingly this is printing")
```

    hehe interestingly this is printing



```python
none_zero_number = 4
none_zero_float = 4.2
none_empty_string = "Text that will be casted to True"
```


```python
if none_zero_number:
    print("This printed because statement was evaluated to True")
```

    This printed because statement was evaluated to True



```python
if none_zero_float:
    print("This printed because statement was evaluated to True")
```

    This printed because statement was evaluated to True



```python
if none_empty_string:
    print("This printed because statement was evaluated to True")
```

    This printed because statement was evaluated to True


The statement went through and was able to print the text. We can also verify this by casting the string and numerical data type to a boolean data type using bool( ).


```python
print(bool(none_zero_number))
print(bool(none_zero_float))
print(bool(none_empty_string))
```

    True
    True
    True


now let's see with empty string and zero numerical


```python
zero_number = 0
zero_float = 0.0
empty_string_double_quote = ""
empty_string_single_quote = ''
none_variable = None
empty_list = []
empty_set = ()
empty_dict = {}
```


```python
print("zero_number: {},zero_float: {},empty_string_double_quote: {},empty_string_single_quote: {},none_variable: {},empty_list: {},empty_set: {},empty_set: {}".format(bool(zero_number),bool(zero_float),bool(empty_string_double_quote),bool(empty_string_single_quote),bool(none_variable),bool(empty_list),bool(empty_set),bool(empty_dict)))
```

    zero_number: False,zero_float: False,empty_string_double_quote: False,empty_string_single_quote: False,none_variable: False,empty_list: False,empty_set: False,empty_set: False


Here we can see that casting the numerical variables with value 0 result in False, same as the empty strings. For none_variable we have assigned it to a None keyword value which exactly means that we are assigning the variable to nothing. This will be interpreted as False when evaluating it with conditional. The other examples are an empty list, set and dictionaries since we have not discussed them yet, just think of them as containers and when those containers are empty they are evaluated as False.

let's use these in an example of the conditional statement


```python
if zero_float:
    print("This will not run")
else:
    print("This will definitely run")
```

    This will definitely run


Such evaluation can be helpful in the case when we are evaluating if an input value has been registered.


```python
your_city = input("Please enter the city where you live ")
```

    Please enter the city where you live 



```python
if your_city:
    print("You have entered {}, Thank you for telling us you city".format(your_city)) #print 1
else:
    print("You have not entered your city") #print 2
```

    You have not entered your city


In this example, nothing was inputted in the your_city variable, so when evaluating this variable False was returned and #print 2 was executed.

##### not

We can also use "not" operator to get the opposite of boolean value.


```python
print(not True)
print(not False)
```

    False
    True


We can use the not operator in a conditional statement and still get the same result, in this case, we only have to change the signs to the opposite signs ( < will be changed to > and > will be changed to < ).


```python
age = int(input("Please enter your age "))
```

    Please enter your age 33



```python
if not(age > 23) or not(age < 85):
    print("You can't be the president of the republic of Burundi")
else:
    print("You are eligible to become president of the republic of Burundi")
```

    You are eligible to become president of the republic of Burundi


Age will be compared to 23 and at the same time with 85. With the addition of the not operator, the results of those two comparisons will be inversed to the corresponding opposite boolean value. Depending on which value 
returned, the indented block code will be executed.

##### in

In the following example, we will be using the "in" keyword


```python
sentence = "On a beach in Hawaii"
```


```python
the_char = input("Please enter one character: ")
if the_char in sentence:
    print("yeap, the character {} was found in the sentene".format(the_char)) #print 1
else:
    print("the character {} was not found".format(the_char)) #print 2
```

    Please enter one character: x
    the character x was not found


In this example above, if the character inputted is not found in the sentence, the evaluation will be False and if the character inputted is found in the sentence, True will be returned. The indented block code will be executed accordingly.

##### not in

we can also use "not in" operator, just like this.


```python
sentence = "On a beach in Hawaii"
```


```python
the_char = input("Please enter one character: ")
if the_char not in sentence:
    print("the character {} was not found".format(the_char)) #print 1
else:
    print("yeap, the character {} was found in the sentene".format(the_char)) #print 2

```

    Please enter one character: B
    the character B was not found


Since we are using not, In this case, we are going to swap the two print statement.

#### Conditional Expressions

conditional expressions are like conditional statements, the only difference is that they return a value that we can evaluate or assign to a variable using only one line of code.


```python
raining = False
print("We will go to the {}".format("beach" if not raining else "library"))
```

    We will go to the beach


In the example above, "if not raining" will be evaluated first, since raining is assigned to False the "not" operator will change it to True then the value in front of the if statement("beach" in our case) will be returned and used in the format( ).

Now let's see the inverse.


```python
raining = True
print("We will go to the {}".format("beach" if not raining else "library"))
```

    We will go to the library


Now raining variable is assigned to True, the if statement will be evaluated and will result in False which means that code in the else statement will be returned ("library" in this case).

Another thing we can do is to directly assign the value to a variable


```python
age = 2
age_range = "adult" if age >=21 else "youngster"
```


```python
print("You are {}".format(age_range))
```

    You are youngster


In this example, after the if statement is evaluated, either "youngster" or "adult" will be returned and assigned to age_range.

#### pass statement

Occasionally, we might find ourselves in a situation where we need to add a placeholder to our code where we will write future code without raising an error. This is called code stubbing. In python, pass is a keyword for that.


```python
raining = True
if raining:
    pass
else:
    print("Let's go to the beach")
```

running this code nothing will happen, why? because the if the statement was passed but after reaching the indented block code, python read the pass keyword as "just pass this line of code but do not raise an error". In these terms, pass keyword will act as a placeholder.

### Challenge 1

Challenge time!!! For your first challenge, we are going to use the challenge from the previous blog post and add to it some useful informations.

The rules of this challenge are:
   * copy the codes from previous challenge
   * if the BMI is below 18.5, we should get this message: "you are underweight, you should add more weight"
   * if the BMI between 18.6 to 23.0, we should get this message: "you are healthy, maintain that weight"
   * if the BMI between 23.1 to 27.5, we should get this message: "you are overweight, you should consider reducing weight"
   * if the BMI is above 27.5, we should get this message: "you are obese, you should seriously consider reducing weight"
   * you are allowed to use any operators of your choice

### Challenge 2

For your second challenge, we are to going to input the name, age and GPA of a prospective student.

Depending on their age and GPA, the program shall grant a scholarship to the students who are applying for a scholarship.

The rules of this challenge are:
* the program should only granted scholarship to the students with GPA of 3.00 and above aged between 18 years to 25 years old. 
* if there is a scholarship granted, print this message: "Congratulation {name}, from the fact that you have a GPA of {gpa} and are {age} years old, you have received a scholarship".
* if there is no scholarship granted, reply politely with the reason why the scholarship was not granted.
* the reasons can be:
    - Your age. print this message: "Unfortunately {name}, even though you have the minimum GPA requirement but are {age} years old we are in the incapacity of giving you a scholarship".
    - Your GPA. print this message: "Unfortunately {name}, even though you are in the right age range but with a GPA of {gpa} we are in the incapacity of giving you a scholarship".
    - Both your GPA and age. print this message: "Unfortunately {name}, with a GPA of {gpa} and the fact that you are {age} years old we are in the incapacity of giving you a scholarship".
* the GPA should be printed as a float number and have 2 decimal digits with 5 as left alignment width, age should be an integer.
* This last rule is optional, use "and" and "not" in your conditional statement.

Now go away and attempt these 2 challenges by your own, ONLY after you have finished or tried them you should come back and compare your solutions with mine.

### Solution to Challenge 1

Since we have already calculated the BMI, let's copy paste the solution from the last post.


```python
weight = float(input("Please enter your weight "))
```

    Please enter your weight 35



```python
height = float(input("Please enter your height "))
```

    Please enter your height 1.65



```python
BMI = weight/(height**2)
```


```python
print("The BMI of a person with {} kg and {} m is {:6.3f}".format(weight,height,BMI))
```

    The BMI of a person with 35.0 kg and 1.65 m is 12.856



```python
if BMI <= 18.5:
    print("you are underweight, you should add some weight")
elif 18.6 <= BMI <= 23.0:
    print("you are healthy, maintain that weight")
elif 23.1 <= BMI <= 27.5:
    print("you are overweight, you should reduce some weight")
else:
    print("you are obese, you should reduce a lot of weight")
```

    you are underweight, you should add some weight


Now that we have BMI we use the conditional statement to evaluate it, let's print the appropriate messages.

### Solution to Challenge 2

We'll first start by getting the name and store it in a variable, after that we're going to input the age and cast it to an integer. after it will be the turn of the GPA and we'll cast it to a float.


```python
name = input("What is is your name? ")
```

    What is is your name? Ondari



```python
age = int(input("What is your age? "))
```

    What is your age? 22



```python
gpa = float(input("Enter your GPA using this format X.XX: "))
```

    Enter your GPA using this format X.XX: 3.65



```python
if 18 <= age <= 25:
    if gpa >= 3.00:
        print("Congratulation {}, from the fact that you have a GPA of {:5.2f} and are {} years old, you have received a scholarship".format(name,gpa,age))
    else:
        print("Unfortunately {}, even though you are in the right age range but with a GPA of {:5.2f} we are in the incapacity of giving you a scholarship".format(name,gpa))
else:
    if gpa >= 3.00:
        print("Unfortunately {}, even though you have the minimum GPA requirement but are {:5.2f} years old we are in the incapacity of giving you a scholarship".format(name,age))
    else:
        print("Unfortunately {}, with a GPA of {:5.2f} and the fact that you are {} years old we are in the incapacity of giving you a scholarship".format(name,gpa,age))

```

    Congratulation Ondari, from the fact that you have a GPA of  3.65 and are 22 years old, you have received a scholarship


Now that we have all the inputs, we are going to first check if age is in the range of 18 and 25 as required to be granted a scholarship. If we are in this range, we compare the GPA to the minimum required GPA of 3.00, if it is greater or equal to 3.00 this means that we have secured all the required to get a scholarship. If one or two of the conditions fail, the program will print the reason why the scholarship is not given accordingly.

Optionally, we can rewrite the program using "and" with "not" operators


```python
if (age >= 18) and (age <= 25):
    if not(gpa <= 3.00):
        print("Congratulation {}, from the fact that you have a GPA of {:5.2f} and are {} years old, you have received a scholarship".format(name,gpa,age))
    else:
        print("Unfortunately {}, even though you are in the right age range but with a GPA of {:5.2f} we are in the incapacity of giving you a scholarship".format(name,gpa))
else:
    if not(gpa >= 3.00):
        print("Unfortunately {}, even though you have the minimum GPA requirement but are {} years old we are in the incapacity of giving you a scholarship".format(name,age))
    else:
        print("Unfortunately {}, with a GPA of {:5.2f} and the fact that you are {} years old we are in the incapacity of giving you a scholarship".format(name,gpa,age))

```

    Congratulation george,from the fact that you have a GPA of  3.66 and your are 22 years old, you have received a scholarship


This is two of the many ways to approach this challenge, we could have checked first the GPA instead of the age the result would be the same, another way to do it would have been with the usage of the "or" operator.

### Conclusion

In this tutorial post, we have seen the conditional statement with the pass statement, which makes it possible to execute a statement or block of code based on the result of an evaluation. All control flow structures are crucial in order to write complex Python code. In the upcoming post, 
we will discuss loops in python.

Thank you for reading this tutorial. Hope you have learned one or two things. If you like this post, please subscribe to stay updated with new posts and if you have a thought or a question, I would love to hear it by commenting below.
