---
title:  "Data types, variables and operators in Python"
image: /assets/post_images/py_tuto.jpeg
excerpt_separator: <!-- more -->
tags:
- python
- tutorial
- programming
---

In this post, we are going to discover all you need to know about data types, operators and variables in python. I'll be using jupyter
notebook, feel free to use any text editor of your choice but I do highly recommend to use jupyter especially if you are interested in data science. I also suggest to type the codes and run it on your system and see your results. This tutorial post is entirely written in Jupyter and exported as a markdown file to Jekyll.<!-- more -->

### print function

We'll start this tutorial by printing the traditional "hello world" example using the print pre-written function. A function is a group of code that performs a specific task. In this case, the function displays anything written between its parenthesis.


```python
print(34)
```

    34


To print text, we have to add double or single quotes


```python
print("Hello world")
```

    Hello world



```python
print('Hello world')
```

    Hello world


We can also print special character


```python
print("$%#@")
```

    $%#@


The print function can be empty, in this instance nothing will be displayed


```python
print("")
```

    


### Data type

data type simply means the type of information stored in computer memory. In python, we have several data types, but for this introductory post, we are going to talk about 3 of them: numerical, string and boolean data type.

#### Numerical data type

##### integers

Integers are numbers with no decimal part in it. In Python 3, there is no limit on how long an integer can be it's only constrained by how memory your system has.


```python
print(1234222)
```

    1234222



```python
print(4)
```

    4


integers can be written in binary, octal, and hexadecimal form.

octal


```python
print(0o10)
print(0o23)
```

    8
    19


binary


```python
print(0b10)
print(0b011110)
```

    2
    30


Hexadecimal


```python
print(0x49)
print(0x10)
```

    73
    16


you can learn more about binary [here](https://en.wikipedia.org/wiki/Binary_number), octal [here](https://en.wikipedia.org/wiki/Octal) and hexadecimal [here](https://en.wikipedia.org/wiki/Hexadecimal)

##### float

floats are numbers with a decimal part. Optionally, the character e or E followed by a positive or negative integer can be appended to it to represent mathematical notation.


```python
print(0.4)
```

    0.4



```python
print(5.1e4)
```

    51000.0



```python
print(4.2e-4)
```

    0.00042


##### complex

Complex numbers are written as "real part"+"imaginary part".j


```python
print(3+2j)
```

    (3+2j)


#### string data type

As we have seen at the beginning, a string is a sequence of characters. it is part of the sequence data type and is used to represent a single, special character or a sequence of characters which form text. String literals are delimited using either single, double or triple quotes.


```python
print("a")
print("This a sequence of characters")
print('I am a delimited by single quotes')
print("I am a delimited by double quotes")
print("""I am a delimited by single quotes""")
```

    a
    This a sequence of characters
    I am a delimited by single quotes
    I am a delimited by double quotes
    I am a delimited by single quotes


But we should not use single quote and double quotes delimiter at the same time this will result in an error.


```python
print('This is wrong")
```


      File "<ipython-input-31-327641873bad>", line 1
        print('This is wrong")
                              ^
    SyntaxError: EOL while scanning string literal



N.B: Errors are your best friend, it is a way python is just telling you that it does not understand what you have written. It also gives useful information on where you have gone wrong by using an arrow point at where you've made a mistake and also tells you what type of error it is. In the case above, It is saying that we have made a SyntaxError. There are many more types of errors in Python and we are going to discuss how to handle them gracefully in the upcoming posts using exception handling. 

if we want to use simple and double quotes at the same time, we have these options:
1. to use a backslash key also called the escape sequence


```python
print("this is kevin\'s house")
```

    this is kevin's house


I explain in detail about the escape sequences in python toward the end of the blog

2. Delimit the string by using double or single quote


```python
print("This is kevin's house")
print('Marcus said:"freedom to the people"')
```

    This is kevin's house
    Marcus said:"freedom to the people"


3. We can also delimit the string with triple quotes this allows to use double and single quote in the same string.


```python
print(""" The chief said:"No, this is Bob's house" """)
print(''' The chief said:"No, this is Bob's house" ''')
```

    The chief said:"No, this is Bob's house" 
    The chief said:"No, this is Bob's house" 


#### boolean data type

Boolean data type, named after the mathematician George Boole, is a type of information that can be expressed as True or False only.

As you will see in upcoming tutorials, expressions in Python 
are often evaluated in Boolean context, meaning they are 
interpreted to represent truth or falsehood.
A value that is true in Boolean context is sometimes said 
to be “truthy,” and one that is false in Boolean context 
is said to be “falsy.” (You may also see “falsy” spelt “falsey.”)

The “truthiness” of an object of the Boolean type is self-evident: Boolean objects that are equal to True are truthy (true), and those equal to False are falsy (false). But non-Boolean objects can be evaluated in a Boolean context as well and determined to be true or false.


```python
print(True)
print(False)
```

You will learn more about the evaluation of objects using Booleans when you encounter logical operators and conditional statements in the upcoming tutorial.

To conclude on this topic, these are 3 basic data types they are many more and we will discuss them as we move to the more advanced topics in python

### Type function

The type function in python returns the data type of a given value is. This function is very useful when we don't know or when we are in doubt of the data type of a given value. we are using the print( ) function to display the data type which was returned by the type( ) function.


```python
print(type(44))
print(type(0o10))
print(type(0b10))
print(type(0x10))
```

    <class 'int'>
    <class 'int'>
    <class 'int'>
    <class 'int'>



```python
print(type(5.1e4))
print(type(8.9e-4))
```

    <class 'float'>
    <class 'float'>



```python
print(type(3+2j))
```

    <class 'complex'>



```python
print(type(True))
print(type(False))
```

    <class 'bool'>
    <class 'bool'>



```python
print(type(""" The chief said:"No, this is Bob's house" """))
print(type("I am delimited by double quotes"))
```

    <class 'str'>
    <class 'str'>


### comments

Comments in Python are part of the code that will be ignored while running(executing) the code. The comments can be useful when giving some explanation to code or when we want to ignore codes without deleting them.


```python
print("This will print")
#comment start with # character. all the text after # will be ignored
#this will be not print
#print("hello word")
print("This will also print")
```

    This will print
    This will also print


### Variables in python

To easily find and recall codes stored in computer memory, we use variables. think of variables as boxes that contain the values and those boxes have labels on them to easily differentiate them. the labels correspond to the names given to the variables. The values(right of the = sign) are stored into variables(left of the = sign). The act of giving a value to a variable using the = sign is called declaring a variable


```python
name = "semasuka"
print(name)
```

    semasuka


Here "name" is the variable name, the value is "semasuka", we can print the value stored in the variable by calling
the name in this case and the value of the name will be printed.

In python, there are conventions to keep in mind while naming a variable:
1. Don't use names that are too long
2. The name should explicitly define the value, for example, avoid calling your variable naive names like x or a, you should be specific as possible
3. The variable name can't start with a number or contain any special character($%&...) except underscore ( _ ) otherwise, this will result in a syntax error
4. Since space is a special character, the pythonic way to represent a space in a variable name is by using the _
5. Python is case-sensitive, for example naming a variable greeting and another one named Greeting, these variables will be considered as two be different variable


```python
#The following code will not run, you should avoid declare variable like this
number% = 8
1_number = 8
this_name_is_too_long_to_be_used = "Not to be used"
x = 34
weshouldputspacebetween = "Not to be used"
```


```python
#the pythonic way to name a variable
number_8 = 8
my_name = "semasuka"
the_total = 34
first_second_name = "Semasuka Stern"
_second_name = "semasuka"
chanelle_no_5 = "urus"
greeting = "Hello"
Greeting = "hi!"
```


```python
#now we can print the varibles
print(number_8)
print(my_name)
print(the_total)
print(first_second_name)
print(greeting)
print(Greeting)
```

    8
    semasuka
    34
    Semasuka Stern
    Hello
    hi!



```python
car = "Porsche"
print(car)
```

    Porsche



```python
car = "Benz"
print(car)
```

    Benz


Also, another thing to mention is the fact that car has been assigned to a new value (Benz), so the old value (Porsche) will no longer be considered.

#### concatenation

We can add together two variables using the + symbol this process is called concatenation.


```python
print("hello"+"world")
```

    helloworld


we can add spacing for a better readability


```python
print("hello"+" "+"world")
```

    hello world



```python
greetings = "hello"
name = "stern"
```


```python
print(greetings+" "+name)
```

    hello stern



```python
city = "Bujumbura"
country = "Burundi"
country_city = country + " " + city
print("The capital city of "+country+" is "+city)
print("country + city: "+country_city)
```

    The capital city of Burundi is Bujumbura
    country + city: Burundi Bujumbura


We can not concatenate two variable with different data types


```python
age = 55
my_name = "semasuka"
print(type(age))
print(type(my_name))
```

    <class 'int'>
    <class 'str'>



```python
print("My name is "+my_name+" and I am "+age+" years old")
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-74-217a319b30e3> in <module>
    ----> 1 print("My name is "+my_name+" and I am "+age+" years old")
    

    TypeError: can only concatenate str (not "int") to str


We are encountering a TypeError, this is due to the fact we want to add together an integer data type to a string data type. To solve this problem, we have to change the integer data type into a string data type, this process is called casting. 

#### casting

casting is transforming one variable data type into another compatible data type.


```python
print("My name is "+my_name+" and I am "+str(age)+" years old")
```

    My name is semasuka and I am 55 years old


So what I did there was to change(cast) the age data type from integer to a string data type using str( ) function.


```python
some_age = 45
print(type(some_age))
some_age = str(some_age)
print(type(some_age))
```

    <class 'int'>
    <class 'str'>


We can clearly see that some_age has changed from int(integer) data type to str(string). 

### input variable

We can input value from the user using the input( ) function and store it in a variable

When we run it, as small box appear that's where we will input the value 


```python
name = input("Please enter your name")
greetings = "hello"
```

    Please enter your nameyve



```python
print(greetings+" "+name)
```

    hello yve



```python
weight = input("Please enter you your weight in kilogram ")
```

    Please enter you your weight in kilogram 74



```python
print("Your weight is "+weight)
```

    Your weight is 74


N.B: Remember all the input are stored as string even though they are numberical input


```python
print(type(weight))
```

    <class 'str'>


### operator

#### arithmetic operators

To perform basic arithmetic operations on variables in python, we use the following


```python
first_num = 12
sec_num = 3
print("Addition of 12 and 3 is "+str(first_num+sec_num))
print("Substraction of 12 and 3 is "+str(first_num-sec_num))
print("Multiplication of 12 and 3 is "+str(first_num*sec_num))
print("division of 12 and 3 is "+str(first_num/sec_num))
print("Truncation Division (also known as floordivision or floor division) of 12 and 3 is "+str(first_num//sec_num))
print("12 exponential 3 is "+str(first_num**sec_num))
print("modulus of 12 and 3 is "+str(12%3))
```

    Addition of 12 and 3 is 15
    Substraction of 12 and 3 is 9
    Multiplication of 12 and 3 is 36
    division of 12 and 3 is 4.0
    Truncation Division (also known as floordivision or floor division) of 12 and 3 is 4
    12 exponential 3 is 1728
    modulus of 12 and 3 is 0


To perform addition we use +, subtraction we use -, multiplication we use *, the division we use /, truncation division we use //. Truncation division returns the result as an integer and ignores the decimal part on the result, exponential we use ** and lastly modulus we use %. Modulus returns the remainder of the division.

#### operator precedence

As it is in math, when different operators are found in the same expression, the operator with the highest precedence will be executed first. this is due to the operator precedence. In Python, it is the same.


```python
print(6 + 15/3 - 4*12)
```

    -37.0


here we might think that the answer would 6 plus 15 which is 21 then divide it by 3 then minus 4 and multiply by 12 to give a result of 36 but the answer is -37. why? well in Python the operators with the highest precedence will be calculated first (in this case division and multiplication will be evaluated first) and then the operators with the lowest precedence. So 15 divided by 3 will be executed first at the same time as 4 times 12, the result from 15 divided by 3 will be then added to 6 which is 11 then substracte 48(4 times 12) to give the final result of -37.

To remove ambiguity caused by operator precedence, we can use parathesis to group together expressions to be executed first.


```python
print((((6+15)/3)-4)*12)
```

    36.0


To learn more about operator precedence in Python, visit [this](https://www.tutorialspoint.com/python/operators_precedence_example.htm) link.

#### assignment operator

As we have already seen, to declare a variable in python we use the = sign, this is called assignment of variables


```python
price = 39.99
```

What if there is an increase in price of 10, we can either declare a new price value and assign it to the variable


```python
price = 49.99
```

But this is not convenient

A better solution is to add 10 to the existing price, then assign it to itself


```python
price = price + 10
print(price)
```

    49.99


There is a short hand form of writing this, which is much more prefered


```python
price = 39.99
price += 10
print(price)
```

    49.99


we can do substraction


```python
price = 39.99
price -= 10
print(price)
```

    29.990000000000002


division


```python
price = 39.99
price /= 10
print(price)
```

    3.999


truncation division


```python
price = 39.99
price //= 10
print(price)
```

    3.0


multiplication


```python
price = 39.99
price *= 2
print(price)
```

    79.98


exponential


```python
price = 39.99
price **= 2
print(price)
```

    1599.2001000000002


modulus


```python
price = 39.99
price %= 10
print(price)
```

    9.990000000000002


#### comparison operator

Comparison operators are used for comparing values, It either returns True or False according to the condition. An operand is a name given to  each value that is being compared

* ">"    Greater that - True if the left operand is greater than the right
* "<"    Less that - True if the left operand is less than the right
* "=="    Equal to - True if both operands are equal
* "!="    Not equal to - True if operands are not equal
* ">="    Greater than or equal to - True if the left operand is greater than or equal to the right
* "<="    Less than or equal to - True if the left operand is less than or equal to the right


```python
num_1 = 45
num_2 = 99
print(num_1 > num_2)
print(num_1 < num_2)
print(num_1 >= num_2)
print(num_2 <= num_2)
```

    False
    True
    False
    True



```python
num_3 = 32
num_4 = 32
print(num_3 == num_4)
print(num_3 != num_4)
```

    True
    False


#### logical operator

Logical operators are "and", "or", "not" operators. These operators evaluate the operands (the values that are being compared) and return True or False.

* "and" return True if both the operands are true
* "or" return True if either of the operands is true
* "not" return True if the operand is False (complements the operand)


```python
#let's use this qualification to define a person
is_boy = False
is_girl = True
is_teen = True
print(is_boy and is_girl)
print(is_girl and is_teen)
```

    False
    True



```python
is_refugee = False
is_unemployed = True
print(is_refugee or is_unemployed)
```

    True



```python
is_refugee = True
is_unemployed = True
print(is_refugee or is_unemployed)
```

    True



```python
girl = True
print(not girl)
```

    False


#### membership operator

"is" and "is not" are the identity operators in Python. They are used to check if two values (or variables) are located on the same part of the memory. Two variables that are equal does not imply that they are identical.

* "is" return True if the operands are identical (refer to the same object in memory).
* "is not" return True if the operands are not identical (do not refer to the same object in memory).


```python
num_1 = 45
num_2 = 45
string_1 = "water"
string_2 = "water"
print(num_1 is num_2)
print(string_1 is string_2)
```

    True
    True


Now let's see an example using a list, We'll see list data type in depth in the upcoming post but bear with me for now and just simply put, a list data type is a container that can store many other types at once.


```python
the_list_1 = ["water",5,True]
the_list_2 = ["water",5,True]
```


```python
print(the_list_1 in the_list_2)
```

    False


As we can see, False was returned. This means that Python stores the_list_1 in a different part of memory as the_list_2 even though the values are the same. We can conclude that Python stores some data type differently.

#### identify operator

"in" and "not in" are the membership operators in Python. They are used to test whether a value or variable is found in a sequence (string, list, tuple, set and dictionary)data type. List, tuple, set and dictionaries will be explained in the upcoming posts.

* "in" return True if value/variable is found in the sequence.
* "not in" return True if value/variable is not found in the sequence.


```python
the_sentence = "We will search a specific sequence of characters in here"
word_1_to_search = "will"
word_2_to_search = "a"
word_3_to_search = "quence"
word_4_to_search = "not in the sentence"
```


```python
print(word_1_to_search in the_sentence)
```

    True



```python
print(word_2_to_search in the_sentence)
```

    True



```python
print(word_3_to_search in the_sentence)
```

    True



```python
print(word_4_to_search in the_sentence)
```

    False



```python
print(word_4_to_search not in the_sentence)
```

    True


N.B: All remember that Python is case sensitive


```python
the_sentence = "python is so much fun"
lowercase_word = "python"
uppercase_word = "Python"
```


```python
print(lowercase_word in the_sentence)
print(uppercase_word in the_sentence)
print(uppercase_word not in the_sentence)
```

    True
    False
    True


### String functions

#### the escaping character

String literals can be enclosed(delimited) by single quote('), double quotes(") or triple quotes("""). if we use single quote to delimit a string character and use at the same time another single quote in the string literal, this will result in an error.


```python
print('let's learn Python')
```


      File "<ipython-input-51-05f322a64dab>", line 1
        print('let's learn Python')
                   ^
    SyntaxError: invalid syntax



In order to use two single quotes as string delimiter and one inside the string literal, we'll have the backslash character.


```python
print('let\'s learn Python')
```

    let's learn Python


The backslash ( \ ) character is used to escape characters that otherwise have a special meaning, such as newline, backslash itself, the quote character, or tab.


```python
print('this is kevin\'s house')
print("****") #this stars are used to separate the output
print("Your coach said:\"You need to exercise more often\"")
print("****")
print("You will send this letter via DHL\\FEDEX")
print("****")
print("If we want to break \
the string \
in a new \
line")
print("****")
print("There is a tab\tcharacter")
print("****")
print("This text have been splited\ninto\ndifferent lines")
print("****")
print("""This is
splitted
in
several
lines""")
```

    this is kevin's house
    ****
    Your coach said:"You need to exercise more often"
    ****
    You will send this letter via DHL\FEDEX
    ****
    If we want to break the string in a new line
    ****
    There is a tab	character
    ****
    This text have been splited
    into
    different lines
    ****
    This is
    splitted
    in
    several
    lines


We can tell python to handle the string as a raw string literal by adding an r or R at the beginning of the string. Python will ignore all the escape sequence found within the string


```python
print(r"this is kevin\ house")
print(r'this is kevin\'s house')
print(r"""this should have been\t a tab space""")
print(r"You will send this letter via DHL\\FEDEX")
print(R"This text will not be splited\ninto\ndifferent lines")
```

    this is kevin\ house
    this is kevin\'s house
    this should have been\t a tab space
    You will send this letter via DHL\\FEDEX
    This text will not be splited\ninto\ndifferent lines


You can check out more about escape characters [here](https://docs.python.org/2.0/ref/strings.html).

#### slicing a string

We can cut and get a substring from a string using Python slicing notation. Slicing works on all sequence data type but because we have not seen the rest of the sequence data types we are going to use it on the string only on this post.

Because the string data type is a sequence of character, we can actually extract a specific character at a specific index starting from index 0 which represent the first character, index 1 represents the second character and so on. we will use this format: variable[index number]


```python
dog_breed = "labrador"
print(dog_breed[3])
print(dog_breed[0])
print(dog_breed[6])
```

    r
    l
    o


to get the last character of a string, we use index -1


```python
print(dog_breed[-1])
```

    r


we can get a range of a world starting from the first character to a specific index using this format: variable[index_start: index_stop]. The character at the index_stop is not included


```python
print(dog_breed[0:5])
```

    labra


The character at index 5 which is character "d" is not printed

This can be also be written like this.


```python
print(dog_breed[:5])
```

    labra


we can also start from a specific index to the last character


```python
print(dog_breed[3:])
```

    rador


we can also print using negative indexes


```python
print(dog_breed[-4:-1])
```

    ado


In the example above counting from the last character, we will print the character number 4 to number 1 which is the last character

we can skip characters by introducing a third factor.


```python
print(dog_breed[0:8:2])
```

    lbao


Every second character(starting from 0) is printed


```python
spaced_number = "1, 2, 3, 4, 5, 6, 7, 8, 9"
print(spaced_number[0:25:3])
```

    123456789


For this example, every third character is printed the rest is ignored

This can be rewritten like this


```python
print(spaced_number[::3])
```

    123456789


#### duplicate string

We can use * to duplicate string


```python
name = "Yvan"
print(name*5)
```

    YvanYvanYvanYvanYvan



```python
name = "Yvan"
dupli_name = name * 5
print(dupli_name)
```

    YvanYvanYvanYvanYvan


#### string formating using format( )

As we have already seen, this example below will not run because of incompatibility of data type


```python
age = 84
print("I am "+age+" years old")
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-5-7eea149da213> in <module>
          1 age = 84
    ----> 2 print("I am "+age+" years old")
    

    TypeError: can only concatenate str (not "int") to str


To fix this error, we can either caste variable age or print using a function called format ( )

format( ) method takes any number of parameters. But, is divided into two types of parameters:

1. Positional parameters - list of parameters that can be accessed with the index of the parameter inside curly braces {index}
2. Keyword parameters - list of parameters of type key=value, that can be accessed with the key of the parameter inside curly braces {key}

1. Positional parameters


```python
age = 84
print("I am {0} years old".format(age))
print("I leave in {0}".format("Africa"))
```

    I am 84 years old
    I leave in Africa


Here the curly { } act as a placeholders in the string literal, and the value that will be placed there will be the variable(s)/value(s) in the .format( ). the numbers written in the { }, are the position of the variable(s)/value(s) in the .format( ) and dont forget that in Python, we count from 0.


```python
husband_car = "benz"
wife_car = "porshe"
print("the husband drives a {0} and the wife drives a {1}".format(husband_car,wife_car))
```

    the husband drives a benz and the wife drives a porshe



```python
city = "Kinshasa"
country = "DRC"
print("I am {0} years old and I live in the capital of {1}, {2}".format("89",country,city))
```

    I am 89 years old and I live in the capital of DRC, Kinshasa


We can omit the indexes in the { } like this


```python
city = "Kinshasa"
country = "DRC"
print("I am {} years old and I live in the capital of {}, {}".format("89",country,city))
```

    I am 89 years old and I live in the capital of DRC, Kinshasa


Python will automatically assign the placeholders to the matching variable(s)/value(s). always remember that the number of variables and values must be greater than the number of placeholders, else this will not run


```python
city = "Kinshasa"
country = "DRC"
print("I am {} years old and I live in the capital of {}, {}".format("89",country))
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-6-b94d74797fcc> in <module>
          1 city = "Kinshasa"
          2 country = "DRC"
    ----> 3 print("I am {} years old and I live in the capital of {}, {}".format("89",country))
    

    IndexError: tuple index out of range


We can also have additional information beside the index


```python
print("Hello {}!, your account balance is ${:9.3f}".format("John",230.8926))
```

    Hello John!, your account balance is $  230.893


in the second placeholder, we can see {:9.3f}. 
* The part before the "." 9 specifies the minimum width/padding of the number (230.2346) including "." In this case, 230.8926 is allotted a minimum of 9 places including the "." if no alignment option is specified, it is aligned to the right of the remaining spaces. (For strings, it is aligned to the left.
* The part after the "." (3) truncates the decimal part (8926) upto the given number. In this case, 8926 is truncated after 3 places. Remaining numbers (26) is rounded off outputting 893.
* f specifies the format is dealing with a float number. If not correctly specified, it will give out an error.

refer to this tableau to see the different formatting of numbers using format( )

![number_format](/blog/assets/post_cont_image/num_format.png)

2. keyword arguments

The keyword arguments work almost the same as positional parameters the difference is that instead of using the index position in the placeholder, we use the variable name.


```python
print("My first name is {first_name} and my last name is {last_name} and I work as a {profession}".format(first_name="semasuka",last_name="stern",profession="software engineer"))
```

    My first name is semasuka and my last name is stern and I work as a Software engineer


The rest is exactly the same as the positional parameters.

##### more example


```python
print("There is {0} days, in the month of {1}, {2}, {3}, {4}, {5}, {6} and {7}".format(31,"January","March","May","July","August","October","December"))
```

    There is 31 days, in the month of January, March, May, July, August, October and December



```python
print("There is {} days, in the month of {}, {}, {}, {}, {}, {} and {}".format(31,"January","March","May",
                                                                                      "July","August","October","December"))
```

    There is 31 days, in the month of January, March, May, July, August, October and December


Depending on which order we want the value to be displayed, we can interchange the index in the placeholder


```python
print("During the month of {2}, {1}, {3}, {5}, {6}, {7} and {4}. there is {0} days, ".format(31,"January","March","May","July","August","October","December"))

```

    During the month of March, January, May, August, October, December and July. there is 31 days, 


integer arguments


```python
print("The number is: {:d}".format(443))
```

    The number is: 443


float arguments


```python
print("The float number is: {:f}".format(123.4567898))
```

    The float number is: 123.456790


octal, binary and hexadecimal format


```python
print("bin form of 30: {0:b}, oct form of 30: {0:o}, hex form of 30: {0:x}".format(30))
```

    bin form of 30: 11110, oct form of 30: 36, hex form of 30: 1e


integer numbers with a minimum width


```python
print("{:5d}".format(12))
```

       12


width will be ignored if used with numbers longer than the width


```python
print("{:2d}".format(1234))
```

    1234


padding for float numbers


```python
print("{:8.3f}".format(12.2346))
```

      12.235


integer numbers with minimum width filled with zeros


```python
print("{:05d}".format(12))
```

    00012


padding for float numbers filled with zeros


```python
print("{:08.3f}".format(12.2346))
```

    0012.235


Explanations of the examples above,

* in the first example, {:5d} takes an integer argument and assigns a minimum width of 5. Since no alignment is specified, it is aligned to the right.
* In the second example, you can see the width (2) is less than the number (1234), so it doesn't take any space to the left but also doesn't truncate the number.
* Unlike integers, floats has both integer and decimal parts. And, the minimum width defined to the number is for both parts as a whole including ".".
* In the third statement, {:8.3f} truncates the decimal part into 3 places rounding off the last 2 digits. And, the number, now 12.235, takes a width of 8 as a whole leaving 2 places to the left.
* If you want to fill the remaining places with zero, placing a zero before the format specifier does this. It works both for integers and floats: {:05d} and {:08.3f}.

##### signed numbers

show the + sign


```python
print("{:+f} {:+f}".format(12.23, -12.23))
```

    +12.230000 -12.230000


show the - sign only


```python
print("{:-f} {:-f}".format(12.23, -12.23))
```

    12.230000 -12.230000


show space for + sign


```python
print("{: f} {: f}".format(12.23, -12.23))
```

     12.230000 -12.230000


##### The operators "<", "^", ">" and "="

 these operators are used for alignment when assigned a certain width to the numbers.

integer numbers with right alignment


```python
print("{:5d}".format(12))
```

       12


float numbers with center alignment


```python
print("{:^10.3f}".format(12.2346))
```

      12.235  


float numbers with center alignment


```python
print("{:=8.3f}".format(-12.2346))
```

    - 12.235


integer left alignment filled with zeros


```python
print("{:<05d}".format(12))
```

    12000


For the last example, Left alignment filled with zeros for integer numbers can cause problems as the 3rd example which returns 12000, rather than 12.

For more information, refer to this table

![sign_format](/blog/assets/post_cont_image/si_format.png)

N.B: I did not talk about the % string literal formatting intentionally because it is depreciated and is not recommended to use in Python 3. 

### Challenge

Challenges are problems that I will be giving you at the end of each tutorial post to solidify your understanding of what you just learned, for your first challenge, you will be to create a small program to calculate and print the Body Mass Index(BMI) and these are the rules to keep in mind:

* The program should get value through the input ( ) function
* Don't use the truncation division while calculating the BMI
* Should display the BMI using format ( )
* When displaying the BMI, use this format "The BMI of a person with XX kg and X.XX m is XX.XXX"
* BMI should a float and have 3 decimal digits with 6 as left alignment width

NB: the formula of BMI is equal to the weight in kilogram divided by the square of the height expressed in meters.

Now go ahead try this challenge on your own, ONLY after you have finished or tried the challenge come back and compare your solution with mine.

### solution

We first get the weight in kilogram of the person


```python
weight_str = input("Please enter your weight ")
```

    Please enter your weight 77


Then the height


```python
height_str = input("Please enter your height ")
```

    Please enter your height 1.88


Since we know that any input is a string data type, we need to cast it in order to use it as numerical data type and do arithmetic operation on it. 


```python
weight = float(weight_str)
height = float(height_str)
```

Now let's calculate the BMI


```python
BMI = weight/(height**2)
```


```python
print("The BMI of a person with {} kg and {} m is {:6.3f}".format(weight,height,BMI))
```

    The BMI of a person with 77.0 kg and 1.88 m is 21.786


Thank you for reading this post. If you like this post, please subscribe to stay updated with new posts and if you have a thought or a question, I would love to hear it by commenting below.