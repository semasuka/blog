---
title:  "Python, The programming language of Machine Learning"
image: /assets/post_images/python-snake.jpg
excerpt_separator: <!-- more -->
tags:
- python
- programming
- tutorial
---
When I say python most of folk will think that I am talking about the snake.No, ain't talking about the snake, I am talking about Python the programming language. Python is an open source high-level programming language(a programming language is a language computers can understand) created by Guido van Rossum and was first released to the public in 1991. Its main features are code readability by using whitespace instead of the curly braces, dynamic typing and automatic memory management. It also supports multiple programming paradigms including object-oriented, imperative, functional and procedural.<!-- more --> The name python comes from the fact that Rossum was influenced by “Monty Python’s Flying Circus”, a comedy series from the 1970s. he thought he needed a name that was short, unique, and slightly mysterious, so he decided to call the language Python.

### The most popular programming language

![fastest growing language](/blog/assets/post_cont_image/fastest-prg-lang.png)
picture credit: [stackoverflow](https://insights.stackoverflow.com/survey/2018)


As the time of writing this blog, Python is the fastest growing language and this trend is just starting thanks to its readability as we can read it from its core philosophy called zen of Python. You can see the entire list if you have python installed by running this command in the python interpreter.

```bash
import this
```

output
![zen of python](/blog/assets/post_cont_image/zen-py.png)

### Why is python popular in machine learning?

the answer is that its syntax is easy to understand and therefore making it an awesome language for prototyping and testing during an ML project. As most of the algorithms used in Machine Learning are a bit complex, we don't need to add another layer of complexity by worrying about a ";" that might cause the whole program not to run.

Another reason why Python wins is the fact that most of the ML frameworks are written in C++ with Python even though Python is not the most rapid programming language in terms of speed. To compensate for this, most of the core codes of the ML frameworks are written in C++ in order to take advantage of its speed and then wrapped with Python code for prototyping purposes.

I am hoping that now you understand why Python is the language to use when it comes to machine learning especially if you are a beginner.

### Installation

Let's install python but before we start, I have to mention one thing about Python, Python comes into two different versions Python 2 and Python 3. They are not compatible and there are some minor differences in syntax between the two. I suggest to always use version 3 since version 2 will be discontinued by 2020. I will only be using Python 3 on this blog.

Go to the official python organization [here](https:https://www.python.org/downloads/) and download the latest version of Python for your platform (windows, mac os or Linux)and install it.

![python download](/blog/assets/post_cont_image/py-web.png)

NB: while you are installing Python on Windows, remember to tick the box "Add Python 3 to PATH" in order to use it in the command line

![tick installation](/blog/assets/post_cont_image/tick-py.png)

Now go to your terminal/command line and type

After the installation, run the following command

```bash
python3
```

you should see the cursor changed to

```bash
>>>
```

### Hello world and basic arithmetics

We gonna print the classic "hello world" on the console using the `print` keyword like this

```bash
print("Hello world")
```

and this will output this

```bash
Hello world
```
the `print` keyword display anything in place in the bracket inside the ""

you can also make some basic operation like addition`+`, subtraction`-`, division`//`(only return whole numbers),division`/`(return float or whole numbers),multiplication`*`,exponential`**` and modulo`%` which is the remainder of the division

```bash
>>>15+2 //output: 17
>>>15-2 //output: 13
>>>15//3 //output: 5
>>>15*2 //output: 30
>>>15/3 //output: 5.0
>>>15**2 //output: 225
>>>15%3 //output: 0
```

This is a brief introduction to python and an overview of its installation. I will go in depth of Python in the upcoming posts. Stay tuned by subscribing!

Thank you for reading this post. If you like this post, please subscribe to stay updated with new posts and if you have a thought or a question, I would love to hear it by commenting below.