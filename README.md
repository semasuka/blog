# Machine Intelligence Blog (MIB)

Machine Intelligence Blog (MIB) is your companion blog to get you up and running with machine learning and data science. On this blog, I will cover topics in the machine learning field ranging from maths, programming, algorithm optimization and data analysis.

## Getting Started

If you are interested in using or contribute to this blog, you will need some tools on your machine.

### Prerequisites

To run this blog on you locale machine, you will need to install [Jekyll](https://jekyllrb.com/"Jekyll"), kindly refer to the installation process on the official website of jekyll [here](https://jekyllrb.com/docs/installation/).

Sometimes it can be hard to know if jekyll is installed, to check if it is successfully installed run this command

```bash
    jekyll -v
```

if successfully installed you will get

```bash
    jekyll X.X.X
```

here the x.x.x is the version of jekyll, and if you get this error

```bash
bad interpreter: No such file or directory
```

 it means that jekyll is not installed and this is probably due to the fact one or more of the dependencies to install jekyll was not installed. Kindly refer to [this](https://jekyllrb.com/docs/troubleshooting/#installation-problems) manual to fix the issue.

### Run

After this, you will clone this repo(or download it) then using the command line, navigate to the root directory and input this for the first time

```bash
    bundle exec jekyll serve
```

to run jekyll after the first time, simply use

```bash
    jekyll serve
```

now you shall access to the blog at 127.0.0.1:4000 (or http://localhost:4000).

## Customize

In order to customize this blog as you wish, you can clone this repo and for the documentation please refer to the official skin github page [here](https://github.com/mmistakes/jekyll-theme-basically-basic).

## Deployment

To host your jekyll blog to github page, use [this](https://www.youtube.com/watch?v=fqFjuX4VZmU) tutorials.

## Built With

* [Jekyll](https://jekyllrb.com/"Jekyll") static site generator.
* [this](https://github.com/mmistakes/jekyll-theme-basically-basic) jekyll skin.
* [Disqus](https://disqus.com"disqus") as comment generator
* [Google analytics](https://analytics.google.com/analytics/web/"Google_analytics") as the site analytics
* [Algolia](https://www.algolia.com/"Algolia") as the search tool within the blog
* [Jekyll feed](https://github.com/jekyll/jekyll-feed) as the atom RSS generator
* [Jekyll seo](https://github.com/jekyll/jekyll-seo-tag) as the blog seo
* [mailchimp](https://mailchimp.com/"mailchimp") as mail newsletter subscriber service

## Authors

* [semasuka](https://github.com/semasuka"semasuka") - author and designer of the blog.

* [mmistakes](https://github.com/mmistakes"mmistakes") - initial jekyll skin creator.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/semasuka/blog/blob/master/LICENSE"LICENSE.md") file for details.