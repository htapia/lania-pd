# Dr. Horacio Tapia-McClung
## Postdoctoral Research @ LANIA funded by CONACyT (2014)
### Notebooks on scientific computing with CUDA (and Python)
============================================================

A set of IPython notebooks on how to use  CUDA with Python to solve some problems.
This repository is organised following a structure that I got from [Carl Boettiger's Lab Notebooks](khttp://carlboettiger.info/lab-notebook.html) with a basic set of directories:  `python_nb`, `src`, `man`, `data`, `inst/examples` and `inst/doc`, a metadata file `DESCRIPTION` and a `README.md` with this information and other description.

The fundamental ideas that I want people to take of this work are 1) it is possible to program a GPU quickly and with very little effort, thus it is possible to solve a large amount of problems in a very fast way, and 2) Python is a simple and efficient framework to solve these problems and provides a good platform for High Performance Computing also with very little effort.

To develop these ideas I created these notebooks using IPython to solve some problems of diverse fields, all these enclosed in the proposed work for the Postdoctoral Research Chronogram; the proposed activities can be enclosed in two general caterogies: Mathematical problems and Image analysis. The first one is very general so we have chosen two problems that are common in computational scientific applications: solving a differential equation and basic linear algebra. These two are not unrelated so we will try to approach them in a complete way, _i.e_ such that we see some closure at the end :)
The second category is also very broad and we have chosen to study basic applications only.

On the las part of these notebooks we also look at simulations of some physical processes, mainly in complex fluid dynamics, using software that has been developed by third parties, using the GPUs. We discuss on the difference between using the GPUs to obtain data from (simulated) physical processes and using the GPUs to analize the data.

Finally some thoughts on the 

1. Math problems: Differential equations and linear algebra; applications to physics
2. Graphics and image analysis.
3. 

# Instructions
To open these notebooks in IPython, download the files in the `python_nb` to a directory on your computer and from that directory run:

    $ ipython notebook

This will open a new page in your browser with a list of the available notebooks, or you can also see the online versions:

* [01_Libreta_1](http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/python_nb/01_Libreta_1.ipynb)
* [02_Libreta_2](http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/python_nb/02_Libreta_2.ipynb)
* [03_Libreta_3](http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/python_nb/03_Libreta_3.ipynb)
* [04_Libreta_4](http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/python_nb/04_Libreta_4.ipynb)
* [05_Libreta_5](http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/python_nb/05_Libreta_5.ipynb)
* [06_Libreta_6](http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/python_nb/06_Libreta_6.ipynb)
* [07_Libreta_7](http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/python_nb/07_Libreta_7.ipynb)
* [08_Libreta_8](http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/python_nb/08_Libreta_8.ipynb)
* [09_Libreta_9](http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/python_nb/09_Libreta_9.ipynb)

License
=======

This work is licensed under a [Creative Commons Attribution 3.0 Unported License.](http://creativecommons.org/licenses/by/3.0/)